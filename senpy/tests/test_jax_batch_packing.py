"""Tests for host-side packing of the optional JAX NUFFT batch interface."""

import numpy as np
import pytest

from senpy import jax_backend as senpy_jax
from senpy.jax_backend import pack_nustft_window_batches


class _FakeJax:
    """Enough JAX surface to exercise masking without optional GPU packages."""

    @staticmethod
    def jit(function):
        return function

    @staticmethod
    def vmap(function, in_axes):
        def mapped(first, second):
            return np.stack([function(a, b) for a, b in zip(first, second)])

        return mapped


def _fake_nufft1(nfft, strengths, points, *, eps, iflag, opts=None):
    del points, eps, iflag, opts
    return np.repeat(np.sum(strengths, axis=1, keepdims=True), nfft, axis=1)


@pytest.fixture(autouse=True)
def _clear_transform_cache():
    """Keep NumPy-backed fakes out of the module-level transform cache.

    ``_nustft_window_batch_transform`` is an unbounded ``lru_cache`` keyed only
    on ``(nfft_padded, detrend, eps)``. A closure built while ``_dependencies``
    is monkeypatched would otherwise outlive the patch and be handed to any
    later test that happens to use the same key.
    """
    senpy_jax._nustft_window_batch_transform.cache_clear()
    yield
    senpy_jax._nustft_window_batch_transform.cache_clear()


def test_packer_makes_static_three_axis_batches_and_marks_final_padding():
    timestamps = np.arange(10, dtype=np.float64) / 2.0
    first = np.column_stack((timestamps, timestamps + 1.0, timestamps + 2.0))
    second = np.column_stack((2.0 * timestamps, 3.0 * timestamps, 4.0 * timestamps))

    batches = pack_nustft_window_batches(
        [(timestamps, first), (timestamps, second)],
        window_s=2.0,
        overlap_s=1.0,
        batch_size=8,
    )

    assert len(batches) == 1
    batch = batches[0]
    assert batch.points.shape == batch.valid.shape == (8, 4)
    assert batch.signals.shape == (8, 3, 4)
    assert batch.nfft_padded == 4
    assert batch.row_valid.tolist() == [True] * 8
    assert batch.recording_indices.tolist() == [0] * 4 + [1] * 4
    assert batch.window_indices.tolist() == [0, 1, 2, 3] * 2
    assert np.all(batch.window_ss[batch.row_valid] > 0.0)


def test_packer_zero_fills_masked_samples_and_final_rows():
    timestamps = np.arange(7, dtype=np.float64) / 2.0
    samples = np.column_stack((timestamps, timestamps + 1.0, timestamps + 2.0))

    batches = pack_nustft_window_batches(
        [(timestamps, samples)],
        window_s=2.0,
        overlap_s=1.0,
        batch_size=4,
    )

    batch = batches[0]
    assert batch.row_valid.tolist() == [True, True, False, False]
    assert batch.valid[0].tolist() == [True, True, True, True]
    assert not batch.valid[-1].any()
    assert np.all(batch.points[-1] == 0.0)
    assert np.all(batch.signals[-1] == 0.0)
    assert batch.recording_indices[-1] == -1
    assert batch.window_indices[-1] == -1
    assert np.isnan(batch.times[-1])
    assert batch.window_ss[-1] == 0.0


def test_packer_rejects_non_three_axis_or_unsorted_recordings():
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="samples\\[N, 3\\]"):
        pack_nustft_window_batches(
            [(timestamps, np.ones((4, 2)))], window_s=2.0, overlap_s=1.0, batch_size=2
        )
    with pytest.raises(ValueError, match="sorted"):
        pack_nustft_window_batches(
            [(timestamps[[0, 2, 1, 3]], np.ones((4, 3)))],
            window_s=2.0,
            overlap_s=1.0,
            batch_size=2,
        )


def test_window_batch_masks_padded_samples_and_rows_without_optional_jax(monkeypatch):
    monkeypatch.setattr(
        senpy_jax,
        "_dependencies",
        lambda: (_FakeJax(), np, _fake_nufft1),
    )
    points = np.array([[-np.pi, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    valid = np.array([[True, True, False, False], [False, False, False, False]])
    signals = np.array(
        [
            [[1.0, 3.0, 1e6, 1e6], [2.0, 6.0, 1e6, 1e6], [3.0, 9.0, 1e6, 1e6]],
            [[1e6, 1e6, 1e6, 1e6], [1e6, 1e6, 1e6, 1e6], [1e6, 1e6, 1e6, 1e6]],
        ]
    )

    coefficients = senpy_jax.compute_nustft_window_batch(
        points, signals, valid, nfft_padded=4, median_fs=np.array([2.0, 1.0])
    )

    assert coefficients.shape == (2, 3, 3)
    assert np.all(np.isfinite(coefficients[0]))
    assert np.all(coefficients[1] == 0.0)


def test_low_energy_window_keeps_its_true_hann_scale(monkeypatch):
    """A sparse window near a taper edge must not be silently renormalized.

    ``sum(hann^2)`` falls below 1.0 whenever the surviving samples sit close to
    a window edge, where the taper goes to zero quadratically. The scale factor
    is ``1/sqrt(fs * sum(hann^2))`` in both CPU backends, with no floor, so the
    batched path must not clamp the divisor either.
    """
    monkeypatch.setattr(
        senpy_jax,
        "_dependencies",
        lambda: (_FakeJax(), np, _fake_nufft1),
    )
    # Four samples in the first 0.4% of the window, plus an all-padding row.
    tau = np.array([0.001, 0.002, 0.003, 0.004])
    points = np.vstack((2.0 * np.pi * tau - np.pi, np.zeros(4)))
    valid = np.array([[True] * 4, [False] * 4])
    signals = np.ones((2, 3, 4))
    median_fs = np.array([50.0, 1.0])

    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau))
    window_ss = float(np.sum(hann * hann))
    assert window_ss < 1.0, "the clamp would be inert if the taper energy exceeded 1"

    coefficients = senpy_jax.compute_nustft_window_batch(
        points, signals, valid, nfft_padded=8, median_fs=median_fs, detrend=False
    )

    # _fake_nufft1 sums the strengths into every mode, so with unit samples and
    # no detrending each coefficient is sum(hann) * phase_correction * scale.
    phase_correction = np.where(np.arange(5) % 2 == 0, 1.0, -1.0)
    expected = np.sum(hann) * phase_correction / np.sqrt(50.0 * window_ss)

    assert coefficients.shape == (2, 3, 5)
    for channel in range(3):
        np.testing.assert_allclose(coefficients[0, channel], expected, rtol=1e-12)
    # The padded row still resolves to exactly zero rather than inf/NaN.
    assert np.all(coefficients[1] == 0.0)
    assert np.all(np.isfinite(coefficients))
