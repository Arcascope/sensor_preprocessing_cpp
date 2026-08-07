"""Tests for host-side packing of the optional JAX NUFFT batch interface."""

import numpy as np
import pytest

from senpy import jax as senpy_jax
from senpy.jax import pack_nustft_window_batches


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


def _fake_nufft1(nfft, strengths, points, *, eps, iflag):
    del points, eps, iflag
    return np.repeat(np.sum(strengths, axis=1, keepdims=True), nfft, axis=1)


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
    senpy_jax._nustft_window_batch_transform.cache_clear()
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
