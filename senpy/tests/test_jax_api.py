"""Optional integration tests for the JAX NUFFT API."""

import pytest


jax = pytest.importorskip("jax")
pytest.importorskip("jax_finufft")
jnp = pytest.importorskip("jax.numpy")

from senpy import jax_backend as senpy_jax


@pytest.mark.parametrize(
    ("output_size", "iflag", "modeord"),
    [(8, 1, 0), (7, -1, 0), (8, 1, 1)],
)
def test_gaussian_type1_nufft_matches_definition(output_size, iflag, modeord):
    """The gridding NUFFT preserves jax-finufft's signs and mode ordering."""
    import numpy as np

    rng = np.random.default_rng(20260921)
    points = rng.uniform(-np.pi, np.pi, size=(2, 7)).astype(np.float32)
    source = (
        rng.standard_normal((2, 3, 7)) + 1j * rng.standard_normal((2, 3, 7))
    ).astype(np.complex64)

    actual = senpy_jax._gaussian_type1_nufft(
        jnp,
        jax,
        jnp.asarray(source),
        jnp.asarray(points),
        output_size=output_size,
        iflag=iflag,
        modeord=modeord,
        eps=1e-6,
    )
    if modeord == 0:
        modes = np.arange(-(output_size // 2), (output_size + 1) // 2)
    else:
        modes = np.concatenate(
            (
                np.arange(0, (output_size + 1) // 2),
                np.arange(-(output_size // 2), 0),
            )
        )
    expected = np.einsum(
        "btm,bmk->btk",
        source,
        np.exp(1j * iflag * points[..., None] * modes),
    )

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=3e-5, atol=3e-5)


def test_compute_nustft_returns_device_arrays_with_known_tone():
    timestamps = jnp.arange(400, dtype=jnp.float32) / 50.0
    signal = jnp.sin(2.0 * jnp.pi * 3.0 * timestamps)

    result = senpy_jax.compute_nustft(
        timestamps,
        signal,
        window_s=4.0,
        overlap_s=2.0,
        target_fs=16.0,
    )

    assert result.axis_order == "time_frequency"
    assert result.coefficients.shape == (3, 33)
    assert result.frequencies.shape == (33,)
    assert result.times.shape == (3,)
    peak = result.frequencies[jnp.argmax(jnp.mean(result.power, axis=0))]
    assert float(peak) == pytest.approx(3.0)


def test_compute_nufft_spectrogram_and_welch_stay_jax_backed():
    timestamps = jnp.arange(400, dtype=jnp.float32) / 50.0
    signal = jnp.sin(2.0 * jnp.pi * 3.0 * timestamps)

    spectrogram = senpy_jax.compute_nufft_spectrogram(
        timestamps, signal, window_s=4.0, overlap_s=2.0, kind="power"
    )
    frequencies, welch = senpy_jax.compute_nufft_welch(
        timestamps, signal, window_s=4.0, overlap_s=2.0
    )

    assert spectrogram.Sxx.shape == (3, 129)
    assert frequencies.shape == welch.shape == (129,)
