"""Optional integration tests for the JAX NUFFT API."""

import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from senpy import jax_backend as senpy_jax


@pytest.mark.parametrize(
    ("output_size", "iflag", "modeord"),
    [(8, 1, 0), (7, -1, 0), (8, 1, 1)],
)
def test_gaussian_type1_nufft_matches_definition(output_size, iflag, modeord):
    """The gridding NUFFT matches the type-1 sum for both signs and orderings."""
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


def _reference_type1(source, points, output_size, iflag, modeord=0):
    import numpy as np

    if modeord == 0:
        modes = np.arange(-(output_size // 2), (output_size + 1) // 2)
    else:
        modes = np.concatenate(
            (np.arange(0, (output_size + 1) // 2), np.arange(-(output_size // 2), 0))
        )
    return source @ np.exp(1j * iflag * np.outer(points, modes))


def test_nufft1_accepts_a_bare_transform_and_keeps_its_rank():
    """source [point] with points [point] -> [mode], the single-signal case."""
    import numpy as np

    rng = np.random.default_rng(7)
    points = rng.uniform(-np.pi, np.pi, size=11)
    source = rng.standard_normal(11) + 1j * rng.standard_normal(11)

    actual = senpy_jax.nufft1(8, jnp.asarray(source), jnp.asarray(points), eps=1e-6)

    assert actual.shape == (8,)
    np.testing.assert_allclose(
        np.asarray(actual), _reference_type1(source, points, 8, 1), rtol=2e-5, atol=2e-5
    )


def test_nufft1_evaluates_a_transform_stack_sharing_one_point_set():
    """source [transform, point] -> [transform, mode]: the 3 accel channels."""
    import numpy as np

    rng = np.random.default_rng(8)
    points = rng.uniform(-np.pi, np.pi, size=11)
    source = rng.standard_normal((3, 11)) + 1j * rng.standard_normal((3, 11))

    actual = senpy_jax.nufft1(8, jnp.asarray(source), jnp.asarray(points), eps=1e-6)

    assert actual.shape == (3, 8)
    np.testing.assert_allclose(
        np.asarray(actual), _reference_type1(source, points, 8, 1), rtol=2e-5, atol=2e-5
    )


def test_nufft1_is_traceable_by_jit_and_vmap():
    """It is an ordinary JAX function, so transforms apply without a rule."""
    import numpy as np

    rng = np.random.default_rng(9)
    points = jnp.asarray(rng.uniform(-np.pi, np.pi, size=(4, 11)))
    source = jnp.asarray(
        rng.standard_normal((4, 3, 11)) + 1j * rng.standard_normal((4, 3, 11))
    )
    one = lambda p, s: senpy_jax.nufft1(8, s, p, eps=1e-6)

    batched = jax.vmap(one, in_axes=(0, 0))(points, source)
    jitted = jax.jit(jax.vmap(one, in_axes=(0, 0)))(points, source)

    assert batched.shape == (4, 3, 8)
    np.testing.assert_allclose(np.asarray(batched), np.asarray(jitted), rtol=1e-6, atol=1e-6)
    # vmapping the wrapper agrees with passing the problem axis directly.
    direct = senpy_jax.nufft1(8, source, points, eps=1e-6)
    np.testing.assert_allclose(np.asarray(batched), np.asarray(direct), rtol=2e-5, atol=2e-5)


def test_nufft1_rejects_mismatched_point_axes():
    with pytest.raises(ValueError, match="point axis"):
        senpy_jax.nufft1(8, jnp.zeros((3, 10), dtype=jnp.complex64), jnp.zeros(11))
