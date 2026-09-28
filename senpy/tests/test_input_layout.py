"""Strided and non-float64 inputs must give the same results as contiguous float64 ones.

The C++ wrappers read their buffers densely, so a column view of an [N, C] array -- the
natural way to pass one channel -- used to be read with the wrong stride.
"""

import numpy as np
import pytest

import senpy

FS = 32.0


@pytest.fixture
def columns():
    rng = np.random.default_rng(4)
    t = np.arange(0.0, 120.0, 1.0 / FS)
    block = np.column_stack([t, rng.normal(size=t.size), rng.normal(size=t.size), rng.normal(size=t.size)])
    return block  # [N, 4]; block[:, k] is a strided view


@pytest.mark.parametrize(
    "call",
    [
        lambda t, x: senpy.compute_nustft(t, x, 10.0, 8.0, empty_windows="keep").coefficients,
        lambda t, x: senpy.compute_nufft_spectrogram(t, x, 10.0, 8.0, empty_windows="keep").Sxx,
        lambda t, x: senpy.compute_nustft_streaming(
            t, x, 10.0, 8.0, subwindow_s=2.0, empty_windows="keep"
        ).coefficients,
        lambda t, x: senpy.compute_magnitude(x, x, x),
        lambda t, x: senpy.compute_jerk(t, x, x, x).jerk,
    ],
)
def test_a_column_view_gives_the_same_result_as_a_copy(columns, call):
    t, x = columns[:, 0], columns[:, 1]
    assert not x.flags.c_contiguous

    np.testing.assert_array_equal(call(t, x), call(np.ascontiguousarray(t), np.ascontiguousarray(x)))


def test_float32_signal_is_converted_not_misread(columns):
    t, x = columns[:, 0], columns[:, 1]
    got = senpy.compute_nustft(t, x.astype(np.float32), 10.0, 8.0, empty_windows="keep")
    want = senpy.compute_nustft(t, x.astype(np.float32).astype(np.float64), 10.0, 8.0, empty_windows="keep")

    np.testing.assert_array_equal(got.coefficients, want.coefficients)
