import numpy as np
import pytest

import senpy
from senpy import api as sp


def _make_accel(duration_s: float = 120.0, fs: float = 50.0) -> sp.AccelerometerData:
    rng = np.random.default_rng(42)
    n = int(duration_s * fs)
    t_s = np.arange(n, dtype=np.float64) / fs
    timestamps_us = (t_s * 1e6).astype(np.int64)
    x = rng.standard_normal(n).astype(np.float64) * 0.5
    y = rng.standard_normal(n).astype(np.float64) * 0.5
    z = 1.0 + rng.standard_normal(n).astype(np.float64) * 0.1
    return sp.AccelerometerData(timestamps_us=timestamps_us, x=x, y=y, z=z)


def test_stacked_spectrograms_default_five_channels():
    accel = _make_accel()
    result = sp.compute_stacked_spectrograms(accel, window_s=30.0, overlap_s=0.0)

    assert result.Sxx.ndim == 3
    T, F, C = result.Sxx.shape
    assert C == 5
    assert result.channels == ["x", "y", "z", "mag", "jerk"]
    assert T > 0
    assert F > 0


def test_stacked_spectrograms_shape_matches_single_axis():
    accel = _make_accel()
    stacked = sp.compute_stacked_spectrograms(
        accel, window_s=30.0, overlap_s=0.0, target_fs=16.0
    )
    single_x = sp.compute_nufft_spectrogram(
        timestamps=accel.timestamps_s,
        signal=accel.x,
        window_s=30.0,
        overlap_s=0.0,
        target_fs=16.0,
    )

    assert stacked.Sxx.shape[0] == single_x.Sxx.shape[0]
    assert stacked.Sxx.shape[1] == single_x.Sxx.shape[1]
    np.testing.assert_array_equal(stacked.frequencies, single_x.frequencies)
    np.testing.assert_array_equal(stacked.times, single_x.times)


def test_stacked_spectrograms_x_channel_matches_standalone():
    accel = _make_accel()
    stacked = sp.compute_stacked_spectrograms(
        accel, window_s=30.0, overlap_s=0.0, target_fs=16.0
    )
    x_idx = stacked.channels.index("x")
    single_x = sp.compute_nufft_spectrogram(
        timestamps=accel.timestamps_s,
        signal=accel.x,
        window_s=30.0,
        overlap_s=0.0,
        target_fs=16.0,
    )
    np.testing.assert_allclose(stacked.Sxx[:, :, x_idx], single_x.Sxx)


def test_stacked_spectrograms_mag_channel_matches_standalone():
    accel = _make_accel()
    mag = sp.compute_magnitude(accel.x, accel.y, accel.z)
    stacked = sp.compute_stacked_spectrograms(
        accel, window_s=30.0, overlap_s=0.0, target_fs=16.0
    )
    mag_idx = stacked.channels.index("mag")
    single_mag = sp.compute_nufft_spectrogram(
        timestamps=accel.timestamps_s,
        signal=mag,
        window_s=30.0,
        overlap_s=0.0,
        target_fs=16.0,
    )
    np.testing.assert_allclose(stacked.Sxx[:, :, mag_idx], single_mag.Sxx)


def test_stacked_spectrograms_subset_of_channels():
    accel = _make_accel()
    result = sp.compute_stacked_spectrograms(
        accel, window_s=30.0, overlap_s=0.0, channels=["x", "z", "jerk"]
    )
    assert result.channels == ["x", "z", "jerk"]
    assert result.Sxx.shape[2] == 3


def test_stacked_spectrograms_rejects_unknown_channel():
    accel = _make_accel()
    with pytest.raises(ValueError, match="Unknown channel"):
        sp.compute_stacked_spectrograms(accel, window_s=30.0, overlap_s=0.0, channels=["x", "bogus"])


def test_stacked_spectrograms_constant_channel_name():
    assert sp.STACKED_SPECTROGRAM_CHANNELS == ["x", "y", "z", "mag", "jerk"]
    assert senpy.STACKED_SPECTROGRAM_CHANNELS == ["x", "y", "z", "mag", "jerk"]
