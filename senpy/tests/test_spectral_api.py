import warnings

import numpy as np
import pytest

import senpy
from senpy import api as sp


def _jittered_tone(freq_hz=3.0, duration_s=60.0, nominal_fs=50.0):
    rng = np.random.default_rng(42)
    n = int(duration_s * nominal_fs)
    t = np.arange(n) / nominal_fs + rng.normal(0.0, 0.001, n)
    t.sort()
    signal = np.sin(2 * np.pi * freq_hz * t)
    return t, signal


def _nustft_direct_reference(timestamps, signal, window_s):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    dt_median = np.sort(diffs)[len(diffs) // 2]
    median_fs = 1.0 / dt_median

    nfft = int(window_s * median_fs)
    nfft_padded = 1
    while nfft_padded < nfft:
        nfft_padded <<= 1
    n_pos_freqs = nfft_padded // 2 + 1

    win_start = timestamps[0]
    in_window = (timestamps >= win_start) & (timestamps < win_start + window_s)
    t_win = timestamps[in_window]
    s_win = signal[in_window]
    tau = (t_win - win_start) / window_s
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau))
    windowed = (s_win - np.mean(s_win)) * hann
    scale = 1.0 / np.sqrt(median_fs * np.sum(hann * hann))

    k = np.arange(n_pos_freqs, dtype=np.float64)
    phase = np.exp(1j * 2.0 * np.pi * np.outer(k, tau))
    coefficients = phase @ windowed * scale
    frequencies = np.arange(n_pos_freqs, dtype=np.float64) / window_s
    return frequencies, coefficients


def _single_window_times(duration_s=8.0, fs=32.0, start_s=0.0):
    return start_s + np.arange(int(duration_s * fs), dtype=np.float64) / fs


def test_compute_nustft_returns_complex_time_major_coefficients():
    t, signal = _jittered_tone()

    result = sp.compute_nustft(t, signal, window_s=8.0, overlap_s=4.0, target_fs=16.0)

    assert result.axis_order == "time_frequency"
    assert result.coefficients.dtype == np.complex128
    assert result.coefficients.shape == (len(result.times), len(result.frequencies))
    assert result.coefficients.shape[0] > 0
    assert result.coefficients.shape[1] > 0


def test_nustft_dc_bin_matches_direct_windowed_sum_with_offset():
    window_s = 8.0
    t = _single_window_times(duration_s=window_s, fs=32.0)
    signal = 2.0 + 0.1 * t + np.cos(2.0 * np.pi * 1.25 * t + 0.4)

    result = sp.compute_nustft(t, signal, window_s=window_s, overlap_s=0.0)
    _, expected = _nustft_direct_reference(t, signal, window_s)

    np.testing.assert_allclose(
        result.coefficients[0, 0],
        expected[0],
        atol=1e-10,
        rtol=1e-10,
    )


def test_nustft_odd_bin_phase_matches_unshifted_direct_transform():
    window_s = 8.0
    t = _single_window_times(duration_s=window_s, fs=64.0)
    freq_hz = 9.0 / window_s
    signal = np.cos(2.0 * np.pi * freq_hz * t + 0.37)

    result = sp.compute_nustft(t, signal, window_s=window_s, overlap_s=0.0)
    _, expected = _nustft_direct_reference(t, signal, window_s)
    bin_idx = 9

    np.testing.assert_allclose(
        result.coefficients[0, bin_idx],
        expected[bin_idx],
        atol=1e-10,
        rtol=1e-10,
    )


def test_nustft_even_bin_phase_matches_unshifted_direct_transform():
    window_s = 8.0
    t = _single_window_times(duration_s=window_s, fs=64.0)
    freq_hz = 10.0 / window_s
    signal = np.cos(2.0 * np.pi * freq_hz * t - 0.82)

    result = sp.compute_nustft(t, signal, window_s=window_s, overlap_s=0.0)
    _, expected = _nustft_direct_reference(t, signal, window_s)
    bin_idx = 10

    np.testing.assert_allclose(
        result.coefficients[0, bin_idx],
        expected[bin_idx],
        atol=1e-10,
        rtol=1e-10,
    )


def test_nustft_magnitudes_match_direct_transform_for_all_bins():
    window_s = 8.0
    rng = np.random.default_rng(7)
    t = _single_window_times(duration_s=window_s + 0.1, fs=48.0)
    t = np.sort(t + rng.normal(0.0, 0.0007, size=t.shape))
    signal = (
        0.8 * np.cos(2.0 * np.pi * 1.125 * t + 0.2)
        + 0.3 * np.sin(2.0 * np.pi * 2.5 * t - 0.6)
        + 0.05 * t
    )

    result = sp.compute_nustft(t, signal, window_s=window_s, overlap_s=0.0)
    frequencies, expected = _nustft_direct_reference(t, signal, window_s)

    np.testing.assert_allclose(result.frequencies, frequencies)
    np.testing.assert_allclose(
        np.abs(result.coefficients[0]),
        np.abs(expected),
        atol=1e-10,
        rtol=1e-10,
    )


def test_nustft_zero_signal_returns_zero_coefficients_without_nans():
    window_s = 8.0
    t = _single_window_times(duration_s=window_s, fs=32.0)
    signal = np.zeros_like(t)

    result = sp.compute_nustft(t, signal, window_s=window_s, overlap_s=0.0)

    assert np.isfinite(result.coefficients.real).all()
    assert np.isfinite(result.coefficients.imag).all()
    np.testing.assert_allclose(result.coefficients, 0.0)


def test_nustft_timestamp_unit_conversion_preserves_coefficients_and_times():
    window_s = 8.0
    t = _single_window_times(duration_s=window_s, fs=32.0, start_s=12.0)
    signal = np.sin(2.0 * np.pi * 1.5 * t + 0.15)

    seconds = sp.compute_nustft(t, signal, window_s=window_s, overlap_s=0.0, ts_unit="s")
    milliseconds = sp.compute_nustft(
        t * 1000.0,
        signal,
        window_s=window_s,
        overlap_s=0.0,
        ts_unit="ms",
    )

    np.testing.assert_allclose(milliseconds.frequencies, seconds.frequencies)
    np.testing.assert_allclose(milliseconds.times, seconds.times)
    np.testing.assert_allclose(milliseconds.coefficients, seconds.coefficients)


def test_nustft_spectrogram_and_welch_are_derived_from_coefficients():
    t, signal = _jittered_tone()
    result = sp.compute_nustft(t, signal, window_s=8.0, overlap_s=4.0, target_fs=16.0)

    magnitude = result.spectrogram(kind="magnitude")
    power = result.spectrogram(kind="power")
    freqs, welch_psd = result.welch(kind="psd")

    np.testing.assert_allclose(magnitude.Sxx, np.abs(result.coefficients))
    np.testing.assert_allclose(power.Sxx, np.abs(result.coefficients) ** 2)
    np.testing.assert_allclose(freqs, result.frequencies)
    np.testing.assert_allclose(welch_psd, np.mean(result.psd, axis=0))


def test_nustft_accepts_mag_alias_and_power_kind():
    t, signal = _jittered_tone()
    result = sp.compute_nustft(t, signal, window_s=8.0, overlap_s=4.0, target_fs=16.0)

    mag = result.spectrogram(kind="mag")
    power = result.spectrogram(kind="power")

    assert mag.kind == "magnitude"
    np.testing.assert_allclose(mag.Sxx, np.abs(result.coefficients))
    np.testing.assert_allclose(power.Sxx, mag.Sxx ** 2)


def test_nustft_detrend_false_preserves_dc_offset():
    window_s = 8.0
    t = _single_window_times(duration_s=window_s, fs=32.0)
    signal = np.full_like(t, 3.0)

    detrended = sp.compute_nustft(t, signal, window_s=window_s, overlap_s=0.0, detrend=True)
    raw = sp.compute_nustft(t, signal, window_s=window_s, overlap_s=0.0, detrend=False)

    np.testing.assert_allclose(detrended.coefficients, 0.0, atol=1e-12)
    assert abs(raw.coefficients[0, 0]) > 1.0


def test_nufft_spectrogram_convenience_resolves_known_tone():
    t, signal = _jittered_tone(freq_hz=3.0)

    spec = sp.compute_nufft_spectrogram(
        t,
        signal,
        window_s=8.0,
        overlap_s=4.0,
        target_fs=16.0,
        kind="magnitude",
    )
    mean_spectrum = np.mean(spec.Sxx, axis=0)
    band = (spec.frequencies >= 2.0) & (spec.frequencies <= 4.0)
    peak = spec.frequencies[band][np.argmax(mean_spectrum[band])]

    assert spec.Sxx.shape == (len(spec.times), len(spec.frequencies))
    assert abs(peak - 3.0) < 0.5


def test_uniform_spectrogram_is_time_major():
    fs = 32.0
    t = np.arange(int(fs * 20.0)) / fs
    signal = np.sin(2 * np.pi * 3.0 * t)

    spec = sp.compute_uniform_spectrogram(signal, fs=fs, nperseg=128, noverlap=64)

    assert spec.axis_order == "time_frequency"
    assert spec.Sxx.shape == (len(spec.times), len(spec.frequencies))


def test_resampled_spectrogram_rejects_rounded_zero_hop():
    fs = 10.0
    t = np.arange(50) / fs
    signal = np.sin(2 * np.pi * 1.0 * t)

    with pytest.raises(ValueError, match="noverlap < nperseg"):
        sp.compute_resampled_spectrogram(
            t,
            signal,
            target_fs=fs,
            window_s=1.0,
            overlap_s=0.99,
        )


def test_nustft_rejects_inputs_shorter_than_one_window():
    t = np.array([0.0, 0.1, 0.2])
    signal = np.array([1.0, 0.0, -1.0])

    with pytest.raises(ValueError, match="at least one window"):
        sp.compute_nustft(t, signal, window_s=10.0, overlap_s=5.0, target_fs=16.0)


def test_compatibility_aliases_warn():
    t, signal = _jittered_tone()

    with pytest.warns(FutureWarning):
        old_nufft = sp.compute_spectrogram_nufft(
            t, signal, secperseg=8.0, secoverlap=4.0, target_fs=16.0
        )
    new_nufft = sp.compute_nufft_spectrogram(
        t, signal, window_s=8.0, overlap_s=4.0, target_fs=16.0, kind="magnitude"
    )
    np.testing.assert_allclose(old_nufft.Sxx, new_nufft.Sxx)

    with pytest.warns(FutureWarning):
        old_uniform = sp.compute_spectrogram(signal, fs=50.0, nperseg=128, noverlap=64)
    new_uniform = sp.compute_uniform_spectrogram(signal, fs=50.0, nperseg=128, noverlap=64)
    np.testing.assert_allclose(old_uniform.Sxx, new_uniform.Sxx)


def test_version_is_consolidated():
    assert senpy.__version__ == sp.__version__ == "1.0.0"
