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


def test_compute_nustft_returns_complex_time_major_coefficients():
    t, signal = _jittered_tone()

    result = sp.compute_nustft(t, signal, window_s=8.0, overlap_s=4.0, target_fs=16.0)

    assert result.axis_order == "time_frequency"
    assert result.coefficients.dtype == np.complex128
    assert result.coefficients.shape == (len(result.times), len(result.frequencies))
    assert result.coefficients.shape[0] > 0
    assert result.coefficients.shape[1] > 0


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
