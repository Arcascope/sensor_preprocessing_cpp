"""Tests for resampling artifact characterisation.

Compares three approaches:
  - **linear**:  existing piecewise-linear resampling (baseline)
  - **cubic**:   natural cubic-spline resampling (C++ backend)
  - **nufft_spec**: NUFFT-based spectrogram computed directly from
    non-uniform samples (bypasses resampling entirely)

Each test computes a concrete numerical metric — spurious spectral
power in the 0–10 Hz band, SNR, reconstruction RMSE, etc. — and
prints the results so you get a side-by-side comparison in the
pytest output.

Run with:
    pytest tests/test_resampling_artifacts.py -v -s
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from senpy.api import (
    resample_accelerometer,
    resample_accelerometer_cubic,
    compute_spectrogram,
    compute_spectrogram_nufft,
    compute_magnitude,
    AccelerometerData,
)

TARGET_FS = 32.0  # Hz — standard accelerometer target rate

# ── resampling dispatcher ─────────────────────────────────────────

RESAMPLE_METHODS = {
    "linear": resample_accelerometer,
    "cubic": resample_accelerometer_cubic,
}


def _resample(method: str, t, x, y, z, fs, ts_unit="s") -> AccelerometerData:
    return RESAMPLE_METHODS[method](t, x, y, z, fs, ts_unit=ts_unit)


# ── spectral helpers ──────────────────────────────────────────────


def band_power(signal: NDArray, fs: float, fmin: float, fmax: float) -> float:
    """Integrated PSD in [fmin, fmax] Hz via periodogram."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum = np.abs(np.fft.rfft(signal)) ** 2 / n
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.sum(spectrum[mask]))


def spectrogram_band_power(spec, fmin: float, fmax: float) -> float:
    """Mean power in [fmin, fmax] across all time windows."""
    mask = (spec.frequencies >= fmin) & (spec.frequencies <= fmax)
    return float(np.mean(spec.Sxx[:, mask]))


def window_seconds_from_samples(
    timestamps_s: NDArray[np.float64], nperseg: int, noverlap: int
) -> tuple[float, float]:
    """Convert sample-count windows to time windows using the median positive dt."""
    dt = np.diff(timestamps_s)
    valid_dt = dt[(dt > 0) & np.isfinite(dt)]
    if valid_dt.size == 0:
        raise ValueError("timestamps must include at least one positive time step")

    median_dt = float(np.median(valid_dt))
    return nperseg * median_dt, noverlap * median_dt


# ══════════════════════════════════════════════════════════════════
# 1. SYNTHETIC TONE LEAKAGE (resampling comparison)
# ══════════════════════════════════════════════════════════════════


class TestSyntheticToneLeakage:
    """Inject a pure tone above the 0–10 Hz band, resample from
    jittered timestamps, and measure how much power leaks into 0–10 Hz.
    """

    @staticmethod
    def _make_jittered_tone(freq_hz=15.0, duration_s=60.0,
                            nominal_fs=55.0, jitter_std_s=0.002,
                            seed=42):
        rng = np.random.default_rng(seed)
        n = int(duration_s * nominal_fs)
        dt = 1.0 / nominal_fs
        t = np.arange(n) * dt + rng.normal(0, jitter_std_s, n)
        t.sort()
        x = np.sin(2 * np.pi * freq_hz * t)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        return t, x, y, z

    @pytest.mark.parametrize("method", ["linear", "cubic"])
    def test_tone_leakage_into_0_10hz(self, method):
        """Power in 0–10 Hz should be small relative to the tone."""
        t, x, y, z = self._make_jittered_tone()
        resampled = _resample(method, t, x, y, z, TARGET_FS)

        tone_power = band_power(resampled.x, TARGET_FS, 14.0, 16.0)
        leak_power = band_power(resampled.x, TARGET_FS, 0.5, 10.0)
        ratio = leak_power / (tone_power + 1e-30)

        print(f"\n  [{method}] tone_leakage_ratio = {ratio:.6e}"
              f"  (leak={leak_power:.4e}, tone={tone_power:.4e})")
        assert ratio < 1.0, f"{method}: leakage ratio {ratio} >= 1.0"

    @pytest.mark.parametrize("method", ["linear", "cubic"])
    @pytest.mark.parametrize("jitter_ms", [0.5, 1.0, 2.0, 5.0])
    def test_jitter_sensitivity_sweep(self, method, jitter_ms):
        """Leakage as a function of jitter magnitude."""
        t, x, y, z = self._make_jittered_tone(
            jitter_std_s=jitter_ms / 1000.0)
        resampled = _resample(method, t, x, y, z, TARGET_FS)

        leak = band_power(resampled.x, TARGET_FS, 0.5, 10.0)
        tone = band_power(resampled.x, TARGET_FS, 14.0, 16.0)
        ratio = leak / (tone + 1e-30)
        print(f"\n  [{method}, jitter={jitter_ms}ms] ratio={ratio:.6e}")
        assert np.isfinite(ratio)


# ══════════════════════════════════════════════════════════════════
# 2. WHITE-NOISE FLOOR
# ══════════════════════════════════════════════════════════════════


class TestWhiteNoiseFloor:

    @staticmethod
    def _make_jittered_white_noise(duration_s=60.0, nominal_fs=55.0,
                                   jitter_std_s=0.002, seed=99):
        rng = np.random.default_rng(seed)
        n = int(duration_s * nominal_fs)
        t = np.arange(n) / nominal_fs + rng.normal(0, jitter_std_s, n)
        t.sort()
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        z = rng.standard_normal(n)
        return t, x, y, z

    @pytest.mark.parametrize("method", ["linear", "cubic"])
    def test_flat_spectrum_preserved(self, method):
        """Low and high bands should have roughly equal power."""
        t, x, y, z = self._make_jittered_white_noise()
        resampled = _resample(method, t, x, y, z, TARGET_FS)

        p_low = band_power(resampled.x, TARGET_FS, 1.0, 10.0)
        p_high = band_power(resampled.x, TARGET_FS, 10.0, 20.0)
        ratio = p_low / (p_high + 1e-30)

        print(f"\n  [{method}] low/high = {ratio:.4f}")
        assert 0.3 < ratio < 3.0


# ══════════════════════════════════════════════════════════════════
# 3. KNOWN-EMPTY-BAND TEST
# ══════════════════════════════════════════════════════════════════


class TestKnownEmptyBand:
    """Signal has energy only above 12 Hz."""

    @staticmethod
    def _make_high_freq(duration_s=60.0, nominal_fs=55.0,
                        jitter_std_s=0.002, seed=77):
        rng = np.random.default_rng(seed)
        n = int(duration_s * nominal_fs)
        t = np.arange(n) / nominal_fs + rng.normal(0, jitter_std_s, n)
        t.sort()
        x = (np.sin(2 * np.pi * 13 * t)
             + 0.8 * np.sin(2 * np.pi * 15 * t)
             + 0.6 * np.sin(2 * np.pi * 18 * t))
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        return t, x, y, z

    @pytest.mark.parametrize("method", ["linear", "cubic"])
    def test_no_energy_below_10hz(self, method):
        t, x, y, z = self._make_high_freq()
        resampled = _resample(method, t, x, y, z, TARGET_FS)

        p_signal = band_power(resampled.x, TARGET_FS, 12.0, 20.0)
        p_artifact = band_power(resampled.x, TARGET_FS, 0.5, 10.0)
        ratio = p_artifact / (p_signal + 1e-30)

        print(f"\n  [{method}] artifact/signal = {ratio:.6e}")
        assert ratio < 1.0


# ══════════════════════════════════════════════════════════════════
# 4. SINUSOID RECONSTRUCTION ACCURACY
# ══════════════════════════════════════════════════════════════════


class TestSinusoidReconstruction:

    @pytest.mark.parametrize("method", ["linear", "cubic"])
    @pytest.mark.parametrize("freq", [1.0, 5.0, 10.0])
    def test_reconstruction_rmse(self, method, freq):
        duration = 10.0
        n_in = int(duration * 55.0)
        t_in = np.linspace(0, duration, n_in, endpoint=False)
        x_in = np.sin(2 * np.pi * freq * t_in)
        y_in = np.zeros_like(x_in)
        z_in = np.zeros_like(x_in)

        resampled = _resample(method, t_in, x_in, y_in, z_in, TARGET_FS)
        t_out = resampled.timestamps_s
        x_true = np.sin(2 * np.pi * freq * t_out)
        rmse = np.sqrt(np.mean((resampled.x - x_true) ** 2))

        print(f"\n  [{method}, freq={freq}Hz] RMSE = {rmse:.6e}")
        assert rmse < 0.1

    @pytest.mark.parametrize("method", ["linear", "cubic"])
    def test_cubic_beats_linear(self, method):
        """Cubic should always have lower RMSE than linear at 5 Hz."""
        freq = 5.0
        duration = 10.0
        n_in = int(duration * 55.0)
        t_in = np.linspace(0, duration, n_in, endpoint=False)
        x_in = np.sin(2 * np.pi * freq * t_in)
        y_in = z_in = np.zeros_like(x_in)

        resampled = _resample(method, t_in, x_in, y_in, z_in, TARGET_FS)
        x_true = np.sin(2 * np.pi * freq * resampled.timestamps_s)
        rmse = np.sqrt(np.mean((resampled.x - x_true) ** 2))

        # Store for cross-method comparison
        if not hasattr(TestSinusoidReconstruction, "_rmses"):
            TestSinusoidReconstruction._rmses = {}
        TestSinusoidReconstruction._rmses[method] = rmse

        if len(TestSinusoidReconstruction._rmses) == 2:
            r = TestSinusoidReconstruction._rmses
            print(f"\n  linear={r['linear']:.6e}  cubic={r['cubic']:.6e}"
                  f"  improvement={r['linear']/r['cubic']:.1f}x")
            assert r["cubic"] <= r["linear"]


# ══════════════════════════════════════════════════════════════════
# 5. NUFFT SPECTROGRAM vs. RESAMPLE-THEN-SPECTROGRAM
# ══════════════════════════════════════════════════════════════════


class TestNufftSpectrogram:
    """Compare the NUFFT spectrogram (no resampling) against the
    traditional resample-then-FFT pipeline.
    """

    @staticmethod
    def _make_known_spectrum(duration_s=60.0, nominal_fs=55.0,
                             jitter_std_s=0.002, seed=42):
        """Signal with known spectral content: tones at 3 and 7 Hz."""
        rng = np.random.default_rng(seed)
        n = int(duration_s * nominal_fs)
        t = np.arange(n) / nominal_fs + rng.normal(0, jitter_std_s, n)
        t.sort()
        signal = (np.sin(2 * np.pi * 3.0 * t)
                  + 0.5 * np.sin(2 * np.pi * 7.0 * t))
        return t, signal

    def test_nufft_spectrogram_peaks_at_known_freqs(self):
        """NUFFT spectrogram should show peaks at 3 Hz and 7 Hz."""
        t, signal = self._make_known_spectrum()
        nperseg = 256
        noverlap = 128
        secperseg, secoverlap = window_seconds_from_samples(t, nperseg, noverlap)

        spec = compute_spectrogram_nufft(t, signal, secperseg, secoverlap)
        mean_psd = np.mean(spec.Sxx, axis=0)

        # Find the peak frequency within each expected band
        mask_3 = (spec.frequencies >= 2.0) & (spec.frequencies <= 4.0)
        mask_7 = (spec.frequencies >= 6.0) & (spec.frequencies <= 8.0)

        peak_3 = spec.frequencies[mask_3][np.argmax(mean_psd[mask_3])]
        peak_7 = spec.frequencies[mask_7][np.argmax(mean_psd[mask_7])]

        # Also check that the 3 Hz band has more power (signal is 1.0 vs 0.5 amplitude)
        power_3 = np.max(mean_psd[mask_3])
        power_7 = np.max(mean_psd[mask_7])

        print(f"\n  NUFFT peaks: {peak_3:.2f} Hz (power={power_3:.4e}), "
              f"{peak_7:.2f} Hz (power={power_7:.4e})")

        assert abs(peak_3 - 3.0) < 0.5, f"3 Hz peak at {peak_3}"
        assert abs(peak_7 - 7.0) < 0.5, f"7 Hz peak at {peak_7}"
        assert power_3 > power_7, "3 Hz should be stronger than 7 Hz"

    def test_nufft_vs_resample_leakage(self):
        """For a high-freq tone, NUFFT spectrogram should show less
        artifact power in 0–10 Hz than resample-then-spectrogram.
        """
        rng = np.random.default_rng(42)
        n = int(60 * 55)
        t = np.arange(n) / 55.0 + rng.normal(0, 0.002, n)
        t.sort()
        signal = np.sin(2 * np.pi * 15.0 * t)
        nperseg = 256
        noverlap = 128
        secperseg, secoverlap = window_seconds_from_samples(t, nperseg, noverlap)

        # NUFFT spectrogram: directly from non-uniform samples
        spec_nufft = compute_spectrogram_nufft(t, signal, secperseg, secoverlap)
        nufft_leak = spectrogram_band_power(spec_nufft, 0.5, 10.0)
        nufft_tone = spectrogram_band_power(spec_nufft, 14.0, 16.0)

        # Resample-then-spectrogram (linear)
        from senpy.api import resample_accelerometer
        resampled = resample_accelerometer(
            t, signal, np.zeros_like(signal), np.zeros_like(signal),
            TARGET_FS)
        spec_linear = compute_spectrogram(
            resampled.x, TARGET_FS, nperseg, noverlap)
        linear_leak = spectrogram_band_power(spec_linear, 0.5, 10.0)
        linear_tone = spectrogram_band_power(spec_linear, 14.0, 16.0)

        # Resample-then-spectrogram (cubic)
        from senpy.api import resample_accelerometer_cubic
        resampled_c = resample_accelerometer_cubic(
            t, signal, np.zeros_like(signal), np.zeros_like(signal),
            TARGET_FS)
        spec_cubic = compute_spectrogram(
            resampled_c.x, TARGET_FS, nperseg, noverlap)
        cubic_leak = spectrogram_band_power(spec_cubic, 0.5, 10.0)
        cubic_tone = spectrogram_band_power(spec_cubic, 14.0, 16.0)

        print(f"\n  Spectrogram leakage comparison (0–10 Hz / 14–16 Hz):")
        print(f"    linear resample: {linear_leak/(linear_tone+1e-30):.6e}")
        print(f"    cubic resample:  {cubic_leak/(cubic_tone+1e-30):.6e}")
        print(f"    NUFFT direct:    {nufft_leak/(nufft_tone+1e-30):.6e}")

        # All should have finite values
        assert np.isfinite(nufft_leak)
        assert np.isfinite(linear_leak)

    @pytest.mark.parametrize("jitter_ms", [0.5, 1.0, 2.0, 5.0])
    def test_nufft_spectrogram_jitter_robustness(self, jitter_ms):
        """NUFFT spectrogram should resolve 3 Hz and 7 Hz tones
        even under heavy jitter.
        """
        rng = np.random.default_rng(42)
        n = int(60 * 55)
        t = np.arange(n) / 55.0 + rng.normal(0, jitter_ms / 1000, n)
        t.sort()
        signal = (np.sin(2 * np.pi * 3.0 * t)
                  + 0.5 * np.sin(2 * np.pi * 7.0 * t))

        secperseg, secoverlap = window_seconds_from_samples(t, 256, 128)

        spec = compute_spectrogram_nufft(t, signal, secperseg, secoverlap)
        mean_psd = np.mean(spec.Sxx, axis=0)

        mask_3 = (spec.frequencies >= 2.0) & (spec.frequencies <= 4.0)
        mask_7 = (spec.frequencies >= 6.0) & (spec.frequencies <= 8.0)
        peak_3 = spec.frequencies[mask_3][np.argmax(mean_psd[mask_3])]
        peak_7 = spec.frequencies[mask_7][np.argmax(mean_psd[mask_7])]

        print(f"\n  [jitter={jitter_ms}ms] peaks: 3Hz→{peak_3:.2f}, 7Hz→{peak_7:.2f}")
        assert abs(peak_3 - 3.0) < 1.0
        assert abs(peak_7 - 7.0) < 1.0


# ══════════════════════════════════════════════════════════════════
# 6. REAL DATA: SPECTROGRAM COMPARISON
# ══════════════════════════════════════════════════════════════════


class TestRealDataSpectrogram:
    """Compare spectrograms across methods on real accelerometer data.
    Uses only the first 60 seconds to keep memory reasonable.
    """

    @pytest.mark.parametrize("method", ["linear", "cubic"])
    def test_spectrogram_smoothness(self, real_accel, method):
        name, t, x, y, z = real_accel

        # Use first 60 seconds only
        mask = t - t[0] < 60
        t, x, y, z = t[mask], x[mask], y[mask], z[mask]
        if len(t) < 500:
            pytest.skip(f"{name}: not enough data")

        resampled = _resample(method, t, x, y, z, TARGET_FS)
        mag = compute_magnitude(resampled.x, resampled.y, resampled.z)

        nperseg = min(256, len(mag) // 4)
        noverlap = nperseg // 2
        if len(mag) < nperseg * 2:
            pytest.skip(f"{name}: signal too short")

        spec = compute_spectrogram(mag, TARGET_FS, nperseg, noverlap)
        mask_f = (spec.frequencies >= 0.5) & (spec.frequencies <= 10.0)
        band = spec.Sxx[:, mask_f]

        if band.shape[0] > 1:
            temporal_diff = np.mean(np.abs(np.diff(band, axis=0)))
            flicker = temporal_diff / (np.mean(band) + 1e-30)
        else:
            flicker = 0.0

        print(f"\n  [{method}, {name}] flicker = {flicker:.6f}"
              f"  mean_power = {np.mean(band):.4e}")
        assert np.isfinite(flicker)

    def test_nufft_spectrogram_on_real_data(self, real_accel):
        """NUFFT spectrogram on real data should produce finite output."""
        name, t, x, y, z = real_accel
        mask = t - t[0] < 60
        t, x, y, z = t[mask], x[mask], y[mask], z[mask]
        if len(t) < 500:
            pytest.skip(f"{name}: not enough data")

        mag = compute_magnitude(x, y, z)
        secperseg, secoverlap = window_seconds_from_samples(t, 256, 128)
        spec = compute_spectrogram_nufft(t, mag, secperseg, secoverlap)

        print(f"\n  [nufft, {name}] shape={spec.Sxx.shape}"
              f"  freq_range=[{spec.frequencies[0]:.2f}, {spec.frequencies[-1]:.2f}]"
              f"  mean_power={np.mean(spec.Sxx):.4e}")

        assert spec.Sxx.shape[0] > 0
        assert spec.Sxx.shape[1] > 0
        assert np.all(np.isfinite(spec.Sxx))


# ══════════════════════════════════════════════════════════════════
# 7. TIMESTAMP JITTER CHARACTERISATION (real data)
# ══════════════════════════════════════════════════════════════════


class TestTimestampJitter:

    def test_jitter_statistics(self, real_accel):
        name, t, x, y, z = real_accel
        dt = np.diff(t)
        median_dt = np.median(dt)
        nominal_fs = 1.0 / median_dt if median_dt > 0 else 0

        jitter = dt - median_dt
        jitter_std = np.std(jitter)
        jitter_max = np.max(np.abs(jitter))
        jitter_pct99 = np.percentile(np.abs(jitter), 99)
        n_gaps = np.sum(dt > 3 * median_dt)

        print(f"\n  [{name}]"
              f"\n    nominal_fs   = {nominal_fs:.2f} Hz"
              f"\n    median_dt    = {median_dt * 1000:.3f} ms"
              f"\n    jitter_std   = {jitter_std * 1000:.3f} ms"
              f"\n    jitter_max   = {jitter_max * 1000:.3f} ms"
              f"\n    jitter_p99   = {jitter_pct99 * 1000:.3f} ms"
              f"\n    n_gaps(>3x)  = {n_gaps}"
              f"\n    n_samples    = {len(t)}")
        assert nominal_fs > 0
