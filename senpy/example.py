import numpy as np
import senpy.api as sp


n_samples = 2000
rng = np.random.default_rng(123)
timestamps_s = np.arange(n_samples) / 50.0 + rng.normal(0.0, 0.001, n_samples)
timestamps_s.sort()
x = np.sin(2 * np.pi * 1.2 * timestamps_s)
y = 0.2 * rng.standard_normal(n_samples)
z = np.ones(n_samples)

jerk = sp.compute_jerk(timestamps_s, x, y, z, ts_unit="s")
print(f"Jerk samples: {len(jerk)}")

nustft = sp.compute_nustft(
    jerk.timestamps_s,
    jerk.jerk,
    window_s=10.0,
    overlap_s=5.0,
    target_fs=16.0,
)
print(f"NUSTFT coefficients: {nustft.coefficients.shape}")
print(f"NUSTFT frequency range: {nustft.frequencies[0]:.2f}-{nustft.frequencies[-1]:.2f} Hz")

spec = nustft.spectrogram(kind="magnitude")
print(f"NUFFT spectrogram: {spec.Sxx.shape} ({spec.kind})")

freqs, psd = nustft.welch(kind="psd")
print(f"Welch-style spectrum: {freqs.shape}, {psd.shape}")

resampled = sp.resample_accelerometer(timestamps_s, x, y, z, target_fs=50.0)
uniform_spec = sp.compute_uniform_spectrogram(
    resampled.x,
    fs=50.0,
    nperseg=256,
    noverlap=128,
)
print(f"Uniform FFT spectrogram: {uniform_spec.Sxx.shape}")
