import numpy as np
import senpy.api as sp

# Example 1: Resample accelerometer data
# Generate sample data (timestamps in microseconds)
n_samples = 1000
timestamps = np.arange(n_samples) * 20000  # 50 Hz sampling (20ms intervals)
x = np.random.randn(n_samples) * 0.1
y = np.random.randn(n_samples) * 0.1
z = np.ones(n_samples) + np.random.randn(n_samples) * 0.1

# Resample to 50 Hz
target_fs = 50.0
resampled = sp.resample_accelerometer(timestamps, x, y, z, target_fs)
print(f"Original samples: {len(timestamps)}")
print(f"Resampled samples: {len(resampled._timestamps_us)}")
print(f"Resampled x shape: {resampled.x.shape}")

# Example 2: Compute jerk
time, jerk = sp.compute_jerk(
    resampled._timestamps_us, resampled.x, resampled.y, resampled.z
)
print(f"\nJerk signal shape: {jerk.shape}")
print(f"Jerk mean: {jerk.mean():.6f}")

# Example 3: Compute magnitude
magnitude = sp.compute_magnitude(resampled.x, resampled.y, resampled.z)
print(f"\nMagnitude shape: {magnitude.shape}")
print(f"Magnitude mean: {magnitude.mean():.6f}")

# Example 4: Compute spectrogram
fs = 50.0  # Sampling frequency
nperseg = 256  # Window size
noverlap = 128  # Overlap

spec = sp.compute_spectrogram(jerk, fs, nperseg, noverlap)
print(f"\nSpectrogram frequencies: {spec.frequencies.shape}")
print(f"Spectrogram times: {spec.times.shape}")
print(f"Spectrogram Sxx: {spec.Sxx.shape}")
print(f"Frequency range: {spec.frequencies[0]:.2f} - {spec.frequencies[-1]:.2f} Hz")

# Example 5: Compute motion features
# For this example, let's create a longer signal
long_jerk = np.random.randn(10000) * 0.5 + 1.0

features = sp.compute_motion_features(
    long_jerk,
    fs=50.0,
    window_size=1500,  # 30 seconds at 50 Hz
    overlap=750,  # 50% overlap
)

print(f"\nMotion Features:")
print(f"  Breathing Rate (BR): {features.breathing_rate.shape}")
print(f"  Heart Rate (HR): {features.heart_rate.shape}")
print(f"  Frequency Sum: {features.frequency_sum.shape}")
print(f"  BR std: {features.breathing_rate_std.shape}")
print(f"  HR std: {features.heart_rate_std.shape}")
print(f"  Mean HR: {features.heart_rate.mean():.1f} BPM")
print(f"  Mean BR: {features.breathing_rate.mean():.1f} BPM")

# Example 6: Utility functions
# Hann window
window = sp.hann_window(256)
print(f"\nHann window shape: {window.shape}")

# Gaussian filter
signal_to_filter = np.random.randn(1000)
filtered = sp.gaussian_filter_1d(signal_to_filter, sigma=2.0)
print(f"Filtered signal shape: {filtered.shape}")

# Find peaks
peaks = sp.find_peaks(magnitude, prominence_threshold=0.1)
print(f"Number of peaks found: {len(peaks)}")
