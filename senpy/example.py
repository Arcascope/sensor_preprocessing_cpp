import numpy as np
import senpy as sp

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
print(f"Resampled samples: {len(resampled['timestamps'])}")
print(f"Resampled x shape: {resampled['x'].shape}")

# Example 2: Compute jerk
jerk_result = sp.compute_jerk(
    resampled['timestamps'], 
    resampled['x'], 
    resampled['y'], 
    resampled['z']
)
print(f"\nJerk signal shape: {jerk_result['jerk'].shape}")
print(f"Jerk mean: {jerk_result['jerk'].mean():.6f}")

# Example 3: Compute magnitude
magnitude = sp.compute_magnitude(resampled['x'], resampled['y'], resampled['z'])
print(f"\nMagnitude shape: {magnitude.shape}")
print(f"Magnitude mean: {magnitude.mean():.6f}")

# Example 4: Compute spectrogram
fs = 50.0  # Sampling frequency
nperseg = 256  # Window size
noverlap = 128  # Overlap

spec = sp.compute_spectrogram(jerk_result['jerk'], fs, nperseg, noverlap)
print(f"\nSpectrogram frequencies: {spec['freqs'].shape}")
print(f"Spectrogram times: {spec['times'].shape}")
print(f"Spectrogram Sxx: {spec['Sxx'].shape}")
print(f"Frequency range: {spec['freqs'][0]:.2f} - {spec['freqs'][-1]:.2f} Hz")

# Example 5: Compute motion features
# For this example, let's create a longer signal
long_jerk = np.random.randn(10000) * 0.5 + 1.0

features = sp.compute_motion_features(
    long_jerk,
    fs=50.0,
    windowSize=1500,  # 30 seconds at 50 Hz
    overlap=750,       # 50% overlap
    brMinHz=0.15,      # 9 BPM
    brMaxHz=0.417,     # 25 BPM
    htMinHz=0.5,       # 30 BPM
    htMaxHz=2.0,       # 120 BPM
)

print(f"\nMotion Features:")
print(f"  Breathing Rate (BR): {features['BR'].shape}")
print(f"  Heart Rate (HR): {features['HR'].shape}")
print(f"  Frequency Sum: {features['freqSum'].shape}")
print(f"  BR std: {features['BR_std'].shape}")
print(f"  HR std: {features['HR_std'].shape}")
print(f"  Mean HR: {features['HR'].mean():.1f} BPM")
print(f"  Mean BR: {features['BR'].mean():.1f} BPM")

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

# Rolling std
rolling = sp.rolling_std(jerk_result['jerk'], window_minutes=1.0, seconds_per_window=0.02)
print(f"Rolling std shape: {rolling.shape}")

# Statistical functions
median = sp.compute_median(magnitude)
p90 = sp.compute_percentile(magnitude, 90.0)
print(f"\nMedian: {median:.6f}")
print(f"90th percentile: {p90:.6f}")

# Next power of 2
n = sp.next_power_of_2(1000)
print(f"Next power of 2 after 1000: {n}")