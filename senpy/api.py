"""Defines a Python API for easier development with the library.

This module is an interface. The Python bindings already exist, but one must read the C++ code to understand how to use them. This module wraps the C++ bindings in a more Pythonic way.
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from numpy.typing import NDArray

# Import the C++ module
import senpy._core as _senpy


# Add a simple test function to verify imports work
def test_import():
    """Test function to verify the C++ module is loaded."""
    print("API module loaded successfully!")
    print(
        "Available functions:",
        [attr for attr in dir(_senpy) if not attr.startswith("_")],
    )
    return True


class AccelerometerData:
    """Container for accelerometer data with timestamps."""

    def __init__(
        self,
        timestamps_us: NDArray[np.int64],
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
    ):
        self.timestamps_us = timestamps_us
        self.x = x
        self.y = y
        self.z = z

    @property
    def timestamps_s(self) -> NDArray[np.float64]:
        """Return timestamps in seconds."""
        return self.timestamps_us.astype(np.float64) / 1e6

    @property
    def shape(self) -> Tuple[int]:
        """Number of samples in the data."""
        return self.timestamps_us.shape

    def __len__(self) -> int:
        return len(self.timestamps_us)
    
    def to_xyz_array(self) -> NDArray[np.float64]:
        """Return accelerometer data as an (N, 3) array."""
        return np.column_stack((self.x, self.y, self.z))
    
    def to_txyz_array(self) -> NDArray[Union[np.int64, np.float64]]:
        """Return accelerometer data as an (N, 4) array with timestamps."""
        return np.column_stack((self.timestamps_us, self.x, self.y, self.z))
    
    def to_pandas(self) -> "pd.DataFrame":
        """Return accelerometer data as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            {
                "timestamp_us": self.timestamps_us,
                "x": self.x,
                "y": self.y,
                "z": self.z,
            }
        )


class JerkData:
    """Container for jerk data with timestamps."""

    def __init__(
        self,
        timestamps_us: NDArray[np.int64],
        jerk: NDArray[np.float64],
    ):
        self.timestamps_us = timestamps_us
        self.jerk = jerk

    @property
    def timestamps_s(self) -> NDArray[np.float64]:
        """Return timestamps in seconds."""
        return self.timestamps_us.astype(np.float64) / 1e6

    @property
    def shape(self) -> Tuple[int]:
        """Number of samples in the data."""
        return self.timestamps_us.shape

    def __len__(self) -> int:
        return len(self.timestamps_us)


class SpectrogramResult:
    """Container for spectrogram computation results.

    Attributes:
        freqs: Array of frequency bins in Hz
        times: Array of time bins in seconds
        Sxx:  (time, frequency) array of power spectral density
    """

    def __init__(
        self,
        frequencies: NDArray[np.float64],
        times: NDArray[np.float64],
        Sxx: NDArray[np.float64],
    ):
        self.frequencies = frequencies
        self.times = times
        self.Sxx = Sxx

    @property
    def frequency_resolution(self) -> float:
        """Frequency resolution in Hz."""
        return (
            self.frequencies[1] - self.frequencies[0]
            if len(self.frequencies) > 1
            else 0.0
        )

    @property
    def time_resolution(self) -> float:
        """Time resolution in seconds."""
        return self.times[1] - self.times[0] if len(self.times) > 1 else 0.0
    
    def find_peaks(
        self,
        prominence_threshold: float,
        relative_prominence: bool = True,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        scaling_factor: float = 60.0,
    ) -> NDArray[np.int32]:
        """Find peaks in each time slice of the spectrogram.

        Args:
            prominence_threshold: Minimum prominence required for peak detection

        Returns:
            List of arrays, each containing peak indices for corresponding time slice
        """
        freq_search = np.ones_like(self.frequencies, dtype=bool)
        if freq_min is not None:
            freq_search &= self.frequencies >= freq_min
        if freq_max is not None:
            freq_search &= self.frequencies <= freq_max
        
        if relative_prominence:
            # Scale prominence threshold based on max and min values in the spectrogram
            max_val = np.max(self.Sxx[:, freq_search])
            min_val = np.min(self.Sxx[:, freq_search])
            prominence_threshold = prominence_threshold * (max_val - min_val)

        search_frequencies = self.frequencies[freq_search]
        search_spectrogram = self.Sxx[:, freq_search]
        peaks = find_spectrogram_peaks(
            Sxx=search_spectrogram,
            prominence_threshold=prominence_threshold,
            frequencies=search_frequencies,
            scaling_factor=scaling_factor
        )

        return peaks

class ShortTimeFTResult:
    """Container for Short-Time Fourier Transform results.

    Attributes:
        stft: Complex STFT array shaped (n_times, n_frequencies, 2) where
              [:, :, 0] contains real parts and [:, :, 1] contains imaginary parts
        freqs: Array of frequency bins in Hz
        times: Array of time bins in seconds
    """

    def __init__(
        self,
        stft: NDArray[np.float64],
        frequencies: NDArray[np.float64],
        times: NDArray[np.float64],
    ):
        self.stft = stft
        self.frequencies = frequencies
        self.times = times

    @property
    def real(self) -> NDArray[np.float64]:
        """Real part of the STFT."""
        return self.stft[:, :, 0]

    @property
    def imag(self) -> NDArray[np.float64]:
        """Imaginary part of the STFT."""
        return self.stft[:, :, 1]

    @property
    def complex(self) -> NDArray[np.complex128]:
        """Complex STFT as a complex array."""
        return self.stft[:, :, 0] + 1j * self.stft[:, :, 1]

    @property
    def magnitude(self) -> NDArray[np.float64]:
        """Magnitude (absolute value) of the STFT."""
        return np.sqrt(self.stft[:, :, 0] ** 2 + self.stft[:, :, 1] ** 2)

    @property
    def phase(self) -> NDArray[np.float64]:
        """Phase angle of the STFT in radians."""
        return np.arctan2(self.stft[:, :, 1], self.stft[:, :, 0])

    @property
    def power(self) -> NDArray[np.float64]:
        """Power spectral density (magnitude squared)."""
        return self.stft[:, :, 0] ** 2 + self.stft[:, :, 1] ** 2

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the STFT array (n_times, n_frequencies, 2)."""
        return self.stft.shape

    @property
    def frequency_resolution(self) -> float:
        """Frequency resolution in Hz."""
        return self.freqs[1] - self.freqs[0] if len(self.freqs) > 1 else 0.0

    @property
    def time_resolution(self) -> float:
        """Time resolution in seconds."""
        return self.times[1] - self.times[0] if len(self.times) > 1 else 0.0


class MotionFeatures:
    """Container for motion feature extraction results."""

    def __init__(
        self,
        breathing_rate: NDArray[np.float64],
        heart_rate: NDArray[np.float64],
        frequency_sum: NDArray[np.float64],
        breathing_rate_std: NDArray[np.float64],
        heart_rate_std: NDArray[np.float64],
        spectrogram: SpectrogramResult,
    ):
        self.breathing_rate = breathing_rate  # BPM
        self.heart_rate = heart_rate  # BPM
        self.frequency_sum = frequency_sum
        self.breathing_rate_std = breathing_rate_std
        self.heart_rate_std = heart_rate_std
        self.spectrogram = spectrogram


def resample_accelerometer(
    timestamps: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    target_fs: float,
    ts_unit: str = "s",
) -> AccelerometerData:
    """
    Resample accelerometer data to a target sampling frequency.

    Args:
        timestamps: Array of timestamps in seconds
        x: X-axis acceleration values
        y: Y-axis acceleration values
        z: Z-axis acceleration values
        target_fs: Target sampling frequency in Hz
        second_scalar: Scalar to convert timestamps to microseconds (default: 1.0 for seconds)

    Returns:
        AccelerometerData: Resampled accelerometer data

    Raises:
        ValueError: If input arrays have different lengths
    """
    print("USING PYTHON API FUNCTION")
    if not (len(timestamps) == len(x) == len(y) == len(z)):
        raise ValueError("All input arrays must have the same length")

    conversion_scalar = 1e6
    if ts_unit == "ms":
        conversion_scalar = 1e3
    elif ts_unit == "us":
        conversion_scalar = 1.0

    # Convert timestamps to microseconds
    timestamps_us = (timestamps * conversion_scalar).astype(np.int64)

    result = _senpy.resample_accelerometer(timestamps_us, x, y, z, target_fs)
    return AccelerometerData(
        timestamps_us=result["timestamps"],
        x=result["x"],
        y=result["y"],
        z=result["z"],
    )


def resample_accelerometer_microseconds(
    timestamps: NDArray[np.int64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    target_fs: float,
) -> AccelerometerData:
    """
    Resample accelerometer data to a target sampling frequency.

    Args:
        timestamps: Array of timestamps in microseconds
        x: X-axis acceleration values
        y: Y-axis acceleration values
        z: Z-axis acceleration values
        target_fs: Target sampling frequency in Hz
        second_scalar: Scalar to convert timestamps to microseconds (default: 1.0 for seconds)

    Returns:
        AccelerometerData: Resampled accelerometer data

    Raises:
        ValueError: If input arrays have different lengths
    """
    print("USING PYTHON API FUNCTION")
    if not (len(timestamps) == len(x) == len(y) == len(z)):
        raise ValueError("All input arrays must have the same length")

    result = _senpy.resample_accelerometer(timestamps, x, y, z, target_fs)
    return AccelerometerData(
        timestamps_us=result["timestamps"], x=result["x"], y=result["y"], z=result["z"]
    )


def compute_jerk(
    timestamps: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    ts_unit: str = "s",
) -> JerkData:
    """
    Compute jerk (derivative of acceleration) from accelerometer data.

    Args:
        timestamps: Array of timestamps in seconds
        x: X-axis acceleration values
        y: Y-axis acceleration values
        z: Z-axis acceleration values
        ts_unit: Unit of the timestamps ('s' for seconds, 'ms' for milliseconds, 'us' for microseconds)
    Returns:
        Tuple of (timestamps, jerk_values)
    Raises:
        ValueError: If input arrays have different lengths
    """
    if not (len(timestamps) == len(x) == len(y) == len(z)):
        raise ValueError("All input arrays must have the same length")

    conversion_scalar = 1e6
    if ts_unit == "ms":
        conversion_scalar = 1e3
    elif ts_unit == "us":
        conversion_scalar = 1.0

    # Convert timestamps to microseconds
    timestamps_us = (timestamps * conversion_scalar).astype(np.int64)

    result = compute_jerk_microseconds(timestamps_us, x, y, z)
    return result


def compute_jerk_microseconds(
    timestamps: NDArray[np.int64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
) -> JerkData:
    """
    Compute jerk (derivative of acceleration) from accelerometer data.

    Args:
        timestamps: Array of timestamps in microseconds
        x: X-axis acceleration values
        y: Y-axis acceleration values
        z: Z-axis acceleration values

    Returns:
        Tuple of (timestamps, jerk_values)

    Raises:
        ValueError: If input arrays have different lengths
    """
    if not (len(timestamps) == len(x) == len(y) == len(z)):
        raise ValueError("All input arrays must have the same length")

    result = _senpy.compute_jerk(timestamps, x, y, z)
    return JerkData(timestamps_us=result["timestamps"], jerk=result["jerk"])


def compute_magnitude(
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute magnitude from x, y, z components.

    Args:
        x: X-axis values
        y: Y-axis values
        z: Z-axis values

    Returns:
        Array of magnitude values: sqrt(x² + y² + z²)

    Raises:
        ValueError: If input arrays have different lengths
    """
    if not (len(x) == len(y) == len(z)):
        raise ValueError("All input arrays must have the same length")

    return _senpy.compute_magnitude(x, y, z)


def compute_spectrogram(
    signal: NDArray[np.float64], fs: float, nperseg: int, noverlap: int
) -> SpectrogramResult:
    """
    Compute spectrogram using FFT-based Short-Time Fourier Transform.

    Args:
        signal: Input signal array
        fs: Sampling frequency in Hz
        nperseg: Length of each segment (window size)
        noverlap: Number of points to overlap between segments

    Returns:
        SpectrogramResult: Contains frequencies, times, and power spectral density matrix

    Note:
        Uses Hann window, constant detrending, and magnitude scaling compatible with scipy.
    """
    result = _senpy.compute_spectrogram(signal, fs, nperseg, noverlap)
    return SpectrogramResult(
        frequencies=result["freqs"], times=result["times"], Sxx=result["Sxx"]
    )


def compute_short_time_ft(
    signal: NDArray[np.float64], fs: float, nperseg: int, noverlap: int
) -> ShortTimeFTResult:
    """
    Compute Short-Time Fourier Transform returning complex values.

    This function performs STFT analysis and returns the complex Fourier coefficients,
    allowing access to both magnitude and phase information. Unlike compute_spectrogram,
    which returns only the power spectral density, this function preserves the full
    complex representation of the signal in the frequency domain.

    Args:
        signal: Input signal array
        fs: Sampling frequency in Hz
        nperseg: Length of each segment (window size)
        noverlap: Number of points to overlap between segments

    Returns:
        ShortTimeFTResult: Container with STFT array shaped (n_times, n_frequencies, 2)
            where the last dimension contains [real, imaginary] parts. Also includes
            frequency and time bin arrays.

    Note:
        - Uses Hann window and constant detrending (mean removal)
        - Only returns positive frequencies (0 to Nyquist)
        - Access magnitude via result.magnitude, phase via result.phase
        - Access complex array via result.complex for numpy operations

    Example:
        >>> result = compute_short_time_ft(signal, fs=50.0, nperseg=256, noverlap=128)
        >>> magnitude = result.magnitude  # Time-frequency magnitude
        >>> phase = result.phase          # Time-frequency phase
        >>> complex_stft = result.complex # Full complex representation
        >>> print(result.shape)           # (n_times, n_frequencies, 2)
    """
    stft_array = _senpy.compute_short_time_ft(signal, fs, nperseg, noverlap)

    # Compute frequency and time arrays (same as in spectrogram)
    n_times, n_freqs = stft_array.shape[0], stft_array.shape[1]
    nfft = nperseg
    step = nperseg - noverlap

    # Generate frequency bins
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)

    # Generate time bins (center of each window)
    times = np.arange(n_times) * step / fs + (nperseg / 2.0) / fs

    return ShortTimeFTResult(stft=stft_array, freqs=freqs, times=times)


def compute_motion_features(
    jerk_signal: NDArray[np.float64],
    fs: float,
    window_size: int = 1500,
    overlap: int = 750,
    breathing_rate_min_hz: float = 0.15,
    breathing_rate_max_hz: float = 0.417,
    heart_rate_min_hz: float = 0.5,
    heart_rate_max_hz: float = 2.0,
    std_window_minutes: float = 5.0,
    smooth_hr_spectrogram: bool = True,
    smooth_br_spectrogram: bool = False,
    spectrogram_smoothing_freq: float = 1.0,
    spectrogram_smoothing_time: float = 2.0,
    hr_max_change_per_sec: float = 15.0,
    br_max_change_per_sec: float = 7.5,
    time_resolution: float = 30.0,
) -> MotionFeatures:
    """
    Extract breathing rate, heart rate, and motion features from jerk signal.

    Args:
        jerk_signal: Input jerk signal array
        fs: Sampling frequency in Hz
        window_size: FFT window size in samples (default: 1500 = 30s @ 50Hz)
        overlap: Window overlap in samples (default: 750 = 50% overlap)
        breathing_rate_min_hz: Minimum breathing rate frequency (default: 0.15 = 9 BPM)
        breathing_rate_max_hz: Maximum breathing rate frequency (default: 0.417 = 25 BPM)
        heart_rate_min_hz: Minimum heart rate frequency (default: 0.5 = 30 BPM)
        heart_rate_max_hz: Maximum heart rate frequency (default: 2.0 = 120 BPM)
        std_window_minutes: Window size for rolling standard deviation in minutes
        smooth_hr_spectrogram: Whether to apply smoothing to HR spectrogram
        smooth_br_spectrogram: Whether to apply smoothing to BR spectrogram
        spectrogram_smoothing_freq: Frequency domain smoothing parameter
        spectrogram_smoothing_time: Time domain smoothing parameter
        hr_max_change_per_sec: Maximum allowed HR change per second (BPM/s)
        br_max_change_per_sec: Maximum allowed BR change per second (BPM/s)
        time_resolution: Time resolution for output features in seconds

    Returns:
        MotionFeatures: Container with breathing rate, heart rate, and derived features
    """
    result = _senpy.compute_motion_features(
        jerk_signal,
        fs,
        window_size,
        overlap,
        breathing_rate_min_hz,
        breathing_rate_max_hz,
        heart_rate_min_hz,
        heart_rate_max_hz,
        std_window_minutes,
        smooth_hr_spectrogram,
        smooth_br_spectrogram,
        spectrogram_smoothing_freq,
        spectrogram_smoothing_time,
        hr_max_change_per_sec,
        br_max_change_per_sec,
        time_resolution,
    )

    spectrogram = SpectrogramResult(
        frequencies=result["spectrogram"]["freqs"],
        times=result["spectrogram"]["times"],
        Sxx=result["spectrogram"]["Sxx"],
    )

    return MotionFeatures(
        breathing_rate=result["BR"],
        heart_rate=result["HR"],
        frequency_sum=result["freqSum"],
        breathing_rate_std=result["BR_std"],
        heart_rate_std=result["HR_std"],
        spectrogram=spectrogram,
    )


# Utility functions
def hann_window(n: int) -> NDArray[np.float64]:
    """
    Generate Hann window of size N.

    Args:
        n: Window size

    Returns:
        Array containing Hann window values
    """
    return _senpy.hann_window(n)


def gaussian_filter_1d(
    data: NDArray[np.float64], sigma: float, truncate: float = 4.0
) -> NDArray[np.float64]:
    """
    Apply 1D Gaussian filter to data.

    Args:
        data: Input data array
        sigma: Standard deviation of Gaussian kernel
        truncate: Truncate filter at this many standard deviations

    Returns:
        Filtered data array
    """
    return _senpy.gaussian_filter_1d(data, sigma, truncate)

def find_spectrogram_peaks(
    Sxx: NDArray[np.float64],
    prominence_threshold: float,
    frequencies: NDArray[np.float64],
    scaling_factor: float = 60.0
) -> NDArray[np.int32]:
    """
    Find peaks in each time slice of spectrogram with prominence threshold.

    Args:
        Sxx: Spectrogram power spectral density array (time, frequency)
        prominence_threshold: Minimum prominence required for peak detection

    Returns:
        List of arrays, each containing peak indices for corresponding time slice
    """
    return _senpy.find_spectrogram_peaks(
        Sxx=Sxx,
        prominence_threshold=prominence_threshold,
        frequencies=frequencies,
        scaling_factor=scaling_factor)


def find_peaks(
    signal: NDArray[np.float64], prominence_threshold: float
) -> NDArray[np.int32]:
    """
    Find peaks in signal with prominence threshold.

    Args:
        signal: Input signal array
        prominence_threshold: Minimum prominence required for peak detection

    Returns:
        Array of peak indices
    """
    return _senpy.find_peaks(signal, prominence_threshold)


def rolling_std(
    data: NDArray[np.float64], window_minutes: float, seconds_per_window: float = 30.0
) -> NDArray[np.float64]:
    """
    Compute rolling standard deviation.

    Args:
        data: Input data array
        window_minutes: Window size in minutes
        seconds_per_window: Time resolution in seconds per sample

    Returns:
        Array of rolling standard deviation values
    """
    return _senpy.rolling_std(data, window_minutes, seconds_per_window)


def next_power_of_2(n: int) -> int:
    """
    Find next power of 2 greater than or equal to n.

    Args:
        n: Input integer

    Returns:
        Next power of 2
    """
    return _senpy.next_power_of_2(n)


def compute_median(data: NDArray[np.float64]) -> float:
    """
    Compute median of data.

    Args:
        data: Input data array

    Returns:
        Median value
    """
    return _senpy.compute_median(data)


def compute_percentile(data: NDArray[np.float64], percentile: float) -> float:
    """
    Compute percentile of data.

    Args:
        data: Input data array
        percentile: Percentile to compute (0-100)

    Returns:
        Percentile value
    """
    return _senpy.compute_percentile(data, percentile)


# Constants for convenience
class FrequencyRanges:
    """Common frequency ranges for physiological signals."""

    # Breathing rate ranges (Hz, N/60 = Breaths/Min)
    BR_RESTING_MIN = 9/60
    BR_RESTING_MAX = 25/60

    # Heart rate ranges (Hz, N/60 = BPM)
    HR_RESTING_MIN = 30/60
    HR_RESTING_MAX = 120/60


class SamplingRates:
    """Common sampling rates for sensor data."""
    ACCELEROMETER_STANDARD = 50.0  # Hz
    ACCELEROMETER_HIGH = 100.0  # Hz


# Version info
__version__ = "1.0.0"
__all__ = [
    "AccelerometerData",
    "SpectrogramResult",
    "JerkData",
    "ShortTimeFTResult",
    "MotionFeatures",
    "resample_accelerometer",
    "compute_jerk",
    "compute_magnitude",
    "compute_spectrogram",
    "compute_short_time_ft",
    "compute_motion_features",
    "hann_window",
    "gaussian_filter_1d",
    "find_peaks",
    "find_spectrogram_peaks",
    "rolling_std",
    "next_power_of_2",
    "compute_median",
    "compute_percentile",
    "FrequencyRanges",
    "SamplingRates",
]
