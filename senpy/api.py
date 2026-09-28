"""Defines a Python API for easier development with the library.

This module is an interface. The Python bindings already exist, but one must read the C++ code to understand how to use them. This module wraps the C++ bindings in a more Pythonic way.
"""

import warnings
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from numpy.typing import NDArray

# Import the C++ module
import senpy._core as _senpy
from ._version import __version__
from ._grid import (
    DEFAULT_MIN_SAMPLES,
    WindowGrid,
    relative_seconds as _relative_seconds,
    resolve_empty_windows as _resolve_empty_windows,
    resolve_origin as _resolve_origin,
    validate_min_samples as _validate_min_samples,
    window_grid,
)


AXIS_ORDER_TIME_FREQUENCY = "time_frequency"


def _row_metadata(
    n_rows: int,
    window_index: Optional[NDArray[np.int64]],
    sample_count: Optional[NDArray[np.int64]],
    valid: Optional[NDArray[np.bool_]],
) -> Tuple[Optional[NDArray[np.int64]], Optional[NDArray[np.int64]], NDArray[np.bool_]]:
    """Validate the per-window metadata a result carries alongside its rows.

    ``window_index`` and ``sample_count`` are ``None`` when unknown, as for a
    result built by hand. ``valid`` defaults to every row being valid.
    """
    def _column(values, dtype, name):
        if values is None:
            return None
        array = np.asarray(values, dtype=dtype)
        if array.shape != (n_rows,):
            raise ValueError(f"{name} must have one entry per time bin ({n_rows}), got {array.shape}")
        return array

    window_index = _column(window_index, np.int64, "window_index")
    sample_count = _column(sample_count, np.int64, "sample_count")
    valid = _column(valid, bool, "valid")
    if valid is None:
        valid = np.ones(n_rows, dtype=bool)
    return window_index, sample_count, valid


def _normalize_spectral_kind(kind: str) -> str:
    normalized = str(kind).replace("-", "_").lower()
    aliases = {"mag": "magnitude", "magnitude": "magnitude", "power": "power", "psd": "psd"}
    if normalized not in aliases:
        raise ValueError("kind must be one of: 'mag', 'magnitude', 'power', 'psd'")
    return aliases[normalized]


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
    """Container for a time-frequency spectral surface.

    Attributes:
        frequencies: Array of frequency bins in Hz.
        times: Array of time bins in seconds.
        Sxx: Array shaped ``(n_times, n_freqs)``.
        kind: Spectral quantity stored in ``Sxx``: ``"magnitude"``,
            ``"power"``, or ``"psd"``.
        method: Computation method, such as ``"nufft"`` or ``"uniform_fft"``.
        window_index: NUFFT results: each row's index on the window grid, so
            gaps show where windows were left out. ``None`` if unknown.
        sample_count: NUFFT results: samples in each row's window. ``None`` if
            unknown.
        valid: ``False`` for rows with too few samples to transform, which
            hold NaN (``empty_windows="keep"``). All ``True`` otherwise.
        origin_s: NUFFT results: absolute start of window 0, in seconds on the
            input timestamps' clock. ``times`` are measured from it.
    """

    def __init__(
        self,
        frequencies: NDArray[np.float64],
        times: NDArray[np.float64],
        Sxx: NDArray[np.float64],
        kind: str = "magnitude",
        method: str = "unknown",
        window_index: Optional[NDArray[np.int64]] = None,
        sample_count: Optional[NDArray[np.int64]] = None,
        valid: Optional[NDArray[np.bool_]] = None,
        origin_s: Optional[float] = None,
    ):
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)
        self.Sxx = np.asarray(Sxx, dtype=np.float64)
        self.kind = kind
        self.method = method
        self.axis_order = AXIS_ORDER_TIME_FREQUENCY
        if self.Sxx.ndim != 2:
            raise ValueError("Sxx must be a 2-D array shaped (n_times, n_freqs)")
        expected_shape = (len(self.times), len(self.frequencies))
        if self.Sxx.shape != expected_shape:
            raise ValueError(
                f"Sxx shape {self.Sxx.shape} does not match "
                f"(len(times), len(frequencies)) {expected_shape}"
            )
        self.window_index, self.sample_count, self.valid = _row_metadata(
            len(self.times), window_index, sample_count, valid
        )
        self.origin_s = None if origin_s is None else float(origin_s)

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
            scaling_factor=scaling_factor,
        )

        return peaks


class NUSTFTResult:
    """Complex non-uniform short-time Fourier transform result.

    ``coefficients`` is always shaped ``(n_times, n_freqs)`` and contains
    scaled complex FINUFFT coefficients. Derived spectra are exposed as views
    so callers can choose magnitude, power, PSD-like density, or Welch-style
    averages without recomputing the transform.

    ``times`` are window centres in seconds after ``origin_s``. ``window_index``,
    ``sample_count``, and ``valid`` describe each row's window; with
    ``empty_windows="keep"`` every window on the grid has a row and those
    without enough samples hold NaN coefficients with ``valid`` False.
    """

    def __init__(
        self,
        frequencies: NDArray[np.float64],
        times: NDArray[np.float64],
        coefficients: NDArray[np.complex128],
        window_index: Optional[NDArray[np.int64]] = None,
        sample_count: Optional[NDArray[np.int64]] = None,
        valid: Optional[NDArray[np.bool_]] = None,
        origin_s: Optional[float] = None,
    ):
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.complex128)
        self.method = "nufft"
        self.axis_order = AXIS_ORDER_TIME_FREQUENCY
        if self.coefficients.ndim != 2:
            raise ValueError("coefficients must be shaped (n_times, n_freqs)")
        expected_shape = (len(self.times), len(self.frequencies))
        if self.coefficients.shape != expected_shape:
            raise ValueError(
                f"coefficients shape {self.coefficients.shape} does not match "
                f"(len(times), len(frequencies)) {expected_shape}"
            )
        self.window_index, self.sample_count, self.valid = _row_metadata(
            len(self.times), window_index, sample_count, valid
        )
        self.origin_s = None if origin_s is None else float(origin_s)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.coefficients.shape

    @property
    def frequency_resolution(self) -> float:
        return (
            self.frequencies[1] - self.frequencies[0]
            if len(self.frequencies) > 1
            else 0.0
        )

    @property
    def time_resolution(self) -> float:
        return self.times[1] - self.times[0] if len(self.times) > 1 else 0.0

    @property
    def magnitude(self) -> NDArray[np.float64]:
        return np.abs(self.coefficients)

    @property
    def power(self) -> NDArray[np.float64]:
        return np.abs(self.coefficients) ** 2

    @property
    def psd(self) -> NDArray[np.float64]:
        # Intentional: "psd" equals "power" and is NOT a missing normalization.
        # The density scaling (1 / sqrt(fs * Σw²), the sample rate times Hann
        # window energy) is already baked into the complex coefficients in the
        # C++ layer, so power = |X|² / (fs * Σw²) already carries units²/Hz --
        # exactly scipy's welch(scaling="density") formula. Do NOT additionally
        # divide by df (the frequency step); that would double-count and is the
        # wrong normalization anyway (a PSD needs ÷(fs·Σw²), not ÷df).
        #
        # Difference from scipy: scipy's one-sided density multiplies every bin
        # except DC/Nyquist by 2 so integrating over [0, fs/2] recovers signal
        # variance. We deliberately omit that ×2 folding factor -- this is our
        # convention, applied equally to power and psd. Do not "fix" to match
        # scipy without updating the documented contract and all consumers.
        return self.power

    def _surface(self, kind: str) -> NDArray[np.float64]:
        kind = _normalize_spectral_kind(kind)
        if kind == "magnitude":
            return self.magnitude
        if kind == "power":
            return self.power
        return self.psd

    def spectrogram(self, kind: str = "psd") -> SpectrogramResult:
        """Return a derived real-valued spectral surface.

        The default is ``"psd"`` to emphasize the new complex-coefficient
        workflow. Compatibility wrappers such as ``compute_nufft_spectrogram``
        may choose ``"magnitude"`` to preserve older return-value expectations.
        """
        kind = _normalize_spectral_kind(kind)
        return SpectrogramResult(
            frequencies=self.frequencies,
            times=self.times,
            Sxx=self._surface(kind),
            kind=kind,
            method="nufft",
            window_index=self.window_index,
            sample_count=self.sample_count,
            valid=self.valid,
            origin_s=self.origin_s,
        )

    def welch(
        self,
        kind: str = "psd",
        average: str = "mean",
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        surface = self._surface(kind)
        if surface.shape[0] == 0:
            return self.frequencies, np.array([], dtype=np.float64)
        if average == "mean":
            spectrum = np.nanmean(surface, axis=0)
        elif average == "median":
            spectrum = np.nanmedian(surface, axis=0)
        else:
            raise ValueError("average must be 'mean' or 'median'")
        return self.frequencies, spectrum


STACKED_SPECTROGRAM_CHANNELS: List[str] = ["x", "y", "z", "mag", "jerk"]


class StackedSpectrogramResult:
    """Multi-channel stacked spectrogram with shape ``(n_times, n_freqs, n_channels)``.

    Attributes:
        frequencies: Frequency bins shared by all channels, in Hz.
        times: Time bins shared by all channels, in seconds.
        Sxx: Array shaped ``(n_times, n_freqs, n_channels)``.
        channels: Ordered list of channel names, one per ``Sxx[:, :, i]``.
        kind: Spectral quantity stored in ``Sxx``.
        window_index, sample_count, valid, origin_s: As on
            :class:`SpectrogramResult`. ``sample_count`` is the reference
            channel's; ``valid`` is ``False`` where any channel has no data.
    """

    def __init__(
        self,
        frequencies: NDArray[np.float64],
        times: NDArray[np.float64],
        Sxx: NDArray[np.float64],
        channels: List[str],
        kind: str = "magnitude",
        window_index: Optional[NDArray[np.int64]] = None,
        sample_count: Optional[NDArray[np.int64]] = None,
        valid: Optional[NDArray[np.bool_]] = None,
        origin_s: Optional[float] = None,
    ):
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)
        self.Sxx = np.asarray(Sxx, dtype=np.float64)
        self.channels = list(channels)
        self.kind = kind
        if self.Sxx.ndim != 3:
            raise ValueError("Sxx must be a 3-D array shaped (n_times, n_freqs, n_channels)")
        expected = (len(self.times), len(self.frequencies), len(self.channels))
        if self.Sxx.shape != expected:
            raise ValueError(
                f"Sxx.shape={self.Sxx.shape} does not match "
                f"(n_times={expected[0]}, n_freqs={expected[1]}, n_channels={expected[2]})"
            )
        self.window_index, self.sample_count, self.valid = _row_metadata(
            len(self.times), window_index, sample_count, valid
        )
        self.origin_s = None if origin_s is None else float(origin_s)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.Sxx.shape


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
        return (
            self.frequencies[1] - self.frequencies[0]
            if len(self.frequencies) > 1
            else 0.0
        )

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
    if not (len(timestamps) == len(x) == len(y) == len(z)):
        raise ValueError("All input arrays must have the same length")

    result = _senpy.resample_accelerometer(timestamps, x, y, z, target_fs)
    return AccelerometerData(
        timestamps_us=result["timestamps"], x=result["x"], y=result["y"], z=result["z"]
    )


# ── Cubic-spline resampling (C++ backend) ─────────────────────────


def resample_accelerometer_cubic(
    timestamps: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    target_fs: float,
    ts_unit: str = "s",
) -> AccelerometerData:
    """Resample accelerometer data using natural cubic spline interpolation.

    Same interface as ``resample_accelerometer`` but uses C3-continuous
    cubic splines instead of piecewise-linear interpolation, giving much
    better high-frequency rolloff (sinc^4-like vs sinc^2).

    Args:
        timestamps: Sample timestamps.
        x, y, z: Acceleration components.
        target_fs: Target sampling frequency in Hz.
        ts_unit: Timestamp unit — ``'s'``, ``'ms'``, or ``'us'``.

    Returns:
        AccelerometerData on a uniform grid at *target_fs*.
    """
    if not (len(timestamps) == len(x) == len(y) == len(z)):
        raise ValueError("All input arrays must have the same length")

    conversion_scalar = {"s": 1e6, "ms": 1e3, "us": 1.0}.get(ts_unit, 1e6)
    timestamps_us = (timestamps * conversion_scalar).astype(np.int64)

    result = _senpy.resample_accelerometer_cubic(timestamps_us, x, y, z, target_fs)
    return AccelerometerData(
        timestamps_us=result["timestamps"],
        x=result["x"],
        y=result["y"],
        z=result["z"],
    )


def resample_accelerometer_cubic_microseconds(
    timestamps: NDArray[np.int64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    target_fs: float,
) -> AccelerometerData:
    """Cubic-spline resampling with microsecond timestamps."""
    if not (len(timestamps) == len(x) == len(y) == len(z)):
        raise ValueError("All input arrays must have the same length")

    result = _senpy.resample_accelerometer_cubic(timestamps, x, y, z, target_fs)
    return AccelerometerData(
        timestamps_us=result["timestamps"],
        x=result["x"],
        y=result["y"],
        z=result["z"],
    )

def _timestamps_to_seconds(
    timestamps: NDArray[np.float64],
    ts_unit: str,
) -> NDArray[np.float64]:
    conversion = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
    if ts_unit not in conversion:
        raise ValueError("ts_unit must be one of: 's', 'ms', 'us'")
    return np.asarray(timestamps, dtype=np.float64) * conversion[ts_unit]


def _validate_time_window(window_s: float, overlap_s: float) -> None:
    if window_s <= 0:
        raise ValueError("window_s must be > 0")
    if overlap_s < 0 or overlap_s >= window_s:
        raise ValueError("overlap_s must satisfy 0 <= overlap_s < window_s")


def _grid_rows(
    rows: NDArray,
    window_indices: NDArray[np.int64],
    grid_sample_counts: NDArray[np.int64],
    *,
    window_s: float,
    overlap_s: float,
    min_samples: int,
    empty_windows: str,
) -> Dict[str, NDArray]:
    """Lay transformed windows out on the grid, dropping or NaN-filling the rest.

    ``rows`` holds one row per entry of ``window_indices`` -- the windows with
    at least ``min_samples`` samples. ``grid_sample_counts`` has one entry per
    window on the grid.
    """
    window_indices = np.asarray(window_indices, dtype=np.int64)
    grid_sample_counts = np.asarray(grid_sample_counts, dtype=np.int64)
    rows = np.asarray(rows)
    if empty_windows == "keep":
        n_windows = grid_sample_counts.size
        dense = np.full((n_windows,) + rows.shape[1:], np.nan, dtype=rows.dtype)
        dense[window_indices] = rows
        rows = dense
        window_indices = np.arange(n_windows, dtype=np.int64)
    sample_count = grid_sample_counts[window_indices]
    hop_s = float(window_s) - float(overlap_s)
    return {
        "rows": rows,
        "times": window_indices * hop_s + float(window_s) / 2.0,
        "window_index": window_indices,
        "sample_count": sample_count,
        "valid": sample_count >= min_samples,
    }


def _validate_sample_window(nperseg: int, noverlap: int) -> None:
    if nperseg <= 0:
        raise ValueError("rounded window_s * target_fs must be at least 1 sample")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError(
            "rounded overlap_s * target_fs must satisfy 0 <= noverlap < nperseg"
        )


def compute_nustft(
    timestamps: NDArray[np.float64],
    signal: NDArray[np.float64],
    window_s: float,
    overlap_s: float,
    ts_unit: str = "s",
    target_fs: Optional[float] = None,
    detrend: bool = True,
    *,
    origin_s: Union[None, float, str] = None,
    empty_windows: Optional[str] = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> NUSTFTResult:
    """Compute complex non-uniform STFT coefficients using FINUFFT.

    This is the primary spectral primitive for jittered or otherwise
    non-uniform timestamps. It bypasses time-domain resampling entirely.

    Args:
        timestamps: Sample timestamps, same length as ``signal``.
        signal: 1-D signal values.
        window_s: Window duration in seconds.
        overlap_s: Window overlap in seconds.
        ts_unit: Timestamp unit: ``"s"``, ``"ms"``, or ``"us"``.
        target_fs: Optional output frequency-grid limit. If specified, output
            bins span ``0`` through ``target_fs / 2`` with spacing
            ``1 / window_s``.
        detrend: If true, subtract each window's mean before applying the Hann
            taper. Set false to preserve DC/low-frequency offsets.
        origin_s: Where window 0 starts. ``None`` (the default) is the first
            sample. A number is an absolute time in seconds on the
            timestamps' clock -- for example the start of a reference
            recording, so every window lines up with it; samples before it
            are ignored. ``"unix"`` anchors the grid a whole number of hops
            from the Unix epoch, at or after the first sample, so windows from
            separate recordings or sessions share one grid.
        empty_windows: ``"drop"`` leaves out windows with fewer than
            ``min_samples`` samples, so ``times`` has gaps. ``"keep"`` reports
            every window on the grid; those rows hold NaN and ``valid`` is
            False. Omitting it means ``"drop"`` and warns: senpy 5.0 changes
            the default to ``"keep"``.
        min_samples: Fewest samples a window needs to be transformed.

    Returns:
        ``NUSTFTResult`` with complex coefficients shaped ``(n_times, n_freqs)``,
        ``times`` measured from ``origin_s``, and per-window ``window_index``,
        ``sample_count`` and ``valid``.
    """
    empty_windows = _resolve_empty_windows(empty_windows)
    return _compute_nustft(
        timestamps,
        signal,
        window_s,
        overlap_s,
        ts_unit=ts_unit,
        target_fs=target_fs,
        detrend=detrend,
        origin_s=origin_s,
        empty_windows=empty_windows,
        min_samples=min_samples,
    )


def _compute_nustft(
    timestamps: NDArray[np.float64],
    signal: NDArray[np.float64],
    window_s: float,
    overlap_s: float,
    *,
    ts_unit: str,
    target_fs: Optional[float],
    detrend: bool,
    origin_s: Union[None, float, str],
    empty_windows: str,
    min_samples: int,
) -> NUSTFTResult:
    if len(timestamps) != len(signal):
        raise ValueError("timestamps and signal must have the same length")
    if len(timestamps) < 2:
        raise ValueError("compute_nustft requires at least two timestamps")
    _validate_time_window(window_s, overlap_s)
    if target_fs is not None and target_fs < 0:
        raise ValueError("target_fs must be >= 0")
    min_samples = _validate_min_samples(min_samples)

    t, origin = _relative_seconds(timestamps, ts_unit, origin_s, float(window_s - overlap_s))
    target_fs_val = target_fs if target_fs is not None else 0.0

    try:
        result_dict = _senpy.compute_nustft(
            t,
            np.asarray(signal, dtype=np.float64),
            float(window_s),
            float(overlap_s),
            float(target_fs_val),
            bool(detrend),
            0.0,
            min_samples,
        )
    except RuntimeError as e:
        raise RuntimeError(f"C++ NUSTFT computation failed: {e}") from e

    grid_counts = result_dict["grid_sample_counts"]
    if (empty_windows == "drop" and len(result_dict["times"]) == 0) or len(grid_counts) == 0:
        raise ValueError("compute_nustft requires enough data for at least one window")

    frequencies = result_dict["freqs"]
    coefficients = result_dict["coefficients"]
    if coefficients.shape[0] == 0:
        coefficients = np.empty((0, len(frequencies)), dtype=np.complex128)
    laid_out = _grid_rows(
        coefficients,
        result_dict["window_indices"],
        grid_counts,
        window_s=window_s,
        overlap_s=overlap_s,
        min_samples=min_samples,
        empty_windows=empty_windows,
    )
    return NUSTFTResult(
        frequencies=frequencies,
        times=laid_out["times"],
        coefficients=laid_out["rows"],
        window_index=laid_out["window_index"],
        sample_count=laid_out["sample_count"],
        valid=laid_out["valid"],
        origin_s=origin,
    )


class StreamingWindow:
    """One finished window from :class:`StreamingNUSTFT`.

    Attributes:
        index: Window index relative to the transform's origin.
        start: Window start in seconds, on the same clock as the pushed timestamps.
        center: Window center in seconds — the value ``compute_nustft`` reports in ``times``,
            before it subtracts the recording's first timestamp.
        sample_count: Samples that landed in the window.
        coefficients: Complex coefficients, one per entry of the transform's ``frequencies``.
    """

    __slots__ = ("index", "start", "center", "sample_count", "coefficients")

    def __init__(
        self,
        index: int,
        start: float,
        center: float,
        sample_count: int,
        coefficients: NDArray[np.complex128],
    ):
        self.index = int(index)
        self.start = float(start)
        self.center = float(center)
        self.sample_count = int(sample_count)
        self.coefficients = np.asarray(coefficients, dtype=np.complex128)

    def magnitude(self) -> NDArray[np.float64]:
        return np.abs(self.coefficients)

    def power(self) -> NDArray[np.float64]:
        return np.abs(self.coefficients) ** 2

    def __repr__(self) -> str:
        return (
            f"StreamingWindow(index={self.index}, start={self.start:g}, "
            f"samples={self.sample_count}, freqs={self.coefficients.size})"
        )


class StreamingNUSTFT:
    """Incremental NUSTFT — the transform of ``compute_nustft``, one chunk at a time.

    ``compute_nustft`` needs the whole recording. This computes the identical coefficients from
    a stream, emitting each window as soon as the data passes its end, and keeping only
    per-subwindow spectra rather than the samples themselves. It is exact, not an approximation:
    the transform is linear and the subwindows partition the window, so a window recombined from
    its parts equals the transform of the whole. See ``README.md`` for the derivation, including
    how the window's Hann taper and mean removal are deferred and reconstructed.

    Args:
        window_s: Window duration, as in ``compute_nustft``'s ``window_s``.
        overlap_s: Window overlap; the hop is ``window_s - overlap_s``.
        subwindow_s: Streaming granularity — the unit of work pushed in, typically one sensor
            packet. ``window_s`` and the hop must both be whole multiples of it, so that no
            subwindow straddles a window edge.
        sample_rate_hz: Nominal stream rate. Used for the magnitude scale factor and to size the
            frequency grid; ``compute_nustft`` derives the same number as the median sample
            spacing over the whole recording, which a stream cannot see.
        fmax: Report only bins up to this frequency. This is what makes the per-sample cost
            small when a narrow band is wanted. ``None`` reports the full grid.
        origin_s: Anchors the window grid — window ``w`` spans
            ``[origin_s + w*hop, origin_s + w*hop + window_s)``. Pass the first timestamp to
            reproduce ``compute_nustft``'s alignment, or a fixed epoch to keep window indices
            meaningful across sessions.
        detrend: Subtract each window's mean before the taper.
        ts_unit: Unit of pushed timestamps — ``"s"``, ``"ms"``, or ``"us"``. ``origin_s``,
            ``window_s`` and the returned times are always in seconds. Note that absolute unix
            seconds in float64 resolve to about half a microsecond; pass times relative to a
            recent origin when sub-microsecond timing matters.
        min_samples: Fewest samples a window needs to be reported. Sparser windows are counted
            by :attr:`skipped_windows`; windows with no samples at all are never seen. Either
            kind shows up as a gap in the reported ``index`` values.

    For a grid shared across sessions, anchor it to the Unix epoch: ``origin_s=0.0`` with Unix
    timestamps puts windows on the same grid ``compute_nustft(..., origin_s="unix")`` uses, with
    each ``index`` counted from the epoch.

    Example:
        >>> transform = StreamingNUSTFT(30.0, 0.0, 1.0, sample_rate_hz=100.0, fmax=5.0)
        >>> for packet_t, packet_x in packets:              # doctest: +SKIP
        ...     for window in transform.push(packet_t, packet_x):
        ...         consume(window.magnitude())
        >>> tail = transform.flush()                        # doctest: +SKIP
    """

    def __init__(
        self,
        window_s: float,
        overlap_s: float,
        subwindow_s: float,
        sample_rate_hz: float,
        fmax: Optional[float] = None,
        origin_s: float = 0.0,
        detrend: bool = True,
        ts_unit: str = "s",
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ):
        _validate_time_window(window_s, overlap_s)
        if subwindow_s <= 0 or subwindow_s > window_s:
            raise ValueError("subwindow_s must satisfy 0 < subwindow_s <= window_s")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if fmax is not None and fmax < 0:
            raise ValueError("fmax must be >= 0")
        self._ts_unit = ts_unit
        self._impl = _senpy.StreamingNUSTFT(
            secperseg=float(window_s),
            secoverlap=float(overlap_s),
            secpersub=float(subwindow_s),
            sample_rate=float(sample_rate_hz),
            fmax=float(fmax) if fmax is not None else 0.0,
            origin=float(origin_s),
            detrend=bool(detrend),
            min_samples=_validate_min_samples(min_samples),
        )
        self.frequencies = self._impl.frequencies()

    def push(
        self,
        timestamps: NDArray[np.float64],
        signal: NDArray[np.float64],
    ) -> List[StreamingWindow]:
        """Appends samples and returns every window that closed as a result.

        Timestamps must be non-decreasing across the life of the object; a sample belonging to a
        subwindow the stream has already passed cannot be folded in and is counted by
        :attr:`dropped_samples` instead.
        """
        t = _timestamps_to_seconds(timestamps, self._ts_unit)
        values = np.asarray(signal, dtype=np.float64)
        if t.shape != values.shape:
            raise ValueError("timestamps and signal must have the same length")
        return [
            StreamingWindow(**entry)
            for entry in self._impl.push(np.ascontiguousarray(t), np.ascontiguousarray(values))
        ]

    def flush(self) -> List[StreamingWindow]:
        """Returns every window still open, however partial, and resets the transform.

        Unlike :meth:`push`, this reports the trailing window ``compute_nustft`` stops short of.
        """
        return [StreamingWindow(**entry) for entry in self._impl.flush()]

    @property
    def dropped_samples(self) -> int:
        """Samples that could not be placed: non-finite, or arriving out of order."""
        return self._impl.dropped_samples

    @property
    def skipped_windows(self) -> int:
        """Windows that held samples, but fewer than ``min_samples``, as ``compute_nustft`` skips."""
        return self._impl.skipped_windows

    @property
    def open_windows(self) -> int:
        return self._impl.open_windows


def compute_nustft_streaming(
    timestamps: NDArray[np.float64],
    signal: NDArray[np.float64],
    window_s: float,
    overlap_s: float,
    subwindow_s: float,
    sample_rate_hz: Optional[float] = None,
    ts_unit: str = "s",
    fmax: Optional[float] = None,
    detrend: bool = True,
    chunk: int = 1024,
    *,
    origin_s: Union[None, float, str] = None,
    empty_windows: Optional[str] = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> NUSTFTResult:
    """Runs a whole array through :class:`StreamingNUSTFT` and returns an ``NUSTFTResult``.

    Mostly a convenience for testing the streaming path against ``compute_nustft``: with
    ``sample_rate_hz`` left to default it reproduces that function's coefficients to roughly
    1e-13 relative. Production callers stream with :class:`StreamingNUSTFT` directly.
    ``origin_s``, ``empty_windows`` and ``min_samples`` are as in ``compute_nustft``, and the
    windows are the same ones.
    """
    empty_windows = _resolve_empty_windows(empty_windows)
    min_samples = _validate_min_samples(min_samples)
    values = np.asarray(signal, dtype=np.float64)
    if len(timestamps) < 2:
        raise ValueError("compute_nustft_streaming requires at least two timestamps")
    # The grid is built from the samples in time order. The stream sees them as given: it drops
    # a sample whose subwindow it has already closed (counted in dropped_samples), but disorder
    # within one subwindow goes unnoticed. The counts below report what it actually used.
    grid = window_grid(
        np.sort(np.asarray(timestamps, dtype=np.float64)),
        window_s,
        overlap_s,
        origin_s=origin_s,
        min_samples=min_samples,
        ts_unit=ts_unit,
    )
    # Stream times relative to the origin, exactly as the batch transforms see them.
    t, _ = _relative_seconds(timestamps, ts_unit, grid.origin_s, grid.hop_s)
    if sample_rate_hz is None:
        sample_rate_hz = 1.0 / float(np.median(np.diff(t)))

    transform = StreamingNUSTFT(
        window_s=window_s,
        overlap_s=overlap_s,
        subwindow_s=subwindow_s,
        sample_rate_hz=sample_rate_hz,
        fmax=fmax,
        origin_s=0.0,
        detrend=detrend,
        min_samples=min_samples,
    )
    windows: List[StreamingWindow] = []
    for start in range(0, len(t), chunk):
        windows.extend(transform.push(t[start : start + chunk], values[start : start + chunk]))
    # push() only reports windows the stream has passed the end of. compute_nustft, which sees
    # where the recording stops, also emits a final window that ends within one sample period
    # of the last timestamp; take that one out of the flush and drop anything past the grid.
    windows.extend(transform.flush())
    windows = [w for w in windows if w.index < grid.n_windows]

    if grid.n_windows == 0 or (empty_windows == "drop" and not windows):
        raise ValueError("compute_nustft_streaming requires enough data for at least one window")
    frequencies = transform.frequencies
    coefficients = (
        np.stack([w.coefficients for w in windows])
        if windows
        else np.empty((0, len(frequencies)), dtype=np.complex128)
    )
    emitted = np.array([w.index for w in windows], dtype=np.int64)
    # Report what the stream actually used: samples it had to drop (out of
    # order, or non-finite) are missing from its counts, so a window the grid
    # thinks is full may have come out sparse or not at all.
    sample_count = grid.sample_count.copy()
    sample_count[emitted] = [w.sample_count for w in windows]
    laid_out = _grid_rows(
        coefficients,
        emitted,
        sample_count,
        window_s=window_s,
        overlap_s=overlap_s,
        min_samples=min_samples,
        empty_windows=empty_windows,
    )
    was_emitted = np.zeros(grid.n_windows, dtype=bool)
    was_emitted[emitted] = True
    return NUSTFTResult(
        frequencies=frequencies,
        times=laid_out["times"],
        coefficients=laid_out["rows"],
        window_index=laid_out["window_index"],
        sample_count=laid_out["sample_count"],
        valid=was_emitted[laid_out["window_index"]],
        origin_s=grid.origin_s,
    )


def compute_nufft_spectrogram(
    timestamps: NDArray[np.float64],
    signal: NDArray[np.float64],
    window_s: float,
    overlap_s: float,
    ts_unit: str = "s",
    target_fs: Optional[float] = None,
    kind: str = "magnitude",
    detrend: bool = True,
    *,
    origin_s: Union[None, float, str] = None,
    empty_windows: Optional[str] = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> SpectrogramResult:
    """Compute a FINUFFT-backed spectrogram from non-uniform samples.

    Use ``compute_nustft`` when phase or custom spectral reductions are needed.
    ``origin_s``, ``empty_windows`` and ``min_samples`` are as there; with
    ``empty_windows="keep"`` rows without enough data hold NaN.
    """
    empty_windows = _resolve_empty_windows(empty_windows)
    return _compute_nufft_spectrogram(
        timestamps,
        signal,
        window_s,
        overlap_s,
        ts_unit=ts_unit,
        target_fs=target_fs,
        kind=kind,
        detrend=detrend,
        origin_s=origin_s,
        empty_windows=empty_windows,
        min_samples=min_samples,
    )


def _compute_nufft_spectrogram(
    timestamps: NDArray[np.float64],
    signal: NDArray[np.float64],
    window_s: float,
    overlap_s: float,
    *,
    ts_unit: str,
    target_fs: Optional[float],
    kind: str,
    detrend: bool,
    origin_s: Union[None, float, str],
    empty_windows: str,
    min_samples: int,
) -> SpectrogramResult:
    if len(timestamps) != len(signal):
        raise ValueError("timestamps and signal must have the same length")
    if len(timestamps) < 2:
        raise ValueError("compute_nufft_spectrogram requires at least two timestamps")
    _validate_time_window(window_s, overlap_s)
    if target_fs is not None and target_fs < 0:
        raise ValueError("target_fs must be >= 0")
    min_samples = _validate_min_samples(min_samples)

    t, origin = _relative_seconds(timestamps, ts_unit, origin_s, float(window_s - overlap_s))
    target_fs_val = target_fs if target_fs is not None else 0.0
    normalized_kind = _normalize_spectral_kind(kind)

    try:
        result = _senpy.compute_nufft_spectrogram(
            t,
            np.asarray(signal, dtype=np.float64),
            float(window_s),
            float(overlap_s),
            float(target_fs_val),
            normalized_kind,
            bool(detrend),
            0.0,
            min_samples,
        )
    except RuntimeError as e:
        raise RuntimeError(f"C++ NUFFT spectrogram computation failed: {e}") from e

    grid_counts = result["grid_sample_counts"]
    if (empty_windows == "drop" and len(result["times"]) == 0) or len(grid_counts) == 0:
        raise ValueError("compute_nufft_spectrogram requires enough data for at least one window")

    Sxx = result["Sxx"]
    if Sxx.shape[0] == 0:
        Sxx = np.empty((0, len(result["freqs"])), dtype=np.float64)
    laid_out = _grid_rows(
        Sxx,
        result["window_indices"],
        grid_counts,
        window_s=window_s,
        overlap_s=overlap_s,
        min_samples=min_samples,
        empty_windows=empty_windows,
    )
    return SpectrogramResult(
        frequencies=result["freqs"],
        times=laid_out["times"],
        Sxx=laid_out["rows"],
        method="nufft",
        kind=normalized_kind,
        window_index=laid_out["window_index"],
        sample_count=laid_out["sample_count"],
        valid=laid_out["valid"],
        origin_s=origin,
    )


def compute_nufft_welch(
    timestamps: NDArray[np.float64],
    signal: NDArray[np.float64],
    window_s: float,
    overlap_s: float,
    ts_unit: str = "s",
    target_fs: Optional[float] = None,
    kind: str = "psd",
    average: str = "mean",
    detrend: bool = True,
    *,
    origin_s: Union[None, float, str] = None,
    empty_windows: Optional[str] = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute a Welch-style average spectrum using FINUFFT windows.

    Empty windows never enter the average: ``"drop"`` leaves them out and
    ``"keep"`` fills them with NaN, which the NaN-aware average skips.
    """
    empty_windows = _resolve_empty_windows(empty_windows)
    return _compute_nustft(
        timestamps,
        signal,
        window_s,
        overlap_s,
        ts_unit=ts_unit,
        target_fs=target_fs,
        detrend=detrend,
        origin_s=origin_s,
        empty_windows=empty_windows,
        min_samples=min_samples,
    ).welch(kind=kind, average=average)


def compute_stacked_spectrograms(
    accel: "AccelerometerData",
    window_s: float,
    overlap_s: float,
    target_fs: Optional[float] = None,
    kind: str = "magnitude",
    detrend: bool = True,
    channels: Optional[List[str]] = None,
    use_diff: bool = True,
    *,
    origin_s: Union[None, float, str] = None,
    empty_windows: Optional[str] = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> StackedSpectrogramResult:
    """Compute per-axis NUFFT spectrograms and stack along the channel axis.

    Computes individual spectrograms for each requested channel from a single
    ``AccelerometerData`` object, aligns them to a shared time grid, and
    stacks the results into a ``(T, F, C)`` tensor.

    Available channels (``"x"``, ``"y"``, ``"z"``, ``"mag"``, ``"jerk"``):
        ``"x"``    – X-axis acceleration spectrogram
        ``"y"``    – Y-axis acceleration spectrogram
        ``"z"``    – Z-axis acceleration spectrogram
        ``"mag"``  – Euclidean magnitude spectrogram, ``||x, y, z||``
        ``"jerk"`` – Scalar jerk (time-derivative of ||acceleration||) spectrogram

    Args:
        accel: Container holding aligned timestamps and x/y/z components.
        window_s: NUFFT window duration in seconds.
        overlap_s: Window overlap in seconds; must satisfy ``0 <= overlap_s < window_s``.
        target_fs: If provided, output frequency bins are capped at ``target_fs / 2`` Hz.
        kind: Spectral quantity — ``"magnitude"``, ``"power"``, or ``"psd"``.
        detrend: Subtract each window's mean before the Hann taper.
        channels: Ordered list of channels to include. Defaults to
            ``["x", "y", "z", "mag", "jerk"]``.
        use_diff: Finite-difference jerk (``True``) or C++ gradient estimator (``False``).
        origin_s, empty_windows, min_samples: As in :func:`compute_nustft`. The
            origin is resolved once from the accelerometer timestamps and
            shared by every channel, so a window index names the same window
            in every channel and rows are matched by it.

    Returns:
        ``StackedSpectrogramResult`` with ``Sxx`` shaped ``(T, F, len(channels))``.
    """
    if channels is None:
        channels = list(STACKED_SPECTROGRAM_CHANNELS)

    empty_windows = _resolve_empty_windows(empty_windows)
    _validate_time_window(window_s, overlap_s)
    t_s = accel.timestamps_s
    origin = _resolve_origin(origin_s, float(t_s[0]), float(window_s - overlap_s))

    def _spec(
        signal: NDArray[np.float64],
        timestamps: NDArray[np.float64] = t_s,
    ) -> SpectrogramResult:
        return _compute_nufft_spectrogram(
            timestamps,
            np.ascontiguousarray(signal, dtype=np.float64),
            window_s,
            overlap_s,
            ts_unit="s",
            target_fs=target_fs,
            kind=kind,
            detrend=detrend,
            origin_s=origin,
            empty_windows=empty_windows,
            min_samples=min_samples,
        )

    channel_specs: Dict[str, SpectrogramResult] = {}
    for ch in channels:
        if ch == "x":
            channel_specs["x"] = _spec(accel.x)
        elif ch == "y":
            channel_specs["y"] = _spec(accel.y)
        elif ch == "z":
            channel_specs["z"] = _spec(accel.z)
        elif ch == "mag":
            channel_specs["mag"] = _spec(compute_magnitude(accel.x, accel.y, accel.z))
        elif ch == "jerk":
            jerk_data = compute_jerk(
                accel.timestamps_s,
                accel.x,
                accel.y,
                accel.z,
                ts_unit="s",
                use_diff=use_diff,
            )
            channel_specs["jerk"] = _spec(jerk_data.jerk, timestamps=jerk_data.timestamps_s)
        else:
            raise ValueError(
                f"Unknown channel {ch!r}. Must be one of: {STACKED_SPECTROGRAM_CHANNELS}"
            )

    ref_ch = next((c for c in channels if c != "jerk"), channels[0])
    ref_spec = channel_specs[ref_ch]
    ref_index = ref_spec.window_index
    frequencies = ref_spec.frequencies
    T = len(ref_index)
    F = len(frequencies)
    C = len(channels)

    # Every channel shares the origin, so a window index names the same window
    # in each. Match rows by index.
    Sxx = np.full((T, F, C), np.nan, dtype=np.float64)
    valid = ref_spec.valid.copy()
    unmatched: Dict[str, int] = {}
    for i, ch in enumerate(channels):
        spec = channel_specs[ch]
        position = np.searchsorted(spec.window_index, ref_index)
        position = np.minimum(position, max(len(spec.window_index) - 1, 0))
        found = (
            spec.window_index[position] == ref_index
            if len(spec.window_index)
            else np.zeros(T, dtype=bool)
        )
        Sxx[found, :, i] = spec.Sxx[position[found]]
        channel_valid = np.zeros(T, dtype=bool)
        channel_valid[found] = spec.valid[position[found]]
        # Rows the reference itself has no data for are expected to be NaN.
        n_unmatched = int(np.count_nonzero(ref_spec.valid & ~channel_valid))
        if n_unmatched:
            unmatched[ch] = n_unmatched
        valid &= channel_valid

    if unmatched:
        detail = ", ".join(
            f"{ch}: {count}/{T} time bins" for ch, count in unmatched.items()
        )
        warnings.warn(
            "compute_stacked_spectrograms produced NaN-filled time bins for "
            f"channels with no data in windows the {ref_ch!r} channel has ({detail}). "
            "These appear as NaN in Sxx with valid=False; downstream reductions must "
            "handle them (e.g. np.nanmean) or filter them out.",
            RuntimeWarning,
            stacklevel=2,
        )

    return StackedSpectrogramResult(
        frequencies=frequencies,
        times=ref_spec.times,
        Sxx=Sxx,
        channels=channels,
        kind=_normalize_spectral_kind(kind),
        window_index=ref_index,
        sample_count=ref_spec.sample_count,
        valid=valid,
        origin_s=origin,
    )


def compute_jerk(
    timestamps: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    ts_unit: str = "s",
    use_diff: bool = True
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

    result = compute_jerk_microseconds(timestamps_us, x, y, z, use_diff)
    return result


def compute_jerk_microseconds(
    timestamps: NDArray[np.int64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    diff: bool
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

    result = _senpy.compute_jerk(timestamps, x, y, z, diff=diff)
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


def compute_uniform_spectrogram(
    signal: NDArray[np.float64], fs: float, nperseg: int, noverlap: int
) -> SpectrogramResult:
    """
    Compute a spectrogram for an already-uniform signal using FFT-based STFT.

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
    result = _senpy.compute_spectrogram(
        np.asarray(signal, dtype=np.float64), fs, nperseg, noverlap
    )
    return SpectrogramResult(
        frequencies=result["freqs"],
        times=result["times"],
        Sxx=result["Sxx"],
        kind="magnitude",
        method="uniform_fft",
    )


def compute_resampled_spectrogram(
    timestamps: NDArray[np.float64],
    signal: NDArray[np.float64],
    target_fs: float,
    window_s: float,
    overlap_s: float,
    ts_unit: str = "s",
    resample_method: str = "linear",
) -> SpectrogramResult:
    """Compute a legacy resample-then-FFT spectrogram for comparison.

    This path performs time-domain interpolation before spectral analysis. It
    remains available for compatibility and controlled comparisons, but
    ``compute_nustft`` is preferred for non-uniform or jittered timestamps.
    """
    _validate_time_window(window_s, overlap_s)
    if target_fs <= 0:
        raise ValueError("target_fs must be > 0")
    nperseg = int(round(window_s * target_fs))
    noverlap = int(round(overlap_s * target_fs))
    _validate_sample_window(nperseg, noverlap)

    zeros = np.zeros_like(signal, dtype=np.float64)
    if resample_method == "linear":
        resampled = resample_accelerometer(
            timestamps, signal, zeros, zeros, target_fs, ts_unit=ts_unit
        )
    elif resample_method == "cubic":
        resampled = resample_accelerometer_cubic(
            timestamps, signal, zeros, zeros, target_fs, ts_unit=ts_unit
        )
    else:
        raise ValueError("resample_method must be 'linear' or 'cubic'")

    return compute_uniform_spectrogram(resampled.x, target_fs, nperseg, noverlap)


def compute_spectrogram(
    signal: NDArray[np.float64], fs: float, nperseg: int, noverlap: int
) -> SpectrogramResult:
    """Compatibility alias for ``compute_uniform_spectrogram``.

    Deprecated in senpy 1.0. This function assumes ``signal`` is already on a
    uniform sample grid. For jittered timestamps, use ``compute_nustft``.
    """
    warnings.warn(
        "compute_spectrogram is deprecated; use compute_uniform_spectrogram "
        "for uniform signals or compute_nustft for non-uniform timestamps",
        FutureWarning,
        stacklevel=2,
    )
    return compute_uniform_spectrogram(signal, fs, nperseg, noverlap)


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
        kind="magnitude",
        method="uniform_fft",
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
    scaling_factor: float = 60.0,
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
        scaling_factor=scaling_factor,
    )


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


def smooth_spectrogram_peaks(
    spectrogram_peaks: np.ndarray,
    sampling_rate: float,
    max_change_per_sec: float = 10.0,
    filter_sigma: float = 2.0,
) -> np.ndarray:
    """
    Smooth the spectrogram peaks using a Gaussian filter,
    then enforce a maximum allowable rate of change per second.

    Parameters:
        spectrogram_peaks: array of peak values (e.g., heart rate in BPM)
        sampling_rate: frequency at which spectrogram peaks are sampled (Hz)
        max_change_per_sec: maximum allowed delta in BPM per second
        filter_sigma: standard deviation for Gaussian filter

    Returns:
        np.ndarray: smoothed peak signal
    """
    return _senpy.smooth_spectrogram_peaks(
        spectrogram_peaks, sampling_rate, max_change_per_sec, filter_sigma
    )


# Constants for convenience
class FrequencyRanges:
    """Common frequency ranges for physiological signals."""

    # Breathing rate ranges (Hz, N/60 = Breaths/Min)
    BR_RESTING_MIN = 9 / 60
    BR_RESTING_MAX = 25 / 60

    # Heart rate ranges (Hz, N/60 = BPM)
    HR_RESTING_MIN = 30 / 60
    HR_RESTING_MAX = 120 / 60


class SamplingRates:
    """Common sampling rates for sensor data."""

    ACCELEROMETER_STANDARD = 50.0  # Hz
    ACCELEROMETER_HIGH = 100.0  # Hz


__all__ = [
    "AccelerometerData",
    "NUSTFTResult",
    "SpectrogramResult",
    "StackedSpectrogramResult",
    "STACKED_SPECTROGRAM_CHANNELS",
    "JerkData",
    "ShortTimeFTResult",
    "MotionFeatures",
    "resample_accelerometer",
    "resample_accelerometer_cubic",
    "resample_accelerometer_cubic_microseconds",
    "compute_nustft",
    "compute_nustft_streaming",
    "StreamingNUSTFT",
    "StreamingWindow",
    "compute_nufft_spectrogram",
    "compute_nufft_welch",
    "compute_stacked_spectrograms",
    "window_grid",
    "WindowGrid",
    "compute_jerk",
    "compute_magnitude",
    "compute_uniform_spectrogram",
    "compute_resampled_spectrogram",
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
    "smooth_spectrogram_peaks",
    "FrequencyRanges",
    "SamplingRates",
]
