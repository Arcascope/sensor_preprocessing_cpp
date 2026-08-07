"""JAX-native non-uniform spectral transforms.

This module is deliberately separate from :mod:`senpy.api`: its result arrays
remain JAX arrays, so a CUDA-enabled JAX installation can execute the NUFFT
without converting signal samples through NumPy or ``senpy._core``.

Install ``jax-finufft`` alongside the JAX build for the desired platform.  A
CUDA-enabled jax-finufft build dispatches ``nufft1`` to cuFINUFFT.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, DefaultDict, List, Literal, Optional, Sequence, Tuple

import numpy as np


AXIS_ORDER_TIME_FREQUENCY = "time_frequency"


def _dependencies() -> Tuple[Any, Any, Any]:
    """Import optional dependencies only when this backend is used."""
    try:
        import jax
        import jax.numpy as jnp
        from jax_finufft import nufft1
    except ImportError as exc:
        raise ImportError(
            "senpy.jax requires JAX and jax-finufft. Install a JAX build for "
            "your device, then install 'senpy[jax]'."
        ) from exc
    return jax, jnp, nufft1


def _normalize_kind(kind: str) -> str:
    normalized = str(kind).replace("-", "_").lower()
    aliases = {"mag": "magnitude", "magnitude": "magnitude", "power": "power", "psd": "psd"}
    if normalized not in aliases:
        raise ValueError("kind must be one of: 'mag', 'magnitude', 'power', 'psd'")
    return aliases[normalized]


@dataclass(frozen=True)
class JaxSpectrogramResult:
    """A JAX-backed time-frequency surface with shape ``(time, frequency)``."""

    frequencies: Any
    times: Any
    Sxx: Any
    kind: str = "magnitude"
    method: str = "jax_finufft"
    axis_order: str = AXIS_ORDER_TIME_FREQUENCY


@dataclass(frozen=True)
class JaxNUSTFTResult:
    """Complex JAX-backed type-1 NUFFT coefficients.

    ``coefficients``, ``frequencies``, and ``times`` stay on the active JAX
    device.  The numerical convention intentionally matches
    :func:`senpy.api.compute_nustft`.
    """

    frequencies: Any
    times: Any
    coefficients: Any
    method: str = "jax_finufft"
    axis_order: str = AXIS_ORDER_TIME_FREQUENCY

    @property
    def shape(self) -> Tuple[int, int]:
        return self.coefficients.shape

    @property
    def magnitude(self) -> Any:
        _, jnp, _ = _dependencies()
        return jnp.abs(self.coefficients)

    @property
    def power(self) -> Any:
        _, jnp, _ = _dependencies()
        return jnp.abs(self.coefficients) ** 2

    @property
    def psd(self) -> Any:
        # This matches the CPU API: coefficient scaling already contains the
        # density normalization, therefore PSD is the squared magnitude.
        return self.power

    def spectrogram(self, kind: str = "psd") -> JaxSpectrogramResult:
        kind = _normalize_kind(kind)
        if kind == "magnitude":
            surface = self.magnitude
        else:
            surface = self.power
        return JaxSpectrogramResult(
            frequencies=self.frequencies,
            times=self.times,
            Sxx=surface,
            kind=kind,
        )

    def welch(
        self,
        kind: str = "psd",
        average: Literal["mean", "median"] = "mean",
    ) -> Tuple[Any, Any]:
        _, jnp, _ = _dependencies()
        surface = self.spectrogram(kind).Sxx
        if average == "mean":
            return self.frequencies, jnp.nanmean(surface, axis=0)
        if average == "median":
            return self.frequencies, jnp.nanmedian(surface, axis=0)
        raise ValueError("average must be 'mean' or 'median'")


@dataclass(frozen=True)
class PackedNUSTFTWindowBatch:
    """One static-shape, cross-recording NUFFT window batch.

    The sample buffers are host NumPy arrays so they can be filled while
    discovering ragged windows.  Pass ``points``, ``signals``, and ``valid``
    directly to :func:`compute_nustft_window_batch`; JAX transfers them once
    to the selected device and keeps the returned coefficients there.

    ``recording_indices`` and ``window_indices`` identify each real row in
    the caller's original order.  A ``-1`` recording index marks final-batch
    padding and must be discarded with ``row_valid``.
    """

    points: np.ndarray
    signals: np.ndarray
    valid: np.ndarray
    row_valid: np.ndarray
    window_ss: np.ndarray
    recording_indices: np.ndarray
    window_indices: np.ndarray
    times: np.ndarray
    median_fs: np.ndarray
    nfft_padded: int
    window_s: float

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the static ``(B, M)`` point-buffer shape."""
        return tuple(self.points.shape)  # type: ignore[return-value]


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _timestamp_scale(ts_unit: str) -> float:
    units = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
    try:
        return units[ts_unit]
    except KeyError as exc:
        raise ValueError("ts_unit must be one of: 's', 'ms', 'us'") from exc


def _validate_nfft_padded(nfft_padded: int) -> int:
    nfft_padded = int(nfft_padded)
    if nfft_padded < 2 or nfft_padded & (nfft_padded - 1):
        raise ValueError("nfft_padded must be a power of two and at least 2")
    return nfft_padded


@lru_cache(maxsize=None)
def _nustft_window_batch_transform(
    nfft_padded: int,
    detrend: bool,
    eps: float,
) -> Any:
    """Build and cache a static-shape batched type-1 NUFFT executable."""
    jax, jnp, nufft1 = _dependencies()
    mode_indices = jnp.concatenate(
        (jnp.arange(nfft_padded // 2, nfft_padded), jnp.array([0]))
    )
    phase_correction = jnp.where(
        jnp.arange(nfft_padded // 2 + 1) % 2 == 0, 1.0, -1.0
    )

    def one_window(points_m: Any, strengths_3m: Any) -> Any:
        # The three accelerometer channels have identical non-uniform points.
        # jax-finufft can therefore lower this as a transform stack (the
        # cuFINUFFT ntrans analogue) rather than three unrelated transforms.
        modes = nufft1(nfft_padded, strengths_3m, points_m, eps=eps, iflag=1)
        return modes[:, mode_indices] * phase_correction

    @jax.jit
    def transform(points: Any, signals: Any, valid: Any, median_fs: Any) -> Any:
        valid_float = valid.astype(signals.dtype)
        counts = jnp.sum(valid_float, axis=1)
        safe_counts = jnp.maximum(counts, 1.0)
        means = jnp.sum(signals * valid_float[:, None, :], axis=2) / safe_counts[:, None]
        centered = signals - means[:, :, None] if detrend else signals

        # ``points`` are local NUFFT coordinates in [-pi, pi).  Invalid
        # samples have zero strength, so their in-range value is irrelevant.
        tau = (points + jnp.pi) / (2.0 * jnp.pi)
        hann = valid_float * 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * tau))
        strengths = (centered * hann[:, None, :]).astype(
            jnp.result_type(signals, 1j)
        )
        coefficients = jax.vmap(one_window, in_axes=(0, 0))(points, strengths)

        window_ss = jnp.sum(hann * hann, axis=1)
        valid_scale = (counts > 0.0) & (window_ss > 0.0) & (median_fs > 0.0)
        scale = jnp.where(
            valid_scale,
            1.0 / jnp.sqrt(median_fs * jnp.maximum(window_ss, 1.0)),
            0.0,
        )
        return coefficients * scale[:, None, None]

    return transform


def compute_nustft_window_batch(
    points: Any,
    signals: Any,
    valid: Any,
    *,
    nfft_padded: int,
    median_fs: Any,
    detrend: bool = True,
    eps: float = 1e-6,
) -> Any:
    """Transform a static batch of packed three-axis NUFFT windows.

    Args:
        points: Local type-1 NUFFT coordinates with shape ``[B, M]``.  Real
            samples must be in ``[-pi, pi)``; invalid coordinates are ignored.
        signals: Un-detrended accelerometer samples shaped ``[B, 3, M]``.
        valid: Boolean sample mask shaped ``[B, M]``.
        nfft_padded: Shared power-of-two output-mode count for this bucket.
        median_fs: Positive scalar or ``[B]`` array used for each row's CPU
            compatible density normalization.
        detrend: Subtract each valid row/channel mean before applying Hann.
        eps: Requested jax-finufft relative accuracy.

    Returns:
        JAX complex coefficients shaped ``[B, 3, nfft_padded // 2 + 1]``.
        Padded rows and samples contribute exactly zero.  The result is never
        converted through NumPy.
    """
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    nfft_padded = _validate_nfft_padded(nfft_padded)
    points_shape = tuple(points.shape)
    signals_shape = tuple(signals.shape)
    valid_shape = tuple(valid.shape)
    if len(points_shape) != 2 or points_shape != valid_shape:
        raise ValueError("points and valid must both have shape [B, M]")
    if len(signals_shape) != 3 or signals_shape != (points_shape[0], 3, points_shape[1]):
        raise ValueError("signals must have shape [B, 3, M] matching points")

    _, jnp, _ = _dependencies()
    median_fs_array = jnp.asarray(median_fs)
    if median_fs_array.ndim > 1 or (
        median_fs_array.ndim == 1 and median_fs_array.shape[0] != points_shape[0]
    ):
        raise ValueError("median_fs must be a scalar or have shape [B]")
    median_fs_array = jnp.broadcast_to(median_fs_array, (points_shape[0],))
    return _nustft_window_batch_transform(nfft_padded, bool(detrend), float(eps))(
        jnp.asarray(points),
        jnp.asarray(signals),
        jnp.asarray(valid, dtype=bool),
        median_fs_array,
    )


def pack_nustft_window_batches(
    recordings: Sequence[Tuple[Any, Any]],
    *,
    window_s: float,
    overlap_s: float,
    batch_size: int,
    ts_unit: str = "s",
) -> Tuple[PackedNUSTFTWindowBatch, ...]:
    """Pack many timestamped ``[N, 3]`` recordings into static NUFFT batches.

    This intentionally performs only window discovery and host-side padding;
    it does not import JAX or run a transform.  Thus data loading can prepare
    batches before a GPU worker consumes them with
    :func:`compute_nustft_window_batch`.  Windows are bucketed by padded FFT
    length and source width.  Their metadata restores recording/window order
    after ``row_valid`` removes the final batch padding.
    """
    if window_s <= 0.0:
        raise ValueError("window_s must be > 0")
    if overlap_s < 0.0 or overlap_s >= window_s:
        raise ValueError("overlap_s must satisfy 0 <= overlap_s < window_s")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    scale = _timestamp_scale(ts_unit)
    groups: DefaultDict[Tuple[int, int], List[Tuple[Any, ...]]] = defaultdict(list)

    for recording_index, (timestamps, samples) in enumerate(recordings):
        t = np.asarray(timestamps, dtype=np.float64)
        s = np.asarray(samples)
        if t.ndim != 1 or s.ndim != 2 or s.shape != (t.size, 3):
            raise ValueError(
                "each recording must be a (timestamps[N], samples[N, 3]) pair"
            )
        if not np.issubdtype(s.dtype, np.number) or np.iscomplexobj(s):
            raise ValueError("accelerometer samples must be real numeric values")
        if t.size < 2:
            raise ValueError("each recording requires at least two timestamps")
        if not np.all(np.isfinite(t)):
            raise ValueError("timestamps must be finite")
        if not np.all(np.isfinite(s)):
            raise ValueError("accelerometer samples must be finite")
        t = (t - t[0]) * scale
        diffs = np.diff(t)
        if np.any(diffs < 0.0):
            raise ValueError("timestamps must be sorted")
        positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if positive_diffs.size == 0:
            raise ValueError("timestamps must contain at least one positive time step")
        median_fs = 1.0 / float(np.sort(positive_diffs)[positive_diffs.size // 2])
        nfft = int(window_s * median_fs)
        if nfft < 2:
            raise ValueError("window_s is too short for the observed sampling density")
        nfft_padded = _next_power_of_two(nfft)
        hop_s = window_s - overlap_s
        start = 0.0
        window_index = 0
        while start + window_s <= t[-1] + 1.0 / median_fs:
            first = int(np.searchsorted(t, start, side="left"))
            last = int(np.searchsorted(t, start + window_s, side="left"))
            if last - first >= 4:
                count = last - first
                source_width = _next_power_of_two(count)
                local_points = 2.0 * np.pi * ((t[first:last] - start) / window_s) - np.pi
                groups[(nfft_padded, source_width)].append(
                    (
                        local_points,
                        np.asarray(s[first:last]),
                        recording_index,
                        window_index,
                        start + window_s / 2.0,
                        median_fs,
                    )
                )
            start += hop_s
            window_index += 1

    packed: List[PackedNUSTFTWindowBatch] = []
    for (nfft_padded, source_width), rows in sorted(groups.items()):
        for chunk_start in range(0, len(rows), batch_size):
            chunk = rows[chunk_start : chunk_start + batch_size]
            dtype = np.result_type(*(row[1].dtype for row in chunk), np.float32)
            points = np.zeros((batch_size, source_width), dtype=dtype)
            signals = np.zeros((batch_size, 3, source_width), dtype=dtype)
            valid = np.zeros((batch_size, source_width), dtype=bool)
            row_valid = np.zeros(batch_size, dtype=bool)
            recording_indices = np.full(batch_size, -1, dtype=np.int64)
            window_indices = np.full(batch_size, -1, dtype=np.int64)
            times = np.full(batch_size, np.nan, dtype=np.float64)
            median_fss = np.ones(batch_size, dtype=dtype)
            for row_index, (row_points, row_signals, recording_index, window_index, time, row_fs) in enumerate(chunk):
                count = row_points.size
                points[row_index, :count] = row_points
                signals[row_index, :, :count] = np.asarray(row_signals, dtype=dtype).T
                valid[row_index, :count] = True
                row_valid[row_index] = True
                recording_indices[row_index] = recording_index
                window_indices[row_index] = window_index
                times[row_index] = time
                median_fss[row_index] = row_fs
            tau = (points + np.pi) / (2.0 * np.pi)
            hann = valid * 0.5 * (1.0 - np.cos(2.0 * np.pi * tau))
            packed.append(
                PackedNUSTFTWindowBatch(
                    points=points,
                    signals=signals,
                    valid=valid,
                    row_valid=row_valid,
                    window_ss=np.sum(hann * hann, axis=1),
                    recording_indices=recording_indices,
                    window_indices=window_indices,
                    times=times,
                    median_fs=median_fss,
                    nfft_padded=nfft_padded,
                    window_s=float(window_s),
                )
            )
    return tuple(packed)


def compute_nustft(
    timestamps: Any,
    signal: Any,
    window_s: float,
    overlap_s: float,
    ts_unit: str = "s",
    target_fs: Optional[float] = None,
    detrend: bool = True,
    eps: float = 1e-6,
) -> JaxNUSTFTResult:
    """Compute a device-resident non-uniform STFT using ``jax-finufft``.

    The transform uses type-1 FINUFFT with the same positive-frequency mode
    order, Hann taper, detrending, and density scaling as the CPU API. The
    input sample arrays are never copied to NumPy. Small scalar/window-index
    metadata is synchronized to construct the ragged sliding windows.

    Args:
        timestamps: Sorted one-dimensional JAX-compatible timestamps.
        signal: One-dimensional JAX-compatible sample values.
        window_s: Window duration in seconds.
        overlap_s: Window overlap in seconds.
        ts_unit: ``"s"``, ``"ms"``, or ``"us"``.
        target_fs: Optional maximum output sample rate. It must not exceed the
            observed median sample rate, so requested bins are exact NUFFT bins.
        detrend: Subtract each window mean before applying the Hann taper.
        eps: Requested jax-finufft relative accuracy. Use ``1e-6`` for typical
            GPU float32 use; enable JAX x64 before importing JAX for float64.
    """
    if not window_s > 0.0:
        raise ValueError("window_s must be > 0")
    if overlap_s < 0.0 or overlap_s >= window_s:
        raise ValueError("overlap_s must satisfy 0 <= overlap_s < window_s")
    if target_fs is not None and target_fs < 0.0:
        raise ValueError("target_fs must be >= 0")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    jax, jnp, nufft1 = _dependencies()
    raw_timestamps = jnp.asarray(timestamps)
    # The CPU API reports times relative to the first sample. Center before the
    # unit conversion for the same convention and to preserve float32 precision
    # when callers provide large absolute timestamps.
    t = (raw_timestamps - raw_timestamps[0]) * _timestamp_scale(ts_unit)
    s = jnp.asarray(signal)
    if t.ndim != 1 or s.ndim != 1:
        raise ValueError("timestamps and signal must be one-dimensional")
    if t.shape != s.shape:
        raise ValueError("timestamps and signal must have the same shape")
    if t.size < 2:
        raise ValueError("compute_nustft requires at least two timestamps")

    # Match the CPU implementation's upper median among finite positive steps.
    diffs = t[1:] - t[:-1]
    positive = jnp.isfinite(diffs) & (diffs > 0.0)
    n_positive = int(jax.device_get(jnp.sum(positive)))
    if n_positive == 0:
        raise ValueError("timestamps must contain at least one positive time step")
    ordered_diffs = jnp.sort(jnp.where(positive, diffs, jnp.inf))
    dt_median = float(jax.device_get(ordered_diffs[n_positive // 2]))
    if not math.isfinite(dt_median) or dt_median <= 0.0:
        raise ValueError("median timestamp spacing must be finite and > 0")
    median_fs = 1.0 / dt_median

    if target_fs is not None and target_fs > median_fs:
        raise ValueError(
            "target_fs cannot exceed the observed median sampling rate in the JAX backend"
        )

    nfft = int(window_s * median_fs)
    if nfft < 2:
        raise ValueError("window_s is too short for the observed sampling density")
    nfft_padded = _next_power_of_two(nfft)
    n_pos_freqs = nfft_padded // 2 + 1

    t_end = float(jax.device_get(t[-1]))
    hop_s = window_s - overlap_s
    starts = []
    start = 0.0
    while start + window_s <= t_end + dt_median:
        starts.append(start)
        start += hop_s

    frequencies = jnp.arange(n_pos_freqs, dtype=t.dtype) / window_s
    if not starts:
        return JaxNUSTFTResult(
            frequencies=frequencies,
            times=jnp.empty((0,), dtype=t.dtype),
            coefficients=jnp.empty((0, n_pos_freqs), dtype=jnp.result_type(s, 1j)),
        )

    starts_device = jnp.asarray(starts, dtype=t.dtype)
    start_indices = jax.device_get(jnp.searchsorted(t, starts_device, side="left"))
    end_indices = jax.device_get(jnp.searchsorted(t, starts_device + window_s, side="left"))
    mode_indices = jnp.concatenate(
        (jnp.arange(nfft_padded // 2, nfft_padded), jnp.array([0]))
    )
    phase_correction = jnp.where(jnp.arange(n_pos_freqs) % 2 == 0, 1.0, -1.0)
    complex_dtype = jnp.result_type(s, 1j)

    # Overlapping jittered windows have ragged source coordinates. Compile one
    # transform per observed source-count and reuse it for matching windows.
    transforms = {}

    def transform_for(n_samples: int):
        if n_samples not in transforms:
            @jax.jit
            def transform(t_window: Any, s_window: Any, window_start: Any) -> Any:
                tau = (t_window - window_start) / window_s
                hann = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * tau))
                centered = s_window - jnp.mean(s_window) if detrend else s_window
                strengths = (centered * hann).astype(complex_dtype)
                points = 2.0 * jnp.pi * tau - jnp.pi
                modes = nufft1(nfft_padded, strengths, points, eps=eps, iflag=1)
                scale = 1.0 / jnp.sqrt(median_fs * jnp.sum(hann * hann))
                return modes[mode_indices] * phase_correction * scale

            transforms[n_samples] = transform
        return transforms[n_samples]

    coefficients = []
    output_times = []
    for index, (first, last) in enumerate(zip(start_indices.tolist(), end_indices.tolist())):
        if last - first < 4:
            continue
        window_start = jnp.asarray(starts[index], dtype=t.dtype)
        coefficients.append(transform_for(last - first)(t[first:last], s[first:last], window_start))
        output_times.append(index * hop_s + window_s / 2.0)

    if not coefficients:
        return JaxNUSTFTResult(
            frequencies=frequencies,
            times=jnp.empty((0,), dtype=t.dtype),
            coefficients=jnp.empty((0, n_pos_freqs), dtype=complex_dtype),
        )

    coefficients_array = jnp.stack(coefficients)
    frequencies_out = frequencies
    if target_fs is not None and target_fs > 0.0:
        n_target_freqs = int(math.floor((target_fs / 2.0) * window_s + 0.5)) + 1
        coefficients_array = coefficients_array[:, :n_target_freqs]
        frequencies_out = frequencies[:n_target_freqs]

    return JaxNUSTFTResult(
        frequencies=frequencies_out,
        times=jnp.asarray(output_times, dtype=t.dtype),
        coefficients=coefficients_array,
    )


def compute_nufft_spectrogram(
    timestamps: Any,
    signal: Any,
    window_s: float,
    overlap_s: float,
    ts_unit: str = "s",
    target_fs: Optional[float] = None,
    kind: str = "magnitude",
    detrend: bool = True,
    eps: float = 1e-6,
) -> JaxSpectrogramResult:
    """Compute a JAX-native NUFFT spectrogram without host array copies."""
    return compute_nustft(
        timestamps=timestamps,
        signal=signal,
        window_s=window_s,
        overlap_s=overlap_s,
        ts_unit=ts_unit,
        target_fs=target_fs,
        detrend=detrend,
        eps=eps,
    ).spectrogram(kind)


def compute_nufft_welch(
    timestamps: Any,
    signal: Any,
    window_s: float,
    overlap_s: float,
    ts_unit: str = "s",
    target_fs: Optional[float] = None,
    kind: str = "psd",
    average: Literal["mean", "median"] = "mean",
    detrend: bool = True,
    eps: float = 1e-6,
) -> Tuple[Any, Any]:
    """Compute a JAX-native Welch-style reduction over NUFFT windows."""
    return compute_nustft(
        timestamps=timestamps,
        signal=signal,
        window_s=window_s,
        overlap_s=overlap_s,
        ts_unit=ts_unit,
        target_fs=target_fs,
        detrend=detrend,
        eps=eps,
    ).welch(kind=kind, average=average)


__all__ = [
    "JaxNUSTFTResult",
    "JaxSpectrogramResult",
    "PackedNUSTFTWindowBatch",
    "compute_nustft_window_batch",
    "pack_nustft_window_batches",
    "compute_nustft",
    "compute_nufft_spectrogram",
    "compute_nufft_welch",
]
