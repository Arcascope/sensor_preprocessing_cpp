"""JAX-native non-uniform spectral transforms.

This module is deliberately separate from :mod:`senpy.api`: its result arrays
remain JAX arrays, so a CUDA-enabled JAX installation can execute the NUFFT
without converting signal samples through NumPy or ``senpy._core``.

Install ``jax-finufft`` alongside the JAX build for the desired platform.  A
CUDA-enabled jax-finufft build dispatches ``nufft1`` to cuFINUFFT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple


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


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _timestamp_scale(ts_unit: str) -> float:
    units = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
    try:
        return units[ts_unit]
    except KeyError as exc:
        raise ValueError("ts_unit must be one of: 's', 'ms', 'us'") from exc


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
    "compute_nustft",
    "compute_nufft_spectrogram",
    "compute_nufft_welch",
]
