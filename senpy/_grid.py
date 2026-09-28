"""The NUSTFT window grid every backend shares.

Window ``k`` (``k >= 0``) spans ``[origin + k * hop, origin + k * hop + window)``
on the timestamps' own clock, in seconds. Its start is computed by
multiplication, never by accumulating ``hop``, so every backend -- the C++
batch transform, the streaming transform, the JAX transform and the JAX packer
-- puts the same sample in the same window. The grid runs while a window ends
no later than one median sample period past the last timestamp, the bound the
batch transforms have always used.

Timestamps are measured from the origin once, by :func:`relative_seconds`,
and every backend is handed those relative times. That is what makes the
backends agree on a sample lying exactly on a window edge, and it keeps
microsecond Unix timestamps exact: differences are taken in the input's own
unit before scaling to seconds.

This module is NumPy only. :func:`window_grid` is re-exported as
``senpy.window_grid``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

#: Windows with fewer samples than this are not transformed.
DEFAULT_MIN_SAMPLES = 4

#: ``empty_windows`` values: ``"drop"`` omits windows with too few samples (the
#: 4.x default); ``"keep"`` reports every window on the grid, NaN-filled where
#: there was too little data.
EMPTY_WINDOW_MODES = ("drop", "keep")

OriginSpec = Union[None, float, str]

# "unix" rounds the first timestamp up to a whole hop. A first sample a hair
# past a hop boundary -- clock jitter, or a float that cannot hold the boundary
# exactly -- still anchors to that boundary rather than to the next one.
_UNIX_SNAP_FRACTION = 1e-6


def resolve_origin(origin_s: OriginSpec, first_timestamp_s: float, hop_s: float) -> float:
    """The absolute start of window 0, in seconds.

    ``None`` anchors the grid at the first sample, as senpy always has.
    ``"unix"`` anchors it to the first whole multiple of ``hop_s`` since the
    Unix epoch at or after the first sample, so windows from different
    recordings and sessions share one grid; the timestamps must then be Unix
    time. A number is used as given, on the timestamps' clock in seconds.
    """
    if origin_s is None:
        return float(first_timestamp_s)
    if isinstance(origin_s, str):
        if origin_s != "unix":
            raise ValueError("origin_s must be None, a number of seconds, or 'unix'")
        steps = math.ceil(float(first_timestamp_s) / hop_s - _UNIX_SNAP_FRACTION)
        return float(steps * hop_s)
    origin = float(origin_s)
    if not math.isfinite(origin):
        raise ValueError("origin_s must be finite")
    return origin


_TIMESTAMP_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
_UNITS_PER_SECOND = {"s": 1.0, "ms": 1e3, "us": 1e6}


def timestamp_scale(ts_unit: str) -> float:
    try:
        return _TIMESTAMP_SCALE[ts_unit]
    except KeyError as exc:
        raise ValueError("ts_unit must be one of: 's', 'ms', 'us'") from exc


def relative_seconds(
    timestamps: NDArray,
    ts_unit: str,
    origin_s: OriginSpec,
    hop_s: float,
) -> tuple:
    """Timestamps in seconds after the resolved origin, and that origin.

    Returns ``(t_relative_s, origin_s)``. Differences are taken in the input's
    own unit -- from the first sample, and from the first sample to the
    origin -- before scaling to seconds, so large absolute timestamps (Unix
    microseconds are ~1.7e15) lose no precision.
    """
    scale = timestamp_scale(ts_unit)
    raw = np.asarray(timestamps, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("timestamps must be one-dimensional and non-empty")
    first = float(raw[0])
    relative = (raw - first) * scale
    if origin_s is None:
        return relative, first * scale
    origin = resolve_origin(origin_s, first * scale, hop_s)
    units_per_second = _UNITS_PER_SECOND[ts_unit]
    offset_s = (first - origin * units_per_second) / units_per_second
    return relative + offset_s, origin


def resolve_empty_windows(empty_windows: Optional[str], stacklevel: int = 3) -> str:
    """Validate ``empty_windows``; ``None`` means ``"drop"`` with a FutureWarning."""
    if empty_windows is None:
        warnings.warn(
            "senpy 5.0 will change the default of empty_windows from 'drop' to 'keep': "
            "windows with too few samples will be reported as NaN rows (valid=False) "
            "instead of being left out. Pass empty_windows='keep' to adopt the new "
            "behavior now, or empty_windows='drop' to keep the current one.",
            FutureWarning,
            stacklevel=stacklevel,
        )
        return "drop"
    if empty_windows not in EMPTY_WINDOW_MODES:
        raise ValueError("empty_windows must be 'drop' or 'keep'")
    return empty_windows


def validate_min_samples(min_samples: int) -> int:
    value = int(min_samples)
    if value != min_samples or value < 1:
        raise ValueError("min_samples must be a positive integer")
    return value


def median_spacing(t_s: NDArray[np.float64]) -> float:
    """Upper median of the finite, positive sample spacings, as every backend uses."""
    diffs = np.diff(np.asarray(t_s, dtype=np.float64))
    positive = np.sort(diffs[np.isfinite(diffs) & (diffs > 0.0)])
    if positive.size == 0:
        raise ValueError("timestamps must contain at least one positive time step")
    return float(positive[positive.size // 2])


def window_count(last_relative_s: float, dt_median_s: float, window_s: float, hop_s: float) -> int:
    """How many windows fit: those with ``k * hop + window <= last + dt_median``.

    ``last_relative_s`` is the last timestamp minus the origin. The estimate is
    corrected against the exact comparison, so the count agrees with a loop
    that tests each window in turn.
    """
    limit = last_relative_s + dt_median_s
    if not math.isfinite(limit) or window_s > limit:
        return 0
    n = int(math.floor((limit - window_s) / hop_s)) + 1
    while n > 0 and (n - 1) * hop_s + window_s > limit:
        n -= 1
    while n * hop_s + window_s <= limit:
        n += 1
    return n


@dataclass(frozen=True)
class WindowGrid:
    """Where each NUSTFT window sits and which samples it holds.

    Attributes:
        origin_s: Absolute start of window 0, on the timestamps' clock, in
            seconds. A result's ``times`` are measured from it.
        window_s: Window duration in seconds.
        hop_s: Window hop in seconds (``window_s - overlap_s``).
        dt_median_s: Upper median sample spacing, which sets the grid's end.
        first: Index of each window's first sample, shape ``(n_windows,)``.
        stop: One past each window's last sample, shape ``(n_windows,)``.
        min_samples: Fewest samples a window needs to be transformed.
    """

    origin_s: float
    window_s: float
    hop_s: float
    dt_median_s: float
    first: NDArray[np.int64]
    stop: NDArray[np.int64]
    min_samples: int = DEFAULT_MIN_SAMPLES

    @property
    def n_windows(self) -> int:
        return int(self.first.size)

    @property
    def window_index(self) -> NDArray[np.int64]:
        return np.arange(self.n_windows, dtype=np.int64)

    @property
    def starts(self) -> NDArray[np.float64]:
        """Window starts relative to ``origin_s``."""
        return self.window_index * self.hop_s

    @property
    def times(self) -> NDArray[np.float64]:
        """Window centres relative to ``origin_s``: what results report as ``times``."""
        return self.starts + self.window_s / 2.0

    @property
    def sample_count(self) -> NDArray[np.int64]:
        return self.stop - self.first

    @property
    def valid(self) -> NDArray[np.bool_]:
        """Windows with enough samples to transform."""
        return self.sample_count >= self.min_samples


def window_grid(
    timestamps: NDArray[np.float64],
    window_s: float,
    overlap_s: float,
    *,
    origin_s: OriginSpec = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    ts_unit: str = "s",
) -> WindowGrid:
    """The window grid the NUSTFT backends use for these timestamps.

    Useful for consumers that assemble results themselves -- for example from
    :func:`senpy.jax_backend.pack_nustft_window_batches`, which packs only the
    windows that have data -- and need the full grid, the per-window sample
    counts, or the origin that ``times`` are measured from.

    Args:
        timestamps: Sorted sample timestamps.
        window_s: Window duration in seconds.
        overlap_s: Window overlap in seconds.
        origin_s: ``None`` (the first sample), a number of seconds on the
            timestamps' clock, or ``"unix"``; see :func:`resolve_origin`.
            Samples before the origin belong to no window.
        min_samples: Fewest samples a window needs to be valid.
        ts_unit: Timestamp unit: ``"s"``, ``"ms"``, or ``"us"``.
    """
    if window_s <= 0:
        raise ValueError("window_s must be > 0")
    if overlap_s < 0 or overlap_s >= window_s:
        raise ValueError("overlap_s must satisfy 0 <= overlap_s < window_s")
    if np.asarray(timestamps).size < 2:
        raise ValueError("timestamps must be one-dimensional with at least two samples")
    hop_s = float(window_s - overlap_s)
    t_relative, origin = relative_seconds(timestamps, ts_unit, origin_s, hop_s)
    dt_median = median_spacing(t_relative)
    return grid_from_relative(
        t_relative,
        origin_s=origin,
        window_s=float(window_s),
        hop_s=hop_s,
        dt_median_s=dt_median,
        min_samples=validate_min_samples(min_samples),
    )


def grid_from_relative(
    t_relative_s: NDArray[np.float64],
    *,
    origin_s: float,
    window_s: float,
    hop_s: float,
    dt_median_s: float,
    min_samples: int,
) -> WindowGrid:
    """Build the grid from timestamps already expressed relative to the origin."""
    n = window_count(float(t_relative_s[-1]), dt_median_s, window_s, hop_s)
    starts = np.arange(n, dtype=np.int64) * hop_s
    first = np.searchsorted(t_relative_s, starts, side="left").astype(np.int64)
    stop = np.searchsorted(t_relative_s, starts + window_s, side="left").astype(np.int64)
    return WindowGrid(
        origin_s=origin_s,
        window_s=window_s,
        hop_s=hop_s,
        dt_median_s=dt_median_s,
        first=first,
        stop=stop,
        min_samples=min_samples,
    )
