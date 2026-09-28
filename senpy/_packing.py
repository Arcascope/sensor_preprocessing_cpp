"""Vectorized window discovery and batch gathering for the JAX NUSTFT.

NumPy only: packing prepares host buffers and never imports JAX, so data
loading can build batches while a device consumes earlier ones. Both
:func:`senpy.jax_backend.pack_nustft_window_batches` and
:func:`senpy.jax_backend.compute_nustft_many` are built on it.

A *recording* here is one timestamp vector with its window grid. Its samples
are transformed three channels at a time -- the batch transform's channel
stack -- so a recording with ``C`` channels contributes ``ceil(C / 3)``
*groups*, each a ``[3, N]`` block sharing the recording's grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from ._grid import WindowGrid, grid_from_relative, median_spacing, relative_seconds

#: Channels per transform stack.
GROUP_CHANNELS = 3


def next_power_of_two(values: np.ndarray) -> np.ndarray:
    """Elementwise smallest power of two ``>= value`` (1 for values <= 1)."""
    values = np.asarray(values, dtype=np.int64)
    out = np.ones_like(values)
    big = values > 1
    # bit_length of (v - 1) is exact in integer arithmetic, unlike log2.
    v = values[big] - 1
    bits = np.zeros_like(v)
    while np.any(v):
        nonzero = v > 0
        bits[nonzero] += 1
        v >>= 1
    out[big] = np.left_shift(1, bits)
    return out


@dataclass
class PackedRecording:
    """One recording's timestamps, grid, and channel groups, ready to gather."""

    t: np.ndarray  # seconds after the origin, float64, with one trailing 0.0 pad
    groups: List[np.ndarray]  # each [3, N + 1], a trailing zero column for padded lanes
    grid: WindowGrid
    nfft_padded: int
    median_fs: float
    rows: np.ndarray  # grid indices of the windows that get transformed
    widths: np.ndarray  # padded source width of each of those windows

    @property
    def n_samples(self) -> int:
        return self.t.size - 1


def discover(
    timestamps: np.ndarray,
    samples: np.ndarray,
    *,
    window_s: float,
    overlap_s: float,
    ts_unit: str,
    origin_s,
    min_samples: int,
    dtype=None,
) -> PackedRecording:
    """Validate one recording and lay its windows on the grid.

    ``samples`` is ``[N, C]``. ``dtype`` is the host dtype the channel blocks
    are stored in; ``None`` keeps the samples' own dtype.
    """
    t = np.asarray(timestamps, dtype=np.float64)
    s = np.asarray(samples)
    if s.ndim == 1:
        s = s[:, None]
    if t.ndim != 1 or s.ndim != 2 or s.shape[0] != t.size or s.shape[1] < 1:
        raise ValueError("each recording must be a (timestamps[N], samples[N, C]) pair")
    if not np.issubdtype(s.dtype, np.number) or np.iscomplexobj(s):
        raise ValueError("accelerometer samples must be real numeric values")
    if t.size < 2:
        raise ValueError("each recording requires at least two timestamps")
    if not np.all(np.isfinite(t)):
        raise ValueError("timestamps must be finite")
    if not np.all(np.isfinite(s)):
        raise ValueError("accelerometer samples must be finite")
    hop_s = float(window_s - overlap_s)
    t, origin = relative_seconds(t, ts_unit, origin_s, hop_s)
    if np.any(np.diff(t) < 0.0):
        raise ValueError("timestamps must be sorted")
    dt_median = median_spacing(t)
    median_fs = 1.0 / dt_median
    nfft = int(window_s * median_fs)
    if nfft < 2:
        raise ValueError("window_s is too short for the observed sampling density")
    nfft_padded = int(next_power_of_two(np.array([nfft]))[0])
    grid = grid_from_relative(
        t,
        origin_s=origin,
        window_s=float(window_s),
        hop_s=hop_s,
        dt_median_s=dt_median,
        min_samples=min_samples,
    )
    rows = np.flatnonzero(grid.valid).astype(np.int64)

    block_dtype = s.dtype if dtype is None else np.dtype(dtype)
    groups = []
    for start in range(0, s.shape[1], GROUP_CHANNELS):
        block = np.zeros((GROUP_CHANNELS, t.size + 1), dtype=block_dtype)
        chunk = s[:, start : start + GROUP_CHANNELS].T
        block[: chunk.shape[0], : t.size] = chunk
        groups.append(block)
    return PackedRecording(
        t=np.append(t, 0.0),
        groups=groups,
        grid=grid,
        nfft_padded=nfft_padded,
        median_fs=median_fs,
        rows=rows,
        widths=next_power_of_two(grid.sample_count[rows]),
    )


#: One transform row: (recording index, group index, grid window index).
RowKeys = Tuple[np.ndarray, np.ndarray, np.ndarray]


def buckets(recordings: Sequence[PackedRecording], group_major: bool) -> Dict[Tuple[int, int], RowKeys]:
    """Rows bucketed by ``(nfft_padded, source_width)``, in a stable order.

    Within a bucket rows run by recording, then -- ``group_major`` -- by
    group then window, or by window only when there is one group each.
    """
    parts: Dict[Tuple[int, int], List[RowKeys]] = {}
    for r, rec in enumerate(recordings):
        for width in np.unique(rec.widths).tolist():
            windows = rec.rows[rec.widths == width]
            n_groups = len(rec.groups) if group_major else 1
            for g in range(n_groups):
                parts.setdefault((rec.nfft_padded, int(width)), []).append(
                    (
                        np.full(windows.size, r, dtype=np.int64),
                        np.full(windows.size, g, dtype=np.int64),
                        windows,
                    )
                )
    return {
        key: tuple(np.concatenate([p[i] for p in chunks]) for i in range(3))  # type: ignore[misc]
        for key, chunks in sorted(parts.items())
    }


def chunks(keys: RowKeys, rows_per_chunk: int) -> Iterator[RowKeys]:
    n = keys[0].size
    for start in range(0, n, rows_per_chunk):
        yield tuple(k[start : start + rows_per_chunk] for k in keys)  # type: ignore[misc]


def gather(
    recordings: Sequence[PackedRecording],
    keys: RowKeys,
    *,
    width: int,
    n_rows: int,
    window_s: float,
    hop_s: float,
    dtype,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(points, signals, valid, median_fs)`` for one batch.

    Rows past ``len(keys[0])`` are zero padding with ``median_fs`` 1. The
    local points use exactly the arithmetic of the original per-window
    packer, so batches are bit-identical to it.
    """
    points = np.zeros((n_rows, width), dtype=dtype)
    signals = np.zeros((n_rows, GROUP_CHANNELS, width), dtype=dtype)
    valid = np.zeros((n_rows, width), dtype=bool)
    median_fs = np.ones(n_rows, dtype=dtype)
    lane = np.arange(width, dtype=np.int64)
    rec_keys, group_keys, window_keys = keys
    for r in np.unique(rec_keys).tolist():
        rec = recordings[r]
        in_rec = np.flatnonzero(rec_keys == r)
        windows = window_keys[in_rec]
        counts = rec.grid.sample_count[windows]
        mask = lane[None, :] < counts[:, None]
        # Padded lanes read the trailing zero sample, so no masking pass is
        # needed on the gathered signals.
        index = np.where(mask, rec.grid.first[windows][:, None] + lane[None, :], rec.n_samples)
        start = windows * hop_s
        local = 2.0 * np.pi * ((rec.t[index] - start[:, None]) / window_s) - np.pi
        local[~mask] = 0.0
        points[in_rec] = local
        valid[in_rec] = mask
        median_fs[in_rec] = rec.median_fs
        for g in np.unique(group_keys[in_rec]).tolist():
            in_group = in_rec[group_keys[in_rec] == g]
            gathered = rec.groups[g][:, index[group_keys[in_rec] == g]]  # [3, rows, width]
            signals[in_group] = np.moveaxis(gathered, 0, 1)
    return points, signals, valid, median_fs


def padded_rows(n: int, rows_per_call: int) -> int:
    """Round a partial batch up to one of a few fixed sizes.

    Every distinct shape is an XLA compile; a short menu keeps compiles to a
    handful per run at the cost of some zero rows.
    """
    for size in (rows_per_call // 8, rows_per_call // 2):
        if size and n <= size:
            return size
    return rows_per_call
