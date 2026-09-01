#!/usr/bin/env python3
"""Benchmark senpy's *streaming* NUSTFT against the batch C++ transform.

`compute_nustft` needs the whole recording in memory and re-spreads every
sample once per window it belongs to. `StreamingNUSTFT` sees the data one
packet at a time, touches each sample once at `O(bins)` regardless of how
many windows cover it, and keeps per-subwindow spectra instead of samples.
The README claims three things for it; this script measures all three on
real accelerometer recordings:

    * exactness  -- coefficients match `compute_nustft` to ~1e-11 relative
    * overlap    -- streaming cost is near-flat in overlap, batch is not
    * bounded    -- resident memory does not grow with recording length

and adds the number a live deployment actually cares about: per-push
latency, i.e. how long the packet that closes a window blocks the caller.

Pipeline shape mirrors the other benchmarks: 30 s windows, a sweep over
overlaps, and an optional `fmax` band limit (the streaming path's main
lever on per-sample cost -- 100 Hz into a 5 Hz band is ~1/10 the bins).

Handles both accelerometer CSV schemas in this dataset collection:
    t,x,y,z                     (epoch milliseconds, e.g. SEN-A)
    TIMESTAMP,ACC_X,ACC_Y,ACC_Z (seconds from recording start, e.g. SURF)

Timestamps are recentered on the host to seconds-since-first-sample before
either path sees them. Absolute unix seconds in float64 resolve to about
half a microsecond, which would show up as phase error in the parity check
and tell us nothing about the two implementations.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
import threading
import time
from pathlib import Path

import numpy as np

import senpy.api as sp

try:
    import pandas as pd
except ImportError:  # the minimal senpy dev venv has numpy only
    pd = None


def read_csv_columns(path: Path) -> dict:
    """Return {lowercased column name: float64 array}, via pandas when available."""
    if pd is not None:
        frame = pd.read_csv(path)
        return {c.lower(): frame[c].to_numpy(dtype=np.float64) for c in frame.columns}
    with open(path) as handle:
        names = [c.strip().lower() for c in handle.readline().split(",")]
    table = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    return {name: table[:, i] for i, name in enumerate(names)}


SCHEMAS = [
    ({"t", "x", "y", "z"}, "t", ("x", "y", "z"), 1e-3),
    ({"timestamp", "acc_x", "acc_y", "acc_z"}, "timestamp", ("acc_x", "acc_y", "acc_z"), 1.0),
]


def load_recording(path: Path):
    """Load one accelerometer CSV as (t_seconds_from_start, xyz), auto-detecting schema."""
    columns = read_csv_columns(path)
    for required, ts_key, xyz_keys, to_seconds in SCHEMAS:
        if required <= set(columns):
            t = columns[ts_key] * to_seconds
            samples = np.column_stack([columns[k] for k in xyz_keys])
            order = np.argsort(t, kind="stable")
            t, samples = t[order], samples[order]
            return t - t[0], samples
    raise ValueError(f"{path}: unrecognized columns {sorted(columns)}")


def select_channel(channel: str, samples: np.ndarray) -> np.ndarray:
    if channel in ("x", "y", "z"):
        return np.ascontiguousarray(samples[:, "xyz".index(channel)])
    if channel == "mag":
        return sp.compute_magnitude(samples[:, 0], samples[:, 1], samples[:, 2])
    raise ValueError(f"unknown channel {channel!r}")


def packet_bounds(t: np.ndarray, packet_s: float) -> np.ndarray:
    """Split indices at every `packet_s` boundary -- a stand-in for sensor packets.

    Computed once, outside the timed region: a real stream gets its packets
    handed to it, and we are not benchmarking searchsorted.
    """
    edges = np.arange(packet_s, t[-1] + packet_s, packet_s)
    cuts = np.searchsorted(t, edges, side="left")
    return np.unique(np.concatenate([[0], cuts, [t.size]]))


class RSSSampler:
    """Peak RSS over a phase, sampled from /proc. Coarse, but it sees the C++ heap.

    tracemalloc would not: everything the streaming transform retains lives
    in the extension module, not in Python objects.
    """

    def __init__(self, interval_s: float = 0.002):
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = None
        self.peak_kb = 0
        self.baseline_kb = 0

    @staticmethod
    def _rss_kb() -> int:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)

    def _run(self):
        while not self._stop.wait(self._interval):
            self.peak_kb = max(self.peak_kb, self._rss_kb())

    def __enter__(self):
        self.baseline_kb = self.peak_kb = self._rss_kb()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_kb = max(self.peak_kb, self._rss_kb())
        return False

    @property
    def growth_mb(self) -> float:
        return (self.peak_kb - self.baseline_kb) / 1024.0


def sweep(value: str):
    """Parse a sweep like "0,15,29" or "0 15 29" into a list of strings.

    One argument rather than argparse's nargs="+": a greedy list flag eats the
    positional path when the flag is written last (`--fmax 25 /data/dir`), and
    fails with "the following arguments are required: path", which points
    nowhere near the actual mistake.
    """
    return [piece for piece in re.split(r"[,\s]+", value.strip()) if piece]


def repeat_call(fn, repeats: int):
    """Time `repeats` warmed calls. Returns (samples_s, last_result).

    Callers warm up explicitly rather than reporting a cold first call: the
    cold cost here is FFTW planning, which is cached process-wide and so
    would land on whichever configuration ran first rather than on the one
    that paid for it.
    """
    samples = []
    result = None
    for _ in range(max(repeats, 1)):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    return samples, result


def fmt(samples_s):
    """Median and spread -- median because a stray scheduler hit on one repeat
    should not move the number the summary table is built from."""
    if not samples_s:
        return "n/a"
    median = statistics.median(samples_s)
    if len(samples_s) > 1:
        return f"{median * 1e3:8.1f} ms (min {min(samples_s) * 1e3:.1f}, max {max(samples_s) * 1e3:.1f})"
    return f"{median * 1e3:8.1f} ms"


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q)) if len(values) else float("nan")


def run_stream(t, signal, bounds, window_s, overlap_s, subwindow_s, fs, fmax,
               collect_latency=True, retain=False):
    """One full pass of the stream.

    `retain` decides whether finished windows are kept. Timed passes drain
    them, as a live consumer would: at a 1 s hop a 10 h recording closes
    ~37k windows, and holding their coefficients would put 600 MB on the
    benchmark's own heap and report it as the transform's footprint. The
    parity pass keeps them and is not timed.
    """
    transform = sp.StreamingNUSTFT(
        window_s=window_s,
        overlap_s=overlap_s,
        subwindow_s=subwindow_s,
        sample_rate_hz=fs,
        fmax=fmax,
        origin_s=float(t[0]),
    )
    windows = []
    n_windows = 0
    latencies = []
    emitting = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:
            continue
        if collect_latency:
            t0 = time.perf_counter()
            produced = transform.push(t[lo:hi], signal[lo:hi])
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            if produced:
                emitting.append(elapsed)
        else:
            produced = transform.push(t[lo:hi], signal[lo:hi])
        n_windows += len(produced)
        if retain:
            windows.extend(produced)
    return windows, n_windows, latencies, emitting, transform


def parity_vs_batch(t, signal, windows, transform, reference, window_s, fs, band_limited):
    """Max relative deviation between streamed and batch coefficients.

    `push` reports only windows the stream has passed the end of; the batch
    transform also emits a final window ending within one sample period of
    the last timestamp, so that one is taken out of the flush. Windows are
    matched on center time rather than position -- a dropout empties a
    window on both paths, but only the batch path renumbers around it.
    """
    limit = float(t[-1]) + 1.0 / fs
    windows = windows + [w for w in transform.flush() if w.start + window_s <= limit]

    by_center = {round(w.center, 6): w for w in windows}
    matched, missing = [], 0
    for row, center in enumerate(reference.times):
        window = by_center.get(round(float(center), 6))
        if window is None:
            missing += 1
            continue
        matched.append((window.coefficients, reference.coefficients[row]))
    if not matched:
        return {"matched": 0, "missing": missing, "extra": len(windows), "rel": float("nan")}

    streamed = np.stack([m[0] for m in matched])
    expected = np.stack([m[1] for m in matched])[:, : streamed.shape[1]]
    if not band_limited:
        # compute_nustft reads the Nyquist bin out of the aliased FINUFFT mode
        # and so reports its conjugate; magnitudes agree. See the README.
        streamed, expected = streamed[:, :-1], expected[:, :-1]
    rel = np.abs(streamed - expected).max() / np.abs(expected).max()
    return {
        "matched": len(matched),
        "missing": missing,
        "extra": len(windows) - len(matched),
        "rel": float(rel),
    }


def bench_recording(name, t, signal, args, overlaps, fmaxes):
    fs = 1.0 / float(np.median(np.diff(t)))
    duration_s = float(t[-1] - t[0])
    bounds = packet_bounds(t, args.packet_s)
    n_packets = len(bounds) - 1
    print(f"=== {name} ===")
    print(
        f"  {t.size} samples, {duration_s:.1f} s ({duration_s / 3600:.2f} h), "
        f"median fs={fs:.3f} Hz, {n_packets} packets of {args.packet_s:g} s"
    )

    rows = []
    for overlap_s in overlaps:
        hop_s = args.window_s - overlap_s
        print(f"  --- overlap_s={overlap_s:g} (hop {hop_s:g} s) ---")

        # One batch run per overlap: compute_nustft has no band-limit argument,
        # so it is identical across the fmax sweep. Warm up first -- FINUFFT
        # plans its FFTs on first use of a given size and caches them
        # process-wide, so an unwarmed "first call" measures planning, not
        # transform, and only for whichever configuration happens to run first.
        def batch():
            return sp.compute_nustft(t, signal, window_s=args.window_s, overlap_s=overlap_s)

        reference = batch()
        with RSSSampler() as batch_rss:
            batch_samples, reference = repeat_call(batch, args.repeats)
        batch_mean = statistics.median(batch_samples)
        print(f"    batch  compute_nustft: {fmt(batch_samples)}")
        print(
            f"           {reference.coefficients.shape[0]} windows x "
            f"{reference.coefficients.shape[1]} bins, peak RSS growth {batch_rss.growth_mb:.1f} MB"
        )

        for fmax in fmaxes:
            band = "full" if fmax is None else f"{fmax:g} Hz"

            def stream(**kwargs):
                return run_stream(
                    t, signal, bounds, args.window_s, overlap_s, args.subwindow_s,
                    fs, fmax, **kwargs,
                )

            stream(collect_latency=False)
            with RSSSampler() as stream_rss:
                stream_samples, last = repeat_call(stream, args.repeats)
            _, n_windows, latencies, emitting, transform = last
            stream_mean = statistics.median(stream_samples)
            n_bins = len(transform.frequencies)
            print(f"    stream band={band:>6}:   {fmt(stream_samples)}")
            print(
                f"           {n_windows} windows x {n_bins} bins, "
                f"{transform.dropped_samples} dropped, {transform.skipped_windows} skipped, "
                f"peak RSS growth {stream_rss.growth_mb:.1f} MB"
            )
            print(
                f"           throughput {t.size / stream_mean / 1e3:.0f}k samples/s "
                f"= {duration_s / stream_mean:.0f}x realtime"
            )
            print(
                f"           push latency: p50 {percentile(latencies, 50) * 1e6:7.1f} us, "
                f"p95 {percentile(latencies, 95) * 1e6:7.1f} us, "
                f"max {max(latencies) * 1e6:7.1f} us  "
                f"(window-closing pushes: p50 {percentile(emitting, 50) * 1e6:.1f} us, "
                f"max {(max(emitting) if emitting else float('nan')) * 1e6:.1f} us)"
            )

            windows, _, _, _, retaining = stream(collect_latency=False, retain=True)
            parity = parity_vs_batch(
                t, signal, windows, retaining, reference, args.window_s, fs,
                band_limited=fmax is not None,
            )
            note = ""
            if parity["missing"] or parity["extra"]:
                note = f"  [{parity['missing']} batch windows unmatched, {parity['extra']} stream-only]"
            print(
                f"           parity vs batch: max rel diff {parity['rel']:.3e} "
                f"over {parity['matched']} windows{note}"
            )

            rows.append({
                "name": name,
                "overlap_s": overlap_s,
                "fmax": fmax,
                "batch_s": batch_mean,
                "stream_s": stream_mean,
                "duration_s": duration_s,
                "samples": t.size,
                "bins": n_bins,
                "windows": n_windows,
                "rel": parity["rel"],
                "p95_us": percentile(latencies, 95) * 1e6,
                "batch_mb": batch_rss.growth_mb,
                "stream_mb": stream_rss.growth_mb,
            })
    print()
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=Path, help="A CSV file, or a directory of them")
    parser.add_argument("--pattern", default="*.csv", help="Glob used when path is a directory")
    parser.add_argument("--max-files", type=int, default=3, help="0 or negative disables the cap")
    parser.add_argument("--max-rows", type=int, default=None, help="Truncate each recording")
    parser.add_argument("--channel", choices=["x", "y", "z", "mag"], default="mag")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument(
        "--overlap-s", default="0,15,29",
        help="Overlaps to sweep, comma- or space-separated; the point of the sweep "
        "is that streaming cost is near-flat across it while batch cost is not",
    )
    parser.add_argument(
        "--subwindow-s", type=float, default=1.0,
        help="Streaming granularity; must divide both window_s and every hop",
    )
    parser.add_argument(
        "--fmax", default="none,5",
        help="Band limits to sweep, comma- or space-separated; 'none' means the "
        "full grid up to fs/2",
    )
    parser.add_argument(
        "--packet-s", type=float, default=1.0,
        help="Packet duration fed to push() -- one call per packet, as a live stream would",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    try:
        overlaps = [float(v) for v in sweep(args.overlap_s)]
        fmaxes = [None if v.lower() in ("none", "full") else float(v) for v in sweep(args.fmax)]
    except ValueError as exc:
        parser.error(str(exc))
    if not overlaps or not fmaxes:
        parser.error("--overlap-s and --fmax each need at least one value")

    for overlap_s in overlaps:
        hop_s = args.window_s - overlap_s
        for value, label in ((args.window_s, "window_s"), (hop_s, f"hop for overlap_s={overlap_s:g}")):
            if abs(round(value / args.subwindow_s) * args.subwindow_s - value) > 1e-9:
                parser.error(
                    f"{label}={value:g} is not a whole multiple of subwindow_s="
                    f"{args.subwindow_s:g}; a subwindow straddling a window edge "
                    "cannot be shared by the windows either side"
                )

    if args.path.is_dir():
        paths = sorted(glob.glob(str(args.path / args.pattern)))
        if args.max_files and args.max_files > 0:
            paths = paths[: args.max_files]
    else:
        paths = [str(args.path)]
    if not paths:
        parser.error(f"no files matching {args.pattern!r} in {args.path}")

    print(
        f"window_s={args.window_s:g} subwindow_s={args.subwindow_s:g} "
        f"packet_s={args.packet_s:g} channel={args.channel!r} repeats={args.repeats}"
    )
    print(
        f"{len(paths)} recording(s), {len(overlaps)} overlap(s) x {len(fmaxes)} band(s) each"
    )
    threads = os.environ.get("OMP_NUM_THREADS")
    if threads == "1":
        print("OMP_NUM_THREADS=1: both paths are single-threaded, so ratios are per-core.")
    else:
        print(
            f"OMP_NUM_THREADS is {threads or f'unset ({os.cpu_count()} cores visible)'}, so "
            "FINUFFT threads each window while the streaming transform stays single-threaded. "
            "Worth running both ways: on short per-window transforms the batch path is often "
            "slower with all cores than with one, and the timings get much noisier."
        )
    print()

    rows = []
    for path in paths:
        t, samples = load_recording(Path(path))
        if args.max_rows:
            t, samples = t[: args.max_rows], samples[: args.max_rows]
        if t.size < 2:
            print(f"=== {os.path.basename(path)} === skipped: fewer than two samples\n")
            continue
        signal = select_channel(args.channel, samples)
        rows.extend(bench_recording(os.path.basename(path), t, signal, args, overlaps, fmaxes))

    if not rows:
        return

    print("=== Summary (median of warmed repeats, per recording) ===")
    header = (
        f"{'recording':<38} {'ovl':>5} {'band':>6} {'bins':>5} "
        f"{'batch':>10} {'stream':>10} {'ratio':>7} {'xRT':>8} {'p95 us':>8} {'rel':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        band = "full" if row["fmax"] is None else f"{row['fmax']:g}"
        ratio = row["batch_s"] / row["stream_s"] if row["stream_s"] > 0 else float("inf")
        print(
            f"{row['name'][:38]:<38} {row['overlap_s']:>5g} {band:>6} {row['bins']:>5} "
            f"{row['batch_s'] * 1e3:>9.1f}m {row['stream_s'] * 1e3:>9.1f}m {ratio:>6.2f}x "
            f"{row['duration_s'] / row['stream_s']:>7.0f}x {row['p95_us']:>8.1f} {row['rel']:>10.2e}"
        )
    print()
    print(
        "ratio > 1 means streaming finished the same recording faster than the batch "
        "transform; xRT is stream duration / wall time, i.e. how many live sensors one "
        "core could keep up with. The batch figure is multithreaded FINUFFT unless "
        "OMP_NUM_THREADS says otherwise; the streaming figure never is."
    )
    print()
    print("=== Peak RSS growth (batch must materialize every window; streaming drains them) ===")
    for row in rows:
        band = "full" if row["fmax"] is None else f"{row['fmax']:g}"
        print(
            f"  {row['name'][:38]:<38} ovl={row['overlap_s']:>4g} band={band:>4}  "
            f"batch {row['batch_mb']:>7.1f} MB   stream {row['stream_mb']:>7.1f} MB "
            f"({row['windows']} windows)"
        )
    print(
        "  RSS is a process-wide high-water mark, so growth reads as 0 once the heap "
        "is already large from an earlier configuration; the first run of each shape "
        "is the informative one."
    )

    by_band = {}
    for row in rows:
        by_band.setdefault(row["fmax"], []).append(row)
    print()
    print("=== Cost of overlap (mean over recordings, normalized to the smallest overlap) ===")
    for fmax, group in by_band.items():
        band = "full" if fmax is None else f"{fmax:g} Hz"
        by_overlap = {}
        for row in group:
            by_overlap.setdefault(row["overlap_s"], []).append(row)
        base = min(by_overlap)
        batch_base = statistics.mean(r["batch_s"] for r in by_overlap[base])
        stream_base = statistics.mean(r["stream_s"] for r in by_overlap[base])
        for overlap_s in sorted(by_overlap):
            batch_mean = statistics.mean(r["batch_s"] for r in by_overlap[overlap_s])
            stream_mean = statistics.mean(r["stream_s"] for r in by_overlap[overlap_s])
            print(
                f"  band={band:>6}  overlap_s={overlap_s:>4g}  "
                f"batch {batch_mean / batch_base:>5.2f}x   stream {stream_mean / stream_base:>5.2f}x"
            )


if __name__ == "__main__":
    main()
