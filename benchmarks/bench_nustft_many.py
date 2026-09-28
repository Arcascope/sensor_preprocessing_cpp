#!/usr/bin/env python3
"""Throughput of the JAX NUSTFT paths across many recordings.

Compares, on the same synthetic nights of jittered accelerometer data:

    packer   -- pack_nustft_window_batches + compute_nustft_window_batch, one batch at a time
    many     -- compute_nustft_many: vectorized gathers, large batches, overlapped host/device
    cpu      -- senpy.compute_nustft per channel (optional, --cpu)

Each path runs once to compile, then is timed. Results are checked against each other.

    python benchmarks/bench_nustft_many.py --recordings 8 --hours 8
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

import senpy
from senpy import jax_backend as senpy_jax


def night(hours: float, fs: float, seed: int, channels: int):
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 * fs)
    t = np.arange(n) / fs + rng.uniform(-2e-3, 2e-3, n)
    t.sort()
    samples = rng.normal(0.0, 0.5, (n, channels)).astype(np.float32)
    return t, samples


def run_packer(recordings, window_s, overlap_s, batch_size, n_keep):
    out = []
    for t, samples in recordings:
        per_group = []
        for start in range(0, samples.shape[1], 3):
            block = np.zeros((t.size, 3), dtype=samples.dtype)
            block[:, : samples[:, start : start + 3].shape[1]] = samples[:, start : start + 3]
            batches = senpy_jax.pack_nustft_window_batches(
                [(t, block)], window_s=window_s, overlap_s=overlap_s, batch_size=batch_size
            )
            rows = []
            for batch in batches:
                c = senpy_jax.compute_nustft_window_batch(
                    batch.points, batch.signals, batch.valid,
                    nfft_padded=batch.nfft_padded, median_fs=batch.median_fs,
                )
                rows.append(np.asarray(c)[batch.row_valid][..., :n_keep])
            per_group.append(rows)
        out.append(per_group)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", type=int, default=8)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--fs", type=float, default=32.0)
    parser.add_argument("--channels", type=int, default=5)
    parser.add_argument("--window-s", type=float, default=10.0)
    parser.add_argument("--overlap-s", type=float, default=8.0)
    parser.add_argument("--target-fs", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=128, help="packer rows per call")
    parser.add_argument("--rows-per-call", type=int, default=senpy_jax.DEFAULT_ROWS_PER_CALL)
    parser.add_argument("--cpu", action="store_true", help="also time the C++ transform")
    args = parser.parse_args()

    import jax

    print(f"devices: {jax.devices()}")
    recordings = [night(args.hours, args.fs, seed, args.channels) for seed in range(args.recordings)]
    windows = sum(senpy.window_grid(t, args.window_s, args.overlap_s).valid.sum() for t, _ in recordings)
    print(f"{args.recordings} x {args.hours} h x {args.channels} channels, {windows} windows per channel")
    n_keep = int(np.floor(args.target_fs / 2 * args.window_s + 0.5)) + 1

    def many():
        return senpy_jax.compute_nustft_many(
            recordings, window_s=args.window_s, overlap_s=args.overlap_s, target_fs=args.target_fs,
            empty_windows="drop", rows_per_call=args.rows_per_call,
        )

    def packer():
        return run_packer(recordings[:1], args.window_s, args.overlap_s, args.batch_size, n_keep) and run_packer(
            recordings, args.window_s, args.overlap_s, args.batch_size, n_keep
        )

    timings = {}
    for name, fn in (("many", many), ("packer", packer)):
        fn()  # compile
        start = time.perf_counter()
        result = fn()
        timings[name] = time.perf_counter() - start
        if name == "many":
            many_result = result
    if args.cpu:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            start = time.perf_counter()
            cpu_first = senpy.compute_nustft(
                recordings[0][0], recordings[0][1][:, 0], args.window_s, args.overlap_s,
                target_fs=args.target_fs, empty_windows="drop",
            )
            timings["cpu (1 recording x 1 channel)"] = time.perf_counter() - start
        gap = np.abs(many_result[0][0].coefficients - cpu_first.coefficients).max()
        print(f"max |many - cpu| on recording 0, channel 0: {gap:.2e}")

    for name, seconds in timings.items():
        print(f"{name:32s} {seconds:8.2f} s")
    print(f"speedup many vs packer: {timings['packer'] / timings['many']:.1f}x")


if __name__ == "__main__":
    main()
