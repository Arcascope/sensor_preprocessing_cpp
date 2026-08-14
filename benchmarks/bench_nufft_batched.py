#!/usr/bin/env python3
"""Benchmark senpy's *batched* JAX NUFFT path against the CPU/C++ backend.

The previous benchmark (bench_nufft_cpu_vs_jax.py) drives senpy.jax's
per-window loop (one nufft1 call per 30s window, one distinct jax.jit
compile per distinct window sample-count). That loop is latency-bound: it
only became competitive with the C++ backend at ~3000+ windows from a
single 30-hour recording, and lost badly at small window counts.

senpy.jax also ships a *packing* API (pack_nustft_window_batches +
compute_nustft_window_batch) built specifically to fix this: it buckets
windows -- across many files at once -- into static-shape [B, M] batches
and runs each bucket as ONE vmapped nufft1 call instead of one call per
window. This script exercises that path across a directory of
accelerometer CSVs (designed for something like /home/eric/data/SURF/
cleaned_accelerometer, which has 100+ files) and compares:

    * CPU/C++    -- senpy.api.compute_nustft, once per (file, channel)
    * JAX/cpu    -- batched path, forced onto CPU
    * JAX/gpu    -- batched path, forced onto GPU (cuFINUFFT)

Same pipeline shape as before: 30s windows / 30s steps (no overlap),
resample to --target-fs Hz as a final CPU-only step (no JAX-native
resample exists in this repo).

Handles both accelerometer CSV schemas seen in this dataset collection:
    t,x,y,z                    (epoch milliseconds, e.g. SEN-A)
    TIMESTAMP,ACC_X,ACC_Y,ACC_Z (seconds relative to recording start, e.g. SURF)

Run this from OUTSIDE the senpy source checkout -- see the shadowing
guard below for why.
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _guard_against_jax_shadowing() -> None:
    import importlib.util

    spec = importlib.util.find_spec("jax")
    if spec is None or spec.origin is None:
        return
    origin = Path(spec.origin)
    if origin.name == "jax.py" and (origin.parent / "api.py").exists():
        raise RuntimeError(
            f"`jax` resolved to {origin}, which looks like senpy's own "
            "jax.py, not the real JAX package. Run this script from a "
            "different working directory."
        )


_guard_against_jax_shadowing()

import senpy.api as sp_cpu  # noqa: E402


SCHEMAS = [
    ({"t", "x", "y", "z"}, "t", ("x", "y", "z"), "ms"),
    ({"timestamp", "acc_x", "acc_y", "acc_z"}, "timestamp", ("acc_x", "acc_y", "acc_z"), "s"),
]


def load_recording(path: Path):
    """Load one accelerometer CSV, auto-detecting its column schema."""
    df = pd.read_csv(path)
    lower_to_actual = {c.lower(): c for c in df.columns}
    for required, ts_key, xyz_keys, ts_unit in SCHEMAS:
        if required <= set(lower_to_actual):
            t = df[lower_to_actual[ts_key]].to_numpy(dtype=np.float64)
            samples = np.column_stack(
                [df[lower_to_actual[k]].to_numpy(dtype=np.float64) for k in xyz_keys]
            )
            order = np.argsort(t, kind="stable")
            return t[order], samples[order], ts_unit
    raise ValueError(f"{path}: unrecognized columns {list(df.columns)}")


def load_directory(data_dir: Path, pattern: str, max_files, max_rows_per_file):
    paths = sorted(glob.glob(str(data_dir / pattern)))
    if max_files:
        paths = paths[:max_files]
    if not paths:
        raise ValueError(f"no files matching {pattern!r} in {data_dir}")

    recordings = []
    ts_unit = None
    total_rows = 0
    for path in paths:
        t, samples, unit = load_recording(Path(path))
        if ts_unit is None:
            ts_unit = unit
        elif unit != ts_unit:
            raise ValueError(
                f"{path} uses ts_unit={unit!r} but earlier files in this "
                f"directory used {ts_unit!r}; mixed schemas aren't supported "
                "in one packed batch run"
            )
        if max_rows_per_file:
            t, samples = t[:max_rows_per_file], samples[:max_rows_per_file]
        recordings.append((t, samples))
        total_rows += len(t)
        print(f"  loaded {os.path.basename(path)}: {len(t)} rows")
    return recordings, ts_unit, total_rows, paths


def time_call(fn, repeats: int):
    t0 = time.perf_counter()
    result = fn()
    first = time.perf_counter() - t0
    steady = []
    for _ in range(repeats - 1):
        t0 = time.perf_counter()
        fn()
        steady.append(time.perf_counter() - t0)
    return first, steady, result


def fmt(samples_s):
    if not samples_s:
        return "n/a"
    mean = statistics.mean(samples_s)
    if len(samples_s) > 1:
        return f"{mean * 1e3:.2f} ms (+/- {statistics.stdev(samples_s) * 1e3:.2f})"
    return f"{mean * 1e3:.2f} ms"


def bench_cpu_baseline(recordings, ts_unit, window_s, step_s, repeats):
    """senpy.api.compute_nustft, once per (recording, channel), summed."""
    overlap_s = window_s - step_s

    def run():
        total_windows = 0
        for t, samples in recordings:
            for ch in range(3):
                result = sp_cpu.compute_nustft(
                    t, samples[:, ch], window_s=window_s, overlap_s=overlap_s, ts_unit=ts_unit,
                )
                total_windows += result.coefficients.shape[0]
        return total_windows

    first, steady, total_windows = time_call(run, repeats)
    return {"first_s": first, "steady_s": steady, "total_windows": total_windows}


def bench_batched(recordings, ts_unit, window_s, step_s, batch_size, eps, device_kind, repeats, enable_x64):
    import jax
    import jax.numpy as jnp
    import senpy.jax as sp_jax

    if enable_x64:
        jax.config.update("jax_enable_x64", True)
    device = jax.devices(device_kind)[0]

    t0 = time.perf_counter()
    packed = sp_jax.pack_nustft_window_batches(
        recordings, window_s=window_s, overlap_s=window_s - step_s,
        batch_size=batch_size, ts_unit=ts_unit,
    )
    pack_s = time.perf_counter() - t0
    n_batches = len(packed)
    total_real_windows = sum(int(b.row_valid.sum()) for b in packed)
    buckets = sorted({(b.nfft_padded, b.points.shape[1]) for b in packed})
    print(
        f"  packed into {n_batches} batch(es) across {len(buckets)} "
        f"(nfft_padded, source_width) bucket(s): {buckets}"
    )
    print(f"  {total_real_windows} real windows (batch_size={batch_size}, host packing took {pack_s * 1e3:.1f} ms)")

    t0 = time.perf_counter()
    device_batches = [
        {
            "points": jax.device_put(jnp.asarray(b.points), device),
            "signals": jax.device_put(jnp.asarray(b.signals), device),
            "valid": jax.device_put(jnp.asarray(b.valid), device),
            "median_fs": jax.device_put(jnp.asarray(b.median_fs), device),
            "nfft_padded": b.nfft_padded,
        }
        for b in packed
    ]
    jax.block_until_ready([db["points"] for db in device_batches])
    transfer_s = time.perf_counter() - t0

    def run():
        outputs = [
            sp_jax.compute_nustft_window_batch(
                db["points"], db["signals"], db["valid"],
                nfft_padded=db["nfft_padded"], median_fs=db["median_fs"], eps=eps,
            )
            for db in device_batches
        ]
        jax.block_until_ready(outputs)
        return outputs

    first, steady, outputs = time_call(run, repeats)
    return {
        "pack_s": pack_s,
        "transfer_s": transfer_s,
        "first_s": first,
        "steady_s": steady,
        "total_windows": total_real_windows,
        "n_batches": n_batches,
        "device": str(device),
        "packed": packed,
        "outputs": outputs,
    }


def bench_resample(recordings, ts_unit, target_fs, repeats):
    def run():
        for t, samples in recordings:
            sp_cpu.resample_accelerometer(
                t, samples[:, 0], samples[:, 1], samples[:, 2], target_fs=target_fs, ts_unit=ts_unit,
            )

    first, steady, _ = time_call(run, repeats)
    return {"first_s": first, "steady_s": steady}


def check_batched_device_parity(cpu_result, gpu_result):
    """Batched CPU vs batched GPU should agree closely -- same code, same
    packed inputs, only the device differs."""
    diffs = []
    for out_cpu, out_gpu in zip(cpu_result["outputs"], gpu_result["outputs"]):
        a = np.asarray(out_cpu)
        b = np.asarray(out_gpu)
        diffs.append(np.abs(a - b).max())
    return max(diffs) if diffs else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="Directory of accelerometer CSVs")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--max-files", type=int, default=8, help="0/omit-negative disables the cap")
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--step-s", type=float, default=30.0)
    parser.add_argument("--target-fs", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--jax-devices", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"],
    )
    parser.add_argument(
        "--float32", action="store_true",
        help="Skip jax_enable_x64 (use JAX's default float32) -- the realistic "
        "GPU-throughput setting, at the cost of an apples-to-apples precision "
        "comparison against the float64 C++ backend.",
    )
    parser.add_argument("--skip-cpu-baseline", action="store_true")
    args = parser.parse_args()

    print(f"Loading recordings from {args.data_dir} (pattern={args.pattern!r}) ...")
    recordings, ts_unit, total_rows, paths = load_directory(
        args.data_dir, args.pattern, args.max_files, args.max_rows_per_file,
    )
    print(f"{len(recordings)} recordings, {total_rows} total rows, ts_unit={ts_unit!r}")
    print(
        f"window_s={args.window_s} step_s={args.step_s} batch_size={args.batch_size} "
        f"eps={args.eps} precision={'float32' if args.float32 else 'float64 (x64 enabled)'}"
    )
    print()

    cpu_baseline = None
    if not args.skip_cpu_baseline:
        print("=== CPU/C++ baseline (senpy.api.compute_nustft, per file per channel) ===")
        cpu_baseline = bench_cpu_baseline(recordings, ts_unit, args.window_s, args.step_s, args.repeats)
        print(f"  total windows (x,y,z summed): {cpu_baseline['total_windows']}")
        print(f"  first call:   {cpu_baseline['first_s'] * 1e3:.1f} ms")
        print(f"  steady-state: {fmt(cpu_baseline['steady_s'])}")
        print()

    batched_results = {}
    for device_kind in args.jax_devices:
        print(f"=== Batched JAX/{device_kind} (pack_nustft_window_batches + compute_nustft_window_batch) ===")
        result = bench_batched(
            recordings, ts_unit, args.window_s, args.step_s, args.batch_size, args.eps,
            device_kind, args.repeats, enable_x64=not args.float32,
        )
        batched_results[device_kind] = result
        print(f"  device: {result['device']}")
        print(f"  host->device transfer (one-time): {result['transfer_s'] * 1e3:.1f} ms")
        print(f"  first call (incl. JIT):  {result['first_s'] * 1e3:.1f} ms")
        print(f"  steady-state:            {fmt(result['steady_s'])}")
        print()

    if "cpu" in batched_results and "gpu" in batched_results:
        max_diff = check_batched_device_parity(batched_results["cpu"], batched_results["gpu"])
        print(f"Numerical parity, batched CPU vs batched GPU (same code path): max abs diff={max_diff:.3e}")
        print()

    print("=== Resample to target_fs (CPU, senpy.api.resample_accelerometer) ===")
    resample = bench_resample(recordings, ts_unit, args.target_fs, args.repeats)
    print(f"  first call:   {resample['first_s'] * 1e3:.1f} ms")
    print(f"  steady-state: {fmt(resample['steady_s'])}")
    print()

    print("=== Summary (steady-state mean, NUFFT/spectrogram stage only) ===")
    if cpu_baseline is not None:
        cpu_mean = statistics.mean(cpu_baseline["steady_s"]) if cpu_baseline["steady_s"] else cpu_baseline["first_s"]
        print(
            f"  CPU/C++:        {cpu_mean * 1e3:8.1f} ms  "
            f"({cpu_baseline['total_windows']} channel-windows)"
        )
    for device_kind, result in batched_results.items():
        mean_s = statistics.mean(result["steady_s"]) if result["steady_s"] else result["first_s"]
        throughput = result["total_windows"] / mean_s if mean_s > 0 else float("inf")
        line = f"  Batched/{device_kind:<4}: {mean_s * 1e3:8.1f} ms  ({result['total_windows']} windows, {throughput:.0f} windows/s)"
        if cpu_baseline is not None:
            ratio = cpu_mean / mean_s if mean_s > 0 else float("inf")
            line += f"  -- {'batched' if ratio > 1 else 'CPU/C++'} {max(ratio, 1 / ratio):.2f}x faster"
        print(line)


if __name__ == "__main__":
    main()
