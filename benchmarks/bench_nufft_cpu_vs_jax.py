#!/usr/bin/env python3
"""Benchmark senpy's C++ FINUFFT backend against the JAX backend.

jax-finufft was rebuilt from source with JAX_FINUFFT_USE_CUDA=ON /
CMAKE_CUDA_ARCHITECTURES=120 against CUDA 13.0 (the PyPI wheel is
CPU-only). This script compares three paths:

    * senpy.api.compute_nustft         -- C++ / FINUFFT, CPU
    * senpy.jax_backend.compute_nustft -- JAX / jax-finufft, forced onto CPU
    * senpy.jax_backend.compute_nustft -- JAX / jax-finufft, forced onto GPU (cuFINUFFT)

Pipeline per the task: load CSV -> NUFFT spectrogram with 30s windows /
30s steps (no overlap) -> resample to 50 Hz at the end. Resampling has no
JAX-native implementation in this repo, so it runs on CPU via senpy.api
for all backends; only the NUFFT/spectrogram stage is actually compared.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

import senpy.api as sp_cpu


def _enable_jax_x64() -> None:
    """Match the CPU backend's float64 precision.

    Timestamps in these CSVs are large epoch-millisecond values; JAX's
    default float32 has too little relative precision to resolve ~10 ms
    sample spacing at that magnitude (catastrophic cancellation), and we
    want a numerically fair speed comparison against the float64 C++
    backend anyway. Must run before any other JAX call.
    """
    import jax

    jax.config.update("jax_enable_x64", True)


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a cleaned_accelerometer CSV: columns t (ms), x, y, z."""
    df = pd.read_csv(path)
    t_ms = df["t"].to_numpy(dtype=np.float64)
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    z = df["z"].to_numpy(dtype=np.float64)
    order = np.argsort(t_ms, kind="stable")
    return t_ms[order], x[order], y[order], z[order]


def select_channel(channel: str, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    if channel == "x":
        return x
    if channel == "y":
        return y
    if channel == "z":
        return z
    if channel == "mag":
        return sp_cpu.compute_magnitude(x, y, z)
    raise ValueError(f"unknown channel {channel!r}")


def time_call(fn, repeats: int):
    """Return (first_call_s, steady_state_samples[repeats-1]) for fn()."""
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


def bench_cpu(t_ms, signal, window_s, step_s, target_fs, kind, repeats):
    overlap_s = window_s - step_s

    def run():
        return sp_cpu.compute_nustft(
            t_ms, signal, window_s=window_s, overlap_s=overlap_s,
            ts_unit="ms", target_fs=target_fs,
        ).spectrogram(kind=kind)

    first, steady, result = time_call(run, repeats)

    def resample():
        return sp_cpu.resample_accelerometer(
            t_ms, signal, np.zeros_like(signal), np.zeros_like(signal),
            target_fs=target_fs, ts_unit="ms",
        )

    r_first, r_steady, _ = time_call(resample, repeats)
    return {
        "nufft_first_s": first,
        "nufft_steady_s": steady,
        "resample_first_s": r_first,
        "resample_steady_s": r_steady,
        "n_windows": result.Sxx.shape[0],
        "n_freqs": result.Sxx.shape[1],
        "Sxx": result.Sxx,
    }


def bench_jax(t_ms, signal, window_s, step_s, target_fs, kind, repeats, device_kind: str):
    _enable_jax_x64()
    import jax
    import jax.numpy as jnp
    import senpy.jax_backend as sp_jax

    overlap_s = window_s - step_s
    # Force these arrays onto the requested device explicitly -- JAX's
    # default device here is the CUDA GPU, and (pre-source-build) nufft1
    # had no registered CUDA lowering and would error rather than fall
    # back to CPU, so explicit placement keeps CPU vs GPU unambiguous.
    device = jax.devices(device_kind)[0]
    # compute_nustft centers host arrays itself, but these are device arrays
    # the caller builds, so recenter in float64 seconds first -- see the
    # _enable_jax_x64 docstring for why this matters.
    t_rel_s = (t_ms - t_ms[0]) / 1e3
    t_device = jax.device_put(jnp.asarray(t_rel_s), device=device)
    s_device = jax.device_put(jnp.asarray(signal), device=device)

    def run():
        result = sp_jax.compute_nustft(
            t_device, s_device, window_s=window_s, overlap_s=overlap_s,
            ts_unit="s", target_fs=target_fs,
        ).spectrogram(kind=kind)
        jax.block_until_ready(result.Sxx)
        return result

    first, steady, result = time_call(run, repeats)

    # No JAX-native resample exists in this repo; run it on CPU for parity
    # with the CPU pipeline's final step.
    def resample():
        return sp_cpu.resample_accelerometer(
            t_ms, signal, np.zeros_like(signal), np.zeros_like(signal),
            target_fs=target_fs, ts_unit="ms",
        )

    r_first, r_steady, _ = time_call(resample, repeats)
    return {
        "nufft_first_s": first,
        "nufft_steady_s": steady,
        "resample_first_s": r_first,
        "resample_steady_s": r_steady,
        "n_windows": result.Sxx.shape[0],
        "n_freqs": result.Sxx.shape[1],
        "device": str(device),
        "Sxx": np.asarray(result.Sxx),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="Path to a cleaned_accelerometer CSV (t,x,y,z)")
    parser.add_argument("--channel", choices=["x", "y", "z", "mag"], default="mag")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--step-s", type=float, default=30.0)
    parser.add_argument("--target-fs", type=float, default=50.0)
    parser.add_argument("--kind", choices=["magnitude", "power", "psd"], default="psd")
    parser.add_argument("--repeats", type=int, default=5, help="Total calls per stage; first is JIT warmup for JAX")
    parser.add_argument("--max-rows", type=int, default=None, help="Truncate input for faster iteration")
    parser.add_argument(
        "--jax-devices", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"],
        help="Which JAX device(s) to benchmark",
    )
    parser.add_argument("--skip-jax", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    t_ms, x, y, z = load_csv(args.csv)
    if args.max_rows:
        t_ms, x, y, z = t_ms[: args.max_rows], x[: args.max_rows], y[: args.max_rows], z[: args.max_rows]
    signal = select_channel(args.channel, x, y, z)
    duration_s = (t_ms[-1] - t_ms[0]) / 1e3
    print(f"{len(t_ms)} samples, {duration_s:.1f} s duration, channel={args.channel!r}")
    print(
        f"window_s={args.window_s} step_s={args.step_s} "
        f"(overlap_s={args.window_s - args.step_s}) target_fs={args.target_fs} kind={args.kind}"
    )
    print()

    print("=== CPU (senpy.api / C++ FINUFFT) ===")
    cpu = bench_cpu(t_ms, signal, args.window_s, args.step_s, args.target_fs, args.kind, args.repeats)
    print(f"  windows x freqs: {cpu['n_windows']} x {cpu['n_freqs']}")
    print(f"  nufft+spectrogram, first call:   {cpu['nufft_first_s'] * 1e3:.2f} ms")
    print(f"  nufft+spectrogram, steady-state: {fmt(cpu['nufft_steady_s'])}")
    print(f"  resample_to_{args.target_fs:g}hz, first call:   {cpu['resample_first_s'] * 1e3:.2f} ms")
    print(f"  resample_to_{args.target_fs:g}hz, steady-state: {fmt(cpu['resample_steady_s'])}")
    print()

    if args.skip_jax:
        return

    jax_results = {}
    for device_kind in args.jax_devices:
        label = f"JAX/{device_kind}"
        print(f"=== {label} (senpy.jax_backend / jax-finufft) ===")
        jx = bench_jax(
            t_ms, signal, args.window_s, args.step_s, args.target_fs, args.kind,
            args.repeats, device_kind,
        )
        jax_results[device_kind] = jx
        print(f"  device: {jx['device']}")
        print(f"  windows x freqs: {jx['n_windows']} x {jx['n_freqs']}")
        print(f"  nufft+spectrogram, first call (incl. JIT):  {jx['nufft_first_s'] * 1e3:.2f} ms")
        print(f"  nufft+spectrogram, steady-state:             {fmt(jx['nufft_steady_s'])}")
        print(f"  resample_to_{args.target_fs:g}hz, first call:   {jx['resample_first_s'] * 1e3:.2f} ms")
        print(f"  resample_to_{args.target_fs:g}hz, steady-state: {fmt(jx['resample_steady_s'])}")

        if cpu["n_windows"] != jx["n_windows"] or cpu["n_freqs"] != jx["n_freqs"]:
            print(
                f"WARNING: CPU and {label} produced different output shapes "
                f"({cpu['n_windows']}x{cpu['n_freqs']} vs {jx['n_windows']}x{jx['n_freqs']}); "
                "speedup comparison below may not be apples-to-apples."
            )
        else:
            diff = np.abs(cpu["Sxx"] - jx["Sxx"])
            denom = np.maximum(np.abs(cpu["Sxx"]), 1e-12)
            print(
                f"  numerical parity vs CPU (kind={args.kind!r}): "
                f"max abs diff={diff.max():.3e}, max rel diff={(diff / denom).max():.3e}"
            )
        print()

    print("=== Summary (steady-state mean, nufft+spectrogram stage only) ===")
    cpu_mean = statistics.mean(cpu["nufft_steady_s"]) if cpu["nufft_steady_s"] else cpu["nufft_first_s"]
    print(f"  CPU/C++:  {cpu_mean * 1e3:.2f} ms")
    for device_kind, jx in jax_results.items():
        jx_mean = statistics.mean(jx["nufft_steady_s"]) if jx["nufft_steady_s"] else jx["nufft_first_s"]
        ratio = cpu_mean / jx_mean if jx_mean > 0 else float("inf")
        faster = f"JAX/{device_kind}" if ratio > 1 else "CPU/C++"
        print(f"  JAX/{device_kind}: {jx_mean * 1e3:.2f} ms  ({jx['device']})  -- {faster} is {max(ratio, 1 / ratio):.2f}x faster")


if __name__ == "__main__":
    main()
