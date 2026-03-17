"""Spectrogram comparison: linear vs. cubic vs. NUFFT-direct.

For each dataset in tests/data/, produces a vertical multi-panel figure with one
row per resampling method (Linear, Cubic spline, NUFFT direct), showing the full
time duration. Each row is a pcolormesh of the spectrogram restricted to 0–15 Hz
(where physiologically relevant sleep-scoring content lives), with its own
colorbar for perfect visual alignment.

Usage (from the senpy/ directory):
    python plot_spectrogram_comparison.py
    python plot_spectrogram_comparison.py --duration full --datasets SleepAccel
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

# Make sure senpy is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from senpy.api import (
    resample_accelerometer,
    resample_accelerometer_cubic,
    compute_spectrogram,
    compute_spectrogram_nufft,
    compute_jerk,
)

DATA_DIR = Path(__file__).parent / "tests" / "data"
OUT_DIR = Path(__file__).parent.parent  # workspace root


# ── data loading ─────────────────────────────────────────────────

def load_csv(path: Path):
    """Return (t, x, y, z) in seconds, auto-detecting format."""
    with open(path) as f:
        first = f.readline()
    if first.strip().upper().startswith("TIMESTAMP"):
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    else:
        data = np.loadtxt(path)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


# ── spectrogram helpers ───────────────────────────────────────────

TARGET_FS = 32.0
NPERSEG   = 512   # ~10 s windows @ 50 Hz
NOVERLAP  = 384   # 75% overlap → ~2.5 s stride
FREQ_MAX  = 15.0  # Hz  (0–15 Hz is the range of interest)


def _spectrogram_linear(t, x, y, z):
    r = resample_accelerometer(t, x, y, z, TARGET_FS)
    # compute_jerk on the uniform grid — timestamps drive the C++ difference
    j = compute_jerk(r.timestamps_s, r.x, r.y, r.z)
    return compute_spectrogram(j.jerk, TARGET_FS, NPERSEG, NOVERLAP)


def _spectrogram_cubic(t, x, y, z):
    r = resample_accelerometer_cubic(t, x, y, z, TARGET_FS)
    j = compute_jerk(r.timestamps_s, r.x, r.y, r.z)
    return compute_spectrogram(j.jerk, TARGET_FS, NPERSEG, NOVERLAP)


def _spectrogram_nufft(t, x, y, z):
    # Compute jerk directly on the raw non-uniform timestamps — no resampling.
    # The C++ jerk computes |Δacc| between consecutive samples; with non-uniform
    # timestamps this is the L2 displacement in acceleration space, which is the
    # same signal the ML pipeline uses (compute_jerk does not normalise by dt).
    #
    # np.ascontiguousarray is essential: column slices from a 2-D array
    # (e.g. data[:,1]) have a non-unit stride, but the C++ wrapper reads the
    # raw pointer assuming contiguous memory.  t is safe because compute_jerk
    # copies it into a new int64 array internally.
    j = compute_jerk(
        t,
        np.ascontiguousarray(x),
        np.ascontiguousarray(y),
        np.ascontiguousarray(z),
    )
    # jerk timestamps come back as microseconds; convert to seconds for NUFFT
    t_jerk_s = j.timestamps_us.astype(np.float64) / 1e6
    return compute_spectrogram_nufft(t_jerk_s, j.jerk, NPERSEG, NOVERLAP)


METHODS = {
    "Linear\n(baseline)": _spectrogram_linear,
    "Cubic spline\n(C++)": _spectrogram_cubic,
    "NUFFT direct\n(no resampling)": _spectrogram_nufft,
}


# ── colour-scale helper ───────────────────────────────────────────

def _percentile_norm(Sxx, plo=2, phi=98):
    """Robust log-scale colour normalisation."""
    flat = Sxx[Sxx > 0].ravel()
    if len(flat) == 0:
        return mcolors.LogNorm(vmin=1e-10, vmax=1.0)
    vmin = np.percentile(flat, plo)
    vmax = np.percentile(flat, phi)
    vmin = max(vmin, vmax * 1e-5)   # clamp dynamic range
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


# ── figure builder ────────────────────────────────────────────────

CMAP = "inferno"

METHOD_COLORS = {
    "Linear\n(baseline)":         "#e07040",
    "Cubic spline\n(C++)":        "#4db8e8",
    "NUFFT direct\n(no resampling)": "#7ddf74",
}


def make_comparison_figure(name: str, t, x, y, z, duration_s: float):
    n_methods = len(METHODS)

    # 3 rows (one per method) × 1 column layout
    fig, axes = plt.subplots(
        n_methods, 1,
        figsize=(12, 3.5 * n_methods),
        constrained_layout=False,
    )
    fig.patch.set_facecolor("#0d0d0d")

    # Ensure axes is always a list (for single-method case compatibility)
    if n_methods == 1:
        axes = [axes]

    # Compute spectrograms for all methods
    specs = {}
    for label, fn in METHODS.items():
        specs[label] = fn(t, x, y, z)

    # ── per-method rows ──────────────────────────────────────────
    # Each row gets its own percentile-based colour normalisation and colorbar
    # so that the structure in every method is independently legible and rows
    # are perfectly aligned in time.
    for row_idx, (ax, (label, fn)) in enumerate(zip(axes, METHODS.items())):
        spec = specs[label]
        fmask = spec.frequencies <= FREQ_MAX
        freqs  = spec.frequencies[fmask]
        times  = spec.times / 60.0          # minutes
        Sxx    = spec.Sxx[:, fmask]

        # Guard against all-zero or near-zero Sxx
        Sxx = np.maximum(Sxx, 1e-30)

        # Per-panel colour normalisation — computed from this method's own Sxx
        norm = _percentile_norm(Sxx, plo=1, phi=99)

        # pcolormesh expects (n_times+1,) and (n_freqs+1,) edge arrays
        if len(times) > 1:
            dt = times[1] - times[0]
            t_edges = np.append(times - dt / 2, times[-1] + dt / 2)
        else:
            t_edges = np.array([0.0, duration_s / 60.0])

        if len(freqs) > 1:
            df = freqs[1] - freqs[0]
            f_edges = np.append(freqs - df / 2, freqs[-1] + df / 2)
        else:
            f_edges = np.array([0.0, FREQ_MAX])

        pcm = ax.pcolormesh(
            t_edges, f_edges, Sxx.T,
            norm=norm,
            cmap=CMAP,
            rasterized=True,
            shading="flat",
        )

        # Annotate leakage: mean power in 0–10 Hz band (text box)
        band_mask = (freqs >= 0.5) & (freqs <= 10.0)
        mean_pw = np.mean(Sxx[:, band_mask]) if np.any(band_mask) else 0.0

        color = METHOD_COLORS[label]

        ax.set_facecolor("#0d0d0d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        ax.tick_params(colors="#cccccc", labelsize=9)
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")

        ax.set_xlabel("Time (min)", fontsize=10, color="#cccccc")
        ax.set_ylabel("Frequency (Hz)", fontsize=10, color="#cccccc")

        ax.set_ylim(0, FREQ_MAX)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(which="minor", length=2, color="#444444")


        # Title with colour accent bar
        ax.set_title(
            f"{label}",
            fontsize=11,
            color=color,
            fontweight="bold",
            pad=8,
        )

        # Inset text: mean 0–10 Hz power
        ax.text(
            0.97, 0.97,
            f"mean 0–10 Hz\n{mean_pw:.2e}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=7.5,
            color="#dddddd",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                      edgecolor="#555555", alpha=0.85),
        )

        # Horizontal reference lines for physio bands
        ax.axhline(9/60, color="#4488ff", lw=0.7, alpha=0.5, ls="--")   # BR min
        ax.axhline(25/60, color="#4488ff", lw=0.7, alpha=0.5, ls="--")  # BR max
        ax.axhline(0.5,   color="#ff8844", lw=0.7, alpha=0.5, ls="--")  # HR min
        ax.axhline(2.0,   color="#ff8844", lw=0.7, alpha=0.5, ls="--")  # HR max

        # ── individual colourbar per row ──────────────────────────
        # Add colourbar to the right of each row, ensuring rows stay aligned
        cbar_ax = fig.add_axes([0.92, 0.12 + (n_methods - row_idx - 1) * (0.78 / n_methods),
                                0.015, (0.78 / n_methods) * 0.95])
        cbar = fig.colorbar(pcm, cax=cbar_ax)
        cbar.set_label("PSD", color="#cccccc", fontsize=8)
        cbar.ax.tick_params(colors="#cccccc", labelsize=7)
        cbar.outline.set_edgecolor("#444444")

    # ── layout adjustment ────────────────────────────────────────
    fig.subplots_adjust(left=0.10, right=0.90, top=0.94, bottom=0.10,
                        hspace=0.35)

    # ── super-title ──────────────────────────────────────────────
    jitter_std_ms = np.std(np.diff(t)) * 1000
    nominal_fs = 1.0 / np.median(np.diff(t))
    fig.suptitle(
        f"{name}  ·  {duration_s/60:.1f} min  ·  "
        f"nominal {nominal_fs:.1f} Hz  ·  jitter σ = {jitter_std_ms:.1f} ms\n"
        f"Magnitude spectrogram — dashed: BR band (blue), HR band (orange)",
        color="#e0e0e0",
        fontsize=10,
        y=0.98,
    )

    return fig


# ── main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot resampling method spectrogram comparison")
    parser.add_argument(
        "--duration", type=str, default="full",
        help="Seconds of data to use per dataset, or 'full' for entire dataset (default: full)")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["SleepAccel", "Weaver", "Dreamt"],
        help="Dataset stem names to process")
    args = parser.parse_args()

    saved = []
    for stem in args.datasets:
        csv_path = DATA_DIR / f"{stem}.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found")
            continue

        print(f"\nLoading {stem}.csv …", flush=True)
        t_all, x_all, y_all, z_all = load_csv(csv_path)

        # Determine duration
        if args.duration.lower() == "full":
            t, x, y, z = t_all, x_all, y_all, z_all
        else:
            duration_s = float(args.duration)
            mask = t_all - t_all[0] < duration_s
            t, x, y, z = t_all[mask], x_all[mask], y_all[mask], z_all[mask]

        if len(t) < 500:
            print(f"  [skip] only {len(t)} samples after trimming")
            continue

        actual_dur = t[-1] - t[0]
        print(f"  {len(t)} samples, {actual_dur:.1f} s, "
              f"jitter σ = {np.std(np.diff(t))*1000:.2f} ms")

        print("  Computing spectrograms …", flush=True)
        fig = make_comparison_figure(stem, t, x, y, z, actual_dur)

        out_path = OUT_DIR / f"spectrogram_comparison_{stem}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {out_path}")
        saved.append(out_path)

    print(f"\nDone. {len(saved)} figure(s) written.")
    return saved


if __name__ == "__main__":
    main()
