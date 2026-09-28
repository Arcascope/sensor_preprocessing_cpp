# senpy

`senpy` is a high-performance C++ library with Python bindings for preprocessing sensor data,
particularly accelerometer data for extracting physiological features. The native routines are
exposed to Python via Pybind11, with an optional pure-JAX backend for device-resident NUFFT work.

## Installation

The package is published to PyPI as **`arcascope-senpy`**. It still imports as `senpy`.

```bash
python -m pip install arcascope-senpy
python -m pip install 'arcascope-senpy[jax]'
```

Prebuilt manylinux wheels are published for CPython 3.12, 3.13, and 3.14. Python 3.11 and other
platforms build from source on install.

To install a specific precompiled wheel from a GitHub release instead:

```bash
python -m pip install https://github.com/Arcascope/sensor_preprocessing_cpp/releases/download/4.0.1/arcascope_senpy-4.0.1-cp312-cp312-manylinux_2_34_x86_64.whl
```

Installing from source, `pip install git+https://github.com/Arcascope/sensor_preprocessing_cpp.git@4.0.1`,
compiles the native extension locally and requires a C++17 toolchain, CMake, and pybind11.

Release wheels are built by `.github/workflows/release-wheel.yml` (attached to a published
release, or run manually to backfill a tag) and published to PyPI via OpenID Connect trusted
publishing. To rehearse that flow without touching PyPI, run `.github/workflows/dry-run-testpypi.yml`
(Actions -> Dry run publish to TestPyPI), which requires a separate pending publisher configured
with the `testpypi` environment.

## The NUSTFT window grid

Every NUSTFT entry point -- `compute_nustft`, `compute_nufft_spectrogram`,
`compute_nufft_welch`, `compute_stacked_spectrograms`, `compute_nustft_streaming`,
the `senpy.jax_backend` transforms, and `pack_nustft_window_batches` -- uses one
window grid, and they all put the same samples in the same windows:

* Window `k` spans `[origin + k*hop, origin + k*hop + window_s)`, with `hop = window_s - overlap_s`.
  Starts are `k*hop`, never an accumulated sum. Samples before the origin belong to no window.
* The grid runs through the last window holding at least `min_samples` samples, so the trailing
  windows the recording stops partway through are transformed like any other window with data.
* `times` are window centres, `k*hop + window_s/2`, measured from the origin.

`senpy.window_grid(timestamps, window_s, overlap_s, origin_s=..., min_samples=...)` returns that
grid -- the origin, each window's first/stop sample index, and its sample count -- without
transforming anything.

**Where window 0 starts** (`origin_s`):

| `origin_s` | Window 0 starts at |
|---|---|
| `None` (default) | the first sample, as in every earlier release |
| a number | that absolute time, in seconds on the timestamps' clock -- for example the start of a reference recording such as PSG, so windows line up with its epochs |
| `"unix"` | the first whole multiple of the hop since the Unix epoch at or after the first sample, so windows from separate recordings and sessions share one grid; the timestamps must be Unix time |

**Windows with too little data** (`empty_windows`, `min_samples`). A window holding fewer than
`min_samples` samples (default 4) is not transformed. With `empty_windows="drop"` it is left out,
so `times` has gaps. With `empty_windows="keep"` every window on the grid gets a row, and the rows
without enough data hold NaN.

Every result carries per-row metadata, whichever mode is used:

* `window_index` -- the row's index on the grid, so gaps in drop mode are explicit;
* `sample_count` -- samples in the row's window;
* `valid` -- `sample_count >= min_samples`; False exactly where a keep-mode row is NaN;
* `origin_s` -- the absolute start of window 0 that `times` are measured from.

```python
result = senpy.compute_nustft(t, x, window_s=10.0, overlap_s=8.0, empty_windows="keep")
result.coefficients[~result.valid]   # all NaN: dropouts, visible on the grid
```

> **Deprecation notice.** `empty_windows` defaults to `"drop"` in 4.x, matching earlier releases,
> and omitting it raises a `FutureWarning`. **senpy 5.0 will change the default to `"keep"`.**
> New code should pass `empty_windows="keep"`; pass `"drop"` explicitly to keep today's output.
>
> **With nothing to report** -- no window on the grid, or none with `min_samples` samples in
> `"drop"` mode -- the CPU and streaming functions raise `ValueError`, while the
> `senpy.jax_backend` transforms return a zero-row result, as they always have. senpy 5.0 will
> make the JAX transforms raise too.

Timestamps are measured from the origin once, in the input's own unit, before scaling to
seconds, so Unix microsecond timestamps (~1.7e15) keep their full precision.

Timestamps must be sorted. `window_grid` and `pack_nustft_window_batches` raise if they are not.
The transforms themselves do not check, as in earlier releases: out-of-order samples near a
window edge are silently assigned to the wrong window.

## JAX NUFFT (CPU / CUDA / Metal)

The regular `senpy` API remains NumPy/C++ based. For a JAX-native NUFFT that
keeps sample arrays on the active JAX device, install the `jax` extra:

```bash
# CPU-only:
python -m pip install 'arcascope-senpy[jax]'
# GPU (CUDA):
python -m pip install 'arcascope-senpy[jax]' 'jax[cuda12]'
```

```python
import jax
import jax.numpy as jnp
from senpy import jax_backend as senpy_jax

timestamps = jnp.arange(3_000, dtype=jnp.float32) / 50.0
signal = jnp.sin(2 * jnp.pi * 3.0 * timestamps)
result = senpy_jax.compute_nustft(
    timestamps, signal, window_s=8.0, overlap_s=4.0, target_fs=16.0
)
print(jax.devices(), result.coefficients.shape)
```

`senpy.jax_backend` implements its own type-1 NUFFT (`nufft1`, Gaussian
gridding + FFT + deconvolution) in pure `jax.numpy`/`jax.vmap` -- there is no
compiled NUFFT dependency and no platform-specific lowering. It runs
unconditionally wherever JAX runs: CPU, CUDA, and Metal all take the same
code path, so there is no separate GPU build to install and no macOS/OpenMP
interaction to work around. It returns JAX arrays rather than
`senpy.api.NUSTFTResult`, so subsequent JAX work stays device-resident.
`eps=1e-6` is the default everywhere (not GPU-specific), giving ~2e-5
relative error against a brute-force reference; the effective eps is floored
at float32 epsilon (~1.19e-7) regardless of platform, so requesting a smaller
value than that has no effect. Enable JAX x64 before importing JAX if the
application needs float64 arithmetic elsewhere in the pipeline. Absolute
epoch timestamps are safe to pass as NumPy arrays -- they are centered on the
first sample in float64 before reaching the device. If you build the
timestamp array with JAX yourself, either enable x64 first or make the values
relative to the first sample; float32 cannot resolve millisecond spacing at
epoch magnitude, and `compute_nustft` rejects such an array rather than
returning a wrongly scaled result. Integer epoch arrays are worse: without x64,
`jnp.asarray` wraps int64 values to int32 before senpy sees them, which cannot
be detected. Spacing survives the wrap but absolute time does not, so pass
NumPy timestamps whenever `origin_s` is a number or `"unix"`.

For high-throughput three-axis work across recordings, pre-pack ragged windows
into a small set of static shapes, then run each batch on the JAX device:

```python
from senpy import jax_backend as senpy_jax

# Each sample array is shaped [N, 3] for x/y/z. The packer only discovers and
# pads windows; it does not import JAX or execute a transform.
batches = senpy_jax.pack_nustft_window_batches(
    recordings, window_s=8.0, overlap_s=4.0, batch_size=128, ts_unit="s"
)
for batch in batches:
    coefficients = senpy_jax.compute_nustft_window_batch(
        batch.points,
        batch.signals,
        batch.valid,
        nfft_padded=batch.nfft_padded,
        median_fs=batch.median_fs,
    )
    real_coefficients = coefficients[batch.row_valid]  # [windows, 3, freqs]
```

For whole datasets, `compute_nustft_many` does all of this for you, at device throughput:

```python
from senpy import jax_backend as senpy_jax

# Each recording is (timestamps[N], samples[N, C]) for any number of channels C.
results = senpy_jax.compute_nustft_many(
    recordings, window_s=10.0, overlap_s=8.0, target_fs=12.0, empty_windows="keep"
)
coefficients = results[0][2].coefficients   # recording 0, channel 2: a senpy.api.NUSTFTResult
```

It gathers windows from every recording into large batches (`rows_per_call`, default 8192) with
vectorized NumPy on `build_threads` host threads that run ahead of the device, keeps up to
`max_in_flight` batches dispatched, and drops bins above `target_fs / 2` on the device before the
copy back. Local sample coordinates are computed on the host in float64, so it agrees with the
C++ transform to ~1e-6 even on long recordings. `benchmarks/bench_nustft_many.py` compares it
with the packer loop below; on one RTX-class GPU, four 8 h nights × 5 channels took 0.61 s
against the loop's 1.55 s. Results change with `rows_per_call` only in float32 rounding.

`recording_indices`, `window_indices`, and `times` in each batch map valid
output rows back to the input order. `window_indices` are grid indices, and
only windows with at least `min_samples` samples are packed; call
`senpy.window_grid` with the same arguments (`origin_s` accepts one origin or
one per recording) for the full grid and its sample counts. Batch sizes remain
a hardware-specific throughput setting: measure with `block_until_ready()` and
a CUDA profiler before claiming GPU saturation.

## Streaming NUSTFT

`StreamingNUSTFT` computes the **same coefficients** from a live stream: 
push samples as they arrive, get each window back as soon as the data 
passes its end, and never retain the samples themselves.

```python
from senpy import StreamingNUSTFT

transform = StreamingNUSTFT(
    window_s=30.0, overlap_s=0.0, subwindow_s=1.0,   # subwindow = one sensor packet
    sample_rate_hz=100.0, fmax=5.0,                  # report DC..5 Hz only
)
for packet_t, packet_x in packets:
    for window in transform.push(packet_t, packet_x):
        consume(window.center, window.magnitude())
tail = transform.flush()                             # the partly-filled final window
```

Against `compute_nustft` on the same samples the coefficients
agree to ~1e-13 relative (`tests/test_streaming_nustft.py`), for any chunking of the input and
with or without window overlap.

### Why it is exact

The transform is linear in the data and the subwindows partition the window, so

$$X_w(f) = \sum_m e^{2\pi i f d_m}\, S_m(f)$$

where $S_m$ is the transform of subwindow $m$ about its own origin and $d_m$ is that origin's
offset into the window. This is decimation-in-time for nonuniformly sampled data: a long
transform is a phase-weighted sum of short ones, with nothing lost. We hold onto the complex 
Fourier coefficients for each window, then scale them with the appropriate Hann window taper.

Note that it does _not_ work to take each spectrogram/PSD on the windows and then combine them.
Averaging $|S_m|^2$ over subwindows pins the frequency resolution at the *subwindow's* $1/T_w$.

### Cost

Each sample is touched once, at `O(bins)`, however many windows it belongs to — so overlap is
nearly free, unlike the batch transform which re-spreads every sample per window. Memory is one
accumulator per open window plus the subwindows in flight; it does not grow with window length or
recording length. A narrow `fmax` is what makes the per-sample constant small: 100 Hz into a 5 Hz
band at 30 s windows costs about 150 000 multiply-accumulates per second of stream.

### Contract and differences from `compute_nustft`

* **Ordering.** Timestamps must be non-decreasing over the object's life. A sample belonging to a
  subwindow the stream has already passed cannot be folded in; `dropped_samples` counts those.
* **Window grid.** `origin_s` anchors it, and window 0 is the earliest — nothing before the origin
  is reported. Pass the first timestamp to reproduce `compute_nustft`'s alignment, or a fixed
  epoch to keep window indices meaningful across sessions and processes: with Unix timestamps,
  `origin_s=0.0` puts windows on the grid `compute_nustft(..., origin_s="unix")` uses.
* **Sparse windows.** A window with fewer than `min_samples` samples is not reported, and neither
  is one the stream saw no samples in at all; both show up as gaps in `index`. `skipped_windows`
  counts the first kind. For a dense grid over a whole array, use `compute_nustft_streaming(...,
  empty_windows="keep")`.
* **Divisibility.** The window and the hop must be whole multiples of `subwindow_s`, so that no
  subwindow straddles a window edge; one that did could not be shared by the windows either side.
* **Sample rate.** Supplied rather than measured: it sets the magnitude scale and the grid size.
  `compute_nustft` takes the median spacing over the whole recording, which a stream cannot see.
* **The trailing windows.** `push` reports only windows the stream has passed the end of, which is
  all a live stream can honestly say. The windows the data stops partway through come out of
  `flush()`, as `compute_nustft` reports them. `compute_nustft_streaming` does both and is the
  function to compare the two paths with.
* **The Nyquist bin** (present only when `fmax` is unset) is the true $+N/2$ coefficient.
  `compute_nustft` reports its conjugate there, an artifact of reading that bin out of the aliased
  FINUFFT mode. Magnitudes are identical.
* **Timestamp precision.** Absolute unix seconds in float64 resolve to about half a microsecond,
  which is a ~1e-5 relative phase error at the top of a 5 Hz band. Pass times relative to a recent
  origin when sub-microsecond timing matters.

