# senpy

`senpy` is a high-performance C++ library with Python bindings for preprocessing sensor data,
particularly accelerometer data for extracting physiological features. The native routines are
exposed to Python via Pybind11, with an optional pure-JAX backend for device-resident NUFFT work.

## Installation

The package is published to PyPI as **`arcascope-senpy`** (the plain `senpy` name is taken by
an unrelated project). It still imports as `senpy`:

```bash
python -m pip install arcascope-senpy
python -m pip install 'arcascope-senpy[jax]'
```

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
returning a wrongly scaled result.

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

`recording_indices`, `window_indices`, and `times` in each batch map valid
output rows back to the input order. Batch sizes remain a hardware-specific
throughput setting: measure with `block_until_ready()` and a CUDA profiler
before claiming GPU saturation.

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
  epoch to keep window indices meaningful across sessions and processes.
* **Divisibility.** The window and the hop must be whole multiples of `subwindow_s`, so that no
  subwindow straddles a window edge; one that did could not be shared by the windows either side.
* **Sample rate.** Supplied rather than measured: it sets the magnitude scale and the grid size.
  `compute_nustft` takes the median spacing over the whole recording, which a stream cannot see.
* **The trailing window.** `push` reports only windows the stream has passed the end of, which is 
  all a live stream can honestly say. `compute_nustft` knows where the recording stops and also 
  emits a final window ending within one sample period of the last timestamp; that one comes out 
  of `flush()`. `compute_nustft_streaming` applies this rule for you and is the function to compare
  the two paths with.
* **The Nyquist bin** (present only when `fmax` is unset) is the true $+N/2$ coefficient.
  `compute_nustft` reports its conjugate there, an artifact of reading that bin out of the aliased
  FINUFFT mode. Magnitudes are identical.
* **Timestamp precision.** Absolute unix seconds in float64 resolve to about half a microsecond,
  which is a ~1e-5 relative phase error at the top of a 5 Hz band. Pass times relative to a recent
  origin when sub-microsecond timing matters.

