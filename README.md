# Sensor Preprocessing C++ Library

This library provides a set of C++ classes and functions for preprocessing sensor data. We split it out as a separate code base to enable reuse across multiple projects. Include this as a git submodule in your project to take advantage of its functionality.

## Dart API
The library exposes a Dart API through FFI (Foreign Function Interface). This allows Dart applications to call the C++ functions for sensor data preprocessing seamlessly.

## Python API
In addition to the Dart API, the library also provides a Python package called `senpy` that provides acces to the C++ routines via Pybind11.

### Precompiled Python wheels

GitHub release wheels are built by `.github/workflows/release-wheel.yml` and attached to a release when it is published. The same workflow can be run manually to backfill an existing tag/release, such as `v1.0.0` or `v2.0.0`.

Use the release asset URL directly when installing a precompiled wheel:

```bash
python -m pip install https://github.com/<owner>/<repo>/releases/download/v2.0.0/senpy-2.0.0-cp311-cp311-linux_x86_64.whl
```

`pip install git+https://github.com/<owner>/<repo>.git@v2.0.0` installs from source and will still compile the native extension locally.

### Streaming NUSTFT

`compute_nustft` needs the whole recording in memory. `StreamingNUSTFT` computes the **same
coefficients** from a live stream: push samples as they arrive, get each window back as soon as
the data passes its end, and never retain the samples themselves.

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

It is exact, not an approximation. Against `compute_nustft` on the same samples the coefficients
agree to ~1e-13 relative (`tests/test_streaming_nustft.py`), for any chunking of the input and
with or without window overlap.

#### Why it is exact

The transform is linear in the data and the subwindows partition the window, so

$$X_w(f) \;=\; \sum_m e^{2\pi i f d_m}\, S_m(f)$$

where $S_m$ is the transform of subwindow $m$ about its own origin and $d_m$ is that origin's
offset into the window. This is decimation-in-time for nonuniformly sampled data: a long
transform is a phase-weighted sum of short ones, with nothing lost.

The contrast worth being clear about is with Bartlett/Welch. Averaging $|S_m|^2$ over subwindows
pins the frequency resolution at the *subwindow's* $1/T_w$ — averaging buys variance reduction,
never resolution. Keeping the complex $S_m$ **and their offsets** recovers the full window's
transform, hence the full window's resolution. Coherence is the whole trick.

Two parts of `compute_nustft` belong to the window rather than to any subwindow, and so are
deferred and reconstructed when the window closes:

**The Hann taper.** A subwindow cannot know its own taper weights until it knows where it sits.
On this frequency grid — spacing $1/T$, because that is the grid the window's own transform lives
on — the taper is exactly one bin wide:

$$h(\tau) = \tfrac12 - \tfrac14 e^{2\pi i \tau} - \tfrac14 e^{-2\pi i \tau}$$

so applying it after the fact is a three-tap convolution,
$0.5\,D(k) - 0.25\,D(k-1) - 0.25\,D(k+1)$. That is why one bin above the reported band is carried
internally; bin $-1$ needs no storage, being the conjugate of bin $+1$ for real input.

**Mean removal.** The window mean is unknown until the window closes, so each subwindow also
carries the transform of the constant 1. Detrending is then $D(k) = X(k) - \bar{x}\,\mathrm{ones}(k)$.
The same accumulator pays for the scale factor too: expanding $h^2$ gives
$0.375 - 0.5\cos 2\pi\tau + 0.125\cos 4\pi\tau$, so

$$\sum_j h(\tau_j)^2 \;=\; 0.375\,N - 0.5\,\mathrm{Re}\,\mathrm{ones}(1) + 0.125\,\mathrm{Re}\,\mathrm{ones}(2)$$

with no second pass over the samples.

#### Cost

Each sample is touched once, at `O(bins)`, however many windows it belongs to — so overlap is
nearly free, unlike the batch transform which re-spreads every sample per window. Memory is one
accumulator per open window plus the subwindows in flight; it does not grow with window length or
recording length. A narrow `fmax` is what makes the per-sample constant small: 100 Hz into a 5 Hz
band at 30 s windows costs about 150 000 multiply-accumulates per second of stream.

#### Contract and differences from `compute_nustft`

* **Ordering.** Timestamps must be non-decreasing over the object's life. A sample belonging to a
  subwindow the stream has already passed cannot be folded in; `dropped_samples` counts those.
* **Window grid.** `origin_s` anchors it, and window 0 is the earliest — nothing before the origin
  is reported. Pass the first timestamp to reproduce `compute_nustft`'s alignment, or a fixed
  epoch to keep window indices meaningful across sessions and processes.
* **Divisibility.** The window and the hop must be whole multiples of `subwindow_s`, so that no
  subwindow straddles a window edge; one that did could not be shared by the windows either side.
* **Sample rate.** Supplied rather than measured: it sets the magnitude scale and the grid size.
  `compute_nustft` takes the median spacing over the whole recording, which a stream cannot see.
* **The trailing window.** `push` reports only windows the stream has passed the end of — all a
  live stream can honestly say. `compute_nustft` knows where the recording stops and also emits a
  final window ending within one sample period of the last timestamp; that one comes out of
  `flush()`. `compute_nustft_streaming` applies this rule for you and is the function to compare
  the two paths with.
* **The Nyquist bin** (present only when `fmax` is unset) is the true $+N/2$ coefficient.
  `compute_nustft` reports its conjugate there, an artefact of reading that bin out of the aliased
  FINUFFT mode. Magnitudes are identical.
* **Timestamp precision.** Absolute unix seconds in float64 resolve to about half a microsecond,
  which is a ~1e-5 relative phase error at the top of a 5 Hz band. Pass times relative to a recent
  origin when sub-microsecond timing matters.

#### Where this came from

The recombination identity and its resolution argument are developed in
`autofish-jax/mobile/litert_spike/stft_recombination.tex`. The first production user is
FoundryWhoopAndroid, which computes 30 s sleep-staging features one strap packet at a time
because it deletes the raw samples after upload; its Kotlin implementation and this one agree to
float32 storage precision on the same fixture.

### JAX / NVIDIA CUDA NUFFT

The regular `senpy` API remains NumPy/C++ based. For a JAX-native NUFFT that
keeps sample arrays on the active JAX device, install `jax-finufft` with the
JAX build appropriate for the machine:

```bash
# First install JAX with its CUDA support using JAX's installation guidance.
# Then, from this repository's senpy/ directory:
python -m pip install '.[jax]'
```

```python
import jax
import jax.numpy as jnp
from senpy import jax as senpy_jax

timestamps = jnp.arange(3_000, dtype=jnp.float32) / 50.0
signal = jnp.sin(2 * jnp.pi * 3.0 * timestamps)
result = senpy_jax.compute_nustft(
    timestamps, signal, window_s=8.0, overlap_s=4.0, target_fs=16.0
)
print(jax.devices(), result.coefficients.shape)
```

`senpy.jax` uses `jax-finufft`'s type-1 transform; a CUDA-enabled
`jax-finufft` build dispatches it to cuFINUFFT. It returns JAX arrays rather
than `senpy.api.NUSTFTResult`, so subsequent JAX work stays device-resident.
The GPU default is `eps=1e-6`; enable JAX x64 before importing JAX if the
application requires float64 precision. For absolute microsecond timestamps,
either enable x64 before creating the JAX arrays or pass timestamps relative to
the first sample.

For high-throughput three-axis work across recordings, pre-pack ragged windows
into a small set of static shapes, then run each batch on the JAX device:

```python
from senpy import jax as senpy_jax

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
