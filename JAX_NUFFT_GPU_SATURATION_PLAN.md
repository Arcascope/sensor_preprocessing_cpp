# JAX NUFFT GPU saturation plan

## Goal

Replace the current one-window-at-a-time JAX NUFFT path with a static-shape,
cross-recording window batch that keeps accelerometer samples on an NVIDIA GPU
and exposes enough independent work to use a large GPU effectively.

This is a throughput plan, not a promise of 100% utilization. The useful
success criterion is substantially higher completed-window throughput with no
avoidable host/device copies, verified by a CUDA profiler on representative
recordings.

## Current state

`senpy.jax_backend.compute_nustft` currently finds windows, then calls a JIT-compiled
type-1 NUFFT once per window. It reuses compilation for equal source counts,
but the Python loop still submits the transforms serially. The three
accelerometer axes are also separate calls at the application level.

## Target batch representation

Pre-window all recordings, concatenate their windows, then process fixed-size
chunks. A chunk has static shapes:

```text
points      float32/float64 [B, M]     local NUFFT positions in [-pi, pi)
signals     float32/float64 [B, 3, M]  x/y/z accelerometer samples
valid       bool            [B, M]     true for observed samples
row_valid   bool            [B]        true for a real, rather than pad, window
window_ss   float32/float64 [B]        sum of squared valid Hann weights
```

- `B` is the window batch size and is fixed for a compiled variant.
- `M` is a source-count bucket, preferably a small set of powers of two. A
  window with fewer than `M` observations is padded.
- The final batch is padded to `B`; its `row_valid` entries are removed after
  the transform.
- Inputs should be normalized to each window's local time coordinate before
  packing. Absolute timestamps are not needed in the GPU transform.

The output is `coefficients[B, 3, F]`, then compacted using `row_valid` and
returned in the original recording/window order.

## Masking rule for padded samples

Padding must not change the transform normalization or detrending:

```text
count       = sum(valid, axis=M)
mean        = sum(signals * valid, axis=M) / count
tau         = local time / window_duration
hann        = valid * 0.5 * (1 - cos(2*pi*tau))
strengths   = (signals - mean[..., None]) * hann[:, None, :]
points      = where(valid, 2*pi*tau - pi, 0)
window_ss   = sum(hann**2, axis=M)
```

Padded strengths are exactly zero and padded points are set to zero, an
in-range NUFFT coordinate. This permits fixed-width transforms without adding
spurious spectral energy.

## Transform layout

Within each window, the three axes share `points[B, M]`. They should be passed
as the transform stack (`ntrans=3` conceptually):

```python
def one_window(points_m, strengths_3m):
    # Returns [3, nfft]. This lets jax-finufft/cuFINUFFT share the point setup
    # across the x, y, and z source-strength vectors.
    return nufft1(nfft, strengths_3m, points_m, eps=eps, iflag=1)

batched_nufft = jax.jit(jax.vmap(one_window, in_axes=(0, 0)))
coefficients = batched_nufft(points, strengths)
```

Use the same positive-frequency extraction, phase correction, and
`1 / sqrt(median_fs * window_ss)` scaling as the CPU implementation.

### Important limitation

The three channels in a window are a native stacked transform because their
nonuniform points are identical. Different windows generally have different
jittered points, so they cannot all become one cuFINUFFT `ntrans` plan.
`vmap` is still the correct first implementation, but it must be profiled: its
lowering may batch effectively, or it may emit multiple independent CUDA calls
on the default stream. Do not infer saturation from the presence of `vmap`.

## Implementation stages

1. Add a low-level `compute_nustft_window_batch(points, signals, valid, ...)`
   API. It accepts already-packed static arrays and returns `[B, 3, F]`; it
   does not discover windows or touch NumPy.
2. Implement a CPU/JAX-compatible packer that converts many recordings into
   `M`-bucketed window batches and records `(recording_id, window_id)` for
   restoring output order.
3. Implement the masked detrend, Hann taper, type-1 transform, mode ordering,
   and normalization in one JITted batch function.
4. Use JAX `vmap` over `B` and the transform-stack axis for the three channels.
   Keep all batch buffers and output coefficients as JAX arrays.
5. Process a long recording set as a sequence of static `B` chunks. Start with
   host-side batch iteration; use `lax.scan` only after profiling establishes
   that host dispatch, rather than NUFFT work, is material.
6. Add a high-level multi-recording API only after the low-level batch result
   agrees with the CPU reference.

## Correctness gates

For identical synthetic and real inputs, compare batched output with the
existing CPU `compute_nustft` implementation:

- Same window centres, frequency bins, output shape, and original ordering.
- Float64: complex-coefficient agreement consistent with the requested
  `eps` and CPU FINUFFT reference.
- Float32 GPU: check complex relative error, magnitude/PSD error, and peak
  frequencies separately; use a tolerance established from representative
  data rather than the CPU's `1e-14` target.
- Padding invariance: adding masked rows/samples must not change valid-window
  output.
- Detrend-off and `target_fs` behavior must match the existing API contract.

## Saturation and throughput experiment

Use a representative set containing thousands of windows. Time only after
`result.block_until_ready()` and record:

- windows/second and samples/second;
- end-to-end latency and GPU-only transform latency;
- peak allocated GPU memory;
- GPU compute and memory utilization from Nsight Systems/Compute, not only
  `nvidia-smi`'s coarse utilization field;
- number and duration of CUDA launches; and
- numerical error against the CPU reference.

Sweep separately for each common `(M, nfft)` bucket:

```text
B = 8, 16, 32, 64, 128, 256, ... until memory or throughput regresses
dtype = float32 first; float64 only when required
channels = 3 stacked transforms per window
```

Select the smallest `B` within roughly 5% of peak throughput to leave memory
headroom for upstream/downstream JAX work. Record the chosen values by GPU
model; a batch that fits one GPU well may be suboptimal on another.

## Escalation if `vmap` does not fill the GPU

First inspect the compiled HLO and CUDA trace. If the trace shows independent
small cuFINUFFT calls rather than useful batching:

1. Group more windows per `B` and reduce the number of `M` buckets where
   padding waste is acceptable.
2. Benchmark a limited number of concurrent CUDA streams, with reusable plans
   and buffers, in a dedicated CUDA FFI extension.
3. Only then consider a custom batched CUDA NUFFT implementation. It must
   handle distinct point sets per window, unlike cuFINUFFT's standard
   shared-point transform stack.

The custom-FFI path is a performance escalation, not the initial API design.
