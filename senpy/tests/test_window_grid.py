"""The NUSTFT window grid: one contract, shared by every backend.

Window ``k`` starts ``k * hop`` after the origin. Every backend must put the
same samples in the same windows, report the same ``window_index`` and
``times``, and -- with ``empty_windows="keep"`` -- the same dense grid with NaN
where a window had too little data.
"""

import warnings

import numpy as np
import pytest

import senpy

WINDOW_S = 10.0
OVERLAP_S = 8.0  # a 2 s hop, the pisces-lite configuration
HOP_S = WINDOW_S - OVERLAP_S
FS = 32.0


def recording(start=0.0, seconds=300.0, gaps=((100.0, 140.0),), seed=3):
    """A jittery 32 Hz recording with a dropout, starting at ``start``."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, 1.0 / FS)
    t = t + rng.uniform(-2e-3, 2e-3, t.size)
    t.sort()
    keep = np.ones(t.size, dtype=bool)
    for lo, hi in gaps:
        keep &= (t < lo) | (t >= hi)
    t = t[keep]
    signal = np.sin(2 * np.pi * 1.1 * t) + 0.2 * rng.standard_normal(t.size)
    return start + t, signal


def nustft(t, x, **kwargs):
    kwargs.setdefault("empty_windows", "keep")
    return senpy.compute_nustft(t, x, WINDOW_S, OVERLAP_S, **kwargs)


# ── the grid itself ─────────────────────────────────────────────────


def test_grid_starts_are_whole_hops_after_the_origin():
    t, _ = recording(start=12.25)
    grid = senpy.window_grid(t, WINDOW_S, OVERLAP_S)

    assert grid.origin_s == t[0]
    np.testing.assert_array_equal(grid.starts, np.arange(grid.n_windows) * HOP_S)
    np.testing.assert_array_equal(grid.times, grid.starts + WINDOW_S / 2.0)
    # Every sample in a window lies inside it, measured from the origin.
    t_rel = t - grid.origin_s
    for k in np.flatnonzero(grid.valid)[:20]:
        inside = t_rel[grid.first[k] : grid.stop[k]]
        assert inside.min() >= grid.starts[k]
        assert inside.max() < grid.starts[k] + WINDOW_S


def test_grid_ends_where_the_batch_transforms_always_have():
    # Evenly spaced samples: the last window may end at most one sample past the data.
    t = np.arange(0.0, 60.0, 0.5)
    grid = senpy.window_grid(t, WINDOW_S, OVERLAP_S)
    last_end = grid.starts[-1] + WINDOW_S

    assert last_end <= t[-1] + 0.5
    assert last_end + HOP_S > t[-1] + 0.5


def test_dropout_windows_are_counted_not_lost():
    t, _ = recording()
    grid = senpy.window_grid(t, WINDOW_S, OVERLAP_S)

    # Exactly the windows lying wholly inside the 100-140 s dropout hold nothing.
    start = grid.origin_s + grid.starts
    inside = (start > t[t < 100.0].max()) & (start + WINDOW_S <= t[t >= 140.0].min())
    np.testing.assert_array_equal(grid.sample_count == 0, inside)
    assert inside.sum() == 15
    assert not grid.valid[inside].any()


# ── every backend on the same grid ─────────────────────────────────


def test_cpu_and_streaming_agree_on_every_window():
    t, x = recording()
    grid = senpy.window_grid(t, WINDOW_S, OVERLAP_S)
    cpu = nustft(t, x)
    streamed = senpy.compute_nustft_streaming(
        t, x, WINDOW_S, OVERLAP_S, subwindow_s=HOP_S, empty_windows="keep"
    )

    for result in (cpu, streamed):
        np.testing.assert_array_equal(result.window_index, grid.window_index)
        np.testing.assert_array_equal(result.times, grid.times)
        np.testing.assert_array_equal(result.sample_count, grid.sample_count)
        np.testing.assert_array_equal(result.valid, grid.valid)
        assert result.origin_s == grid.origin_s
    np.testing.assert_array_equal(np.isnan(cpu.coefficients), np.isnan(streamed.coefficients))
    # The top (Nyquist) bin is left out: the streaming transform reports the conjugate of the
    # batch transform's value there, a documented difference (README, "Streaming NUSTFT").
    np.testing.assert_allclose(
        streamed.coefficients[grid.valid, :-1],
        cpu.coefficients[grid.valid, :-1],
        rtol=1e-9,
        atol=1e-12,
    )


def test_jax_backends_agree_with_the_grid():
    pytest.importorskip("jax")
    from senpy import jax_backend as senpy_jax

    t, x = recording()
    grid = senpy.window_grid(t, WINDOW_S, OVERLAP_S)
    result = senpy_jax.compute_nustft(t, x, WINDOW_S, OVERLAP_S, empty_windows="keep")

    np.testing.assert_array_equal(result.window_index, grid.window_index)
    np.testing.assert_allclose(np.asarray(result.times), grid.times)
    np.testing.assert_array_equal(result.sample_count, grid.sample_count)
    np.testing.assert_array_equal(np.isnan(np.asarray(result.coefficients)).all(axis=1), ~grid.valid)

    samples = np.column_stack([x, x, x])
    batches = senpy_jax.pack_nustft_window_batches(
        [(t, samples)], window_s=WINDOW_S, overlap_s=OVERLAP_S, batch_size=64
    )
    packed = np.sort(np.concatenate([b.window_indices[b.row_valid] for b in batches]))
    np.testing.assert_array_equal(packed, np.flatnonzero(grid.valid))
    counts = np.concatenate([b.valid[b.row_valid].sum(axis=1) for b in batches])
    order = np.concatenate([b.window_indices[b.row_valid] for b in batches])
    np.testing.assert_array_equal(counts, grid.sample_count[order])


def test_cpu_and_jax_coefficients_agree_in_keep_mode():
    pytest.importorskip("jax")
    from senpy import jax_backend as senpy_jax

    t, x = recording()
    cpu = nustft(t, x)
    jax_result = senpy_jax.compute_nustft(t, x, WINDOW_S, OVERLAP_S, empty_windows="keep")
    got = np.asarray(jax_result.coefficients)

    assert got.shape == cpu.coefficients.shape
    # Layout is what is under test; the tolerance is the float32 device transform's.
    np.testing.assert_allclose(got[cpu.valid], cpu.coefficients[cpu.valid], rtol=1e-3, atol=1e-3)


# ── keep versus drop ────────────────────────────────────────────────


def test_keep_is_drop_laid_out_on_the_grid():
    t, x = recording()
    kept = nustft(t, x, empty_windows="keep")
    dropped = nustft(t, x, empty_windows="drop")

    np.testing.assert_array_equal(dropped.window_index, kept.window_index[kept.valid])
    np.testing.assert_array_equal(dropped.times, kept.times[kept.valid])
    np.testing.assert_array_equal(dropped.coefficients, kept.coefficients[kept.valid])
    assert np.isnan(kept.coefficients[~kept.valid]).all()
    assert dropped.valid.all()


def test_keep_mode_nan_rows_reach_the_spectrogram():
    t, x = recording()
    spec = senpy.compute_nufft_spectrogram(
        t, x, WINDOW_S, OVERLAP_S, target_fs=12.0, kind="psd", empty_windows="keep"
    )

    assert spec.Sxx.shape == (spec.valid.size, 61)  # 0..6 Hz in 0.1 Hz bins
    assert np.isnan(spec.Sxx[~spec.valid]).all()
    assert np.isfinite(spec.Sxx[spec.valid]).all()
    np.testing.assert_array_equal(
        nustft(t, x).spectrogram("psd").valid, spec.valid
    )


def test_welch_ignores_empty_windows_either_way():
    t, x = recording()
    kept = senpy.compute_nufft_welch(t, x, WINDOW_S, OVERLAP_S, empty_windows="keep")
    dropped = senpy.compute_nufft_welch(t, x, WINDOW_S, OVERLAP_S, empty_windows="drop")

    np.testing.assert_allclose(kept[1], dropped[1], rtol=1e-12)


def test_min_samples_raises_the_bar_for_a_window():
    t, x = recording()
    default = nustft(t, x)
    strict = nustft(t, x, min_samples=300)

    assert strict.valid.sum() < default.valid.sum()
    np.testing.assert_array_equal(strict.valid, strict.sample_count >= 300)
    with pytest.raises(ValueError, match="min_samples"):
        nustft(t, x, min_samples=0)


# ── origins ─────────────────────────────────────────────────────────


def test_an_explicit_origin_before_the_data_leads_with_empty_windows():
    # A reference recording (say, PSG) started 15 s before the accelerometer.
    t, x = recording(start=1_000.0)
    origin = 985.0
    result = nustft(t, x, origin_s=origin)

    assert result.origin_s == origin
    np.testing.assert_array_equal(result.times[:4], [5.0, 7.0, 9.0, 11.0])
    # Windows 0-2 (0-10, 2-12, 4-14 s) end before the data begins at 15 s; window 3
    # (6-16 s) holds its last second.
    np.testing.assert_array_equal(result.sample_count[:3], 0)
    assert not result.valid[:3].any()
    assert np.isnan(result.coefficients[:3]).all()
    assert result.valid[3]


def test_an_explicit_origin_after_the_data_starts_ignores_earlier_samples():
    t, x = recording()
    result = nustft(t, x, origin_s=3.0)
    grid = senpy.window_grid(t, WINDOW_S, OVERLAP_S, origin_s=3.0)

    first_used = grid.first[0]
    assert t[first_used] >= 3.0 and t[first_used - 1] < 3.0
    np.testing.assert_array_equal(result.sample_count, grid.sample_count)


def test_unix_origin_puts_every_recording_on_one_grid():
    epoch = 1_758_000_000.0  # a whole number of 2 s hops since the Unix epoch
    a = nustft(*recording(start=epoch + 0.37, seed=1), origin_s="unix")
    b = nustft(*recording(start=epoch + 5_411.9, seed=2), origin_s="unix")

    for result, first in ((a, epoch + 0.37), (b, epoch + 5_411.9)):
        assert result.origin_s % HOP_S == 0.0
        assert first - 2e-3 <= result.origin_s + 1e-9  # at or after the (jittered) first sample
        assert result.origin_s < first + HOP_S
    # Absolute window centres are all odd seconds for 10 s windows every 2 s.
    for result in (a, b):
        centres = result.origin_s + result.times
        np.testing.assert_allclose(np.mod(centres, HOP_S), 1.0, atol=1e-6)


def test_unix_origin_snaps_to_a_boundary_the_first_sample_barely_missed():
    t = 1_758_000_000.0 + 1e-9 + np.arange(0.0, 60.0, 1.0 / FS)
    grid = senpy.window_grid(t, WINDOW_S, OVERLAP_S, origin_s="unix")

    assert grid.origin_s == 1_758_000_000.0


def test_microsecond_epoch_timestamps_keep_their_spacing():
    t, x = recording(start=0.0)
    epoch_s = 1_758_000_000.0
    offsets_us = np.round(t * 1e6)
    absolute = nustft(epoch_s * 1e6 + offsets_us, x, ts_unit="us", origin_s="unix")
    # The same grid, expressed on a clock that starts at the epoch.
    relative = nustft(offsets_us, x, ts_unit="us", origin_s=absolute.origin_s - epoch_s)

    assert absolute.origin_s % HOP_S == 0.0
    np.testing.assert_array_equal(absolute.sample_count, relative.sample_count)
    # Differences are taken in microseconds, so the epoch costs no precision at all.
    np.testing.assert_array_equal(absolute.coefficients, relative.coefficients)


def test_bad_origins_are_refused():
    t, x = recording()
    with pytest.raises(ValueError, match="origin_s"):
        nustft(t, x, origin_s="psg")
    with pytest.raises(ValueError, match="origin_s"):
        nustft(t, x, origin_s=float("nan"))


# ── stacked spectrograms ────────────────────────────────────────────


def stacked(accel, **kwargs):
    kwargs.setdefault("empty_windows", "keep")
    return senpy.compute_stacked_spectrograms(
        accel, WINDOW_S, OVERLAP_S, target_fs=12.0, kind="psd", channels=["mag", "jerk"], **kwargs
    )


def accel_recording():
    t, _ = recording()
    rng = np.random.default_rng(9)
    return senpy.AccelerometerData(
        timestamps_us=np.round(t * 1e6).astype(np.int64),
        x=rng.normal(0, 0.5, t.size),
        y=rng.normal(0, 0.5, t.size),
        z=1 + rng.normal(0, 0.1, t.size),
    )


def test_stacked_channels_share_one_grid():
    accel = accel_recording()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # keep-mode NaN rows are not "unmatched"
        result = stacked(accel)
    grid = senpy.window_grid(accel.timestamps_s, WINDOW_S, OVERLAP_S)

    np.testing.assert_array_equal(result.window_index, grid.window_index)
    assert result.origin_s == grid.origin_s
    # Where the magnitude channel has data, jerk does too: they are the same windows.
    assert np.isfinite(result.Sxx[result.valid]).all()
    assert np.isnan(result.Sxx[~result.valid]).all()


def test_stacked_drop_mode_matches_keep_on_valid_rows():
    accel = accel_recording()
    kept = stacked(accel)
    dropped = stacked(accel, empty_windows="drop")

    np.testing.assert_array_equal(dropped.window_index, kept.window_index[kept.valid])
    np.testing.assert_array_equal(dropped.Sxx, kept.Sxx[kept.valid])


# ── nothing to report ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"empty_windows": "drop", "min_samples": 100_000},  # no window has enough samples
        {"empty_windows": "keep", "origin_s": 10_000.0},  # no window on the grid at all
        {"empty_windows": "drop", "origin_s": 10_000.0},
    ],
)
def test_cpu_and_streaming_raise_when_there_is_nothing_to_report(kwargs):
    t, x = recording()
    with pytest.raises(ValueError, match="at least one window"):
        nustft(t, x, **kwargs)
    with pytest.raises(ValueError, match="at least one window"):
        senpy.compute_nustft_streaming(t, x, WINDOW_S, OVERLAP_S, subwindow_s=HOP_S, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"empty_windows": "drop", "min_samples": 100_000},
        {"empty_windows": "keep", "origin_s": 10_000.0},
    ],
)
def test_jax_returns_zero_rows_when_there_is_nothing_to_report(kwargs):
    # Its historical behavior, kept through 4.x; senpy 5.0 makes it raise like the CPU API.
    pytest.importorskip("jax")
    from senpy import jax_backend as senpy_jax

    t, x = recording()
    result = senpy_jax.compute_nustft(t, x, WINDOW_S, OVERLAP_S, **kwargs)

    assert result.coefficients.shape[0] == 0
    assert result.window_index.size == result.sample_count.size == result.valid.size == 0


def test_keep_mode_with_every_window_too_sparse_is_all_nan_not_an_error():
    t, x = recording()
    result = nustft(t, x, empty_windows="keep", min_samples=100_000)

    assert result.valid.size == senpy.window_grid(t, WINDOW_S, OVERLAP_S).n_windows
    assert not result.valid.any()
    assert np.isnan(result.coefficients).all()


# ── the 5.0 default change ──────────────────────────────────────────


@pytest.mark.parametrize(
    "call",
    [
        lambda t, x: senpy.compute_nustft(t, x, WINDOW_S, OVERLAP_S),
        lambda t, x: senpy.compute_nufft_spectrogram(t, x, WINDOW_S, OVERLAP_S),
        lambda t, x: senpy.compute_nufft_welch(t, x, WINDOW_S, OVERLAP_S),
        lambda t, x: senpy.compute_nustft_streaming(t, x, WINDOW_S, OVERLAP_S, subwindow_s=HOP_S),
    ],
)
def test_omitting_empty_windows_warns_about_the_5_0_default(call):
    t, x = recording()
    with pytest.warns(FutureWarning, match="senpy 5.0"):
        result = call(t, x)
    del result


def test_an_explicit_choice_does_not_warn():
    t, x = recording()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        nustft(t, x, empty_windows="drop")
        nustft(t, x, empty_windows="keep")


def test_the_default_still_drops():
    t, x = recording()
    with pytest.warns(FutureWarning):
        default = senpy.compute_nustft(t, x, WINDOW_S, OVERLAP_S)

    assert default.valid.all()
    np.testing.assert_array_equal(default.coefficients, nustft(t, x, empty_windows="drop").coefficients)


def test_unknown_empty_windows_is_refused():
    t, x = recording()
    with pytest.raises(ValueError, match="empty_windows"):
        nustft(t, x, empty_windows="fill")


def test_streaming_wrapper_reports_the_counts_the_stream_used():
    # Samples out of order are dropped by the stream; the reported counts and validity
    # must describe the windows it actually computed, not the grid's ideal.
    t, x = recording()
    order = np.arange(t.size)
    order[2000:2100] = order[2000:2100][::-1]
    result = senpy.compute_nustft_streaming(
        t[order], x[order], WINDOW_S, OVERLAP_S, subwindow_s=HOP_S, empty_windows="keep"
    )
    grid = senpy.window_grid(t, WINDOW_S, OVERLAP_S)

    assert (result.sample_count < grid.sample_count).any()
    assert (result.sample_count <= grid.sample_count).all()  # never more than the window holds
    np.testing.assert_array_equal(result.valid, ~np.isnan(result.coefficients).all(axis=1))
    assert (result.sample_count[result.valid] >= 4).all()


def test_window_grid_refuses_unsorted_timestamps():
    t, _ = recording()
    with pytest.raises(ValueError, match="sorted"):
        senpy.window_grid(t[::-1], WINDOW_S, OVERLAP_S)


# ── streaming class ─────────────────────────────────────────────────


def test_streaming_min_samples_matches_the_batch_rule():
    t, x = recording()
    transform = senpy.StreamingNUSTFT(
        WINDOW_S, OVERLAP_S, HOP_S, sample_rate_hz=FS, origin_s=float(t[0]), min_samples=300
    )
    windows = transform.push(t, x) + transform.flush()

    assert windows and all(w.sample_count >= 300 for w in windows)
    assert transform.skipped_windows > 0
