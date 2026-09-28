"""The throughput JAX path: the vectorized packer and :func:`compute_nustft_many`.

The packer was rewritten with vectorized NumPy; it must produce the same batches, bit for bit,
as the per-window loop it replaced, which is kept below as the reference. The driver must put
the same windows on the same grid as every other backend and agree with the CPU transform.
"""

from collections import defaultdict
from typing import Any, DefaultDict, List, Tuple

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import senpy
from senpy import jax_backend as senpy_jax
from senpy._grid import grid_from_relative, median_spacing, relative_seconds
from senpy.jax_backend import PackedNUSTFTWindowBatch, _next_power_of_two

WINDOW_S = 10.0
OVERLAP_S = 8.0
HOP_S = WINDOW_S - OVERLAP_S


def reference_pack(recordings, *, window_s, overlap_s, batch_size, ts_unit="s", origin_s=None, min_samples=4):
    """The per-window packer the vectorized one replaced, verbatim."""
    recordings = list(recordings)
    if isinstance(origin_s, (list, tuple, np.ndarray)):
        origins = list(origin_s)
    else:
        origins = [origin_s] * len(recordings)
    hop_s = window_s - overlap_s
    groups: DefaultDict[Tuple[int, int], List[Tuple[Any, ...]]] = defaultdict(list)

    for recording_index, (timestamps, samples) in enumerate(recordings):
        t = np.asarray(timestamps, dtype=np.float64)
        s = np.asarray(samples)
        if t.ndim != 1 or s.ndim != 2 or s.shape != (t.size, 3):
            raise ValueError(
                "each recording must be a (timestamps[N], samples[N, 3]) pair"
            )
        if not np.issubdtype(s.dtype, np.number) or np.iscomplexobj(s):
            raise ValueError("accelerometer samples must be real numeric values")
        if t.size < 2:
            raise ValueError("each recording requires at least two timestamps")
        if not np.all(np.isfinite(t)):
            raise ValueError("timestamps must be finite")
        if not np.all(np.isfinite(s)):
            raise ValueError("accelerometer samples must be finite")
        t, origin = relative_seconds(t, ts_unit, origins[recording_index], hop_s)
        diffs = np.diff(t)
        if np.any(diffs < 0.0):
            raise ValueError("timestamps must be sorted")
        dt_median = median_spacing(t)
        median_fs = 1.0 / dt_median
        nfft = int(window_s * median_fs)
        if nfft < 2:
            raise ValueError("window_s is too short for the observed sampling density")
        nfft_padded = _next_power_of_two(nfft)
        grid = grid_from_relative(
            t,
            origin_s=origin,
            window_s=float(window_s),
            hop_s=float(hop_s),
            dt_median_s=dt_median,
            min_samples=min_samples,
        )
        for window_index in np.flatnonzero(grid.valid).tolist():
            first = int(grid.first[window_index])
            last = int(grid.stop[window_index])
            start = window_index * hop_s
            source_width = _next_power_of_two(last - first)
            local_points = 2.0 * np.pi * ((t[first:last] - start) / window_s) - np.pi
            groups[(nfft_padded, source_width)].append(
                (
                    local_points,
                    np.asarray(s[first:last]),
                    recording_index,
                    window_index,
                    start + window_s / 2.0,
                    median_fs,
                )
            )

    packed: List[PackedNUSTFTWindowBatch] = []
    for (nfft_padded, source_width), rows in sorted(groups.items()):
        for chunk_start in range(0, len(rows), batch_size):
            chunk = rows[chunk_start : chunk_start + batch_size]
            dtype = np.result_type(*(row[1].dtype for row in chunk), np.float32)
            points = np.zeros((batch_size, source_width), dtype=dtype)
            signals = np.zeros((batch_size, 3, source_width), dtype=dtype)
            valid = np.zeros((batch_size, source_width), dtype=bool)
            row_valid = np.zeros(batch_size, dtype=bool)
            recording_indices = np.full(batch_size, -1, dtype=np.int64)
            window_indices = np.full(batch_size, -1, dtype=np.int64)
            times = np.full(batch_size, np.nan, dtype=np.float64)
            median_fss = np.ones(batch_size, dtype=dtype)
            for row_index, (row_points, row_signals, recording_index, window_index, time, row_fs) in enumerate(chunk):
                count = row_points.size
                points[row_index, :count] = row_points
                signals[row_index, :, :count] = np.asarray(row_signals, dtype=dtype).T
                valid[row_index, :count] = True
                row_valid[row_index] = True
                recording_indices[row_index] = recording_index
                window_indices[row_index] = window_index
                times[row_index] = time
                median_fss[row_index] = row_fs
            tau = (points + np.pi) / (2.0 * np.pi)
            hann = valid * 0.5 * (1.0 - np.cos(2.0 * np.pi * tau))
            packed.append(
                PackedNUSTFTWindowBatch(
                    points=points,
                    signals=signals,
                    valid=valid,
                    row_valid=row_valid,
                    window_ss=np.sum(hann * hann, axis=1),
                    recording_indices=recording_indices,
                    window_indices=window_indices,
                    times=times,
                    median_fs=median_fss,
                    nfft_padded=nfft_padded,
                    window_s=float(window_s),
                )
            )
    return tuple(packed)



def recording(seconds=300.0, fs=32.0, start=0.0, gaps=((100.0, 140.0),), channels=3, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, 1.0 / fs) + rng.uniform(-2e-3, 2e-3, int(round(seconds * fs)))
    t.sort()
    keep = np.ones(t.size, dtype=bool)
    for lo, hi in gaps:
        keep &= (t < lo) | (t >= hi)
    t = t[keep]
    samples = (np.sin(2 * np.pi * 1.1 * t)[:, None] + 0.3 * rng.standard_normal((t.size, channels)))
    return start + t, samples.astype(dtype)


# ── the vectorized packer ───────────────────────────────────────────


def assert_same_batches(got, want):
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert a.nfft_padded == b.nfft_padded and a.window_s == b.window_s
        for field in PackedNUSTFTWindowBatch.__dataclass_fields__:
            x, y = getattr(a, field), getattr(b, field)
            if isinstance(x, np.ndarray):
                assert x.dtype == y.dtype, field
                np.testing.assert_array_equal(x, y, err_msg=field)


@pytest.mark.parametrize("batch_size", [7, 128, 4096])
def test_packer_matches_the_per_window_loop_bit_for_bit(batch_size):
    recordings = [
        recording(seed=1),
        recording(seconds=200.0, fs=25.0, seed=2),  # another rate: another nfft bucket
        recording(seconds=120.0, fs=50.0, gaps=(), seed=3, dtype=np.float32),
    ]
    kwargs = dict(window_s=WINDOW_S, overlap_s=OVERLAP_S, batch_size=batch_size)

    assert_same_batches(
        senpy_jax.pack_nustft_window_batches(recordings, **kwargs),
        reference_pack(recordings, **kwargs),
    )


def test_packer_matches_with_origins_units_and_min_samples():
    a, b = recording(start=1_758_000_000.37, seed=4), recording(start=1_758_000_512.9, seed=5)
    recordings = [(np.round(t * 1e3), s) for t, s in (a, b)]
    kwargs = dict(
        window_s=WINDOW_S,
        overlap_s=OVERLAP_S,
        batch_size=64,
        ts_unit="ms",
        origin_s=["unix", 1_758_000_500.0],
        min_samples=200,
    )

    assert_same_batches(
        senpy_jax.pack_nustft_window_batches(recordings, **kwargs),
        reference_pack(recordings, **kwargs),
    )


def test_packer_still_rejects_bad_recordings():
    t = np.arange(8.0)
    with pytest.raises(ValueError, match=r"samples\[N, 3\]"):
        senpy_jax.pack_nustft_window_batches([(t, np.ones((8, 2)))], window_s=2.0, overlap_s=1.0, batch_size=2)
    with pytest.raises(ValueError, match="sorted"):
        senpy_jax.pack_nustft_window_batches(
            [(t[[0, 2, 1, 3, 4, 5, 6, 7]], np.ones((8, 3)))], window_s=2.0, overlap_s=1.0, batch_size=2
        )


# ── compute_nustft_many ─────────────────────────────────────────────


def many(recordings, **kwargs):
    kwargs.setdefault("empty_windows", "keep")
    return senpy_jax.compute_nustft_many(recordings, window_s=WINDOW_S, overlap_s=OVERLAP_S, **kwargs)


@pytest.mark.parametrize("channels", [1, 2, 3, 5])
def test_many_matches_the_cpu_transform_on_every_channel(channels):
    recordings = [recording(channels=channels, seed=6), recording(seconds=200.0, fs=25.0, channels=channels, seed=7)]
    results = many(recordings, target_fs=12.0)

    assert len(results) == 2
    for (t, samples), per_channel in zip(recordings, results):
        assert len(per_channel) == channels
        for c, got in enumerate(per_channel):
            want = senpy.compute_nustft(t, samples[:, c], WINDOW_S, OVERLAP_S, target_fs=12.0, empty_windows="keep")
            np.testing.assert_array_equal(got.window_index, want.window_index)
            np.testing.assert_array_equal(got.sample_count, want.sample_count)
            np.testing.assert_array_equal(got.valid, want.valid)
            np.testing.assert_array_equal(got.times, want.times)
            # The CPU target_fs grid is built by repeated addition, so it drifts in the last digit.
            np.testing.assert_allclose(got.frequencies, want.frequencies, rtol=1e-13)
            assert got.origin_s == want.origin_s
            assert np.isnan(got.coefficients[~got.valid]).all()
            # float32 device arithmetic on host-computed float64 coordinates.
            np.testing.assert_allclose(
                got.coefficients[got.valid], want.coefficients[want.valid], rtol=0, atol=5e-5
            )


def test_a_1d_samples_array_is_one_channel():
    t, samples = recording(channels=1, seed=8)
    (one,), = many([(t, samples[:, 0])])
    (two,), = many([(t, samples)])

    np.testing.assert_array_equal(one.coefficients, two.coefficients)


def test_drop_mode_is_keep_mode_without_the_empty_rows():
    recordings = [recording(seed=9)]
    (kept,) = many(recordings)
    (dropped,) = many(recordings, empty_windows="drop")

    for k, d in zip(kept, dropped):
        np.testing.assert_array_equal(d.window_index, k.window_index[k.valid])
        np.testing.assert_array_equal(d.coefficients, k.coefficients[k.valid])
        assert d.valid.all()


@pytest.mark.parametrize(
    "schedule",
    [dict(max_in_flight=1, build_threads=1), dict(max_in_flight=2, build_threads=3), dict(max_in_flight=5, build_threads=8)],
)
def test_threads_and_queue_depth_do_not_change_a_bit(schedule):
    recordings = [recording(seed=10), recording(seconds=150.0, fs=25.0, channels=4, seed=11)]
    baseline = many(recordings, rows_per_call=64)
    other = many(recordings, rows_per_call=64, **schedule)

    for a_rec, b_rec in zip(baseline, other):
        for a, b in zip(a_rec, b_rec):
            np.testing.assert_array_equal(a.coefficients, b.coefficients)


@pytest.mark.parametrize("rows_per_call", [16, 100, 8192])
def test_batch_size_changes_only_float32_rounding(rows_per_call):
    # A different batch shape is a different XLA program, whose float32 reductions may round
    # differently; the windows and their layout must not change.
    recordings = [recording(seed=10), recording(seconds=150.0, fs=25.0, channels=4, seed=11)]
    baseline = many(recordings, rows_per_call=64)
    other = many(recordings, rows_per_call=rows_per_call)

    for a_rec, b_rec in zip(baseline, other):
        for a, b in zip(a_rec, b_rec):
            np.testing.assert_array_equal(np.isnan(a.coefficients), np.isnan(b.coefficients))
            np.testing.assert_allclose(a.coefficients, b.coefficients, rtol=0, atol=5e-6)


def test_many_agrees_with_the_packer_and_batch_transform():
    t, samples = recording(seed=12)
    (results,) = many([(t, samples)], empty_windows="drop")
    batches = senpy_jax.pack_nustft_window_batches(
        [(t, samples)], window_s=WINDOW_S, overlap_s=OVERLAP_S, batch_size=512
    )
    for batch in batches:
        coefficients = np.asarray(
            senpy_jax.compute_nustft_window_batch(
                batch.points, batch.signals, batch.valid,
                nfft_padded=batch.nfft_padded, median_fs=batch.median_fs,
            )
        )
        rows = np.flatnonzero(batch.row_valid)
        positions = np.searchsorted(results[0].window_index, batch.window_indices[rows])
        for c in range(3):
            np.testing.assert_allclose(
                results[c].coefficients[positions], coefficients[rows, c], rtol=0, atol=1e-5
            )


def test_many_honours_per_recording_origins():
    a, b = recording(start=985.0, seed=13), recording(start=2_000.0, seed=14)
    ra, rb = many([a, b], origin_s=[970.0, None])

    assert ra[0].origin_s == 970.0 and rb[0].origin_s == b[0][0]
    assert not ra[0].valid[:3].any()  # windows 0-2 end before the data starts at 985 s


def test_many_with_nothing_to_report_gives_zero_rows():
    (per_channel,) = many([recording(seed=15)], empty_windows="drop", min_samples=100_000)
    assert all(r.coefficients.shape[0] == 0 for r in per_channel)


def test_many_warns_about_the_5_0_default():
    with pytest.warns(FutureWarning, match="senpy 5.0"):
        senpy_jax.compute_nustft_many([recording(seed=16)], window_s=WINDOW_S, overlap_s=OVERLAP_S)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(rows_per_call=0), "rows_per_call"),
        (dict(max_in_flight=1.5), "max_in_flight"),
        (dict(build_threads=-1), "build_threads"),
        (dict(target_fs=100.0), "target_fs"),
        (dict(origin_s=[None, None]), "one per recording"),
    ],
)
def test_many_refuses_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        many([recording(seed=17)], **kwargs)
