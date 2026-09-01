"""The streaming NUSTFT must reproduce ``compute_nustft`` exactly, not approximately.

Every test here compares against the batch transform on the same samples. The streaming path
sees the data a chunk at a time and keeps only per-subwindow spectra, so agreement to roughly
machine precision is the whole claim being made.
"""

import numpy as np
import pytest

import senpy

FS = 100.0
WINDOW_S = 30.0
SUBWINDOW_S = 1.0


def strap_stream(seconds=300, gaps=((95, 165), (250, 251)), seed=5):
    """A jittery 100 Hz stream in one-second packets, with dropouts.

    Each packet carries its own small clock offset, which is what makes the sample grid
    genuinely nonuniform across packet boundaries rather than merely noisy.
    """
    rng = np.random.default_rng(seed)
    times = []
    for second in range(seconds):
        if any(lo <= second < hi for lo, hi in gaps):
            continue
        offset = rng.integers(-3_000, 3_000) / 1e6
        times.append(second + offset + np.arange(100) / FS)
    t = np.concatenate(times)
    signal = (
        np.sin(2 * np.pi * 1.3 * t)
        + 0.3 * np.sin(2 * np.pi * 7.7 * t)
        + 0.05 * rng.standard_normal(t.size)
    )
    return t, signal


def stream_through(t, signal, chunk=137, **kwargs):
    options = dict(
        window_s=WINDOW_S,
        overlap_s=0.0,
        subwindow_s=SUBWINDOW_S,
        sample_rate_hz=FS,
        origin_s=float(t[0]),
    )
    options.update(kwargs)
    transform = senpy.StreamingNUSTFT(**options)
    windows = []
    for start in range(0, t.size, chunk):
        windows.extend(transform.push(t[start : start + chunk], signal[start : start + chunk]))
    return transform, windows


def batch_equivalent(t, signal, chunk=137, **kwargs):
    """Streams the whole array and returns the windows ``compute_nustft`` would report.

    ``push`` only reports windows the stream has passed the end of, which is all a live stream
    can honestly say. ``compute_nustft`` knows where the recording stops, so it also emits a
    final window ending within one sample period of the last timestamp; that one comes out of
    the flush.
    """
    transform, windows = stream_through(t, signal, chunk=chunk, **kwargs)
    limit = float(t[-1]) + 1.0 / FS
    windows = windows + [w for w in transform.flush() if w.start + WINDOW_S <= limit]
    return transform, windows


def relative_deviation(actual, expected):
    return np.abs(actual - expected).max() / np.abs(expected).max()


@pytest.mark.parametrize("overlap_s", [0.0, 15.0, 29.0])
def test_matches_compute_nustft(overlap_s):
    t, signal = strap_stream()
    reference = senpy.compute_nustft(t, signal, window_s=WINDOW_S, overlap_s=overlap_s)
    _, windows = batch_equivalent(t, signal, overlap_s=overlap_s)

    assert len(windows) == len(reference.times)
    np.testing.assert_allclose([w.center - t[0] for w in windows], reference.times, atol=1e-9)

    streamed = np.stack([w.coefficients for w in windows])
    # The last bin is excluded deliberately: compute_nustft reads the Nyquist bin out of the
    # aliased FINUFFT mode and so reports its conjugate. See test_nyquist_bin_is_conjugated.
    assert relative_deviation(streamed[:, :-1], reference.coefficients[:, :-1]) < 1e-11


def test_nyquist_bin_is_conjugated():
    t, signal = strap_stream(seconds=90, gaps=())
    reference = senpy.compute_nustft(t, signal, window_s=WINDOW_S, overlap_s=0.0)
    _, windows = batch_equivalent(t, signal)

    streamed = np.stack([w.coefficients for w in windows])[:, -1]
    expected = reference.coefficients[:, -1]
    assert relative_deviation(streamed, np.conj(expected)) < 1e-11


@pytest.mark.parametrize("chunk", [1, 13, 100, 4096])
def test_chunking_does_not_change_the_result(chunk):
    t, signal = strap_stream()
    _, reference = stream_through(t, signal, chunk=100)
    _, windows = stream_through(t, signal, chunk=chunk)

    assert [w.index for w in windows] == [w.index for w in reference]
    for produced, expected in zip(windows, reference):
        np.testing.assert_array_equal(produced.coefficients, expected.coefficients)


def test_band_limit_matches_the_full_grid():
    t, signal = strap_stream()
    _, full = stream_through(t, signal)
    _, band = stream_through(t, signal, fmax=5.0)

    transform = senpy.StreamingNUSTFT(
        WINDOW_S, 0.0, SUBWINDOW_S, sample_rate_hz=FS, fmax=5.0, origin_s=float(t[0])
    )
    assert transform.frequencies[-1] == pytest.approx(5.0)
    assert len(transform.frequencies) == 151

    kept = len(band[0].coefficients)
    for narrow, wide in zip(band, full):
        np.testing.assert_allclose(narrow.coefficients, wide.coefficients[:kept], atol=0, rtol=0)


def test_detrend_off_matches_compute_nustft():
    t, signal = strap_stream(seconds=120, gaps=())
    reference = senpy.compute_nustft(t, signal, window_s=WINDOW_S, overlap_s=0.0, detrend=False)
    _, windows = batch_equivalent(t, signal, detrend=False)

    streamed = np.stack([w.coefficients for w in windows])
    assert relative_deviation(streamed[:, :-1], reference.coefficients[:, :-1]) < 1e-11


def test_dropouts_leave_holes_rather_than_shifting_windows():
    t, signal = strap_stream()
    _, windows = stream_through(t, signal)
    indices = [w.index for w in windows]

    # The 70 s dropout starting at 95 s empties the window covering [120, 150).
    assert 4 not in indices
    assert 3 in indices and 5 in indices
    # Window 3 is only partly covered, so it is reported from what it does have.
    assert windows[indices.index(3)].sample_count < 3000


def test_flush_reports_the_trailing_window_compute_nustft_stops_short_of():
    t, signal = strap_stream(seconds=125, gaps=())
    transform, windows = stream_through(t, signal)
    reference = senpy.compute_nustft(t, signal, window_s=WINDOW_S, overlap_s=0.0)

    assert len(windows) == len(reference.times)
    tail = transform.flush()
    assert [w.index for w in tail] == [4]
    assert tail[0].sample_count == 500


def test_convenience_wrapper_reproduces_compute_nustft():
    t, signal = strap_stream()
    reference = senpy.compute_nustft(t, signal, window_s=WINDOW_S, overlap_s=0.0)
    streamed = senpy.compute_nustft_streaming(
        t, signal, window_s=WINDOW_S, overlap_s=0.0, subwindow_s=SUBWINDOW_S
    )

    np.testing.assert_allclose(streamed.frequencies, reference.frequencies)
    np.testing.assert_allclose(streamed.times, reference.times, atol=1e-9)
    assert relative_deviation(
        streamed.coefficients[:, :-1], reference.coefficients[:, :-1]
    ) < 1e-11


def test_late_samples_are_counted_not_silently_folded():
    t, signal = strap_stream(seconds=90, gaps=())
    transform, _ = stream_through(t, signal)
    before = transform.dropped_samples

    # Re-push the first packet, long after the stream moved past that second.
    transform.push(t[:100], signal[:100])
    assert transform.dropped_samples == before + 100


def test_subwindow_must_divide_the_window_and_the_hop():
    with pytest.raises(RuntimeError, match="whole multiple"):
        senpy.StreamingNUSTFT(30.0, 0.0, 7.0, sample_rate_hz=FS)
    with pytest.raises(RuntimeError, match="whole multiple"):
        senpy.StreamingNUSTFT(30.0, 12.5, 1.0, sample_rate_hz=FS)


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        senpy.StreamingNUSTFT(30.0, 30.0, 1.0, sample_rate_hz=FS)
    with pytest.raises(ValueError):
        senpy.StreamingNUSTFT(30.0, 0.0, 31.0, sample_rate_hz=FS)
    with pytest.raises(ValueError):
        senpy.StreamingNUSTFT(30.0, 0.0, 1.0, sample_rate_hz=0.0)
