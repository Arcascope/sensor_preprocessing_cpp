/// Originally written by Franco Tavella
/// https://github.com/Arcascope/ArcaWatch-WearOS/pull/17

#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <numeric>

// Conditionally include finufft if available
#ifdef USE_FINUFFT
#include <fftw3.h>
#include <finufft.h>
#endif

// Logging macros for Android
// #define LOG_TAG "SensorProcessing"
// #define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
// #define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
// #define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
// #define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct SpectrogramResult
{
    std::vector<double> freqs;            // Hz
    std::vector<double> times;            // Relative time in seconds
    std::vector<std::vector<double>> Sxx; // [times x frequencies]
};

struct MotionFeaturesResult
{
    std::vector<std::vector<double>> features; // BR, HR, freqSum, BR_std, HR_std
    SpectrogramResult spectrogram;
};

class SensorProcessor
{
private:
    // Bit-reverse permutation for FFT
    static void bitReverse(std::vector<std::complex<double>> &data)
    {
        int n = data.size();
        int j = 0;
        for (int i = 1; i < n; ++i)
        {
            int bit = n >> 1;
            while (j & bit)
            {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if (i < j)
            {
                std::swap(data[i], data[j]);
            }
        }
    }

    // Optimized radix-2 FFT implementation
    static void fastFFT(std::vector<std::complex<double>> &data)
    {
        int n = data.size();
        // Assumes n is always a power of 2 (guaranteed by nextPowerOf2)
        if (n <= 1)
            return;

        // Bit-reverse the input
        bitReverse(data);

        // Cooley-Tukey FFT
        for (int len = 2; len <= n; len <<= 1)
        {
            double angle = -2.0 * M_PI / len;
            std::complex<double> wlen(cos(angle), sin(angle));

            for (int i = 0; i < n; i += len)
            {
                std::complex<double> w(1, 0);
                for (int j = 0; j < len / 2; ++j)
                {
                    std::complex<double> u = data[i + j];
                    std::complex<double> v = data[i + j + len / 2] * w;
                    data[i + j] = u + v;
                    data[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
    }

public:
    static std::vector<double> hannWindow(int N)
    {
        std::vector<double> window(N);
        for (int i = 0; i < N; ++i)
            window[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / N)); // Periodic window
        return window;
    }

    // Find the next power of 2 greater than or equal to n
    static int nextPowerOf2(int n)
    {
        int power = 1;
        while (power < n)
            power <<= 1;
        return power;
    }

    static std::vector<std::complex<double>> calculateFFT(const std::vector<double> &signal, int nperseg)
    {
        // nfft = nperseg (no zero-padding)
        std::vector<std::complex<double>> complexData(nperseg, {0.0, 0.0});
        for (size_t i = 0; i < signal.size() && i < nperseg; ++i)
        {
            complexData[i] = std::complex<double>(signal[i], 0.0);
        }

        if ((nperseg & (nperseg - 1)) == 0)
        {
            // nperseg is power of 2, use fast FFT
            fastFFT(complexData);
        }
        else
        {
            // For large sizes, use Bluestein's algorithm
            if (nperseg >= 1000)
            {
                bluesteinFFT(complexData);
            }
            else
            {
                // nperseg is not power of 2 and small, use optimized DFT
                discreteFourierTransform(complexData);
            }
        }
        return complexData;
    }

    // Bluestein's FFT algorithm for arbitrary-size FFT (much faster than DFT for large N)
    static void bluesteinFFT(std::vector<std::complex<double>> &data)
    {
        int n = data.size();
        if (n <= 1)
            return;

        // Find next power of 2 that's at least 2*n-1
        int m = 1;
        while (m < 2 * n - 1)
            m <<= 1;

        // Pre-compute chirp factors
        std::vector<std::complex<double>> chirp(n);
        for (int k = 0; k < n; ++k)
        {
            double angle = M_PI * k * k / n;
            chirp[k] = std::complex<double>(cos(angle), -sin(angle));
        }

        // Prepare sequences for convolution
        std::vector<std::complex<double>> a(m, {0.0, 0.0});
        std::vector<std::complex<double>> b(m, {0.0, 0.0});

        for (int k = 0; k < n; ++k)
        {
            a[k] = data[k] * chirp[k];
        }

        b[0] = chirp[0];
        for (int k = 1; k < n; ++k)
        {
            b[k] = std::conj(chirp[k]);
            b[m - k] = std::conj(chirp[k]);
        }

        // Perform convolution via FFT
        fastFFT(a);
        fastFFT(b);

        for (int k = 0; k < m; ++k)
        {
            a[k] *= b[k];
        }

        // IFFT
        for (int k = 0; k < m; ++k)
        {
            a[k] = std::conj(a[k]);
        }
        fastFFT(a);
        for (int k = 0; k < m; ++k)
        {
            a[k] = std::conj(a[k]) / static_cast<double>(m);
        }

        // Extract result and apply final chirp
        for (int k = 0; k < n; ++k)
        {
            data[k] = a[k] * chirp[k];
        }
    }

    // Optimized Discrete Fourier Transform for non-power-of-2 sizes (SciPy compatible)
    static void discreteFourierTransform(std::vector<std::complex<double>> &data)
    {
        int n = data.size();
        std::vector<std::complex<double>> temp(n);

        // Pre-compute twiddle factors for reuse
        std::vector<std::complex<double>> twiddle(n);
        for (int k = 0; k < n; ++k)
        {
            double angle = -2.0 * M_PI * k / n;
            twiddle[k] = std::complex<double>(cos(angle), sin(angle));
        }

        // Optimized DFT with reduced trigonometric calculations
        for (int k = 0; k < n; ++k)
        {
            std::complex<double> sum(0, 0);
            for (int j = 0; j < n; ++j)
            {
                // Use modular arithmetic to reuse twiddle factors
                int twiddle_idx = (k * j) % n;
                sum += data[j] * twiddle[twiddle_idx];
            }
            temp[k] = sum;
        }

        data = std::move(temp); // Use move semantics
    }

    // Matches scipy's spectrogram function with mode='magnitude', scaling='density',
    // window='hann', zero-padding, and detrend='constant'
    static SpectrogramResult computeSpectrogram(
        const std::vector<double> &signal,
        double fs,
        int nperseg,
        int noverlap)
    {
        int step = nperseg - noverlap;
        int n_segments = (signal.size() - nperseg) / step + 1;
        int nfft = nperseg;
        int n_freqs = nfft / 2 + 1; // Frequency bins

        SpectrogramResult result;
        result.freqs.resize(n_freqs);
        result.times.resize(n_segments);
        result.Sxx.assign(n_segments, std::vector<double>(n_freqs, 0.0));

        // Frequencies based on nperseg
        for (int k = 0; k < n_freqs; ++k)
            result.freqs[k] = (fs * k) / nfft;

        // Pre-compute window (avoid recomputation)
        auto window = hannWindow(nperseg);

        // Pre-compute scaling factor
        double window_power_sum = std::inner_product(window.begin(), window.end(), window.begin(), 0.0);
        double scale_factor = 1.0 / std::sqrt(fs * window_power_sum);

        // Pre-allocate segment buffer (reuse memory)
        std::vector<double> segment(nperseg);

        // Loop over segments
        for (int seg = 0; seg < n_segments; ++seg)
        {
            int start = seg * step;

            double segment_sum = 0.0;
            for (int i = 0; i < nperseg; ++i)
            {
                segment[i] = signal[start + i];
                segment_sum += segment[i];
            }

            // Detrend: subtract mean
            double segment_mean = segment_sum / nperseg;
            for (int i = 0; i < nperseg; ++i)
            {
                segment[i] = (segment[i] - segment_mean) * window[i]; // Detrend and window in one step
            }

            auto fft_result = calculateFFT(segment, nperseg);

            for (int k = 0; k < n_freqs; ++k)
            {
                double real = fft_result[k].real();
                double imag = fft_result[k].imag();
                result.Sxx[seg][k] = std::sqrt(real * real + imag * imag) * scale_factor;
            }

            result.times[seg] = (start + nperseg / 2.0) / fs; // center time in sec
        }
        return result;
    }

    // Compute spectrogram from non-uniformly sampled data using NUFFT
    // Bypasses resampling to avoid spectral artifacts in the 0-10 Hz band
    // secperseg and secoverlap are in seconds (not samples)
    // target_fs: if > 0, resample output to fixed freq grid (fmax = target_fs/2, spacing = 1/secperseg)
    static SpectrogramResult computeSpectrogramNUFFT(
        const std::vector<double> &timestamps,
        const std::vector<double> &signal,
        double secperseg,
        double secoverlap,
        double target_fs = 0.0)
    {
#ifndef USE_FINUFFT
        throw std::runtime_error("computeSpectrogramNUFFT requires finufft support (USE_FINUFFT not defined)");
#endif

        if (timestamps.size() != signal.size())
            throw std::runtime_error("timestamps and signal must have the same length");

        // Estimate median sampling rate
        std::vector<double> diffs;
        for (size_t i = 1; i < timestamps.size(); ++i)
            diffs.push_back(timestamps[i] - timestamps[i-1]);

        std::sort(diffs.begin(), diffs.end());
        double dt_median = diffs[diffs.size() / 2];
        double median_fs = 1.0 / dt_median;

        // Window duration and hop in seconds (input is already in seconds)
        double win_dur = secperseg;
        double hop_dur = secperseg - secoverlap;

        // Compute nfft based on actual window duration and median sampling rate
        int nfft = (int)(secperseg * median_fs);
        // Round up to next power of 2 for efficiency
        int nfft_padded = nfft;
        {
            int p = 1;
            while (p < nfft) p <<= 1;
            nfft_padded = p;
        }

        // Frequency axis for positive frequencies
        int n_pos_freqs = nfft_padded / 2;
        std::vector<double> freqs(n_pos_freqs);
        for (int i = 0; i < n_pos_freqs; ++i)
            freqs[i] = i / win_dur;

        // Prepare output
        std::vector<double> window_centres;
        std::vector<std::vector<double>> spectra;

        double t_start = timestamps[0];
        double t_end = timestamps[timestamps.size() - 1];

        // Sliding window loop
        double win_start = t_start;
        while (win_start + win_dur <= t_end + dt_median)
        {
            double win_end = win_start + win_dur;

            // Select samples in window
            std::vector<double> t_win, s_win;
            for (size_t i = 0; i < timestamps.size(); ++i)
            {
                if (timestamps[i] >= win_start && timestamps[i] < win_end)
                {
                    t_win.push_back(timestamps[i]);
                    s_win.push_back(signal[i]);
                }
            }

            if (t_win.size() < 4)
            {
                // Skip windows with too few samples
                win_start += hop_dur;
                continue;
            }

            // Compute tau and Hann window
            std::vector<double> tau(t_win.size());
            std::vector<double> hann(t_win.size());
            double s_mean = 0.0;
            for (size_t i = 0; i < s_win.size(); ++i)
                s_mean += s_win[i];
            s_mean /= s_win.size();

            // Apply Hann window and detrend
            for (size_t i = 0; i < t_win.size(); ++i)
            {
                tau[i] = (t_win[i] - win_start) / win_dur;
                hann[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * tau[i]));
                s_win[i] = (s_win[i] - s_mean) * hann[i];
            }

            // Map to NUFFT domain [-pi, pi)
            std::vector<double> x(t_win.size());
            for (size_t i = 0; i < tau.size(); ++i)
                x[i] = 2.0 * M_PI * tau[i] - M_PI;

#ifdef USE_FINUFFT
            // Compute NUFFT using finufft
            // Allocate arrays for finufft (uses std::complex<double>)
            std::vector<std::complex<double>> c_complex(t_win.size());
            std::vector<std::complex<double>> fhat_complex(nfft_padded);

            // Fill input array with signal values as complex
            for (size_t i = 0; i < t_win.size(); ++i)
                c_complex[i] = std::complex<double>(s_win[i], 0.0);

            // Make plan and execute
            int ier = 0;
            finufft_plan plan = nullptr;
            finufft_opts opts;
            finufft_default_opts(&opts);

            long nfft_long = nfft_padded;
            ier = finufft_makeplan(1, 1, &nfft_long, +1, 1, 1e-14, &plan, &opts);
            if (ier != 0)
                throw std::runtime_error("finufft_makeplan failed");

            ier = finufft_setpts(plan, (long)t_win.size(), x.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr);
            if (ier != 0)
                throw std::runtime_error("finufft_setpts failed");

            ier = finufft_execute(plan, c_complex.data(), fhat_complex.data());
            if (ier != 0)
                throw std::runtime_error("finufft_execute failed");

            finufft_destroy(plan);

            // Extract positive frequencies and compute PSD
            std::vector<double> psd(n_pos_freqs);
            double win_ss = 0.0;
            for (size_t i = 0; i < hann.size(); ++i)
                win_ss += hann[i] * hann[i];

            if (win_ss > 0.0)
            {
                // Positive half starts at index nfft_padded/2
                for (int k = 0; k < n_pos_freqs; ++k)
                {
                    int idx = nfft_padded / 2 + k;
                    double real = fhat_complex[idx].real();
                    double imag = fhat_complex[idx].imag();
                    double mag_sq = real * real + imag * imag;
                    psd[k] = mag_sq / (median_fs * win_ss);
                    // One-sided: DC unchanged, k>0 × 2
                    if (k > 0)
                        psd[k] *= 2.0;
                }
            }
#else
            // Fallback: return empty spectrum when finufft not available
            std::vector<double> psd(n_pos_freqs, 0.0);
#endif

            spectra.push_back(psd);
            window_centres.push_back(win_start + win_dur / 2.0);

            win_start += hop_dur;
        }

        // Build result
        SpectrogramResult result;
        result.freqs = freqs;

        if (spectra.empty())
        {
            result.times.clear();
            result.Sxx.clear();
        }
        else
        {
            result.times.resize(window_centres.size());
            for (size_t i = 0; i < window_centres.size(); ++i)
                result.times[i] = window_centres[i] - t_start;
            result.Sxx = spectra;
        }

        // If target_fs is specified, resample to a common frequency grid via cubic spline
        if (target_fs > 0.0 && !result.Sxx.empty())
        {
            double fmax = target_fs / 2.0;
            double freq_spacing = 1.0 / secperseg;

            // Build target frequency grid
            std::vector<double> target_freqs;
            for (double f = 0; f <= fmax + freq_spacing / 2; f += freq_spacing)
                target_freqs.push_back(f);

            // Resample each time window using cubic spline
            std::vector<std::vector<double>> resampled_Sxx(result.Sxx.size());

            for (size_t t = 0; t < result.Sxx.size(); ++t)
            {
                const auto &psd_original = result.Sxx[t];

                // Compute cubic spline coefficients
                std::vector<double> freqs_dbl(freqs.begin(), freqs.end());
                auto coeffs = computeNaturalCubicSplineCoeffs(freqs_dbl, psd_original);

                // Evaluate at target frequencies
                auto resampled = evaluateCubicSpline(freqs_dbl, psd_original, coeffs, target_freqs);
                resampled_Sxx[t] = resampled;
            }

            result.freqs = target_freqs;
            result.Sxx = resampled_Sxx;
        }

        return result;
    }

    // Compute Short-Time Fourier Transform returning complex values
    // Returns shape: (n_times, n_frequencies, 2) where last dimension is [real, imag]
    static std::vector<std::vector<std::vector<double>>> computeShortTimeFT(
        const std::vector<double> &signal,
        double fs,
        int nperseg,
        int noverlap)
    {
        int step = nperseg - noverlap;
        int n_segments = (signal.size() - nperseg) / step + 1;
        int nfft = nperseg;
        int n_freqs = nfft / 2 + 1; // Frequency bins

        // Result shape: (n_times, n_frequencies, 2)
        std::vector<std::vector<std::vector<double>>> result(
            n_segments,
            std::vector<std::vector<double>>(n_freqs, std::vector<double>(2, 0.0)));

        // Pre-compute window (avoid recomputation)
        auto window = hannWindow(nperseg);

        // Pre-allocate segment buffer (reuse memory)
        std::vector<double> segment(nperseg);

        // Loop over segments
        for (int seg = 0; seg < n_segments; ++seg)
        {
            int start = seg * step;

            double segment_sum = 0.0;
            for (int i = 0; i < nperseg; ++i)
            {
                segment[i] = signal[start + i];
                segment_sum += segment[i];
            }

            // Detrend: subtract mean
            double segment_mean = segment_sum / nperseg;
            for (int i = 0; i < nperseg; ++i)
            {
                segment[i] = (segment[i] - segment_mean) * window[i]; // Detrend and window in one step
            }

            auto fft_result = calculateFFT(segment, nperseg);

            // Store real and imaginary parts
            for (int k = 0; k < n_freqs; ++k)
            {
                result[seg][k][0] = fft_result[k].real();
                result[seg][k][1] = fft_result[k].imag();
            }
        }
        return result;
    }

    static std::vector<double> findSpectrogramPeaks(
        const std::vector<std::vector<double>> &Sxx,
        const std::vector<double> &frequencies,
        double relative_prominence,
        double scaling_factor = 60.0f)
    {
        int n_times = Sxx.size();
        int n_freqs = Sxx[0].size();
        std::vector<double> peaks(n_times);
        auto sxx_rescaled = rescaleSpectrogram(Sxx);
        double sxx_min = computeMinMax2D(sxx_rescaled, false);

        for (int t = 0; t < n_times; ++t)
        {
            std::vector<double> Sxx_time_slice(n_freqs);
            for (size_t f = 0; f < n_freqs; ++f)
            {
                Sxx_time_slice[f] = sxx_rescaled[t][f];
            }
            auto peak_indices = findPeaks(Sxx_time_slice, relative_prominence * (1 - sxx_min));
            int best_idx = 0;
            if (!peak_indices.empty())
            {
                // Find the peak with maximum power
                best_idx = peak_indices[0];
                double max_power = Sxx_time_slice[best_idx];
                for (int idx : peak_indices)
                {
                    if (Sxx_time_slice[idx] > max_power)
                    {
                        max_power = Sxx_time_slice[idx];
                        best_idx = idx;
                    }
                }
            }
            peaks[t] = scaling_factor * frequencies[best_idx];
        }
        return peaks;
    }

    // Function to compute the min or max of a 2D vector
    static double computeMinMax2D(const std::vector<std::vector<double>> &data, bool findMax = true)
    {
        double result = findMax ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
        for (const auto &row : data)
        {
            for (double val : row)
            {
                if (findMax)
                {
                    result = std::max(result, val);
                }
                else
                {
                    result = std::min(result, val);
                }
            }
        }
        return result;
    }

    // Enhanced function to extract peaks from spectrum with prominence and filtering
    static std::pair<std::vector<double>, std::vector<double>> extractMotionPeaks(
        const std::vector<std::vector<double>> &Sxx,
        const std::vector<double> &frequencies,
        double brMinHz,
        double brMaxHz,
        double htMinHz,
        double htMaxHz,
        double spectrogram_smoothing_freq = 1.0,
        double spectrogram_smoothing_time = 2.0,
        double hr_max_change_per_sec = 15.0,
        double br_max_change_per_sec = 7.5,
        double time_resolution = 30.0)
    {
        int n_times = Sxx.size();
        int n_freqs = Sxx[0].size();
        double sampling_rate = 1.0 / time_resolution;

        // Find frequency indices for breathing and heart rate ranges
        int br_start = 0, br_end = 0, hr_start = 0, hr_end = 0;
        for (size_t i = 0; i < frequencies.size(); ++i)
        {
            if (frequencies[i] >= brMinHz && br_start == 0)
                br_start = i;
            if (frequencies[i] <= brMaxHz)
                br_end = i;
            if (frequencies[i] >= htMinHz && hr_start == 0)
                hr_start = i;
            if (frequencies[i] <= htMaxHz)
                hr_end = i;
        }

        // Extract and rescale HR spectrogram section
        auto [p_hr, f_hr] = extractAndRescaleSpectrogramSection(Sxx, frequencies, hr_start, hr_end);

        // Extract and rescale BR spectrogram section
        auto [p_br, f_br] = extractAndRescaleSpectrogramSection(Sxx, frequencies, br_start, br_end);

        // // Calculate prominence thresholds
        // // max is always 1.0 after extractAndRescale
        // double hr_max = computeMinMax2D(p_hr, true);
        // double hr_min = computeMinMax2D(p_hr, false);
        // double br_max = computeMinMax2D(p_br, true);
        // double br_min = computeMinMax2D(p_br, false);
        // double HR_prominence_threshold = (hr_max - hr_min) * 0.1;
        // double BR_prominence_threshold = (br_max - br_min) * 0.1;
        double relative_prominence_threshold = 0.1;

        std::vector<double> HR_peaks = findSpectrogramPeaks(p_hr, f_hr, relative_prominence_threshold);
        std::vector<double> BR_peaks = findSpectrogramPeaks(p_br, f_br, relative_prominence_threshold);

        // Apply smoothing with rate constraints
        auto HR_smooth = smoothPeaksWithRateConstraint(HR_peaks, hr_max_change_per_sec, sampling_rate);
        auto BR_smooth = smoothPeaksWithRateConstraint(BR_peaks, br_max_change_per_sec, sampling_rate);

        return {BR_smooth, HR_smooth};
    }

    // Helper function to compute rolling standard deviation
    static std::vector<double> rollingStd(const std::vector<double> &data, double window_minutes, double seconds_per_window = 30.0)
    {
        std::vector<double> result(data.size(), 0.0);

        // Convert window_minutes to sample indices
        int window_idx = static_cast<int>(window_minutes * 60.0 / seconds_per_window);

        for (size_t i = 0; i < data.size(); ++i)
        {
            int start = std::max(0, static_cast<int>(i) - window_idx);
            int end = std::min(static_cast<int>(data.size()), static_cast<int>(i) + window_idx);

            // Compute mean
            double mean = 0.0;
            int count = end - start;
            for (int j = start; j < end; ++j)
            {
                mean += data[j];
            }
            mean /= count;

            // Compute standard deviation
            double variance = 0.0;
            for (int j = start; j < end; ++j)
            {
                variance += (data[j] - mean) * (data[j] - mean);
            }
            result[i] = std::sqrt(variance / count);
        }

        return result;
    }

    // Helper function to compute median
    static double computeMedian(std::vector<double> data)
    {
        if (data.empty())
            return 0.0;

        std::sort(data.begin(), data.end());
        size_t n = data.size();

        if (n % 2 == 0)
        {
            return (data[n / 2 - 1] + data[n / 2]) / 2.0;
        }
        else
        {
            return data[n / 2];
        }
    }

    // Helper function to compute percentile
    static double computePercentile(std::vector<double> data, double percentile)
    {
        if (data.empty())
            return 0.0;

        std::sort(data.begin(), data.end());
        size_t n = data.size();

        double index = (percentile / 100.0) * (n - 1);
        size_t lower = static_cast<size_t>(std::floor(index));
        size_t upper = static_cast<size_t>(std::ceil(index));

        if (lower == upper)
        {
            return data[lower];
        }
        else
        {
            double weight = index - lower;
            return data[lower] * (1.0 - weight) + data[upper] * weight;
        }
    }

    // Helper function to extract and rescale a frequency range from spectrogram
    static std::pair<std::vector<std::vector<double>>, std::vector<double>> extractAndRescaleSpectrogramSection(
        const std::vector<std::vector<double>> &Sxx,
        const std::vector<double> &frequencies,
        int start_idx,
        int end_idx)
    {
        // Extract spectrogram section
        int n_times = Sxx.size();
        std::vector<std::vector<double>> Sxx_section(
            n_times,
            std::vector<double>(end_idx - start_idx + 1));
        std::vector<double> f_section;

        for (int f = start_idx; f <= end_idx; ++f)
        {
            for (int t = 0; t < n_times; t++)
            {
                Sxx_section[t][f - start_idx] = Sxx[t][f];
            }
            f_section.push_back(frequencies[f]);
        }

        // Rescale the extracted section
        auto rescaled_section = rescaleSpectrogram(Sxx_section);

        return {rescaled_section, f_section};
    }

    // Helper function to rescale spectrogram (normalize by max along frequency axis)
    static std::vector<std::vector<double>> rescaleSpectrogram(
        const std::vector<std::vector<double>> &Sxx,
        double epsilon = 1e-12)
    {
        /// ax_1_max =np.max(Sxx, axis=1)
        /// Sxx /= ax_1_max
        int n_times = Sxx.size();
        int n_freqs = Sxx[0].size();
        std::vector<std::vector<double>> rescaled(
            n_times,
            std::vector<double>(n_freqs));

        // Find max value for each time bin
        for (int t = 0; t < n_times; ++t)
        {
            double max_val = epsilon;
            for (int f = 0; f < n_freqs; ++f)
            {
                max_val = std::max(max_val, Sxx[t][f]);
            }

            // Normalize by max value
            for (int f = 0; f < n_freqs; ++f)
            {
                rescaled[t][f] = Sxx[t][f] / max_val;
            }
        }

        return rescaled;
    }

    static std::vector<int> findPeaks(
        const std::vector<double> &signal,
        double prominence_threshold)
    {
        std::vector<int> peaks;
        int n = signal.size();

        for (int i = 1; i < n - 1; ++i)
        {
            // Check if it's a local maximum
            if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1])
            {
                // Calculate prominence
                double left_min = signal[i];
                double right_min = signal[i];

                // Find minimum to the left
                for (int j = i - 1; j >= 0; --j)
                {
                    left_min = std::min(left_min, signal[j]);
                    if (signal[j] > signal[i])
                        break;
                }

                // Find minimum to the right
                for (int j = i + 1; j < n; ++j)
                {
                    right_min = std::min(right_min, signal[j]);
                    if (signal[j] > signal[i])
                        break;
                }

                double prominence = signal[i] - std::max(left_min, right_min);
                if (prominence >= prominence_threshold)
                {
                    peaks.push_back(i);
                }
            }
        }

        return peaks;
    }

    // Helper function to smooth peaks with rate-of-change constraints
    static std::vector<double> smoothPeaksWithRateConstraint(
        const std::vector<double> &peaks,
        double max_change_per_sec,
        double sampling_rate)
    {
        if (peaks.empty())
            return peaks;

        // Apply maximum change constraint (clamp derivative)
        double max_delta = max_change_per_sec / sampling_rate;
        std::vector<double> constrained = peaks;

        for (size_t i = 1; i < peaks.size(); ++i)
        {
            double delta = constrained[i] - constrained[i - 1];
            if (std::abs(delta) > max_delta)
            {
                constrained[i] = constrained[i - 1] + (delta > 0 ? max_delta : -max_delta);
            }
        }

        return constrained;
    }

    // Helper function to smooth spectrogram peaks with Gaussian filter and rate constraints
    static std::vector<double> smoothSpectrogramPeaks(
        const std::vector<double> &peaks,
        double sampling_rate,
        double max_change_per_sec = 10.0,
        double filter_sigma = 2.0)
    {
        if (peaks.empty())
            return peaks;

        // First, apply Gaussian smoothing
        std::vector<double> smoothed = gaussianFilter1D(peaks, filter_sigma);

        // Then apply rate-of-change constraint
        std::vector<double> constrained = smoothPeaksWithRateConstraint(smoothed, max_change_per_sec, sampling_rate);

        return constrained;
    }

    // Helper function to apply Gaussian filter (1D smoothing)
    static std::vector<double> gaussianFilter1D(const std::vector<double> &data, double sigma, double truncate = 4.0)
    {
        if (data.empty())
            return data;

        // SciPy-compatible kernel size calculation
        int lw = static_cast<int>(truncate * sigma + 0.5); // radius
        int kernelSize = 2 * lw + 1;                       // full kernel size (always odd)

        std::vector<double> kernel(kernelSize);
        int center = lw; // center is at radius position
        double sum = 0.0;
        double sigma_sq_2 = 2.0 * sigma * sigma;

        // Generate Gaussian kernel (SciPy-compatible)
        for (int i = 0; i < kernelSize; ++i)
        {
            int x = i - center;
            kernel[i] = std::exp(-(x * x) / sigma_sq_2);
            sum += kernel[i];
        }

        // Normalize kernel
        for (double &k : kernel)
        {
            k /= sum;
        }

        // Apply convolution with reflection padding (SciPy mode="reflect")
        std::vector<double> result(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            double value = 0.0;
            for (int j = 0; j < kernelSize; ++j)
            {
                int dataIndex = static_cast<int>(i) + j - center;

                // SciPy-style reflection padding
                if (dataIndex < 0)
                {
                    dataIndex = -dataIndex - 1;
                }
                else if (dataIndex >= static_cast<int>(data.size()))
                {
                    dataIndex = 2 * static_cast<int>(data.size()) - 1 - dataIndex;
                }

                // Clamp to valid range (safety check)
                dataIndex = std::max(0, std::min(dataIndex, static_cast<int>(data.size()) - 1));

                value += data[dataIndex] * kernel[j];
            }
            result[i] = value;
        }

        return result;
    }

    // Helper function to compute frequency sum
    static std::vector<double> computeFrequencySum(const std::vector<std::vector<double>> &Sxx)
    {
        int n_times = Sxx.size();
        int n_freqs = Sxx[0].size();
        std::vector<double> freq_sum(n_times, 0.0);

        for (int t = 0; t < n_times; ++t)
        {
            for (int f = 0; f < n_freqs; ++f)
            {
                freq_sum[t] += Sxx[t][f];
            }
        }

        return freq_sum;
    }

    // Helper function to process frequency sum
    static std::vector<double> processFrequencySum(const std::vector<double> &freq_sum)
    {
        if (freq_sum.empty())
            return freq_sum;

        // Step 1: Subtract median
        double median = computeMedian(freq_sum);
        std::vector<double> watch_freq_sum(freq_sum.size());
        for (size_t i = 0; i < freq_sum.size(); ++i)
        {
            watch_freq_sum[i] = freq_sum[i] - median;
        }

        // Step 2: Ensure non-negative
        for (double &val : watch_freq_sum)
        {
            if (val < 0.0)
                val = 0.0;
        }

        // Step 3: Remove low values
        double percentile10 = computePercentile(watch_freq_sum, 10.0);
        for (double &val : watch_freq_sum)
        {
            if (val < percentile10)
                val = 0.0;
        }

        return watch_freq_sum;
    }

    static std::pair<std::vector<long>, std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>> resampleAccelerometer(
        const std::vector<long> &timestamps,
        const std::vector<double> &x,
        const std::vector<double> &y,
        const std::vector<double> &z,
        double targetFs)
    {
        size_t n = timestamps.size();
        if (n < 2)
        {
            return {std::vector<long>(), std::make_tuple(std::vector<double>(), std::vector<double>(), std::vector<double>())};
        }

        // Timestamps are in microseconds
        long startTimeUs = timestamps.front();
        long endTimeUs = timestamps.back();

        // Calculate number of samples needed
        double durationSec = (endTimeUs - startTimeUs) / 1000000.0;
        int numSamples = static_cast<int>(durationSec * targetFs);
        if (numSamples < 1)
            numSamples = 1;

        std::vector<long> resampledTime(numSamples);
        std::vector<double> rx(numSamples), ry(numSamples), rz(numSamples);

        // Generate new time points in microseconds
        double intervalUs = 1000000.0 / targetFs; // microseconds per sample
        for (int i = 0; i < numSamples; ++i)
        {
            if (numSamples == 1)
            {
                resampledTime[i] = startTimeUs;
            }
            else
            {
                // Generate evenly spaced time points at target frequency
                resampledTime[i] = startTimeUs + static_cast<long>(i * intervalUs);
            }
        }

        size_t origIdx = 0;

        for (int i = 0; i < numSamples; ++i)
        {
            long t = resampledTime[i];

            // Handle extrapolation cases first
            if (t <= timestamps[0])
            {
                // Left extrapolation: use first value
                rx[i] = x[0];
                ry[i] = y[0];
                rz[i] = z[0];
                continue;
            }
            else if (t >= timestamps[n - 1])
            {
                // Right extrapolation: use last value
                rx[i] = x[n - 1];
                ry[i] = y[n - 1];
                rz[i] = z[n - 1];
                continue;
            }

            // Advance origIdx to the correct interval (optimized single-pass)
            // Since resampledTime is sorted, we can start from where we left off
            while (origIdx < n - 1 && timestamps[origIdx + 1] < t)
            {
                origIdx++;
            }

            // Ensure we're in a valid interval
            if (origIdx >= n - 1)
            {
                origIdx = n - 2;
            }

            // Linear interpolation
            long t1 = timestamps[origIdx];
            long t2 = timestamps[origIdx + 1];
            double alpha = (t2 > t1) ? static_cast<double>(t - t1) / (t2 - t1) : 0.0;

            rx[i] = x[origIdx] + alpha * (x[origIdx + 1] - x[origIdx]);
            ry[i] = y[origIdx] + alpha * (y[origIdx + 1] - y[origIdx]);
            rz[i] = z[origIdx] + alpha * (z[origIdx + 1] - z[origIdx]);
        }

        return {resampledTime, std::make_tuple(rx, ry, rz)};
    }

    // ── Cubic-spline helpers ──────────────────────────────────────────
    // Solve for natural cubic spline coefficients given knot values y
    // at (not-necessarily-uniform) knot positions t.
    // Returns vectors b, c, d such that on interval [t_i, t_i+1]:
    //   S(x) = y_i + b_i*(x-t_i) + c_i*(x-t_i)^2 + d_i*(x-t_i)^3
    struct CubicSplineCoeffs
    {
        std::vector<double> b, c, d;
    };

    static CubicSplineCoeffs computeNaturalCubicSplineCoeffs(
        const std::vector<double> &t,
        const std::vector<double> &y)
    {
        size_t n = t.size();
        // n >= 2 guaranteed by caller

        std::vector<double> h(n - 1);
        for (size_t i = 0; i < n - 1; ++i)
            h[i] = t[i + 1] - t[i];

        // Solve tridiagonal system for c (natural boundary: c[0]=c[n-1]=0)
        std::vector<double> alpha(n, 0.0);
        for (size_t i = 1; i < n - 1; ++i)
        {
            alpha[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
        }

        std::vector<double> c(n, 0.0);
        std::vector<double> l(n, 1.0), mu(n, 0.0), z(n, 0.0);

        for (size_t i = 1; i < n - 1; ++i)
        {
            l[i] = 2.0 * (t[i + 1] - t[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        // Back-substitution
        for (int i = static_cast<int>(n) - 2; i >= 0; --i)
        {
            c[i] = z[i] - mu[i] * c[i + 1];
        }

        std::vector<double> b(n - 1), d(n - 1);
        for (size_t i = 0; i < n - 1; ++i)
        {
            d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
            b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
        }

        // Trim c to n-1 entries to match b, d
        c.resize(n - 1);

        return {b, c, d};
    }

    // Evaluate cubic spline at a set of sorted query points
    static std::vector<double> evaluateCubicSpline(
        const std::vector<double> &t,          // knot positions (sorted)
        const std::vector<double> &y,          // knot values
        const CubicSplineCoeffs &coeffs,
        const std::vector<double> &queries)    // query positions (sorted)
    {
        size_t n = t.size();
        size_t m = queries.size();
        const auto &b = coeffs.b;
        const auto &c = coeffs.c;
        const auto &d = coeffs.d;

        std::vector<double> result(m);
        size_t seg = 0;

        for (size_t j = 0; j < m; ++j)
        {
            double q = queries[j];

            // Clamp to data range
            if (q <= t[0])
            {
                result[j] = y[0];
                continue;
            }
            if (q >= t[n - 1])
            {
                result[j] = y[n - 1];
                continue;
            }

            // Advance segment pointer (queries are sorted)
            while (seg < n - 2 && t[seg + 1] < q)
                seg++;

            double dx = q - t[seg];
            result[j] = y[seg] + dx * (b[seg] + dx * (c[seg] + dx * d[seg]));
        }

        return result;
    }

    // ── Cubic-spline resampling ───────────────────────────────────────
    static std::pair<std::vector<long>, std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>
    resampleAccelerometerCubic(
        const std::vector<long> &timestamps,
        const std::vector<double> &x,
        const std::vector<double> &y,
        const std::vector<double> &z,
        double targetFs)
    {
        size_t n = timestamps.size();
        if (n < 2)
        {
            return {std::vector<long>(), std::make_tuple(std::vector<double>(), std::vector<double>(), std::vector<double>())};
        }

        long startTimeUs = timestamps.front();
        long endTimeUs = timestamps.back();
        double durationSec = (endTimeUs - startTimeUs) / 1000000.0;
        int numSamples = static_cast<int>(durationSec * targetFs);
        if (numSamples < 1)
            numSamples = 1;

        // Build uniform output time grid (microseconds)
        double intervalUs = 1000000.0 / targetFs;
        std::vector<long> resampledTime(numSamples);
        std::vector<double> queryT(numSamples);
        for (int i = 0; i < numSamples; ++i)
        {
            resampledTime[i] = startTimeUs + static_cast<long>(i * intervalUs);
            queryT[i] = static_cast<double>(resampledTime[i]);
        }

        // Convert timestamps to double for spline computation
        std::vector<double> tDouble(n);
        for (size_t i = 0; i < n; ++i)
            tDouble[i] = static_cast<double>(timestamps[i]);

        // Compute spline coefficients for each axis
        auto coeffsX = computeNaturalCubicSplineCoeffs(tDouble, x);
        auto coeffsY = computeNaturalCubicSplineCoeffs(tDouble, y);
        auto coeffsZ = computeNaturalCubicSplineCoeffs(tDouble, z);

        // Evaluate
        auto rx = evaluateCubicSpline(tDouble, x, coeffsX, queryT);
        auto ry = evaluateCubicSpline(tDouble, y, coeffsY, queryT);
        auto rz = evaluateCubicSpline(tDouble, z, coeffsZ, queryT);

        return {resampledTime, std::make_tuple(rx, ry, rz)};
    }

    static std::pair<std::vector<long>, std::vector<double>> computeJerk(
        const std::vector<long> &timestamps,
        const std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> &accelerometerData)
    {
        const auto &[x, y, z] = accelerometerData;
        size_t numSamples = x.size();

        if (numSamples < 2)
        {
            return {std::vector<long>(), std::vector<double>()};
        }

        std::vector<long> jerkTimes(numSamples);
        std::vector<double> jerkOut(numSamples);

        // First value is 0 (matching Python behavior)
        jerkTimes[0] = timestamps[0];
        jerkOut[0] = 0.0f;

        // Jerk computation (derivative of acceleration) starting from index 1
        for (size_t i = 1; i < numSamples; i++)
        {
            double dvx = (x[i] - x[i - 1]);
            double dvy = (y[i] - y[i - 1]);
            double dvz = (z[i] - z[i - 1]);
            jerkOut[i] = std::sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
            jerkTimes[i] = timestamps[i]; // Use timestamp of current sample
        }

        return {jerkTimes, jerkOut};
    }

    static std::vector<double> computeMagnitude(
        const std::vector<double> &x,
        const std::vector<double> &y,
        const std::vector<double> &z)
    {
        std::vector<double> magnitude;
        magnitude.reserve(x.size());

        for (size_t i = 0; i < x.size(); ++i)
        {
            magnitude.push_back(std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]));
        }

        return magnitude;
    }

    // Main motion features computation function (enhanced to match Python)
    static MotionFeaturesResult computeMotionFeatures(
        const std::vector<double> &jerkSignal,
        double fs,
        int windowSize = 1500,  // Default: 30 seconds * 50 Hz
        int overlap = 750,      // Default: 50% overlap
        double brMinHz = 0.15,  // 9 BPM (9/60 Hz)
        double brMaxHz = 0.417, // 25 BPM (25/60 Hz)
        double htMinHz = 0.5,   // 30 BPM (30/60 Hz)
        double htMaxHz = 2.0,   // 120 BPM (120/60 Hz)
        double std_window_minutes = 5.0,
        bool smooth_HR_spectrogram = true,
        bool smooth_BR_spectrogram = false,
        double spectrogram_smoothing_freq = 1.0,
        double spectrogram_smoothing_time = 2.0,
        double hr_max_change_per_sec = 15.0,
        double br_max_change_per_sec = 7.5,
        double time_resolution = 30.0)
    {
        MotionFeaturesResult result;

        // Reserve memory to avoid reallocations
        result.features.reserve(5);

        // 1. Compute spectrogram using FFT-based STFT
        auto spec = computeSpectrogram(jerkSignal, fs, windowSize, overlap);

        // Store the spectrogram in the result
        result.spectrogram = std::move(spec);

        // 2. Extract BR and HR peaks using enhanced method (matches Python pipeline)
        auto [BR_smooth, HR_smooth] = extractMotionPeaks(
            result.spectrogram.Sxx,
            result.spectrogram.freqs,
            brMinHz, brMaxHz, htMinHz, htMaxHz,
            spectrogram_smoothing_freq, spectrogram_smoothing_time,
            hr_max_change_per_sec, br_max_change_per_sec,
            time_resolution);

        // 3. Compute rolling std for BR and HR peaks (using BPM values, window in minutes)
        auto BR_std = rollingStd(BR_smooth, std_window_minutes, time_resolution);
        auto HR_std = rollingStd(HR_smooth, std_window_minutes, time_resolution);

        // 4. Compute frequency sum and apply post-processing
        auto freqSum = computeFrequencySum(result.spectrogram.Sxx);
        auto processedFreqSum = processFrequencySum(freqSum);

        // 5. Package into result (use move semantics to avoid copies)
        result.features.emplace_back(std::move(BR_smooth)); // BR in BPM
        result.features.emplace_back(std::move(HR_smooth)); // HR in BPM
        result.features.emplace_back(std::move(processedFreqSum));
        result.features.emplace_back(std::move(BR_std));
        result.features.emplace_back(std::move(HR_std));

        return result;
    }
};

// FFI exports for Dart
extern "C"
{
    void *resample_accelerometer(int64_t *timestamps, double *x, double *y, double *z, int length, double targetFs, int *outLength)
    {
        std::vector<long> ts(length);
        for (int i = 0; i < length; i++)
            ts[i] = timestamps[i];
        std::vector<double> ax(x, x + length);
        std::vector<double> ay(y, y + length);
        std::vector<double> az(z, z + length);

        auto result = SensorProcessor::resampleAccelerometer(ts, ax, ay, az, targetFs);
        auto &resampledTime = result.first;
        auto &[rx, ry, rz] = result.second;

        *outLength = resampledTime.size();

        // Allocate output: timestamps (int64) + x + y + z (doubles)
        // Layout: [timestamps as int64...] [x as double...] [y as double...] [z as double...]
        size_t timestampBytes = resampledTime.size() * sizeof(int64_t);
        size_t doubleBytes = resampledTime.size() * sizeof(double);
        size_t totalBytes = timestampBytes + (3 * doubleBytes);

        char *output = new char[totalBytes];
        int64_t *timestampPtr = reinterpret_cast<int64_t *>(output);
        double *xPtr = reinterpret_cast<double *>(output + timestampBytes);
        double *yPtr = reinterpret_cast<double *>(output + timestampBytes + doubleBytes);
        double *zPtr = reinterpret_cast<double *>(output + timestampBytes + (2 * doubleBytes));

        for (size_t i = 0; i < resampledTime.size(); i++)
        {
            timestampPtr[i] = resampledTime[i];
            xPtr[i] = rx[i];
            yPtr[i] = ry[i];
            zPtr[i] = rz[i];
        }

        return output;
    }

    double *compute_jerk(int64_t *timestamps, double *x, double *y, double *z, int length, int *outLength)
    {
        std::vector<long> ts(length);
        for (int i = 0; i < length; i++)
            ts[i] = timestamps[i];
        std::vector<double> ax(x, x + length);
        std::vector<double> ay(y, y + length);
        std::vector<double> az(z, z + length);

        auto accelerometerData = std::make_tuple(ax, ay, az);
        auto result = SensorProcessor::computeJerk(ts, accelerometerData);
        const auto &jerkValues = result.second;

        *outLength = jerkValues.size();
        double *output = new double[jerkValues.size()];
        std::copy(jerkValues.begin(), jerkValues.end(), output);

        return output;
    }

    void free_memory(void *ptr)
    {
        delete[] static_cast<double *>(ptr);
    }

    void free_double_memory(void *ptr)
    {
        delete[] static_cast<double *>(ptr);
    }

    void *compute_spectrogram(double *signal, int length, double fs, int nperseg, int noverlap, int *out_n_freqs, int *out_n_times)
    {
        std::vector<double> signal_vec(signal, signal + length);

        auto result = SensorProcessor::computeSpectrogram(signal_vec, fs, nperseg, noverlap);

        int n_freqs = result.freqs.size();
        int n_times = result.times.size();

        *out_n_freqs = n_freqs;
        *out_n_times = n_times;

        if (n_freqs == 0 || n_times == 0)
        {
            return nullptr;
        }

        // Allocate a single block of memory for freqs, times, and Sxx
        size_t total_size = n_freqs + n_times + (n_freqs * n_times);
        double *output = new double[total_size];

        // Copy freqs
        std::copy(result.freqs.begin(), result.freqs.end(), output);

        // Copy times
        std::copy(result.times.begin(), result.times.end(), output + n_freqs);

        // Copy Sxx (flattened)
        double *sxx_ptr = output + n_freqs + n_times;
        for (int f = 0; f < n_freqs; ++f)
        {
            std::copy(result.Sxx[f].begin(), result.Sxx[f].end(), sxx_ptr + (f * n_times));
        }

        return output;
    }
}

// Adapter functions to force emission of an externally-linkable symbol
// These call the inline/class-defined implementation above and return a
// heap-allocated MotionFeaturesResult pointer so callers from other TUs
// can use the result without pulling inline symbols into their TU.
// Note: these are plain C++ functions (not extern "C") and live in this
// translation unit so they will be present in sensor_processing.o.

// Caller must delete the returned pointer using free_motion_features_cpp
MotionFeaturesResult *compute_motion_features_cpp(
    const double *signal,
    int length,
    double fs,
    int windowSize,
    int overlap,
    double brMinHz,
    double brMaxHz,
    double htMinHz,
    double htMaxHz,
    double std_window_minutes,
    bool smooth_HR_spectrogram,
    bool smooth_BR_spectrogram,
    double spectrogram_smoothing_freq,
    double spectrogram_smoothing_time,
    double hr_max_change_per_sec,
    double br_max_change_per_sec,
    double time_resolution)
{
    std::vector<double> sig(signal, signal + length);
    MotionFeaturesResult res = SensorProcessor::computeMotionFeatures(
        sig, fs, windowSize, overlap, brMinHz, brMaxHz, htMinHz, htMaxHz,
        std_window_minutes, smooth_HR_spectrogram, smooth_BR_spectrogram,
        spectrogram_smoothing_freq, spectrogram_smoothing_time,
        hr_max_change_per_sec, br_max_change_per_sec, time_resolution);

    return new MotionFeaturesResult(std::move(res));
}

void free_motion_features_cpp(MotionFeaturesResult *ptr)
{
    delete ptr;
}

// --- pybind11 bindings (moved here so all implementations are in one TU) ---

#ifdef PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

namespace py = pybind11;

namespace py = pybind11;

// Wrapper functions to convert between NumPy arrays and std::vector
py::dict resampleAccelerometer_wrapper(
    py::array_t<int64_t> timestamps,
    py::array_t<double> x,
    py::array_t<double> y,
    py::array_t<double> z,
    double targetFs)
{
    auto ts_buf = timestamps.request();
    auto x_buf = x.request();
    auto y_buf = y.request();
    auto z_buf = z.request();

    if (ts_buf.size != x_buf.size || ts_buf.size != y_buf.size || ts_buf.size != z_buf.size)
    {
        throw std::runtime_error("Input arrays must have the same length");
    }

    std::vector<long> ts_vec(static_cast<long *>(ts_buf.ptr),
                             static_cast<long *>(ts_buf.ptr) + ts_buf.size);
    std::vector<double> x_vec(static_cast<double *>(x_buf.ptr),
                              static_cast<double *>(x_buf.ptr) + x_buf.size);
    std::vector<double> y_vec(static_cast<double *>(y_buf.ptr),
                              static_cast<double *>(y_buf.ptr) + y_buf.size);
    std::vector<double> z_vec(static_cast<double *>(z_buf.ptr),
                              static_cast<double *>(z_buf.ptr) + z_buf.size);

    auto result = SensorProcessor::resampleAccelerometer(ts_vec, x_vec, y_vec, z_vec, targetFs);
    auto &resampled_time = result.first;
    auto &[rx, ry, rz] = result.second;

    py::array_t<int64_t> out_timestamps(resampled_time.size());
    py::array_t<double> out_x(rx.size());
    py::array_t<double> out_y(ry.size());
    py::array_t<double> out_z(rz.size());

    std::copy(resampled_time.begin(), resampled_time.end(),
              static_cast<int64_t *>(out_timestamps.request().ptr));
    std::copy(rx.begin(), rx.end(), static_cast<double *>(out_x.request().ptr));
    std::copy(ry.begin(), ry.end(), static_cast<double *>(out_y.request().ptr));
    std::copy(rz.begin(), rz.end(), static_cast<double *>(out_z.request().ptr));

    py::dict result_dict;
    result_dict["timestamps"] = out_timestamps;
    result_dict["x"] = out_x;
    result_dict["y"] = out_y;
    result_dict["z"] = out_z;

    return result_dict;
}

py::dict resampleAccelerometerCubic_wrapper(
    py::array_t<int64_t> timestamps,
    py::array_t<double> x,
    py::array_t<double> y,
    py::array_t<double> z,
    double targetFs)
{
    auto ts_buf = timestamps.request();
    auto x_buf = x.request();
    auto y_buf = y.request();
    auto z_buf = z.request();

    if (ts_buf.size != x_buf.size || ts_buf.size != y_buf.size || ts_buf.size != z_buf.size)
    {
        throw std::runtime_error("Input arrays must have the same length");
    }

    std::vector<long> ts_vec(static_cast<long *>(ts_buf.ptr),
                             static_cast<long *>(ts_buf.ptr) + ts_buf.size);
    std::vector<double> x_vec(static_cast<double *>(x_buf.ptr),
                              static_cast<double *>(x_buf.ptr) + x_buf.size);
    std::vector<double> y_vec(static_cast<double *>(y_buf.ptr),
                              static_cast<double *>(y_buf.ptr) + y_buf.size);
    std::vector<double> z_vec(static_cast<double *>(z_buf.ptr),
                              static_cast<double *>(z_buf.ptr) + z_buf.size);

    auto result = SensorProcessor::resampleAccelerometerCubic(ts_vec, x_vec, y_vec, z_vec, targetFs);
    auto &resampled_time = result.first;
    auto &[rx, ry, rz] = result.second;

    py::array_t<int64_t> out_timestamps(resampled_time.size());
    py::array_t<double> out_x(rx.size());
    py::array_t<double> out_y(ry.size());
    py::array_t<double> out_z(rz.size());

    std::copy(resampled_time.begin(), resampled_time.end(),
              static_cast<int64_t *>(out_timestamps.request().ptr));
    std::copy(rx.begin(), rx.end(), static_cast<double *>(out_x.request().ptr));
    std::copy(ry.begin(), ry.end(), static_cast<double *>(out_y.request().ptr));
    std::copy(rz.begin(), rz.end(), static_cast<double *>(out_z.request().ptr));

    py::dict result_dict;
    result_dict["timestamps"] = out_timestamps;
    result_dict["x"] = out_x;
    result_dict["y"] = out_y;
    result_dict["z"] = out_z;

    return result_dict;
}

py::dict computeJerk_wrapper(
    py::array_t<int64_t> timestamps,
    py::array_t<double> x,
    py::array_t<double> y,
    py::array_t<double> z)
{
    auto ts_buf = timestamps.request();
    auto x_buf = x.request();
    auto y_buf = y.request();
    auto z_buf = z.request();

    if (ts_buf.size != x_buf.size || ts_buf.size != y_buf.size || ts_buf.size != z_buf.size)
    {
        throw std::runtime_error("Input arrays must have the same length");
    }

    std::vector<long> ts_vec(static_cast<long *>(ts_buf.ptr),
                             static_cast<long *>(ts_buf.ptr) + ts_buf.size);
    std::vector<double> x_vec(static_cast<double *>(x_buf.ptr),
                              static_cast<double *>(x_buf.ptr) + x_buf.size);
    std::vector<double> y_vec(static_cast<double *>(y_buf.ptr),
                              static_cast<double *>(y_buf.ptr) + y_buf.size);
    std::vector<double> z_vec(static_cast<double *>(z_buf.ptr),
                              static_cast<double *>(z_buf.ptr) + z_buf.size);

    auto accelerometerData = std::make_tuple(x_vec, y_vec, z_vec);
    auto result = SensorProcessor::computeJerk(ts_vec, accelerometerData);

    py::array_t<int64_t> out_timestamps(result.first.size());
    py::array_t<double> out_jerk(result.second.size());

    std::copy(result.first.begin(), result.first.end(),
              static_cast<int64_t *>(out_timestamps.request().ptr));
    std::copy(result.second.begin(), result.second.end(),
              static_cast<double *>(out_jerk.request().ptr));

    py::dict result_dict;
    result_dict["timestamps"] = out_timestamps;
    result_dict["jerk"] = out_jerk;

    return result_dict;
}

py::array_t<double> computeMagnitude_wrapper(
    py::array_t<double> x,
    py::array_t<double> y,
    py::array_t<double> z)
{
    auto x_buf = x.request();
    auto y_buf = y.request();
    auto z_buf = z.request();

    if (x_buf.size != y_buf.size || x_buf.size != z_buf.size)
    {
        throw std::runtime_error("Input arrays must have the same length");
    }

    std::vector<double> x_vec(static_cast<double *>(x_buf.ptr),
                              static_cast<double *>(x_buf.ptr) + x_buf.size);
    std::vector<double> y_vec(static_cast<double *>(y_buf.ptr),
                              static_cast<double *>(y_buf.ptr) + y_buf.size);
    std::vector<double> z_vec(static_cast<double *>(z_buf.ptr),
                              static_cast<double *>(z_buf.ptr) + z_buf.size);

    auto magnitude = SensorProcessor::computeMagnitude(x_vec, y_vec, z_vec);

    py::array_t<double> out_magnitude(magnitude.size());
    std::copy(magnitude.begin(), magnitude.end(),
              static_cast<double *>(out_magnitude.request().ptr));

    return out_magnitude;
}

py::dict computeSpectrogram_wrapper(
    py::array_t<double> signal,
    double fs,
    int nperseg,
    int noverlap)
{
    auto sig_buf = signal.request();
    std::vector<double> signal_vec(static_cast<double *>(sig_buf.ptr),
                                   static_cast<double *>(sig_buf.ptr) + sig_buf.size);

    auto result = SensorProcessor::computeSpectrogram(signal_vec, fs, nperseg, noverlap);

    // Convert frequencies to NumPy array
    py::array_t<double> freqs(result.freqs.size());
    std::copy(result.freqs.begin(), result.freqs.end(),
              static_cast<double *>(freqs.request().ptr));

    // Convert times to NumPy array
    py::array_t<double> times(result.times.size());
    std::copy(result.times.begin(), result.times.end(),
              static_cast<double *>(times.request().ptr));

    // Convert Sxx to 2D NumPy array (freqs x times)
    size_t n_freqs = result.Sxx.size();
    size_t n_times = result.Sxx[0].size();
    py::array_t<double> Sxx({n_freqs, n_times});
    auto Sxx_buf = Sxx.request();
    double *Sxx_ptr = static_cast<double *>(Sxx_buf.ptr);

    for (size_t f = 0; f < n_freqs; ++f)
    {
        for (size_t t = 0; t < n_times; ++t)
        {
            Sxx_ptr[f * n_times + t] = result.Sxx[f][t];
        }
    }

    py::dict result_dict;
    result_dict["freqs"] = freqs;
    result_dict["times"] = times;
    result_dict["Sxx"] = Sxx;

    return result_dict;
}

py::array_t<double> computeShortTimeFT_wrapper(
    py::array_t<double> signal,
    double fs,
    int nperseg,
    int noverlap)
{
    auto sig_buf = signal.request();
    std::vector<double> signal_vec(static_cast<double *>(sig_buf.ptr),
                                   static_cast<double *>(sig_buf.ptr) + sig_buf.size);

    auto result = SensorProcessor::computeShortTimeFT(signal_vec, fs, nperseg, noverlap);

    // Result shape: (n_times, n_frequencies, 2)
    size_t n_times = result.size();
    size_t n_freqs = result[0].size();

    // Create 3D NumPy array
    py::array_t<double> output({n_times, n_freqs, (size_t)2});
    auto output_buf = output.request();
    double *output_ptr = static_cast<double *>(output_buf.ptr);

    // Copy data: output[t, f, 0] = real, output[t, f, 1] = imag
    for (size_t t = 0; t < n_times; ++t)
    {
        for (size_t f = 0; f < n_freqs; ++f)
        {
            size_t idx = (t * n_freqs + f) * 2;
            output_ptr[idx] = result[t][f][0];     // real
            output_ptr[idx + 1] = result[t][f][1]; // imag
        }
    }

    return output;
}

#ifdef USE_FINUFFT
py::dict computeSpectrogramNUFFT_wrapper(
    py::array_t<double> timestamps,
    py::array_t<double> signal,
    double secperseg,
    double secoverlap,
    double target_fs = 0.0)
{
    auto ts_buf = timestamps.request();
    auto sig_buf = signal.request();
    std::vector<double> timestamps_vec(static_cast<double *>(ts_buf.ptr),
                                       static_cast<double *>(ts_buf.ptr) + ts_buf.size);
    std::vector<double> signal_vec(static_cast<double *>(sig_buf.ptr),
                                   static_cast<double *>(sig_buf.ptr) + sig_buf.size);

    auto result = SensorProcessor::computeSpectrogramNUFFT(timestamps_vec, signal_vec, secperseg, secoverlap, target_fs);

    // Convert frequencies to NumPy array
    py::array_t<double> freqs(result.freqs.size());
    std::copy(result.freqs.begin(), result.freqs.end(),
              static_cast<double *>(freqs.request().ptr));

    // Convert times to NumPy array
    py::array_t<double> times(result.times.size());
    std::copy(result.times.begin(), result.times.end(),
              static_cast<double *>(times.request().ptr));

    // Convert Sxx to 2D NumPy array (times x freqs)
    size_t n_times = result.Sxx.size();
    size_t n_freqs = (n_times > 0) ? result.Sxx[0].size() : 0;
    py::array_t<double> Sxx({n_times, n_freqs});

    if (n_times > 0 && n_freqs > 0)
    {
        auto Sxx_buf = Sxx.request();
        double *Sxx_ptr = static_cast<double *>(Sxx_buf.ptr);
        for (size_t t = 0; t < n_times; ++t)
        {
            for (size_t f = 0; f < n_freqs; ++f)
            {
                Sxx_ptr[t * n_freqs + f] = result.Sxx[t][f];
            }
        }
    }

    py::dict result_dict;
    result_dict["freqs"] = freqs;
    result_dict["times"] = times;
    result_dict["Sxx"] = Sxx;

    return result_dict;
}
#endif

py::dict computeMotionFeatures_wrapper(
    py::array_t<double> jerkSignal,
    double fs,
    int windowSize = 1500,
    int overlap = 750,
    double brMinHz = 0.15,
    double brMaxHz = 0.417,
    double htMinHz = 0.5,
    double htMaxHz = 2.0,
    double std_window_minutes = 5.0,
    bool smooth_HR_spectrogram = true,
    bool smooth_BR_spectrogram = false,
    double spectrogram_smoothing_freq = 1.0,
    double spectrogram_smoothing_time = 2.0,
    double hr_max_change_per_sec = 15.0,
    double br_max_change_per_sec = 7.5,
    double time_resolution = 30.0)
{
    auto sig_buf = jerkSignal.request();
    std::vector<double> signal_vec(static_cast<double *>(sig_buf.ptr),
                                   static_cast<double *>(sig_buf.ptr) + sig_buf.size);

    auto result = SensorProcessor::computeMotionFeatures(
        signal_vec, fs, windowSize, overlap,
        brMinHz, brMaxHz, htMinHz, htMaxHz,
        std_window_minutes, smooth_HR_spectrogram, smooth_BR_spectrogram,
        spectrogram_smoothing_freq, spectrogram_smoothing_time,
        hr_max_change_per_sec, br_max_change_per_sec, time_resolution);

    // Convert features to NumPy arrays
    py::list features_list;
    for (const auto &feature : result.features)
    {
        py::array_t<double> feature_array(feature.size());
        std::copy(feature.begin(), feature.end(),
                  static_cast<double *>(feature_array.request().ptr));
        features_list.append(feature_array);
    }

    // Convert spectrogram
    py::dict spectrogram;

    py::array_t<double> freqs(result.spectrogram.freqs.size());
    std::copy(result.spectrogram.freqs.begin(), result.spectrogram.freqs.end(),
              static_cast<double *>(freqs.request().ptr));

    py::array_t<double> times(result.spectrogram.times.size());
    std::copy(result.spectrogram.times.begin(), result.spectrogram.times.end(),
              static_cast<double *>(times.request().ptr));

    size_t n_freqs = result.spectrogram.Sxx.size();
    size_t n_times = result.spectrogram.Sxx[0].size();
    py::array_t<double> Sxx({n_freqs, n_times});
    auto Sxx_buf = Sxx.request();
    double *Sxx_ptr = static_cast<double *>(Sxx_buf.ptr);

    for (size_t f = 0; f < n_freqs; ++f)
    {
        for (size_t t = 0; t < n_times; ++t)
        {
            Sxx_ptr[f * n_times + t] = result.spectrogram.Sxx[f][t];
        }
    }

    spectrogram["freqs"] = freqs;
    spectrogram["times"] = times;
    spectrogram["Sxx"] = Sxx;

    py::dict result_dict;
    result_dict["BR"] = features_list[0];
    result_dict["HR"] = features_list[1];
    result_dict["freqSum"] = features_list[2];
    result_dict["BR_std"] = features_list[3];
    result_dict["HR_std"] = features_list[4];
    result_dict["spectrogram"] = spectrogram;

    return result_dict;
}

py::array_t<double> hannWindow_wrapper(int N)
{
    auto window = SensorProcessor::hannWindow(N);
    py::array_t<double> out_window(window.size());
    std::copy(window.begin(), window.end(),
              static_cast<double *>(out_window.request().ptr));
    return out_window;
}

py::array_t<double> gaussianFilter1D_wrapper(
    py::array_t<double> data,
    double sigma,
    double truncate = 4.0)
{
    auto data_buf = data.request();
    std::vector<double> data_vec(static_cast<double *>(data_buf.ptr),
                                 static_cast<double *>(data_buf.ptr) + data_buf.size);

    auto filtered = SensorProcessor::gaussianFilter1D(data_vec, sigma, truncate);

    py::array_t<double> out_filtered(filtered.size());
    std::copy(filtered.begin(), filtered.end(),
              static_cast<double *>(out_filtered.request().ptr));

    return out_filtered;
}

py::array_t<double> findSpectrogramPeaks_wrapper(
    py::array_t<double> Sxx,
    py::array_t<double> frequencies,
    double prominence_threshold,
    double scaling_factor = 60.0)
{
    auto Sxx_buf = Sxx.request();
    auto freq_buf = frequencies.request();

    if (Sxx_buf.ndim != 2)
    {
        throw std::runtime_error("Sxx must be a 2D array");
    }

    size_t n_times = Sxx_buf.shape[0];
    size_t n_freqs = Sxx_buf.shape[1];

    if (freq_buf.size != n_freqs)
    {
        throw std::runtime_error("Frequencies array size must match Sxx frequency dimension");
    }

    // Convert 2D NumPy array to std::vector<std::vector<double>>
    std::vector<std::vector<double>> Sxx_vec(n_times, std::vector<double>(n_freqs));
    double *Sxx_ptr = static_cast<double *>(Sxx_buf.ptr);

    for (size_t t = 0; t < n_times; ++t)
    {
        for (size_t f = 0; f < n_freqs; ++f)
        {
            Sxx_vec[t][f] = Sxx_ptr[t * n_freqs + f];
        }
    }

    // Convert frequencies
    std::vector<double> freq_vec(static_cast<double *>(freq_buf.ptr),
                                 static_cast<double *>(freq_buf.ptr) + freq_buf.size);

    auto peaks = SensorProcessor::findSpectrogramPeaks(Sxx_vec, freq_vec, prominence_threshold, scaling_factor);

    py::array_t<double> out_peaks(peaks.size());
    std::copy(peaks.begin(), peaks.end(),
              static_cast<double *>(out_peaks.request().ptr));

    return out_peaks;
}

py::array_t<int> findPeaks_wrapper(
    py::array_t<double> signal,
    double prominence_threshold)
{
    auto sig_buf = signal.request();
    std::vector<double> signal_vec(static_cast<double *>(sig_buf.ptr),
                                   static_cast<double *>(sig_buf.ptr) + sig_buf.size);

    auto peaks = SensorProcessor::findPeaks(signal_vec, prominence_threshold);

    py::array_t<int> out_peaks(peaks.size());
    std::copy(peaks.begin(), peaks.end(),
              static_cast<int *>(out_peaks.request().ptr));

    return out_peaks;
}

py::array_t<double> rollingStd_wrapper(
    py::array_t<double> data,
    double window_minutes,
    double seconds_per_window = 30.0)
{
    auto data_buf = data.request();
    std::vector<double> data_vec(static_cast<double *>(data_buf.ptr),
                                 static_cast<double *>(data_buf.ptr) + data_buf.size);

    auto std_vals = SensorProcessor::rollingStd(data_vec, window_minutes, seconds_per_window);

    py::array_t<double> out_std(std_vals.size());
    std::copy(std_vals.begin(), std_vals.end(),
              static_cast<double *>(out_std.request().ptr));

    return out_std;
}

py::array_t<double> smoothSpectrogramPeaks_wrapper(
    py::array_t<double> peaks,
    double sampling_rate,
    double max_change_per_sec = 10.0,
    double filter_sigma = 2.0)
{
    auto peaks_buf = peaks.request();
    std::vector<double> peaks_vec(static_cast<double *>(peaks_buf.ptr),
                                  static_cast<double *>(peaks_buf.ptr) + peaks_buf.size);

    auto smoothed = SensorProcessor::smoothSpectrogramPeaks(peaks_vec, sampling_rate, max_change_per_sec, filter_sigma);

    py::array_t<double> out_smoothed(smoothed.size());
    std::copy(smoothed.begin(), smoothed.end(),
              static_cast<double *>(out_smoothed.request().ptr));

    return out_smoothed;
}

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Sensor processing module with FFT-based signal analysis";

    // Main processing functions
    m.def("resample_accelerometer", &resampleAccelerometer_wrapper,
          "Resample accelerometer data to target sampling frequency",
          py::arg("timestamps"),
          py::arg("x"),
          py::arg("y"),
          py::arg("z"),
          py::arg("targetFs"));

    m.def("resample_accelerometer_cubic", &resampleAccelerometerCubic_wrapper,
          "Resample accelerometer data using natural cubic spline interpolation",
          py::arg("timestamps"),
          py::arg("x"),
          py::arg("y"),
          py::arg("z"),
          py::arg("targetFs"));

    m.def("compute_jerk", &computeJerk_wrapper,
          "Compute jerk (derivative of acceleration) from accelerometer data",
          py::arg("timestamps"),
          py::arg("x"),
          py::arg("y"),
          py::arg("z"));

    m.def("compute_magnitude", &computeMagnitude_wrapper,
          "Compute magnitude from x, y, z components",
          py::arg("x"),
          py::arg("y"),
          py::arg("z"));

    m.def("compute_spectrogram", &computeSpectrogram_wrapper,
          "Compute spectrogram using FFT-based STFT",
          py::arg("signal"),
          py::arg("fs"),
          py::arg("nperseg"),
          py::arg("noverlap"));

#ifdef USE_FINUFFT
    m.def("compute_spectrogram_nufft", &computeSpectrogramNUFFT_wrapper,
          "Compute spectrogram from non-uniformly sampled data using finufft",
          py::arg("timestamps"),
          py::arg("signal"),
          py::arg("secperseg"),
          py::arg("secoverlap"),
          py::arg("target_fs") = 0.0);
#endif

    m.def("compute_short_time_ft", &computeShortTimeFT_wrapper,
          "Compute Short-Time Fourier Transform returning complex values (n_times, n_frequencies, 2)",
          py::arg("signal"),
          py::arg("fs"),
          py::arg("nperseg"),
          py::arg("noverlap"));

    m.def("compute_motion_features", &computeMotionFeatures_wrapper,
          "Extract breathing rate, heart rate, and motion features from jerk signal",
          py::arg("jerkSignal"),
          py::arg("fs"),
          py::arg("windowSize") = 1500,
          py::arg("overlap") = 750,
          py::arg("brMinHz") = 0.15,
          py::arg("brMaxHz") = 0.417,
          py::arg("htMinHz") = 0.5,
          py::arg("htMaxHz") = 2.0,
          py::arg("std_window_minutes") = 5.0,
          py::arg("smooth_HR_spectrogram") = true,
          py::arg("smooth_BR_spectrogram") = false,
          py::arg("spectrogram_smoothing_freq") = 1.0,
          py::arg("spectrogram_smoothing_time") = 2.0,
          py::arg("hr_max_change_per_sec") = 15.0,
          py::arg("br_max_change_per_sec") = 7.5,
          py::arg("time_resolution") = 30.0);

    // Utility functions
    m.def("hann_window", &hannWindow_wrapper,
          "Generate Hann window of size N",
          py::arg("N"));

    m.def("gaussian_filter_1d", &gaussianFilter1D_wrapper,
          "Apply 1D Gaussian filter to data",
          py::arg("data"),
          py::arg("sigma"),
          py::arg("truncate") = 4.0);

    m.def("find_peaks", &findPeaks_wrapper,
          "Find peaks in signal with prominence threshold",
          py::arg("signal"),
          py::arg("prominence_threshold"));

    m.def("find_spectrogram_peaks", &findSpectrogramPeaks_wrapper,
          "Find peaks in spectrogram with prominence threshold and scaling",
          py::arg("Sxx"),
          py::arg("frequencies"),
          py::arg("prominence_threshold"),
          py::arg("scaling_factor") = 60.0);

    m.def("rolling_std", &rollingStd_wrapper,
          "Compute rolling standard deviation",
          py::arg("data"),
          py::arg("window_minutes"),
          py::arg("seconds_per_window") = 30.0);

    m.def("smooth_spectrogram_peaks", &smoothSpectrogramPeaks_wrapper,
          "Smooth spectrogram peaks using Gaussian filter and rate constraints",
          py::arg("peaks"),
          py::arg("sampling_rate"),
          py::arg("max_change_per_sec") = 10.0,
          py::arg("filter_sigma") = 2.0);

    m.def("next_power_of_2", [](int n)
          { return SensorProcessor::nextPowerOf2(n); }, "Find next power of 2 greater than or equal to n", py::arg("n"));

    m.def("compute_median", [](py::array_t<double> data)
          {
              auto buf = data.request();
              std::vector<double> vec(static_cast<double*>(buf.ptr), 
                                      static_cast<double*>(buf.ptr) + buf.size);
              return SensorProcessor::computeMedian(vec); }, "Compute median of data", py::arg("data"));

    m.def("compute_percentile", [](py::array_t<double> data, double percentile)
          {
              auto buf = data.request();
              std::vector<double> vec(static_cast<double*>(buf.ptr), 
                                      static_cast<double*>(buf.ptr) + buf.size);
              return SensorProcessor::computePercentile(vec, percentile); }, "Compute percentile of data", py::arg("data"), py::arg("percentile"));
}

// Adapter for computePercentile: returns the percentile computed by
// SensorProcessor::computePercentile for an array of doubles.
double compute_percentile_cpp(const double *data, int length, double percentile)
{
    std::vector<double> vec(data, data + length);
    return SensorProcessor::computePercentile(vec, percentile);
}
#endif
