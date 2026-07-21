// ============================================================================
// Shared log-Mel frontend — implementation. See src/voxtral-mel.h.
//
// The DFT / filterbank / normalization arithmetic here is a verbatim move of the
// historical in-lined compute_mel_spectrogram() body, so batch output is
// bit-for-bit unchanged. The incremental frontend calls the exact same kernel,
// which is what guarantees batch-vs-incremental numerical parity.
// ============================================================================

#include "voxtral-mel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

// ----------------------------------------------------------------------------
// Mel filterbank (Slaney-style) + Hann window + reflect map + DFT plan.
// ----------------------------------------------------------------------------

static float voxtral_hertz_to_mel(float freq_hz) {
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel   = 15.0f;
    const float logstep       = 27.0f / logf(6.4f);
    float mels = 3.0f * freq_hz / 200.0f;
    if (freq_hz >= min_log_hertz) {
        mels = min_log_mel + logf(freq_hz / min_log_hertz) * logstep;
    }
    return mels;
}

static float voxtral_mel_to_hertz(float mels) {
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel   = 15.0f;
    const float logstep       = logf(6.4f) / 27.0f;
    float freq = 200.0f * mels / 3.0f;
    if (mels >= min_log_mel) {
        freq = min_log_hertz * expf(logstep * (mels - min_log_mel));
    }
    return freq;
}

void voxtral_mel_build_slaney_filters(std::vector<float> & filters) {
    constexpr int32_t n_freq = VOXTRAL_MEL_N_FREQ;
    constexpr int32_t n_mel  = VOXTRAL_MEL_N_MEL;

    filters.resize((size_t) n_freq * n_mel, 0.0f);

    std::vector<float> fft_freqs(n_freq);
    for (int32_t i = 0; i < n_freq; i++) {
        fft_freqs[i] = (float)(VOXTRAL_SAMPLE_RATE / 2) * (float)i / (float)(n_freq - 1);
    }

    const float mel_min = voxtral_hertz_to_mel(0.0f);
    const float mel_max = voxtral_hertz_to_mel(8000.0f);

    std::vector<float> mel_pts(n_mel + 2);
    for (int32_t i = 0; i < n_mel + 2; i++) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * (float)i / (float)(n_mel + 1);
    }

    std::vector<float> filter_freqs(n_mel + 2);
    for (int32_t i = 0; i < n_mel + 2; i++) {
        filter_freqs[i] = voxtral_mel_to_hertz(mel_pts[i]);
    }

    for (int32_t m = 0; m < n_mel; m++) {
        const float f_left   = filter_freqs[m];
        const float f_center = filter_freqs[m + 1];
        const float f_right  = filter_freqs[m + 2];
        const float enorm    = 2.0f / (f_right - f_left);

        for (int32_t k = 0; k < n_freq; k++) {
            const float f = fft_freqs[k];
            float down_slope = -(f - f_center) / (f_center - f_left);
            float up_slope   =  (f_right - f)  / (f_right - f_center);

            float val = std::max(0.0f, std::min(down_slope, up_slope));
            filters[(size_t) k * n_mel + m] = val * enorm;
        }
    }
}

void voxtral_mel_build_hann_window(std::vector<float> & hann) {
    hann.resize(VOXTRAL_MEL_N_FFT);
    for (int32_t i = 0; i < VOXTRAL_MEL_N_FFT; i++) {
        // Match torch.hann_window(W, periodic=True).
        hann[i] = 0.5f * (1.0f - cosf(2.0f * VOXTRAL_MEL_PI * (float) i / (float) (VOXTRAL_MEL_N_FFT)));
    }
}

int32_t voxtral_reflect_index(int32_t idx, int32_t len) {
    if (len <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= len) {
        if (idx < 0) {
            idx = -idx;
        } else {
            idx = 2 * len - 2 - idx;
        }
    }
    return idx;
}

const voxtral_stft_plan & voxtral_get_stft_plan() {
    static voxtral_stft_plan plan = []() {
        voxtral_stft_plan p;
        p.n_fft  = VOXTRAL_MEL_N_FFT;
        p.n_bins = VOXTRAL_MEL_N_FREQ;
        p.cos_table.resize((size_t) p.n_bins * (size_t) p.n_fft);
        p.sin_table.resize((size_t) p.n_bins * (size_t) p.n_fft);
        for (int32_t k = 0; k < p.n_bins; ++k) {
            for (int32_t n = 0; n < p.n_fft; ++n) {
                const float angle = 2.0f * VOXTRAL_MEL_PI * (float) k * (float) n / (float) p.n_fft;
                const size_t idx = (size_t) k * (size_t) p.n_fft + (size_t) n;
                p.cos_table[idx] = cosf(angle);
                p.sin_table[idx] = sinf(angle);
            }
        }
        return p;
    }();
    return plan;
}

// ----------------------------------------------------------------------------
// Per-frame kernel — the single copy of the DFT + filterbank + normalization.
// ----------------------------------------------------------------------------
void voxtral_mel_frame_from_window(
    const float * window,
    const float * hann_window,
    const float * mel_filters,
    float       * out_frame)
{
    constexpr int32_t n_freq = VOXTRAL_MEL_N_FREQ;
    constexpr int32_t n_mel  = VOXTRAL_MEL_N_MEL;
    constexpr int32_t n_fft  = VOXTRAL_MEL_N_FFT;

    const voxtral_stft_plan & plan = voxtral_get_stft_plan();

    float windowed[n_fft];
    for (int32_t i = 0; i < n_fft; ++i) {
        windowed[i] = window[i] * hann_window[i];
    }

    float power[n_freq];
    for (int32_t k = 0; k < n_freq; ++k) {
        const float * cos_row = plan.cos_table.data() + (size_t) k * (size_t) n_fft;
        const float * sin_row = plan.sin_table.data() + (size_t) k * (size_t) n_fft;
        float re = 0.0f;
        float im = 0.0f;

        int32_t i = 0;
        for (; i + 3 < n_fft; i += 4) {
            const float x0 = windowed[i + 0];
            const float x1 = windowed[i + 1];
            const float x2 = windowed[i + 2];
            const float x3 = windowed[i + 3];

            re += x0 * cos_row[i + 0] + x1 * cos_row[i + 1] + x2 * cos_row[i + 2] + x3 * cos_row[i + 3];
            im -= x0 * sin_row[i + 0] + x1 * sin_row[i + 1] + x2 * sin_row[i + 2] + x3 * sin_row[i + 3];
        }
        for (; i < n_fft; ++i) {
            const float x = windowed[i];
            re += x * cos_row[i];
            im -= x * sin_row[i];
        }

        power[k] = re * re + im * im;
    }

    float mel_accum[n_mel];
    for (int32_t m = 0; m < n_mel; ++m) {
        mel_accum[m] = 0.0f;
    }
    for (int32_t k = 0; k < n_freq; ++k) {
        const float * w  = mel_filters + (size_t) k * (size_t) n_mel;
        const float   pk = power[k];
        for (int32_t m = 0; m < n_mel; ++m) {
            mel_accum[m] += w[m] * pk;
        }
    }

    for (int32_t m = 0; m < n_mel; ++m) {
        float val = mel_accum[m];
        val = std::max(val, 1e-10f);
        val = log10f(val);
        val = std::max(val, VOXTRAL_GLOBAL_LOG_MEL_MAX - 8.0f);
        val = (val + 4.0f) / 4.0f;
        out_frame[m] = val;
    }
}

// ----------------------------------------------------------------------------
// Batch spectrogram over the shared kernel (bit-for-bit legacy behavior).
// ----------------------------------------------------------------------------
int32_t voxtral_mel_batch_frame_count(int32_t n_samples) {
    if (n_samples <= 0) {
        return 0;
    }
    const int32_t n_stft_frames = n_samples / VOXTRAL_MEL_HOP + 1;
    const int32_t n_frames = n_stft_frames - 1;   // drop last frame (Python [:-1])
    return n_frames > 0 ? n_frames : 0;
}

void voxtral_mel_compute_batch(
    const float * audio,
    int32_t       n_samples,
    const float * mel_filters,
    const float * hann_window,
    float       * mel_out,
    int32_t     * out_n_frames)
{
    const int32_t n_frames = voxtral_mel_batch_frame_count(n_samples);
    *out_n_frames = n_frames;

    constexpr int32_t n_mel = VOXTRAL_MEL_N_MEL;
    constexpr int32_t n_fft = VOXTRAL_MEL_N_FFT;
    constexpr int32_t hop   = VOXTRAL_MEL_HOP;
    constexpr int32_t pad   = VOXTRAL_MEL_PAD;

    if (n_frames <= 0) {
        return;
    }

    // Reflect padding once (equivalent to center=True, pad_mode="reflect"), so
    // each frame reads a contiguous window — preserves the batch access pattern
    // and performance exactly.
    const int32_t centered_len = n_samples + 2 * pad;
    std::vector<float> centered((size_t) centered_len, 0.0f);
    if (n_samples > 0) {
        for (int32_t i = 0; i < centered_len; ++i) {
            const int32_t src = i - pad;
            const int32_t ridx = (src >= 0 && src < n_samples) ? src : voxtral_reflect_index(src, n_samples);
            centered[(size_t) i] = audio[(size_t) ridx];
        }
    }

    float frame_buf[n_mel];
    for (int32_t frame = 0; frame < n_frames; ++frame) {
        const int32_t start = frame * hop;
        voxtral_mel_frame_from_window(centered.data() + (size_t) start, hann_window, mel_filters, frame_buf);
        for (int32_t m = 0; m < n_mel; ++m) {
            mel_out[(size_t) m * (size_t) n_frames + (size_t) frame] = frame_buf[m];
        }
    }
    (void) n_fft;
}

// ============================================================================
// Incremental Mel frontend
// ============================================================================

struct voxtral_mel_frontend {
    const float * hann    = nullptr;   // [n_fft], borrowed
    const float * filters = nullptr;   // [n_freq*n_mel], borrowed

    // Rolling PCM buffer: holds absolute samples [pcm_base, pcm_base+pcm.size()).
    std::vector<float> pcm;
    int64_t pcm_base       = 0;
    int64_t total_received = 0;   // pcm_base + pcm.size() (absolute samples fed)

    int64_t next_frame          = 0;   // next mel frame index to emit
    int64_t frames_during_feed  = 0;
    int64_t frames_at_finish    = 0;
    int64_t dft_frames_computed = 0;
    int64_t pcm_peak_retained   = 0;
    bool    finalized           = false;

    // Frame-major accumulated mel: frame f occupies [f*n_mel, (f+1)*n_mel).
    std::vector<float> mel_frames;
};

namespace {

constexpr int32_t FE_N_FFT = VOXTRAL_MEL_N_FFT;
constexpr int32_t FE_HOP   = VOXTRAL_MEL_HOP;
constexpr int32_t FE_PAD   = VOXTRAL_MEL_PAD;
constexpr int32_t FE_N_MEL = VOXTRAL_MEL_N_MEL;

// Highest absolute sample index read by frame f. For f == 0 the leftmost window
// sample (centered index 0 -> signal -pad) reflects to +pad, one past the direct
// window high (f*hop + n_fft-1 - pad); every other frame's direct high dominates.
inline int64_t frame_hi_index(int64_t f) {
    const int64_t direct_hi = f * FE_HOP + (FE_N_FFT - 1) - FE_PAD;   // f*160 + 199
    const int64_t lowest_s  = f * FE_HOP - FE_PAD;                     // f*160 - 200
    const int64_t reflect_hi = (lowest_s < 0) ? (-lowest_s) : direct_hi;
    return direct_hi > reflect_hi ? direct_hi : reflect_hi;
}

// Lowest absolute sample index any frame >= f will read. When the window extends
// left of 0 (f <= 1) the direct part still reads index 0, so 0 is the floor.
inline int64_t frame_lo_index(int64_t f) {
    const int64_t lowest_s = f * FE_HOP - FE_PAD;   // f*160 - 200
    return lowest_s < 0 ? 0 : lowest_s;
}

// Fill window[0..n_fft) for frame f against total length N, reading resolved
// samples from the rolling buffer. Mirrors the batch centered[] access exactly.
void fill_window(const voxtral_mel_frontend * fe, int64_t f, int32_t N, float * window) {
    const int32_t start = (int32_t) (f * FE_HOP);
    const int64_t base  = fe->pcm_base;
    const float * data  = fe->pcm.data();
    for (int32_t i = 0; i < FE_N_FFT; ++i) {
        const int32_t src  = start + i - FE_PAD;
        const int32_t ridx = (src >= 0 && src < N) ? src : voxtral_reflect_index(src, N);
        window[i] = data[(size_t) ((int64_t) ridx - base)];
    }
}

void compute_and_store_frame(voxtral_mel_frontend * fe, int64_t f, int32_t N) {
    float window[FE_N_FFT];
    fill_window(fe, f, N, window);
    const size_t base = fe->mel_frames.size();   // == f * n_mel (frames appended in order)
    fe->mel_frames.resize(base + FE_N_MEL);
    voxtral_mel_frame_from_window(window, fe->hann, fe->filters, fe->mel_frames.data() + base);
    fe->dft_frames_computed++;
}

// Emit every frame whose window is fully within the received samples (and hence
// invariant to future samples / the final length).
void emit_stable_frames(voxtral_mel_frontend * fe) {
    while (frame_hi_index(fe->next_frame) <= fe->total_received - 1) {
        compute_and_store_frame(fe, fe->next_frame, (int32_t) fe->total_received);
        fe->next_frame++;
        fe->frames_during_feed++;
    }
}

// Drop samples no future frame needs; keep a bounded tail. Cheap: it moves only
// the (< n_fft) retained tail, and only once per feed, not per frame.
void compact(voxtral_mel_frontend * fe) {
    const int64_t keep_from = frame_lo_index(fe->next_frame);
    if (keep_from > fe->pcm_base) {
        const size_t drop = (size_t) (keep_from - fe->pcm_base);
        if (drop >= fe->pcm.size()) {
            fe->pcm.clear();
        } else {
            fe->pcm.erase(fe->pcm.begin(), fe->pcm.begin() + (std::ptrdiff_t) drop);
        }
        fe->pcm_base = keep_from;
    }
    const int64_t retained = (int64_t) fe->pcm.size();
    if (retained > fe->pcm_peak_retained) {
        fe->pcm_peak_retained = retained;
    }
}

} // namespace

voxtral_mel_frontend * voxtral_mel_frontend_create(const float * hann_window, const float * mel_filters) {
    if (!hann_window || !mel_filters) {
        return nullptr;
    }
    auto * fe = new (std::nothrow) voxtral_mel_frontend();
    if (!fe) {
        return nullptr;
    }
    fe->hann    = hann_window;
    fe->filters = mel_filters;
    return fe;
}

void voxtral_mel_frontend_destroy(voxtral_mel_frontend * fe) {
    delete fe;
}

void voxtral_mel_frontend_reset(voxtral_mel_frontend * fe) {
    if (!fe) return;
    fe->pcm.clear();
    fe->pcm_base            = 0;
    fe->total_received      = 0;
    fe->next_frame          = 0;
    fe->frames_during_feed  = 0;
    fe->frames_at_finish    = 0;
    fe->dft_frames_computed = 0;
    fe->pcm_peak_retained   = 0;
    fe->finalized           = false;
    fe->mel_frames.clear();
    // Keep pcm / mel_frames capacity for cheap reuse (no shrink_to_fit).
}

bool voxtral_mel_frontend_feed(voxtral_mel_frontend * fe, const float * pcm, size_t n) {
    if (!fe || fe->finalized) return false;
    if (n == 0) return true;
    if (!pcm) return false;
    fe->pcm.insert(fe->pcm.end(), pcm, pcm + n);
    fe->total_received += (int64_t) n;
    emit_stable_frames(fe);
    compact(fe);
    return true;
}

bool voxtral_mel_frontend_feed_silence(voxtral_mel_frontend * fe, size_t n) {
    if (!fe || fe->finalized) return false;
    if (n == 0) return true;
    fe->pcm.insert(fe->pcm.end(), n, 0.0f);
    fe->total_received += (int64_t) n;
    emit_stable_frames(fe);
    compact(fe);
    return true;
}

void voxtral_mel_frontend_finish(voxtral_mel_frontend * fe) {
    if (!fe || fe->finalized) return;
    fe->finalized = true;

    const int64_t final_total = fe->total_received;
    const int64_t n_total     = final_total / FE_HOP;   // floor; == batch frame count
    while (fe->next_frame < n_total) {
        compute_and_store_frame(fe, fe->next_frame, (int32_t) final_total);
        fe->next_frame++;
        fe->frames_at_finish++;
    }

    // Frames are complete; the retained tail is no longer needed.
    fe->pcm.clear();
    fe->pcm_base = final_total;
}

voxtral_mel_metrics voxtral_mel_frontend_metrics(const voxtral_mel_frontend * fe) {
    voxtral_mel_metrics m;
    if (!fe) return m;
    m.frames_total        = fe->next_frame;
    m.frames_during_feed  = fe->frames_during_feed;
    m.frames_at_finish    = fe->frames_at_finish;
    m.dft_frames_computed = fe->dft_frames_computed;
    m.pcm_retained        = (int64_t) fe->pcm.size();
    m.pcm_peak_retained   = fe->pcm_peak_retained;
    m.pcm_base            = fe->pcm_base;
    m.total_received      = fe->total_received;
    m.finalized           = fe->finalized;
    return m;
}

int64_t voxtral_mel_frontend_frame_count(const voxtral_mel_frontend * fe) {
    return fe ? fe->next_frame : 0;
}

const float * voxtral_mel_frontend_frames_data(const voxtral_mel_frontend * fe) {
    return (fe && !fe->mel_frames.empty()) ? fe->mel_frames.data() : nullptr;
}

void voxtral_mel_frontend_assemble_raw(const voxtral_mel_frontend * fe,
                                       std::vector<float> & out, int32_t & out_n_frames) {
    out.clear();
    out_n_frames = 0;
    if (!fe) return;
    const int32_t n_frames = (int32_t) fe->next_frame;
    out_n_frames = n_frames;
    if (n_frames <= 0) return;
    out.resize((size_t) FE_N_MEL * (size_t) n_frames);
    for (int32_t m = 0; m < FE_N_MEL; ++m) {
        for (int32_t f = 0; f < n_frames; ++f) {
            out[(size_t) m * n_frames + f] = fe->mel_frames[(size_t) f * FE_N_MEL + m];
        }
    }
}

void voxtral_mel_frontend_assemble_even(const voxtral_mel_frontend * fe,
                                        std::vector<float> & out, int32_t & out_n_frames) {
    out.clear();
    out_n_frames = 0;
    if (!fe) return;
    const int32_t n_total = (int32_t) fe->next_frame;
    const bool drop_first = (n_total % 2 != 0);
    const int32_t n_out = drop_first ? (n_total - 1) : n_total;
    out_n_frames = n_out;
    if (n_out <= 0) return;
    out.resize((size_t) FE_N_MEL * (size_t) n_out);
    const int32_t src_off = drop_first ? 1 : 0;
    for (int32_t m = 0; m < FE_N_MEL; ++m) {
        for (int32_t j = 0; j < n_out; ++j) {
            const int32_t f = j + src_off;
            out[(size_t) m * n_out + j] = fe->mel_frames[(size_t) f * FE_N_MEL + m];
        }
    }
}
