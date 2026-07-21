#ifndef VOXTRAL_MEL_H
#define VOXTRAL_MEL_H

// ============================================================================
// Shared log-Mel frontend (INTERNAL, UNSTABLE).
// ----------------------------------------------------------------------------
// This module is the single source of truth for the Voxtral STFT / log-Mel
// math. Both the batch path (voxtral_transcribe_audio -> compute_mel_even) and
// the incremental streaming frontend (voxtral-stream) compute Mel frames through
// the SAME per-frame kernel, so they cannot numerically diverge.
//
// It contains exactly one implementation each of:
//   * the DFT (precomputed cos/sin tables, see voxtral_get_stft_plan),
//   * the Slaney mel filterbank,
//   * the Hann window,
//   * the reflect ("center", pad_mode="reflect") padding index map,
//   * the log10 + clamp + (x+4)/4 normalization.
//
// Layout / semantics reproduced bit-for-bit from the original in-lined
// compute_mel_spectrogram():
//   * n_frames        = n_samples / hop            (torch.stft frames minus the
//                       dropped last one: (n/hop + 1) - 1)
//   * frame f window  = centered[f*hop .. f*hop + n_fft), where
//                       centered = reflect-pad(signal, pad=n_fft/2) on both ends
//   * batch mel_out   = channel-major [n_mel, n_frames], element (m,f) at
//                       m*n_frames + f
//   * normalization   = max(v,1e-10) -> log10 -> max(v, LOG_MEL_MAX-8) -> (v+4)/4
//
// The header is deliberately not part of the public C++ surface in
// include/voxtral.h.
// ============================================================================

#include "voxtral.h"   // VOXTRAL_WINDOW_SIZE / HOP_LENGTH / NUM_MEL_BINS / SAMPLE_RATE / ...

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Frontend constants (derived from the public audio constants). Defined here so
// there is one authoritative copy shared by voxtral.cpp and the mel module.
// ----------------------------------------------------------------------------
static constexpr float   VOXTRAL_MEL_PI     = 3.14159265358979323846f;
static constexpr int32_t VOXTRAL_MEL_N_FFT  = VOXTRAL_WINDOW_SIZE;         // 400
static constexpr int32_t VOXTRAL_MEL_HOP    = VOXTRAL_HOP_LENGTH;          // 160
static constexpr int32_t VOXTRAL_MEL_PAD    = VOXTRAL_MEL_N_FFT / 2;       // 200 (center pad)
static constexpr int32_t VOXTRAL_MEL_N_FREQ = VOXTRAL_MEL_N_FFT / 2 + 1;   // 201
static constexpr int32_t VOXTRAL_MEL_N_MEL  = VOXTRAL_NUM_MEL_BINS;        // 128

// ----------------------------------------------------------------------------
// Precomputed DFT twiddle-factor tables ([n_bins][n_fft], row-major by bin).
// ----------------------------------------------------------------------------
struct voxtral_stft_plan {
    int32_t n_fft  = 0;
    int32_t n_bins = 0;
    std::vector<float> cos_table;
    std::vector<float> sin_table;
};

// Process-wide, lazily initialised, immutable after init (thread-safe first call).
const voxtral_stft_plan & voxtral_get_stft_plan();

// PyTorch pad(mode="reflect") index map for a signal of length `len`.
int32_t voxtral_reflect_index(int32_t idx, int32_t len);

// Build the deterministic filter/window tables (single implementation).
void voxtral_mel_build_hann_window(std::vector<float> & hann);           // [n_fft]
void voxtral_mel_build_slaney_filters(std::vector<float> & filters);     // [n_freq*n_mel]

// ----------------------------------------------------------------------------
// The per-frame kernel. `window` is a contiguous [n_fft] block of already
// reflect-resolved input samples for one STFT frame; `out_frame` receives the
// [n_mel] normalized log-mel values for that frame (frame-major/contiguous).
// This is the ONLY place the DFT + filterbank + normalization arithmetic lives.
// ----------------------------------------------------------------------------
void voxtral_mel_frame_from_window(
    const float * window,        // [n_fft]
    const float * hann_window,   // [n_fft]
    const float * mel_filters,   // [n_freq * n_mel]
    float       * out_frame);    // [n_mel]

// ----------------------------------------------------------------------------
// Batch spectrogram. Bit-for-bit equivalent to the historical in-lined
// compute_mel_spectrogram(): channel-major [n_mel, n_frames] output. Retained as
// the canonical batch entry point and as the reference for parity tests.
// mel_out must hold at least n_mel * (n_samples/hop) floats.
// ----------------------------------------------------------------------------
void voxtral_mel_compute_batch(
    const float * audio,
    int32_t       n_samples,
    const float * mel_filters,   // [n_freq * n_mel]
    const float * hann_window,   // [n_fft]
    float       * mel_out,       // [n_mel, n_frames]
    int32_t     * out_n_frames);

// Number of batch mel frames for a signal of `n_samples` (n_samples/hop, >= 0).
int32_t voxtral_mel_batch_frame_count(int32_t n_samples);

// ============================================================================
// Incremental Mel frontend.
// ----------------------------------------------------------------------------
// Feed PCM in arbitrary chunks; stable frames (those that can no longer change
// with future samples) are emitted during feed. finish() flushes the remaining
// frames using right reflect padding against the final length. The result is
// bit-for-bit identical to voxtral_mel_compute_batch() over the concatenation of
// all fed samples, independent of the chunk boundaries.
//
// Only a bounded PCM tail is retained: after every normal feed the retained
// sample count is < n_fft, independent of stream duration (a single very large
// feed transiently holds that feed's samples, then compacts on return).
// ----------------------------------------------------------------------------
struct voxtral_mel_frontend;   // defined in voxtral-mel.cpp

struct voxtral_mel_metrics {
    int64_t frames_total       = 0;  // mel frames produced so far (0-based next index)
    int64_t frames_during_feed = 0;  // frames emitted while feeding
    int64_t frames_at_finish   = 0;  // frames flushed by finish()
    int64_t dft_frames_computed= 0;  // DFT invocations (must equal frames_total)
    int64_t pcm_retained       = 0;  // samples currently retained in the rolling buffer
    int64_t pcm_peak_retained  = 0;  // max retained (post-compaction) over the session
    int64_t pcm_base           = 0;  // absolute index of the oldest retained sample
    int64_t total_received     = 0;  // total samples fed (incl. injected silence)
    bool    finalized          = false;
};

// hann/mel_filters must remain valid for the lifetime of the frontend and are
// used by every frame (the caller owns them: the model's ctx buffers, or a test
// harness's tables). Neither is copied.
voxtral_mel_frontend * voxtral_mel_frontend_create(
    const float * hann_window,   // [n_fft]
    const float * mel_filters);  // [n_freq * n_mel]

void voxtral_mel_frontend_destroy(voxtral_mel_frontend * fe);

// Return to the freshly-created state, retaining allocated capacity for reuse.
void voxtral_mel_frontend_reset(voxtral_mel_frontend * fe);

// Append real samples (nullptr valid only when n == 0) and emit newly-stable
// frames. Rejected (and returns false) after finish().
bool voxtral_mel_frontend_feed(voxtral_mel_frontend * fe, const float * pcm, size_t n);

// Append `n` zero samples (silence) efficiently, without materialising them in
// the caller. Used for the streaming left/right zero padding.
bool voxtral_mel_frontend_feed_silence(voxtral_mel_frontend * fe, size_t n);

// Flush the final frames with right reflect padding against the final length.
// Idempotent: a second call is a no-op and recomputes nothing.
void voxtral_mel_frontend_finish(voxtral_mel_frontend * fe);

voxtral_mel_metrics voxtral_mel_frontend_metrics(const voxtral_mel_frontend * fe);

// Number of mel frames produced so far (== metrics.frames_total).
int64_t voxtral_mel_frontend_frame_count(const voxtral_mel_frontend * fe);

// Borrowed, frame-major view of the accumulated stable Mel frames: frame f (for
// 0 <= f < voxtral_mel_frontend_frame_count()) occupies [f*n_mel, (f+1)*n_mel).
// Used by the incremental encoder to drain newly-stable frames during feed. The
// pointer is INVALIDATED by the next feed/finish/reset (the buffer may grow), so
// consume it immediately. Returns nullptr when no frames exist yet.
const float * voxtral_mel_frontend_frames_data(const voxtral_mel_frontend * fe);

// Assemble the accumulated frames into a channel-major [n_mel, n_frames] matrix.
//   *_raw   : all produced frames, no even trim (parity reference).
//   *_even  : batch-equivalent output of compute_mel_even() — drops the first
//             frame when the frame count is odd (conv stride-2 alignment).
// Both require finish() to have been called. Return the frame count via out.
void voxtral_mel_frontend_assemble_raw (const voxtral_mel_frontend * fe,
                                        std::vector<float> & out, int32_t & out_n_frames);
void voxtral_mel_frontend_assemble_even(const voxtral_mel_frontend * fe,
                                        std::vector<float> & out, int32_t & out_n_frames);

#endif // VOXTRAL_MEL_H
