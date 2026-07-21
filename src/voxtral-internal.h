#ifndef VOXTRAL_INTERNAL_H
#define VOXTRAL_INTERNAL_H

// ============================================================================
// Internal (non-public) inference entry points shared between voxtral.cpp and
// the streaming layer (voxtral-stream.cpp). Not part of include/voxtral.h.
// ============================================================================

#include "voxtral.h"

// Realtime inference from a precomputed, even-trimmed log-Mel matrix in the
// canonical channel-major [n_mel, n_frames] layout (exactly what
// compute_mel_even / voxtral_mel_frontend_assemble_even produce). Runs the
// encoder / adapter / decoder and fills `result`. This is the single Mel->text
// path: voxtral_transcribe_audio() computes a batch Mel and calls it; the
// streaming finish() passes its incrementally accumulated Mel to the same path.
//
// Realtime models only (the offline arch keeps its own windowed path).
bool voxtral_transcribe_mel_internal(
    voxtral_context & ctx,
    const float     * mel,        // [n_mel, n_frames], channel-major
    int32_t           n_frames,
    int32_t           max_tokens,
    voxtral_result  & result);

// Borrowed views of the context's CPU-side frontend tables (stable for the
// context's lifetime). Used by the streaming Mel frontend and test harnesses so
// they compute Mel with the exact same Hann window / mel filterbank as the batch
// path. Return nullptr if unavailable.
const float * voxtral_ctx_hann_window (const voxtral_context * ctx);  // [n_fft]
const float * voxtral_ctx_mel_filters (const voxtral_context * ctx);  // [n_freq*n_mel]

#endif // VOXTRAL_INTERNAL_H
