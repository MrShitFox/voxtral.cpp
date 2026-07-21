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

// Second half of the realtime Mel->text path: adapter + decoder over a host-side,
// channel-major [enc_dim, enc_frames] encoder output (exactly what the batch
// encoder leaves in ctx.encoder_output, and what the streaming incremental encoder
// accumulates during feed). Trims to a multiple of the downsample factor. The
// streaming finish() calls this instead of re-running the encoder over the Mel.
bool voxtral_transcribe_encoder_output_internal(
    voxtral_context & ctx,
    const float     * enc_output, // [enc_dim, enc_frames], channel-major
    int32_t           enc_frames,
    int32_t           max_tokens,
    voxtral_result  & result);

// Encoder-only batch pass: runs the (chunked) encoder over an even-trimmed Mel and
// returns ctx.encoder_output as a host, channel-major [enc_dim, enc_seq_used]
// buffer. The reference for batch-vs-incremental encoder tensor parity.
bool voxtral_encode_mel_batch_internal(
    voxtral_context    & ctx,
    const float        * mel,     // [n_mel, n_frames], channel-major
    int32_t              n_frames,
    std::vector<float> & out_enc,
    int32_t            & out_frames);

// ============================================================================
// Incremental causal audio encoder (streaming). Bounded-window recomputation that
// replays the batch run_encoder_chunked schedule from stable Mel frames, sharing
// the same transformer graph as the batch path. See voxtral.cpp and
// docs/architecture/streaming-runtime.md. Not part of the public surface.
// ============================================================================
struct voxtral_encoder_stream;   // defined in voxtral.cpp

struct voxtral_encoder_metrics {
    int64_t melFramesReceived            = 0;  // stable Mel frames pushed to the encoder
    int64_t encoderExecutions            = 0;  // chunk graph runs
    int64_t encoderInputFramesProcessed  = 0;  // sum of chunk Mel lengths run (recompute-inclusive)
    int64_t encoderNewFramesProduced     = 0;  // == emitted enc frames
    int64_t encoderFramesRecomputed      = 0;  // input frames beyond unique (bounded-window overhead)
    int32_t encoderMaxWindowFrames       = 0;  // max chunk Mel length run (<= CHUNK_MEL)
    int64_t encoderContextFramesRetained = 0;  // current Mel window size, frames
    int64_t encoderPeakContextFrames     = 0;  // peak Mel window size, frames
    int64_t encoderFramesBeforeFinish    = 0;  // enc frames emitted during feed
    int64_t encoderFramesFlushedAtFinish = 0;  // enc frames emitted by finish()
    int64_t encoderOutputFrames          = 0;  // total enc frames accumulated
    int64_t encoderStateBytes            = 0;  // bounded per-stream state (Mel window + scratch)
    int64_t encoderOutputAccumulatedBytes= 0;  // linear accumulated encoder output
    int32_t curChunk                     = 0;
    bool    finalized                    = false;
};

// Encoder frames produced from `mel_frames` Mel frames through the conv stem
// (conv0 k3s1 + conv1 k3s2) and the left-trunc to a multiple of downsample_factor.
// Model-free; exposed so the encoder unit test can lock the chunk-schedule
// arithmetic against the same function the batch/streaming encoder uses.
int32_t voxtral_enc_frames_for_mel_internal(int32_t mel_frames);

// Owns a bounded Mel window + accumulated encoder output; drives run_encoder_chunk
// against `ctx`. One per stream; created lazily, freed on stream destroy.
voxtral_encoder_stream * voxtral_encoder_stream_create(voxtral_context * ctx);
void    voxtral_encoder_stream_destroy(voxtral_encoder_stream * es);
void    voxtral_encoder_stream_reset  (voxtral_encoder_stream * es);

// Push newly-stable frame-major Mel frames [base_frame, base_frame+n) (each n_mel
// contiguous). base_frame must equal the running stable count (contiguous). Runs
// any chunks that became available and accumulates their output. Returns false on
// a contiguity violation or a backend failure.
bool voxtral_encoder_stream_push_mel(voxtral_encoder_stream * es,
                                     const float * frames, int64_t base_frame, int32_t n);

// Run the last (incomplete) chunk exactly as the batch tail. Idempotent.
bool voxtral_encoder_stream_finish(voxtral_encoder_stream * es);

// Accumulated encoder output: host, channel-major [enc_dim, output_frames].
const float * voxtral_encoder_stream_output       (const voxtral_encoder_stream * es);
int32_t       voxtral_encoder_stream_output_frames(const voxtral_encoder_stream * es);
voxtral_encoder_metrics voxtral_encoder_stream_metrics(const voxtral_encoder_stream * es);

// Borrowed views of the context's CPU-side frontend tables (stable for the
// context's lifetime). Used by the streaming Mel frontend and test harnesses so
// they compute Mel with the exact same Hann window / mel filterbank as the batch
// path. Return nullptr if unavailable.
const float * voxtral_ctx_hann_window (const voxtral_context * ctx);  // [n_fft]
const float * voxtral_ctx_mel_filters (const voxtral_context * ctx);  // [n_freq*n_mel]

#endif // VOXTRAL_INTERNAL_H
