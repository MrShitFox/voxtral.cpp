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

// Fixed decoder self-attention K/V allocation owned by one inference context.
int64_t voxtral_context_decoder_kv_bytes_internal(const voxtral_context * ctx);

// ============================================================================
// Incremental causal audio encoder (streaming). Production uses a per-layer
// bounded KV ring and static logical/physical microbatch scheduling; the legacy
// bounded-window recomputation strategy remains an explicit reference mode.
// See voxtral.cpp and docs/architecture/streaming-runtime.md. Not public.
// ============================================================================
struct voxtral_encoder_stream;   // defined in voxtral.cpp
struct voxtral_stream;           // defined in voxtral-stream.cpp

struct voxtral_encoder_metrics {
    int64_t melFramesReceived            = 0;  // stable Mel frames pushed to the encoder
    int64_t encoderExecutions            = 0;  // chunk / KV graph runs
    int64_t encoderInputFramesProcessed  = 0;  // reference: sum of chunk Mel lengths run (recompute-inclusive)
    int64_t encoderNewFramesProduced     = 0;  // == emitted enc frames
    int64_t encoderFramesRecomputed      = 0;  // input frames beyond unique (bounded-window overhead)
    int32_t encoderMaxWindowFrames       = 0;  // reference: max chunk Mel length run (<= CHUNK_MEL)
    int64_t encoderContextFramesRetained = 0;  // current retained Mel window size, frames
    int64_t encoderPeakContextFrames     = 0;  // peak retained Mel window size, frames
    int64_t encoderFramesBeforeFinish    = 0;  // enc frames emitted during feed
    int64_t encoderFramesFlushedAtFinish = 0;  // enc frames emitted by finish()
    int64_t encoderOutputFrames          = 0;  // total enc frames accumulated
    int64_t encoderStateBytes            = 0;  // bounded per-stream host state (Mel tail + scratch)
    int64_t encoderOutputAccumulatedBytes= 0;  // linear accumulated encoder output
    int32_t curChunk                     = 0;
    bool    finalized                    = false;

    // ---- Strategy identity ------------------------------------------------
    // "per-layer-kv" (Session 6.2 production default) or "bounded-window-recompute".
    const char * strategy                = "per-layer-kv";

    // ---- Per-layer KV-cache work instrumentation (Stage 16) --------------
    int64_t encoderUniqueFrames             = 0;  // distinct enc frames that ever existed (== emitted)
    int64_t encoderTransformerFramesComputed= 0;  // enc frames run through the transformer (each once for KV)
    int64_t encoderFrameLayerEvaluations    = 0;  // transformer frames * enc_layers
    int64_t encoderKvAppends                = 0;  // enc frames appended to the ring
    int64_t encoderKvEvictions              = 0;  // enc frames dropped past the window
    int64_t encoderKvWraps                  = 0;  // ring physical wrap events (write or read)
    int64_t encoderKvMaterializedFrames     = 0;  // frames copied by wrap gather (materialization cost)
    int64_t encoderGraphExecutions          = 0;  // KV transformer graph runs
    int32_t encoderMaxNewFramesPerExecution = 0;  // max new frames in one KV graph run
    int64_t encoderFramesComputedDuringFinish=0;  // transformer frames evaluated by finish()

    // ---- Per-layer KV-cache memory instrumentation (Stage 17) ------------
    int64_t encoderKvAllocatedBytes         = 0;  // device bytes held by the K+V ring (all layers)
    int64_t encoderKvLogicalFrames          = 0;  // logical frames currently in the cache (<= window)
    int64_t encoderKvPeakLogicalFrames      = 0;  // peak logical frames
    int32_t encoderKvCapacityFrames         = 0;  // ring capacity (frames)
    int32_t encoderKvElementSize            = 0;  // bytes per K/V element (4 = F32)
    int64_t encoderMelRetainedBytes         = 0;  // host Mel tail retained for the conv stem
    int64_t encoderMelRetainedFrames        = 0;  // host Mel tail frames retained
    int64_t encoderMelPeakRetainedFrames    = 0;  // peak host Mel tail frames
    int64_t encoderOutputQueuedFrames       = 0;  // accumulated (not-yet-consumed) encoder output frames
    int64_t encoderOutputPeakQueuedFrames   = 0;  // peak accumulated encoder output frames

    // Scheduler shape/work accounting. `logical` counts real frames submitted
    // per graph; `physical` is the fixed transformer query-row shape (including
    // masked dummy rows). These counters are intentionally separate so padding
    // can never be mistaken for useful encoder work.
    const char * encoderScheduler            = "static";
    int32_t encoderLogicalBatchFrames        = 4;
    int32_t encoderPhysicalQueryRows         = 32;
    int64_t encoderLogicalFramesSubmitted    = 0;
    int64_t encoderPhysicalQueryRowsEvaluated = 0;
    int64_t encoderPaddingRowsEvaluated       = 0;
    int32_t encoderWarmupFrames               = 0;

    // Optional timing collector (enabled by VOXTRAL_ENCODER_TELEMETRY=1).
    // Values are aggregate percentiles; no per-frame array crosses the API.
    double encoderFirstFrameAbsoluteMs       = 0.0;
    double encoderFirstFrameResidenceMs      = 0.0;
    double encoderResidenceP50Ms             = 0.0;
    double encoderResidenceP95Ms             = 0.0;
    double encoderResidenceP99Ms             = 0.0;
    double encoderResidenceMaxMs             = 0.0;
    double firstAdapterGroupAbsoluteMs       = 0.0;
    double firstAdapterGroupResidenceMs      = 0.0;
    double firstEightFrameGroupAbsoluteMs    = 0.0;
    double firstEightFrameGroupResidenceMs   = 0.0;
    double adapterGroupResidenceP50Ms        = 0.0;
    double adapterGroupResidenceP95Ms        = 0.0;
    double adapterGroupResidenceP99Ms        = 0.0;
    double adapterGroupResidenceMaxMs        = 0.0;
    double encoderComputeP50Ms               = 0.0;
    double encoderComputeP95Ms               = 0.0;
    double encoderComputeP99Ms               = 0.0;
    double encoderComputeMaxMs               = 0.0;
    double encoderComputeWarmMaxMs           = 0.0;  // excludes first graph/shader warmup
};

// Encoder frames produced from `mel_frames` Mel frames through the conv stem
// (conv0 k3s1 + conv1 k3s2) and the left-trunc to a multiple of downsample_factor.
// Model-free; exposed so the encoder unit test can lock the chunk-schedule
// arithmetic against the same function the batch/streaming encoder uses.
int32_t voxtral_enc_frames_for_mel_internal(int32_t mel_frames);

// ============================================================================
// Per-layer encoder KV-cache: model-free ring planner (Session 6.1).
// ----------------------------------------------------------------------------
// All ring bookkeeping is derived by a single pure function so the risky
// off-by-one / wrap / eviction arithmetic can be locked by a backend-free unit
// test (tests/cpp/test_encoder_kv.cpp) before any GPU graph relies on it. The
// GGML graph consumes exactly this plan; there is no second copy of the logic.
// ============================================================================

// Ring capacity (frames) and the max new enc frames processed by one KV graph
// execution. capacity == window + max_new so (a) the causal window of a whole
// batch (<= window-1 + max_new frames) always fits contiguously in logical space
// and (b) writing a batch's K/V never clobbers a still-live window frame.
int32_t voxtral_enc_kv_capacity_internal();   // VOXTRAL_ENC_KV_CAP
int32_t voxtral_enc_kv_max_new_internal();    // VOXTRAL_ENC_KV_MAX_NEW
int32_t voxtral_enc_kv_window_internal();     // VOXTRAL_ENC_WINDOW
int32_t voxtral_enc_kv_logical_batch_internal();
int32_t voxtral_enc_kv_physical_rows_internal();

// One physical ring segment: `len` contiguous frames starting at ring slot `slot`.
struct voxtral_enc_kv_seg { int32_t slot = 0; int32_t len = 0; };

// Everything the KV graph needs for one execution over new enc frames
// [q_start, q_start + n_new). Positions are ABSOLUTE (RoPE / eviction), slots are
// physical (mod capacity). Segments are listed in logical (ascending-position)
// order; a wrapped span has two, otherwise one.
struct voxtral_enc_kv_plan_t {
    int64_t q_start        = 0;   // absolute pos of the first new query (== emitted before batch)
    int32_t n_new          = 0;   // new enc frames this execution (1 <= n_new <= max_new)

    // Causal KV window [win_start, win_end) required by the batch of queries:
    // win_start = max(0, q_start-(window-1)); win_end = q_start + n_new.
    int64_t win_start      = 0;
    int64_t win_end        = 0;
    int32_t win_len        = 0;   // win_end - win_start  (<= window-1 + n_new <= capacity)

    voxtral_enc_kv_seg read_seg[2];   // physical segments for the window READ (gather)
    int32_t read_nseg      = 0;
    bool    read_wraps     = false;

    voxtral_enc_kv_seg write_seg[2];  // physical segments for the new-frame K/V WRITE
    int32_t write_nseg     = 0;
    bool    write_wraps    = false;

    // Conv input: feed Mel [conv_mel_start, conv_mel_end) (conv_mel_start EVEN),
    // then take conv1 output local frames [conv_slice_start, conv_slice_start+n_new).
    int64_t conv_mel_start = 0;
    int64_t conv_mel_end   = 0;
    int32_t conv_slice_start = 0;

    int64_t evicted        = 0;   // frames whose window membership ended at this batch
};

// Build the plan for new enc frames [q_start, q_start+n_new). Requires
// 1 <= n_new <= capacity-window (== max_new); returns false otherwise (the caller
// splits larger batches into <= max_new sub-batches). capacity/window are passed
// explicitly so the unit test can exercise small synthetic rings.
bool voxtral_enc_kv_plan(int64_t q_start, int32_t n_new,
                         int32_t capacity, int32_t window,
                         voxtral_enc_kv_plan_t & out);

// Causal-window mask predicate shared by the runner and the unit test: may a query
// at absolute position q_abs attend a key at absolute position kv_abs?
bool voxtral_enc_kv_mask_allows(int64_t kv_abs, int64_t q_abs, int32_t window);

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

// Attach the wall-clock/audio timeline used by the optional residence collector.
// `sample_count` is the number of real PCM samples available at this feed;
// `arrival_ns` is the monotonic time at which that payload became available.
void voxtral_encoder_stream_note_audio(voxtral_encoder_stream * es,
                                        int64_t sample_count, int64_t arrival_ns,
                                        int64_t mel_frames_ready, int64_t mel_ready_ns,
                                        int64_t timeline_start_ns,
                                        int64_t left_pad_samples);

// Test/instrumentation seam: anchor the stream's monotonic timeline before the
// first paced feed. Production callers leave this unset and the stream starts
// its timeline at the first payload arrival.
void voxtral_stream_set_timeline_start_internal(voxtral_stream * stream,
                                                int64_t timeline_start_ns);

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
