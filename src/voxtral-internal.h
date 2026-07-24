#ifndef VOXTRAL_INTERNAL_H
#define VOXTRAL_INTERNAL_H

// ============================================================================
// Internal (non-public) inference entry points shared between voxtral.cpp and
// the streaming layer (voxtral-stream.cpp). Not part of include/voxtral.h.
// ============================================================================

#include "voxtral-cpp.h"

#include <array>

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

// Thin public-C-adapter hooks. These expose no private layout and perform no
// inference. Public stream operations are externally serialized in API v1.
voxtral_model * voxtral_context_model_internal(const voxtral_context * ctx);
bool voxtral_context_supports_incremental_internal(const voxtral_context * ctx);
bool voxtral_context_try_acquire_public_stream_internal(voxtral_context * ctx);
void voxtral_context_release_public_stream_internal(voxtral_context * ctx);
bool voxtral_context_has_public_stream_internal(const voxtral_context * ctx);
void voxtral_context_set_public_error_internal(
    voxtral_context * ctx,
    voxtral_status status,
    int32_t backend_code,
    const char * message) noexcept;
voxtral_status voxtral_context_public_last_status_internal(
    const voxtral_context * ctx);
int32_t voxtral_context_public_backend_code_internal(
    const voxtral_context * ctx);
const std::string & voxtral_context_public_last_error_internal(
    const voxtral_context * ctx);

// First-step numerical diagnostics. Empty unless
// VOXTRAL_NUMERICAL_DIAGNOSTICS=1 was set before context creation.
const std::vector<float> & voxtral_context_diagnostic_first_hidden_internal(
    const voxtral_context * ctx);
const std::vector<float> & voxtral_context_diagnostic_first_logits_internal(
    const voxtral_context * ctx);
const std::vector<float> & voxtral_context_diagnostic_encoder_output_internal(
    const voxtral_context * ctx);
const std::vector<float> & voxtral_context_diagnostic_adapter_output_internal(
    const voxtral_context * ctx);

// ============================================================================
// Session 8: bounded production-pipeline profiler.
//
// VOXTRAL_PROFILE=1 enables exact aggregate counters plus a fixed-capacity,
// allocation-free sample reservoir for percentiles.  The reservoir is allocated
// with the context (never per feed/graph/decoder step), so profiling cannot turn
// a sustained stream into unbounded host accumulation.  Execute timings include
// the synchronous backend wait; backend_synchronize is also reported separately
// as a subset so command submission vs GPU completion can be diagnosed.
// ============================================================================

enum class voxtral_profile_stage : size_t {
    mel_compute = 0,
    encoder_graph_build,
    encoder_graph_execute,
    encoder_device_copy,
    adapter_graph_build,
    adapter_graph_execute,
    decoder_prefill_graph_build,
    decoder_prefill_graph_execute,
    decoder_step_graph_build,
    decoder_step_graph_execute,
    argmax,
    token_readback,
    backend_synchronize,
    event_processing,
    pipeline_feed,
    count,
};

struct voxtral_stage_profile {
    uint64_t count   = 0;
    double   totalMs = 0.0;
    double   meanMs  = 0.0;
    double   p50Ms   = 0.0;
    double   p95Ms   = 0.0;
    double   p99Ms   = 0.0;
    double   maxMs   = 0.0;
};

struct voxtral_runtime_profile {
    bool enabled = false;
    std::array<voxtral_stage_profile,
               static_cast<size_t>(voxtral_profile_stage::count)> stages{};

    uint64_t encoderGraphBuildCount = 0;
    uint64_t adapterGraphBuildCount = 0;
    uint64_t decoderGraphBuildCount = 0;

    // Scheduler allocation passes.  GGML may allocate several transient tensors
    // inside one pass; these counters intentionally measure the hot-path calls
    // whose growth/reuse is under this runtime's control.
    uint64_t encoderAllocations = 0;
    uint64_t adapterAllocations = 0;
    uint64_t decoderAllocations = 0;
    uint64_t graphAllocations   = 0;

    uint64_t backendSyncCount   = 0;
    uint64_t commandSubmitCount = 0;
    uint64_t tensorSetCount     = 0;
    uint64_t tensorGetCount     = 0;

    double totalGpuComputeMs      = 0.0;
    double totalPipelineComputeMs = 0.0;

    // Filled by the KV implementations.  A production FP16 plan must report no
    // full-size F32 shadow cache here.
    int64_t kvF16Bytes            = 0;
    int64_t temporaryF32KvBytes   = 0;

    int64_t decoderKvCapacity                = 0;
    int64_t decoderKvUsed                    = 0;
    int64_t decoderKvWraps                   = 0;
    int64_t decoderKvEvictions               = 0;
    int64_t decoderKvBytesMoved              = 0;
    int64_t decoderKvFullBufferMoves         = 0;
    int64_t decoderOldestAbsolutePosition    = 0;
    int64_t decoderNextAbsolutePosition      = 0;
    int32_t decoderKvElementSize             = 0;
    int64_t decoderFirstWrapAbsolutePosition = -1;
    double  decoderPreWrapP99Ms              = 0.0;
    double  decoderWrapStepMs                = 0.0;
    double  decoderPostWrapP99Ms             = 0.0;
};

// Allocation-free scalar subset used by the stable public metrics adapter.
// Reading this snapshot neither sorts profiling reservoirs nor synchronizes a
// backend; public API v1 externally serializes it with inference operations.
struct voxtral_public_context_metrics_internal {
    bool profile_enabled = false;
    double total_pipeline_compute_ms = 0.0;
    int64_t decoder_kv_wraps = 0;
    int64_t decoder_kv_evictions = 0;
    int64_t decoder_kv_bytes_moved = 0;
};

const char * voxtral_profile_stage_name(voxtral_profile_stage stage);
voxtral_runtime_profile voxtral_context_runtime_profile_internal(const voxtral_context * ctx);
voxtral_public_context_metrics_internal
voxtral_context_public_metrics_internal(const voxtral_context * ctx);
void voxtral_context_profile_record_internal(voxtral_context * ctx,
                                             voxtral_profile_stage stage,
                                             double milliseconds);
void voxtral_context_profile_note_allocation_internal(voxtral_context * ctx,
                                                      voxtral_profile_stage stage);
void voxtral_context_profile_note_submit_internal(voxtral_context * ctx);
void voxtral_context_profile_note_tensor_set_internal(voxtral_context * ctx);
void voxtral_context_profile_note_tensor_get_internal(voxtral_context * ctx);
void voxtral_context_profile_reset_internal(voxtral_context * ctx);

// Decoder self-attention cache metadata.  RoPE and token/event positions are
// always absolute; only storage addressing is modulo `capacity`.
struct voxtral_decoder_kv_ring {
    int64_t capacity = 0;
    int64_t used = 0;
    int64_t oldest_absolute_position = 0;
    int64_t next_absolute_position = 0;
    int64_t wraps = 0;
    int64_t evictions = 0;
    int64_t bytes_moved = 0;       // rollover-only movement; new K/V writes excluded
    int64_t full_buffer_moves = 0;
};

struct voxtral_decoder_kv_seg {
    int32_t slot = 0;
    int32_t len  = 0;
};

struct voxtral_decoder_kv_plan_t {
    int64_t absolute_start = 0;
    int32_t n_new = 0;
    voxtral_decoder_kv_seg write_seg[2];
    int32_t write_nseg = 0;
    voxtral_decoder_kv_seg read_seg[2];  // oldest -> newest logical order
    int32_t read_nseg = 0;
    int64_t used_after = 0;
    int64_t oldest_after = 0;
    int64_t next_after = 0;
    int64_t evicted = 0;
    bool wrapped = false;
};

bool voxtral_decoder_kv_plan(const voxtral_decoder_kv_ring & ring,
                             int64_t absolute_start, int32_t n_new,
                             voxtral_decoder_kv_plan_t & out);

// ============================================================================
// Session 7: device-resident incremental adapter + decoder. The encoder writes
// its output into a persistent device ring (see voxtral.cpp); these entry points
// let the streaming layer advance the adapter and decoder during feed() without
// copying encoder output or audio embeddings across the device boundary. Only a
// 4-byte token id is read back per decoder step. Not public.
// ----------------------------------------------------------------------------

// Run the adapter on-device over the complete groups [group_start, group_start+
// n_groups) currently resident in the encoder-output ring, writing their audio
// embeddings into the audio-embedding ring (splitting internally at ring wraps).
// Returns n_groups on success, -1 on a backend failure. group == audio position.
int32_t voxtral_ctx_adapter_commit(voxtral_context * ctx, int64_t group_start, int32_t n_groups);

// Route the realtime decoder graphs to the audio-embedding ring and clear KV.
void voxtral_ctx_decoder_begin_incremental(voxtral_context * ctx);
// Prefill the prompt tokens once (positions [0, n_tokens)); no logits readback.
bool voxtral_ctx_decoder_prefill_incremental(voxtral_context * ctx, const int32_t * token_ids, int32_t n_tokens);
// One decoder step at absolute `position` (audio_pos == position). *token_out gets
// the on-device greedy argmax token via a 4-byte readback.
bool voxtral_ctx_decoder_step_incremental(voxtral_context * ctx, int32_t token_id, int32_t position, int32_t * token_out);
// Detach the incremental audio source and reset KV (safe between streams).
void voxtral_ctx_decoder_reset_incremental(voxtral_context * ctx);

// Capacities (frames) of the device-resident rings, for the streaming scheduler's
// backpressure math. Zero if the context has no rings.
int32_t voxtral_ctx_enc_out_ring_frames(const voxtral_context * ctx);  // ENC_OUT_RING_CAP
int32_t voxtral_ctx_aemb_ring_frames(const voxtral_context * ctx);     // AEMB_RING_CAP

// Test-only bounded readback used to calculate chunk-plan-invariant SHA-256
// diagnostics.  The caller reads newly produced rows immediately, before the
// fixed device ring can wrap; production does not call these entry points.
bool voxtral_ctx_read_enc_out_ring_internal(
    const voxtral_context * ctx, int64_t absolute_start, int32_t rows, float * dst);
bool voxtral_ctx_read_aemb_ring_internal(
    const voxtral_context * ctx, int64_t absolute_start, int32_t rows, float * dst);

// Enable the per-batch encoder-output ring copy. Off by default so the finish-only
// reference path does no extra work; the incremental stream turns it on before its
// first feed so the adapter can read groups from the device ring.
void voxtral_ctx_set_enc_out_ring_active(voxtral_context * ctx, bool active);

// Read one token's raw UTF-8 byte piece (empty for specials / out-of-range), so the
// streaming layer can build partial text incrementally with the same bytes the batch
// detokenizer (decode_tokens) uses. Borrowed, valid for the model's lifetime.
const std::string & voxtral_token_piece_internal(const voxtral_model * model, int32_t token_id);

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
    int64_t encoderOutputAccumulatedBytes= 0;  // linear accumulated encoder output (host)
    int64_t encoderOutputD2hBytes        = 0;  // actual encoder-output device->host bytes (0 in incremental)
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
// Compile/allocate the exact production encoder, adapter and decoder graph
// families against scratch state, then restore a fresh logical stream. Device
// caches and reusable buffers remain hot; no token/transcript state is created.
bool    voxtral_encoder_stream_warmup(voxtral_encoder_stream * es);

// Push newly-stable frame-major Mel frames [base_frame, base_frame+n) (each n_mel
// contiguous). base_frame must equal the running stable count (contiguous). Runs
// any chunks that became available and accumulates their output. Returns false on
// a contiguity violation or a backend failure.
bool voxtral_encoder_stream_push_mel(voxtral_encoder_stream * es,
                                     const float * frames, int64_t base_frame, int32_t n);
// Push the complete terminal Mel suffix and evaluate it as bounded finish work.
// Unlike ordinary push_mel(), this preserves the terminal batch boundary so the
// encoder can amortise graph submission without replaying prior frames.
bool voxtral_encoder_stream_push_mel_final(voxtral_encoder_stream * es,
                                           const float * frames,
                                           int64_t base_frame,
                                           int32_t n);

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
// True iff this encoder uses the per-layer KV path (which mirrors output into the
// device ring). The incremental adapter/decoder requires it; the reference
// (bounded-window recompute) encoder does not feed the ring.
bool voxtral_encoder_stream_uses_kv(const voxtral_encoder_stream * es);

// Borrowed views of the context's CPU-side frontend tables (stable for the
// context's lifetime). Used by the streaming Mel frontend and test harnesses so
// they compute Mel with the exact same Hann window / mel filterbank as the batch
// path. Return nullptr if unavailable.
const float * voxtral_ctx_hann_window (const voxtral_context * ctx);  // [n_fft]
const float * voxtral_ctx_mel_filters (const voxtral_context * ctx);  // [n_freq*n_mel]

#endif // VOXTRAL_INTERNAL_H
