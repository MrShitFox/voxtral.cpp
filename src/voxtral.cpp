#include "voxtral.h"
#include "voxtral-mel.h"
#include "voxtral-internal.h"
#include "gguf.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Internal constants
// ============================================================================

// STFT / Mel constants live in voxtral-mel.h (VOXTRAL_MEL_*); the DFT + mel
// filterbank + normalization math is shared with the incremental frontend.
static constexpr int32_t VOXTRAL_ENC_CHUNK_MEL     = 3000;  // mel frames per encoder chunk
static constexpr int32_t VOXTRAL_ENC_CHUNK_OVERLAP  = 750;  // overlap in encoder-token space (= window)
static constexpr int32_t VOXTRAL_MAX_ENC_CHUNK      = 2000; // max enc tokens per single chunk

// Per-layer encoder KV-cache (Session 6.1/6.2). MAX_NEW is architectural ring headroom;
// realtime scheduling uses an independent logical microbatch and fixed physical query shape.
// Absolute physical blocks preserve feed-plan-invariant reduction order while allowing
// adapter-aligned 4-frame launches. The ring capacity is window + MAX_NEW so a full
// physical block's K/V write never clobbers a still-live window frame. K/V are
// FP16 in production, with an explicit F32 numerical-oracle override.
static constexpr int32_t VOXTRAL_ENC_KV_MAX_NEW = 128;
static constexpr int32_t VOXTRAL_ENC_KV_CAP     = VOXTRAL_ENC_WINDOW + VOXTRAL_ENC_KV_MAX_NEW; // 878

// Session 7 device-resident incremental adapter/decoder rings. Both are fixed,
// persistent, device-resident, and drained every feed slice so they never grow
// with the utterance. ENC_OUT_RING holds encoder output frames until the adapter
// consumes them in groups of DOWNSAMPLE_FACTOR; its capacity must exceed the
// 128-frame startup burst plus one drain slice's worth, and MUST be a multiple of
// DOWNSAMPLE_FACTOR so a 4-frame adapter group never straddles the ring wrap.
// AEMB_RING holds one audio embedding per group until the frame-synchronous
// decoder consumes it 1:1.
static constexpr int32_t VOXTRAL_ENC_OUT_RING_CAP = 256;   // enc frames; %4 == 0
static constexpr int32_t VOXTRAL_AEMB_RING_CAP    = 256;   // audio embeddings
static_assert(VOXTRAL_ENC_OUT_RING_CAP % VOXTRAL_DOWNSAMPLE_FACTOR == 0,
              "enc-output ring capacity must be a multiple of the adapter group size");
static_assert(VOXTRAL_ENC_OUT_RING_CAP >= VOXTRAL_ENC_KV_MAX_NEW + 64,
              "enc-output ring must absorb the startup burst plus a drain slice");
// Logical scheduling and physical graph shape are deliberately independent.
// The realtime default is the profiled low-query 4/4 segmented graph; 32-row
// flash and 128/128 remain benchmark/reference families. Physical rows are
// capped by the ring headroom.
static constexpr int32_t VOXTRAL_ENC_KV_DEFAULT_LOGICAL  = 4;
static constexpr int32_t VOXTRAL_ENC_KV_DEFAULT_PHYSICAL = 4;
// Below 32 rows the production graph uses segmented explicit attention. At 32+
// the fused Vulkan flash family remains available for throughput/reference work.
static constexpr int32_t VOXTRAL_ENC_KV_ATTN_MIN_ROWS = 32;

struct encoder_kv_schedule {
    int32_t logical  = VOXTRAL_ENC_KV_DEFAULT_LOGICAL;
    int32_t physical = VOXTRAL_ENC_KV_DEFAULT_PHYSICAL;
};

static int32_t env_positive_i32(const char * name, int32_t fallback) {
    const char * value = std::getenv(name);
    if (!value || !*value) return fallback;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int32_t>::max()) {
        return fallback;
    }
    return (int32_t) parsed;
}

static ggml_type kv_type_from_env(const char * name, ggml_type fallback) {
    if (const char * value = std::getenv(name)) {
        if (std::strcmp(value, "f16") == 0 || std::strcmp(value, "F16") == 0) {
            return GGML_TYPE_F16;
        }
        if (std::strcmp(value, "f32") == 0 || std::strcmp(value, "F32") == 0) {
            return GGML_TYPE_F32;
        }
    }
    return fallback;
}

static ggml_type decoder_kv_type_from_env() {
    // Session 8.1 production: decoder FP16 is exact on both real fixtures and
    // retains the dominant KV memory saving. F32 remains the numerical oracle.
    return kv_type_from_env("VOXTRAL_DECODER_KV_TYPE", GGML_TYPE_F16);
}

static ggml_type encoder_kv_type_from_env() {
    // Session 8.1 production: encoder FP16 caused the only measured real-audio
    // token drift; F32 is therefore the quality-preserving default. FP16 stays
    // available for the explicit B/D acceptance variants.
    return kv_type_from_env("VOXTRAL_ENCODER_KV_TYPE", GGML_TYPE_F32);
}

static int32_t normalize_physical_rows(int32_t rows) {
    if (rows <= 4)  return 4;
    if (rows <= 8)  return 8;
    if (rows <= 16) return 16;
    if (rows <= 32) return 32;
    if (rows <= 64) return 64;
    return 128;
}

static bool encoder_kv_uses_segmented_attention(int32_t physical_rows) {
    // Vulkan flash attention remains the throughput/reference graph family.
    // Small fixed query shapes use explicit attention so a wrapped FP16 ring can
    // be consumed as logical segments without materializing a contiguous K/V
    // window.  The env knob keeps the same path available as an oracle at 32+.
    return physical_rows < VOXTRAL_ENC_KV_ATTN_MIN_ROWS ||
           std::getenv("VOXTRAL_ENC_KV_MANUAL") != nullptr;
}

static encoder_kv_schedule encoder_kv_schedule_from_env() {
    encoder_kv_schedule out;
    const char * mode = std::getenv("VOXTRAL_ENC_KV_MODE");
    if (mode && (std::strcmp(mode, "throughput") == 0 || std::strcmp(mode, "offline") == 0)) {
        out.logical = VOXTRAL_ENC_KV_MAX_NEW;
        out.physical = VOXTRAL_ENC_KV_MAX_NEW;
        return out;
    }

    // Keep the old knob source-compatible: values below the physical floor are
    // logical-only tuning, while 64/128 historically meant a full grid.
    const char * legacy = std::getenv("VOXTRAL_ENC_KV_GRID");
    if (legacy && *legacy) {
        const int32_t v = env_positive_i32("VOXTRAL_ENC_KV_GRID", out.logical);
        out.logical = std::min(v, VOXTRAL_ENC_KV_MAX_NEW);
        if (v >= VOXTRAL_ENC_KV_ATTN_MIN_ROWS) out.physical = normalize_physical_rows(v);
    }
    out.logical = std::min(env_positive_i32("VOXTRAL_ENC_KV_LOGICAL_BATCH", out.logical),
                           VOXTRAL_ENC_KV_MAX_NEW);
    out.physical = normalize_physical_rows(env_positive_i32("VOXTRAL_ENC_KV_PHYSICAL_ROWS", out.physical));
    out.logical = std::max<int32_t>(1, out.logical);
    // The physical block must contain an integral number of logical batches so
    // no real query crosses a kernel-shape boundary. Fall back conservatively
    // if a hand-written env combination violates that invariant.
    if (out.logical > out.physical || (out.physical % out.logical) != 0) {
        out.logical = std::min(out.logical, out.physical);
        while (out.logical > 1 && (out.physical % out.logical) != 0) --out.logical;
    }
    return out;
}

// ============================================================================
// Logging helper
// ============================================================================

#define LOG(ctx_ptr, lvl, ...) \
    do { \
        if ((ctx_ptr)->logger && static_cast<int>(lvl) <= static_cast<int>((ctx_ptr)->log_level)) { \
            char _buf[2048]; \
            snprintf(_buf, sizeof(_buf), __VA_ARGS__); \
            (ctx_ptr)->logger(lvl, std::string(_buf)); \
        } \
    } while (0)

#define LOG_INFO(ctx_ptr, ...)  LOG(ctx_ptr, voxtral_log_level::info,  __VA_ARGS__)
#define LOG_WARN(ctx_ptr, ...)  LOG(ctx_ptr, voxtral_log_level::warn,  __VA_ARGS__)
#define LOG_ERR(ctx_ptr, ...)   LOG(ctx_ptr, voxtral_log_level::error, __VA_ARGS__)
#define LOG_DBG(ctx_ptr, ...)   LOG(ctx_ptr, voxtral_log_level::debug, __VA_ARGS__)

// ============================================================================
// Weight structures (internal)
// ============================================================================

struct voxtral_encoder_layer {
    ggml_tensor * attn_norm_weight;  // [enc_dim]
    ggml_tensor * attn_norm_bias = nullptr;  // [enc_dim] — offline LayerNorm only
    ggml_tensor * attn_q_weight;     // [enc_heads*enc_head_dim, enc_dim]
    ggml_tensor * attn_q_bias;       // [enc_heads*enc_head_dim]
    ggml_tensor * attn_k_weight;     // [enc_kv_heads*enc_head_dim, enc_dim]
    ggml_tensor * attn_v_weight;     // [enc_kv_heads*enc_head_dim, enc_dim]
    ggml_tensor * attn_v_bias;       // [enc_kv_heads*enc_head_dim]
    ggml_tensor * attn_o_weight;     // [enc_dim, enc_heads*enc_head_dim]
    ggml_tensor * attn_o_bias;       // [enc_dim]
    ggml_tensor * ffn_norm_weight;   // [enc_dim]
    ggml_tensor * ffn_norm_bias = nullptr;   // [enc_dim] — offline LayerNorm only
    ggml_tensor * ffn_w1_weight;     // [enc_hidden, enc_dim]
    ggml_tensor * ffn_w1_bias = nullptr;     // [enc_hidden] - offline GELU MLP only
    ggml_tensor * ffn_w2_weight;     // [enc_dim, enc_hidden]
    ggml_tensor * ffn_w2_bias;       // [enc_dim]
    ggml_tensor * ffn_w3_weight = nullptr;   // [enc_hidden, enc_dim] — realtime SwiGLU only
};

struct voxtral_decoder_layer {
    ggml_tensor * attn_norm_weight;  // [dec_dim]
    ggml_tensor * attn_q_weight;     // [dec_heads*dec_head_dim, dec_dim]
    ggml_tensor * attn_k_weight;     // [dec_kv_heads*dec_head_dim, dec_dim]
    ggml_tensor * attn_v_weight;     // [dec_kv_heads*dec_head_dim, dec_dim]
    ggml_tensor * attn_o_weight;     // [dec_dim, dec_heads*dec_head_dim]
    ggml_tensor * ffn_norm_weight;   // [dec_dim]
    ggml_tensor * ffn_w1_weight;     // [dec_hidden, dec_dim]
    ggml_tensor * ffn_w2_weight;     // [dec_dim, dec_hidden]
    ggml_tensor * ffn_w3_weight;     // [dec_hidden, dec_dim]
    ggml_tensor * ada0_weight;       // [ada_dim, dec_dim]
    ggml_tensor * ada2_weight;       // [dec_dim, ada_dim]
};

// ============================================================================
// Model structure
// ============================================================================

// Runtime model hyperparameters from GGUF metadata at load time with the VOXTRAL_* as fallbacks.
struct voxtral_hparams {
    bool    is_offline      = false;  // general.architecture == "voxtral" (offline) vs "voxtral_realtime"

    // Encoder
    int32_t enc_dim         = VOXTRAL_ENC_DIM;
    int32_t enc_layers      = VOXTRAL_ENC_LAYERS;
    int32_t enc_heads       = VOXTRAL_ENC_HEADS;
    int32_t enc_head_dim    = VOXTRAL_ENC_HEAD_DIM;
    int32_t enc_hidden      = VOXTRAL_ENC_HIDDEN;
    int32_t enc_kv_heads    = VOXTRAL_ENC_KV_HEADS;
    bool    enc_causal      = true;                 // realtime: causal+window; offline: full bidirectional
    float   enc_norm_eps    = VOXTRAL_ENC_NORM_EPS;
    float   enc_rope_theta  = VOXTRAL_ENC_ROPE_THETA;

    // Decoder
    int32_t dec_dim         = VOXTRAL_DEC_DIM;
    int32_t dec_layers      = VOXTRAL_DEC_LAYERS;
    int32_t dec_heads       = VOXTRAL_DEC_HEADS;
    int32_t dec_head_dim    = VOXTRAL_DEC_HEAD_DIM;
    int32_t dec_hidden      = VOXTRAL_DEC_HIDDEN;
    int32_t dec_kv_heads    = VOXTRAL_DEC_KV_HEADS;
    float   dec_norm_eps    = VOXTRAL_DEC_NORM_EPS;
    float   dec_rope_theta  = VOXTRAL_DEC_ROPE_THETA;
    bool    ada_t_cond      = true;                 // realtime adaptive RMS norm time-conditioning

    int32_t vocab_size      = VOXTRAL_VOCAB_SIZE;

    // Audio
    int32_t downsample_factor = VOXTRAL_DOWNSAMPLE_FACTOR;

    // Special tokens
    int32_t tok_bos         = VOXTRAL_TOKEN_BOS;
    int32_t tok_eos         = VOXTRAL_TOKEN_EOS;
    int32_t tok_audio       = VOXTRAL_TOKEN_AUDIO;
    int32_t tok_begin_audio = VOXTRAL_TOKEN_BEGIN_AUDIO;
    int32_t tok_transcribe  = 34;   // [TRANSCRIBE]; offline prompt only
    int32_t tok_inst        = 3;    // [INST]
    int32_t tok_inst_end    = 4;    // [/INST]
};

struct voxtral_model {
    voxtral_hparams hp;

    // Encoder conv stem
    ggml_tensor * enc_conv0_weight;  // [enc_dim, num_mel_bins, 3]
    ggml_tensor * enc_conv0_bias;    // [enc_dim]
    ggml_tensor * enc_conv1_weight;  // [enc_dim, enc_dim, 3]
    ggml_tensor * enc_conv1_bias;    // [enc_dim]
    std::vector<voxtral_encoder_layer> enc_layers;
    ggml_tensor * enc_norm_weight;   // [enc_dim]
    ggml_tensor * enc_norm_bias = nullptr;  // [enc_dim] — offline LayerNorm only
    ggml_tensor * enc_pos_embedding = nullptr;  // [enc_dim, max_pos] — offline Whisper sinusoids
    ggml_tensor * output_weight = nullptr;  // [vocab, dec_dim] — offline untied output proj

    // Adapter
    ggml_tensor * adapter_0_weight;  // [dec_dim, enc_dim*downsample]
    ggml_tensor * adapter_2_weight;  // [dec_dim, dec_dim]

    // Decoder
    ggml_tensor * tok_embeddings_weight; // [vocab_size, dec_dim]
    std::vector<voxtral_decoder_layer> dec_layers;
    ggml_tensor * dec_norm_weight;   // [dec_dim]

    // Mel filters (stored in GGUF)
    ggml_tensor * mel_filters;       // [n_freq, n_mel] = [201, 128]

    // Tokenizer (Tekken vocab)
    int32_t tokenizer_num_special_tokens = 1000;
    std::unordered_set<int32_t> tokenizer_special_ranks;
    std::vector<std::string> tokenizer_vocab_b64;
    mutable std::unordered_map<int32_t, std::string> tokenizer_bytes_cache;

    // Owning contexts
    ggml_context * ctx_gguf   = nullptr;
    gguf_context * gguf_ctx   = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;
    ggml_backend_t         backend_weights = nullptr;
    bool                   weights_on_gpu = false;
    voxtral_gpu_backend    gpu_type = voxtral_gpu_backend::none;
};

// ============================================================================
// Context structure
// ============================================================================

// Enough for every 80 ms stage invocation in a 30-minute run (22,500 samples)
// with headroom.  Once full, deterministic reservoir replacement keeps the
// percentile sample representative without allocating or growing.
static constexpr size_t VOXTRAL_PROFILE_RESERVOIR = 32768;

struct voxtral_profile_series_internal {
    uint64_t seen   = 0;
    size_t   stored = 0;
    double   total_ms = 0.0;
    double   max_ms   = 0.0;
    std::vector<double> samples;
};

struct voxtral_runtime_profile_internal_state {
    bool enabled = false;
    std::array<voxtral_profile_series_internal,
               static_cast<size_t>(voxtral_profile_stage::count)> stages;
    uint64_t encoder_graph_build_count = 0;
    uint64_t adapter_graph_build_count = 0;
    uint64_t decoder_graph_build_count = 0;
    uint64_t encoder_allocations = 0;
    uint64_t adapter_allocations = 0;
    uint64_t decoder_allocations = 0;
    uint64_t graph_allocations   = 0;
    uint64_t backend_sync_count   = 0;
    uint64_t command_submit_count = 0;
    uint64_t tensor_set_count     = 0;
    uint64_t tensor_get_count     = 0;
    int64_t kv_f16_bytes          = 0;
    int64_t temporary_f32_kv_bytes= 0;
};

struct voxtral_adapter_graph_cache {
    int32_t groups = 0;
    std::vector<uint8_t> meta;
    ggml_context * gctx = nullptr;
    ggml_cgraph * graph = nullptr;
    bool allocated = false;
    std::vector<int32_t> encoder_rows;
    std::vector<int32_t> audio_rows;
};

struct voxtral_context {
    voxtral_model        * model     = nullptr;
    voxtral_log_level      log_level = voxtral_log_level::info;
    voxtral_log_callback   logger    = nullptr;
    int32_t                n_threads = 4;

    // Backend
    ggml_backend_t         backend      = nullptr;
    ggml_backend_t         backend_cpu  = nullptr;
    ggml_backend_t         blas_backend = nullptr;
    voxtral_gpu_backend    gpu_type     = voxtral_gpu_backend::none;

    // Persistent device tensors (allocated once)
    ggml_context       * ctx_persistent = nullptr;
    ggml_backend_buffer_t buf_persistent = nullptr;

    // Per-chunk encoder output (fixed size, reused each chunk)
    ggml_tensor * encoder_chunk_output = nullptr;  // [enc_dim, MAX_ENC_CHUNK]
    ggml_tensor * decoder_logits  = nullptr;  // [vocab_size]
    ggml_tensor * decoder_argmax  = nullptr;  // [1] i32 — greedy token, computed on device
    ggml_tensor * decoder_hidden_diagnostic = nullptr; // [dec_dim] F32, opt-in test capture

    // KV cache: [kv_heads*head_dim, dec_window, dec_layers]
    ggml_tensor * kv_self_k       = nullptr;
    ggml_tensor * kv_self_v       = nullptr;

    // Full accumulated encoder output (dynamic, allocated per utterance ON DEVICE)
    ggml_context       * ctx_enc_full = nullptr;
    ggml_backend_buffer_t buf_enc_full = nullptr;
    ggml_tensor        * encoder_output = nullptr;  // [enc_dim, total_enc_tokens]
    int32_t total_enc_tokens = 0;

    // Dynamic decoder memory (allocated per utterance ON DEVICE). Used by the
    // batch/offline/finish-only paths, which materialize the whole utterance.
    ggml_context       * ctx_dec_mem = nullptr;
    ggml_backend_buffer_t buf_dec_mem = nullptr;
    ggml_tensor        * decoder_memory = nullptr;  // [dec_dim, dec_seq]

    // Session 7: device-resident incremental adapter/decoder rings (persistent,
    // fixed capacity). enc_out_ring accumulates encoder output frames on device so
    // the incremental adapter reads complete groups without any D2H; audio_emb_ring
    // holds the adapter's audio embeddings until the incremental decoder consumes
    // them. The realtime decoder graphs read audio embeddings from dec_audio_src:
    // when dec_audio_cap>0 it is audio_emb_ring and positions index modulo capacity;
    // when null/0 they fall back to the linear decoder_memory (batch/finish path).
    // Kept in their OWN device buffer (NOT buf_persistent): clear_kv_cache() clears
    // the whole persistent buffer, which must not wipe live encoder output / audio
    // embeddings when the incremental decoder resets its KV.
    ggml_context       * ctx_rings      = nullptr;
    ggml_backend_buffer_t buf_rings     = nullptr;
    ggml_tensor        * enc_out_ring   = nullptr;  // [enc_dim, ENC_OUT_RING_CAP]
    ggml_tensor        * audio_emb_ring = nullptr;  // [dec_dim, AEMB_RING_CAP]
    ggml_tensor        * dec_audio_src  = nullptr;  // audio-embedding source for realtime decoder graphs
    int32_t              dec_audio_cap  = 0;        // >0 => ring modulo indexing
    bool                 want_enc_out_ring = false; // incremental stream mirrors enc output into the ring

    // Actual sizes (set per utterance)
    int32_t enc_seq_len  = 0;  // after conv, before left-trunc
    int32_t enc_seq_used = 0;  // after left-trunc (multiple of downsample_factor)
    int32_t dec_seq_len  = 0;  // adapter output length
    int64_t encoder_kv_allocated_bytes = 0;
    ggml_type encoder_kv_storage_type = GGML_TYPE_COUNT;

    // Decoder KV is a true fixed-size ring. The previous shift-left path evicted
    // position zero along with every other oldest row, so the accepted semantics
    // are fully sliding: there is no pinned prompt prefix or attention sink.
    // Absolute positions drive RoPE and eviction; physical slots are
    // absolute_position % capacity. No rollover path shifts or clears K/V.
    voxtral_decoder_kv_ring decoder_kv = {
        /*capacity=*/ VOXTRAL_DEC_WINDOW,
        /*used=*/ 0,
        /*oldest_absolute_position=*/ 0,
        /*next_absolute_position=*/ 0,
    };
    // Production is always the model window. A smaller value is accepted only
    // through the explicitly named test seam so rollover kernels can be
    // exercised quickly without changing the backing allocation.
    int32_t decoder_kv_configured_capacity = VOXTRAL_DEC_WINDOW;

    struct decoder_rollover_profile {
        static constexpr size_t span = 63; // 5.04 s at the 80 ms decoder cadence
        std::array<double, span> recent{};
        size_t recent_count = 0;
        size_t recent_next = 0;
        std::array<double, span> post{};
        size_t post_count = 0;
        int64_t first_wrap_position = -1;
        double pre_wrap_p99_ms = 0.0;
        double wrap_step_ms = 0.0;
    } decoder_rollover;

    // Schedulers
    ggml_backend_sched_t sched_encoder  = nullptr;
    ggml_backend_sched_t sched_encoder_steady = nullptr;
    ggml_backend_sched_t sched_adapter  = nullptr;
    ggml_backend_sched_t sched_adapter_group = nullptr;
    ggml_backend_sched_t sched_adapter_batch = nullptr;
    ggml_backend_sched_t sched_dec_pre  = nullptr;
    ggml_backend_sched_t sched_dec_step = nullptr;

    // Reusable single-token decoder graph. Slot selection is data (I32 inputs),
    // not graph topology: SET_ROWS appends K/V at the current physical slot and
    // GET_ROWS selects the audio ring slot. One bounded masked-to-unmasked graph
    // transition occurs when the cache fills; neither topology grows per step.
    std::vector<uint8_t> decoder_step_graph_meta;
    ggml_context       * decoder_step_graph_ctx = nullptr;
    ggml_cgraph        * decoder_step_graph = nullptr;
    bool                 decoder_step_graph_allocated = false;
    bool                 decoder_step_graph_full = false;
    std::vector<float>   decoder_step_mask;
    int32_t              decoder_step_mask_valid = -1;
    voxtral_adapter_graph_cache adapter_group_graph;
    voxtral_adapter_graph_cache adapter_batch_graph;
    std::vector<uint8_t> encoder_steady_graph_meta;
    ggml_context       * encoder_steady_graph_ctx = nullptr;
    ggml_cgraph        * encoder_steady_graph = nullptr;
    bool                 encoder_steady_graph_allocated = false;
    std::vector<float>   encoder_steady_mask;
    std::array<int32_t, 4> encoder_steady_positions{};
    std::array<int32_t, 4> encoder_steady_kv_rows{};
    std::array<int32_t, 4> encoder_steady_output_rows{};

    // CPU scratch
    std::vector<float> hann_window;     // [window_size]
    std::vector<float> mel_filters_cpu; // [n_freq * n_mel]
    std::vector<float> time_emb_cpu;    // [dec_dim]

    // Optional, fixed-capacity Session-8 stage profiler.  No hot-path sample
    // recording allocates after context initialization.
    voxtral_runtime_profile_internal_state profile;

    // Explicit numerical-acceptance mode. Production never reads these tensors
    // back; the first decoder step only is captured when the env flag is set.
    bool numerical_diagnostics = false;
    bool capture_encoder_diagnostics = false;
    bool capture_adapter_diagnostics = false;
    bool capture_decoder_diagnostics = false;
    std::vector<float> diagnostic_encoder_output;
    std::vector<float> diagnostic_first_hidden;
    std::vector<float> diagnostic_first_logits;
    std::vector<float> diagnostic_adapter_output;
};

static size_t profile_index(voxtral_profile_stage stage) {
    return static_cast<size_t>(stage);
}

const char * voxtral_profile_stage_name(voxtral_profile_stage stage) {
    switch (stage) {
        case voxtral_profile_stage::mel_compute:                    return "mel_compute";
        case voxtral_profile_stage::encoder_graph_build:            return "encoder_graph_build";
        case voxtral_profile_stage::encoder_graph_execute:          return "encoder_graph_execute";
        case voxtral_profile_stage::encoder_device_copy:            return "encoder_device_copy";
        case voxtral_profile_stage::adapter_graph_build:             return "adapter_graph_build";
        case voxtral_profile_stage::adapter_graph_execute:           return "adapter_graph_execute";
        case voxtral_profile_stage::decoder_prefill_graph_build:     return "decoder_prefill_graph_build";
        case voxtral_profile_stage::decoder_prefill_graph_execute:   return "decoder_prefill_graph_execute";
        case voxtral_profile_stage::decoder_step_graph_build:        return "decoder_step_graph_build";
        case voxtral_profile_stage::decoder_step_graph_execute:      return "decoder_step_graph_execute";
        case voxtral_profile_stage::argmax:                          return "argmax";
        case voxtral_profile_stage::token_readback:                  return "token_readback";
        case voxtral_profile_stage::backend_synchronize:             return "backend_synchronize";
        case voxtral_profile_stage::event_processing:                return "event_processing";
        case voxtral_profile_stage::pipeline_feed:                   return "pipeline_feed";
        case voxtral_profile_stage::count:                           break;
    }
    return "unknown";
}

static uint64_t profile_mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

void voxtral_context_profile_record_internal(voxtral_context * ctx,
                                             voxtral_profile_stage stage,
                                             double milliseconds) {
    if (!ctx || !ctx->profile.enabled || stage == voxtral_profile_stage::count) return;
    if (!std::isfinite(milliseconds) || milliseconds < 0.0) return;
    auto & s = ctx->profile.stages[profile_index(stage)];
    ++s.seen;
    s.total_ms += milliseconds;
    s.max_ms = std::max(s.max_ms, milliseconds);
    if (s.stored < s.samples.size()) {
        s.samples[s.stored++] = milliseconds;
    } else if (!s.samples.empty()) {
        const uint64_t slot = profile_mix64(s.seen) % s.seen;
        if (slot < s.samples.size()) s.samples[(size_t) slot] = milliseconds;
    }

    switch (stage) {
        case voxtral_profile_stage::encoder_graph_build:
            ++ctx->profile.encoder_graph_build_count; break;
        case voxtral_profile_stage::adapter_graph_build:
            ++ctx->profile.adapter_graph_build_count; break;
        case voxtral_profile_stage::decoder_prefill_graph_build:
        case voxtral_profile_stage::decoder_step_graph_build:
            ++ctx->profile.decoder_graph_build_count; break;
        case voxtral_profile_stage::backend_synchronize:
            ++ctx->profile.backend_sync_count; break;
        default: break;
    }
}

void voxtral_context_profile_note_allocation_internal(voxtral_context * ctx,
                                                      voxtral_profile_stage stage) {
    if (!ctx || !ctx->profile.enabled) return;
    ++ctx->profile.graph_allocations;
    switch (stage) {
        case voxtral_profile_stage::encoder_graph_execute:
        case voxtral_profile_stage::encoder_device_copy:
            ++ctx->profile.encoder_allocations; break;
        case voxtral_profile_stage::adapter_graph_execute:
            ++ctx->profile.adapter_allocations; break;
        case voxtral_profile_stage::decoder_prefill_graph_execute:
        case voxtral_profile_stage::decoder_step_graph_execute:
            ++ctx->profile.decoder_allocations; break;
        default: break;
    }
}

void voxtral_context_profile_note_submit_internal(voxtral_context * ctx) {
    if (ctx && ctx->profile.enabled) ++ctx->profile.command_submit_count;
}

void voxtral_context_profile_note_tensor_set_internal(voxtral_context * ctx) {
    if (ctx && ctx->profile.enabled) ++ctx->profile.tensor_set_count;
}

void voxtral_context_profile_note_tensor_get_internal(voxtral_context * ctx) {
    if (ctx && ctx->profile.enabled) ++ctx->profile.tensor_get_count;
}

void voxtral_context_profile_reset_internal(voxtral_context * ctx) {
    if (!ctx) return;
    auto & p = ctx->profile;
    p.encoder_graph_build_count = 0;
    p.adapter_graph_build_count = 0;
    p.decoder_graph_build_count = 0;
    p.encoder_allocations = p.adapter_allocations = p.decoder_allocations = 0;
    p.graph_allocations = p.backend_sync_count = p.command_submit_count = 0;
    p.tensor_set_count = p.tensor_get_count = 0;
    p.kv_f16_bytes = 0;
    p.temporary_f32_kv_bytes = 0;
    for (auto & s : p.stages) {
        s.seen = 0;
        s.stored = 0;
        s.total_ms = 0.0;
        s.max_ms = 0.0;
        if (p.enabled) {
            if (s.samples.size() != VOXTRAL_PROFILE_RESERVOIR) {
                s.samples.assign(VOXTRAL_PROFILE_RESERVOIR, 0.0);
            } else {
                std::fill(s.samples.begin(), s.samples.end(), 0.0);
            }
        } else {
            s.samples.clear();
        }
    }
}

static double profile_percentile(const voxtral_profile_series_internal & s, double p) {
    if (s.stored == 0) return 0.0;
    std::vector<double> values(s.samples.begin(), s.samples.begin() + (std::ptrdiff_t) s.stored);
    std::sort(values.begin(), values.end());
    const double x = p * (double) (values.size() - 1);
    const size_t lo = (size_t) x;
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (x - (double) lo);
}

template <size_t N>
static double fixed_percentile(const std::array<double, N> & source,
                               size_t count, double p) {
    count = std::min(count, N);
    if (count == 0) return 0.0;
    std::vector<double> values(source.begin(), source.begin() + (std::ptrdiff_t) count);
    std::sort(values.begin(), values.end());
    const double x = p * (double) (values.size() - 1);
    const size_t lo = (size_t) x;
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (x - (double) lo);
}

voxtral_runtime_profile voxtral_context_runtime_profile_internal(const voxtral_context * ctx) {
    voxtral_runtime_profile out;
    if (!ctx) return out;
    const auto & p = ctx->profile;
    out.enabled = p.enabled;
    for (size_t i = 0; i < p.stages.size(); ++i) {
        const auto & src = p.stages[i];
        auto & dst = out.stages[i];
        dst.count = src.seen;
        dst.totalMs = src.total_ms;
        dst.meanMs = src.seen ? src.total_ms / (double) src.seen : 0.0;
        dst.p50Ms = profile_percentile(src, 0.50);
        dst.p95Ms = profile_percentile(src, 0.95);
        dst.p99Ms = profile_percentile(src, 0.99);
        dst.maxMs = src.max_ms;
    }
    out.encoderGraphBuildCount = p.encoder_graph_build_count;
    out.adapterGraphBuildCount = p.adapter_graph_build_count;
    out.decoderGraphBuildCount = p.decoder_graph_build_count;
    out.encoderAllocations = p.encoder_allocations;
    out.adapterAllocations = p.adapter_allocations;
    out.decoderAllocations = p.decoder_allocations;
    out.graphAllocations = p.graph_allocations;
    out.backendSyncCount = p.backend_sync_count;
    out.commandSubmitCount = p.command_submit_count;
    out.tensorSetCount = p.tensor_set_count;
    out.tensorGetCount = p.tensor_get_count;
    out.kvF16Bytes = p.kv_f16_bytes;
    if (ctx->kv_self_k && ctx->kv_self_v &&
        ctx->kv_self_k->type == GGML_TYPE_F16 &&
        ctx->kv_self_v->type == GGML_TYPE_F16) {
        out.kvF16Bytes += (int64_t) ggml_nbytes(ctx->kv_self_k) +
                          (int64_t) ggml_nbytes(ctx->kv_self_v);
    }
    if (ctx->encoder_kv_storage_type == GGML_TYPE_F16) {
        out.kvF16Bytes += ctx->encoder_kv_allocated_bytes;
    }
    out.temporaryF32KvBytes = p.temporary_f32_kv_bytes;
    out.decoderKvCapacity = ctx->decoder_kv.capacity;
    out.decoderKvUsed = ctx->decoder_kv.used;
    out.decoderKvWraps = ctx->decoder_kv.wraps;
    out.decoderKvEvictions = ctx->decoder_kv.evictions;
    out.decoderKvBytesMoved = ctx->decoder_kv.bytes_moved;
    out.decoderKvFullBufferMoves = ctx->decoder_kv.full_buffer_moves;
    out.decoderOldestAbsolutePosition = ctx->decoder_kv.oldest_absolute_position;
    out.decoderNextAbsolutePosition = ctx->decoder_kv.next_absolute_position;
    out.decoderKvElementSize = ctx->kv_self_k
        ? (int32_t) ggml_type_size(ctx->kv_self_k->type) : 0;
    out.decoderFirstWrapAbsolutePosition =
        ctx->decoder_rollover.first_wrap_position;
    out.decoderPreWrapP99Ms = ctx->decoder_rollover.pre_wrap_p99_ms;
    out.decoderWrapStepMs = ctx->decoder_rollover.wrap_step_ms;
    out.decoderPostWrapP99Ms = fixed_percentile(
        ctx->decoder_rollover.post, ctx->decoder_rollover.post_count, 0.99);
    const auto stage_total = [&](voxtral_profile_stage stage) {
        return p.stages[profile_index(stage)].total_ms;
    };
    out.totalGpuComputeMs =
        stage_total(voxtral_profile_stage::encoder_graph_execute) +
        stage_total(voxtral_profile_stage::encoder_device_copy) +
        stage_total(voxtral_profile_stage::adapter_graph_execute) +
        stage_total(voxtral_profile_stage::decoder_prefill_graph_execute) +
        stage_total(voxtral_profile_stage::decoder_step_graph_execute);
    out.totalPipelineComputeMs = stage_total(voxtral_profile_stage::pipeline_feed);
    return out;
}

// ============================================================================
// Time embedding (sinusoidal, matches Python compute_time_embedding)
// ============================================================================

static void compute_time_embedding(std::vector<float> & out, float t, int32_t dim) {
    // Python: inv_freq = exp(-log(10000) * arange(half) / half)
    //         emb = t * inv_freq;  return cat([cos(emb), sin(emb)])
    out.resize(dim);
    const int32_t half = dim / 2;
    for (int32_t i = 0; i < half; i++) {
        const float inv_freq = expf(-logf(10000.0f) * (float)i / (float)half);
        const float angle = t * inv_freq;
        out[i]        = cosf(angle);   // cos first half
        out[i + half] = sinf(angle);   // sin second half
    }
}

static double elapsed_ms(const std::chrono::steady_clock::time_point & t0) {
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ============================================================================
// Token decode helpers (Tekken vocab from GGUF metadata)
// ============================================================================

static std::vector<uint8_t> base64_decode(const std::string & in) {
    static const std::array<int8_t, 256> table = [] {
        std::array<int8_t, 256> t{};
        t.fill(-1);
        for (int c = 'A'; c <= 'Z'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(c - 'A');
        for (int c = 'a'; c <= 'z'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(26 + (c - 'a'));
        for (int c = '0'; c <= '9'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(52 + (c - '0'));
        t[static_cast<size_t>('+')] = 62;
        t[static_cast<size_t>('/')] = 63;
        return t;
    }();

    std::vector<uint8_t> out;
    out.reserve((in.size() * 3) / 4 + 4);

    uint32_t acc = 0;
    int bits = 0;

    for (char ch : in) {
        if (ch == '=') {
            break;
        }

        const uint8_t uch = static_cast<uint8_t>(ch);
        const int8_t val = table[uch];
        if (val < 0) {
            continue;
        }

        acc = (acc << 6) | static_cast<uint32_t>(val);
        bits += 6;

        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((acc >> bits) & 0xFF));
        }
    }

    return out;
}

static const std::string & token_bytes_for_id(const voxtral_model & model, int32_t token_id) {
    auto it_cached = model.tokenizer_bytes_cache.find(token_id);
    if (it_cached != model.tokenizer_bytes_cache.end()) {
        return it_cached->second;
    }

    std::string decoded;
    if (token_id >= 0 &&
        token_id >= model.tokenizer_num_special_tokens &&
        model.tokenizer_special_ranks.find(token_id) == model.tokenizer_special_ranks.end()) {
        const int64_t vocab_id = static_cast<int64_t>(token_id) -
                                 static_cast<int64_t>(model.tokenizer_num_special_tokens);
        if (vocab_id >= 0 && vocab_id < static_cast<int64_t>(model.tokenizer_vocab_b64.size())) {
            const std::vector<uint8_t> bytes =
                base64_decode(model.tokenizer_vocab_b64[static_cast<size_t>(vocab_id)]);
            decoded.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        }
    }

    auto [it_new, _] = model.tokenizer_bytes_cache.emplace(token_id, std::move(decoded));
    return it_new->second;
}

static std::string decode_tokens(const voxtral_model & model, const std::vector<int32_t> & tokens) {
    if (model.tokenizer_vocab_b64.empty()) {
        return {};
    }

    std::string out;
    out.reserve(tokens.size() * 3);

    for (int32_t token : tokens) {
        if (token < model.tokenizer_num_special_tokens) {
            continue;
        }
        if (model.tokenizer_special_ranks.find(token) != model.tokenizer_special_ranks.end()) {
            continue;
        }

        const std::string & token_bytes = token_bytes_for_id(model, token);
        out.append(token_bytes);
    }

    return out;
}

// Session 7: one token's raw UTF-8 byte piece for the incremental detokenizer.
// token_bytes_for_id already returns an empty string for special / out-of-range
// ids, so appending this for every emitted token reproduces decode_tokens()
// byte-for-byte (which likewise contributes nothing for those ids).
const std::string & voxtral_token_piece_internal(const voxtral_model * model, int32_t token_id) {
    static const std::string empty;
    if (!model || model->tokenizer_vocab_b64.empty()) return empty;
    return token_bytes_for_id(*model, token_id);
}

// ============================================================================
// WAV file loading (16-bit PCM or 32-bit float, mono/stereo)
// ============================================================================

static bool load_wav_file(const std::string & path, std::vector<float> & audio_out) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) return false;

    // RIFF header
    char riff[4]; fin.read(riff, 4);
    if (memcmp(riff, "RIFF", 4) != 0) return false;

    uint32_t chunk_size; fin.read(reinterpret_cast<char*>(&chunk_size), 4);
    char wave[4]; fin.read(wave, 4);
    if (memcmp(wave, "WAVE", 4) != 0) return false;

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, data_size = 0;
    bool found_fmt = false, found_data = false;

    while (fin.good() && !(found_fmt && found_data)) {
        char sub_id[4]; fin.read(sub_id, 4);
        uint32_t sub_size; fin.read(reinterpret_cast<char*>(&sub_size), 4);
        if (!fin.good()) break;

        if (memcmp(sub_id, "fmt ", 4) == 0) {
            fin.read(reinterpret_cast<char*>(&audio_format),    2);
            fin.read(reinterpret_cast<char*>(&num_channels),    2);
            fin.read(reinterpret_cast<char*>(&sample_rate),     4);
            uint32_t byte_rate; fin.read(reinterpret_cast<char*>(&byte_rate), 4);
            uint16_t block_align; fin.read(reinterpret_cast<char*>(&block_align), 2);
            fin.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            if (sub_size > 16) fin.seekg(sub_size - 16, std::ios::cur);
            found_fmt = true;
        } else if (memcmp(sub_id, "data", 4) == 0) {
            data_size = sub_size;
            found_data = true;
        } else {
            fin.seekg(sub_size, std::ios::cur);
        }
    }

    if (!found_fmt || !found_data) return false;
    if (audio_format != 1 && audio_format != 3) return false; // 1=PCM, 3=IEEE float

    const int32_t n_samples_total = data_size / (bits_per_sample / 8);
    const int32_t n_samples = n_samples_total / num_channels;

    if (audio_format == 1 && bits_per_sample == 16) {
        std::vector<int16_t> raw(n_samples_total);
        fin.read(reinterpret_cast<char*>(raw.data()), data_size);
        audio_out.resize(n_samples);
        for (int32_t i = 0; i < n_samples; i++) {
            float sum = 0.0f;
            for (int32_t c = 0; c < num_channels; c++) {
                sum += (float)raw[i * num_channels + c] / 32768.0f;
            }
            audio_out[i] = sum / num_channels;
        }
    } else if (audio_format == 3 && bits_per_sample == 32) {
        std::vector<float> raw(n_samples_total);
        fin.read(reinterpret_cast<char*>(raw.data()), data_size);
        audio_out.resize(n_samples);
        for (int32_t i = 0; i < n_samples; i++) {
            float sum = 0.0f;
            for (int32_t c = 0; c < num_channels; c++) {
                sum += raw[i * num_channels + c];
            }
            audio_out[i] = sum / num_channels;
        }
    } else {
        return false;
    }

    return true;
}

// Batch Mel spectrogram: thin wrapper over the shared frontend kernel. See
// voxtral_mel_compute_batch() in voxtral-mel.cpp for the (unchanged) STFT math.

// ============================================================================
// GGUF tensor loading helper
// ============================================================================

static ggml_tensor * get_tensor(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "voxtral: tensor '%s' not found in GGUF\n", name);
    }
    return t;
}

// Optional tensor: returns nullptr without warning if absent (arch-dependent tensors).
static ggml_tensor * get_tensor_opt(ggml_context * ctx, const char * name) {
    return ggml_get_tensor(ctx, name);
}

// ============================================================================
// Backend selection via the ggml backend registry
// ============================================================================

// Case-insensitive substring test (registry names vary by ggml version/build,
// e.g. the Metal backend is "MTL" in the bundled build but "Metal" elsewhere).
static bool name_has(const char * hay, const char * needle_lc) {
    if (!hay) return false;
    std::string h(hay);
    for (char & c : h) if (c >= 'A' && c <= 'Z') c += 32;
    return h.find(needle_lc) != std::string::npos;
}

// Does a registry name belong to the requested GPU family? (auto matches any GPU.)
static bool gpu_reg_matches(const char * rn, voxtral_gpu_backend req) {
    switch (req) {
        case voxtral_gpu_backend::metal:  return name_has(rn, "metal") || name_has(rn, "mtl");
        case voxtral_gpu_backend::cuda:   return name_has(rn, "cuda");
        case voxtral_gpu_backend::vulkan: return name_has(rn, "vulkan") || name_has(rn, "vk");
        default:                          return true; // auto_detect
    }
}

// Map a registry name back to our backend enum.
static voxtral_gpu_backend gpu_from_reg(const char * rn) {
    if (name_has(rn, "cuda")) return voxtral_gpu_backend::cuda;
    if (name_has(rn, "vulkan") || name_has(rn, "vk")) return voxtral_gpu_backend::vulkan;
    return voxtral_gpu_backend::metal; // metal/mtl, or a generic GPU
}

// First GPU/IGPU device matching the requested family (any GPU for auto_detect).
static ggml_backend_dev_t find_gpu_device(voxtral_gpu_backend req) {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (gpu_reg_matches(reg ? ggml_backend_reg_name(reg) : "", req)) {
            return dev;
        }
    }
    return nullptr;
}

// Find an accelerator device (e.g. BLAS/Accelerate) used alongside the CPU.
static ggml_backend_dev_t find_accel_device() {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            return dev;
        }
    }
    return nullptr;
}

// Set the thread count on a backend via the registry proc-address (no-op if the
// backend does not support it, e.g. GPU backends).
static void backend_set_threads(ggml_backend_t backend, int n_threads) {
    if (!backend) return;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) return;
    auto set_threads = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_threads) set_threads(backend, n_threads);
}

// Initialize the CPU backend via the registry.
static ggml_backend_t init_cpu_backend() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    return dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
}

// Load all dynamically-linked ggml backend modules into the registry, exactly
// once. Required when linking a system ggml (its backends are separate modules);
// a no-op for the bundled static build whose backends self-register.
static void ensure_backends_loaded() {
    static const bool loaded = [] { ggml_backend_load_all(); return true; }();
    (void) loaded;
}

// ==========================================================================
// Model loading
// ============================================================================

voxtral_model * voxtral_model_load_from_file(
    const std::string    & path,
    voxtral_log_callback   logger,
    voxtral_gpu_backend    gpu)
{
    ensure_backends_loaded();
    auto log_info = [&](const std::string & msg) {
        if (logger) logger(voxtral_log_level::info, msg);
    };

    const auto t_load_start = std::chrono::steady_clock::now();
    log_info("loading model from " + path);

    ggml_context * ctx_meta = nullptr;
    gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &ctx_meta,
    };

    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "voxtral: failed to open GGUF file: %s\n", path.c_str());
        return nullptr;
    }

    voxtral_model * model = new voxtral_model();
    model->gguf_ctx  = gguf_ctx;
    model->ctx_gguf  = ctx_meta;

    // ---- Populate runtime hyperparameters from GGUF metadata (fallback to defaults) ----
    {
        voxtral_hparams & hp = model->hp;
        auto gi32 = [&](const char * k, int32_t & dst) {
            const int64_t kid = gguf_find_key(gguf_ctx, k);
            if (kid >= 0) dst = gguf_get_val_i32(gguf_ctx, kid);
        };
        auto gf32 = [&](const char * k, float & dst) {
            const int64_t kid = gguf_find_key(gguf_ctx, k);
            if (kid >= 0) dst = gguf_get_val_f32(gguf_ctx, kid);
        };
        auto gbool = [&](const char * k, bool & dst) {
            const int64_t kid = gguf_find_key(gguf_ctx, k);
            if (kid >= 0) dst = gguf_get_val_bool(gguf_ctx, kid);
        };

        const int64_t arch_kid = gguf_find_key(gguf_ctx, "general.architecture");
        if (arch_kid >= 0) {
            const std::string arch = gguf_get_val_str(gguf_ctx, arch_kid);
            hp.is_offline = (arch == "voxtral");          // offline arch string
        }

        gi32("voxtral.encoder.dim",          hp.enc_dim);
        gi32("voxtral.encoder.n_layers",     hp.enc_layers);
        gi32("voxtral.encoder.n_heads",      hp.enc_heads);
        gi32("voxtral.encoder.head_dim",     hp.enc_head_dim);
        gi32("voxtral.encoder.hidden_dim",   hp.enc_hidden);
        gi32("voxtral.encoder.n_kv_heads",   hp.enc_kv_heads);
        gf32("voxtral.encoder.norm_eps",     hp.enc_norm_eps);
        gf32("voxtral.encoder.rope_theta",   hp.enc_rope_theta);
        // encoder causality: explicit key if present, else from params.*, else infer from window
        hp.enc_causal = !hp.is_offline;
        gbool("voxtral.encoder.causal", hp.enc_causal);
        gbool("voxtral.params.multimodal.whisper_model_args.encoder_args.causal", hp.enc_causal);

        gi32("voxtral.decoder.dim",          hp.dec_dim);
        gi32("voxtral.decoder.n_layers",     hp.dec_layers);
        gi32("voxtral.decoder.n_heads",      hp.dec_heads);
        gi32("voxtral.decoder.head_dim",     hp.dec_head_dim);
        gi32("voxtral.decoder.hidden_dim",   hp.dec_hidden);
        gi32("voxtral.decoder.n_kv_heads",   hp.dec_kv_heads);
        gf32("voxtral.decoder.norm_eps",     hp.dec_norm_eps);
        gf32("voxtral.decoder.rope_theta",   hp.dec_rope_theta);

        hp.ada_t_cond = !hp.is_offline;
        gbool("voxtral.ada_rms_norm_t_cond", hp.ada_t_cond);

        gi32("voxtral.vocab_size",           hp.vocab_size);
        gi32("voxtral.audio.downsample_factor", hp.downsample_factor);
        gi32("voxtral.token.bos",            hp.tok_bos);
        gi32("voxtral.token.eos",            hp.tok_eos);
        gi32("voxtral.token.audio",          hp.tok_audio);
        gi32("voxtral.token.begin_audio",    hp.tok_begin_audio);
    }

    // Allocate a backend buffer for all the weights. The GPU backend (if any) is
    // selected from the ggml backend registry; weights are allocated on it.
    ggml_backend_t weights_backend = nullptr;
    voxtral_gpu_backend resolved_gpu = voxtral_gpu_backend::none;

    if (gpu != voxtral_gpu_backend::none) {
        ggml_backend_dev_t gdev = find_gpu_device(gpu);
        if (gdev) {
            weights_backend = ggml_backend_dev_init(gdev, nullptr);
        }
        if (weights_backend) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(gdev);
            resolved_gpu = gpu_from_reg(reg ? ggml_backend_reg_name(reg) : "");
        } else {
            log_info("no GPU backend available, using CPU");
        }
    }

    if (!weights_backend) {
        weights_backend = init_cpu_backend();
    }

    model->backend_weights = weights_backend;
    model->weights_on_gpu = (resolved_gpu != voxtral_gpu_backend::none);
    model->gpu_type = resolved_gpu;
    model->buf_weights = ggml_backend_alloc_ctx_tensors(ctx_meta, weights_backend);

    if (!model->buf_weights) {
        fprintf(stderr, "voxtral: failed to allocate weight buffer\n");
        if (model->backend_weights) {
            ggml_backend_free(model->backend_weights);
            model->backend_weights = nullptr;
        }
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        delete model;
        return nullptr;
    }

    // Load tensor data from file into buffer
    {
        FILE * fp = fopen(path.c_str(), "rb");
        if (!fp) {
            fprintf(stderr, "voxtral: failed to open file for reading weights\n");
            voxtral_model_free(model);
            return nullptr;
        }

        const int n_tensors = gguf_get_n_tensors(gguf_ctx);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
            if (!t) continue;

            const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);
            const size_t nbytes = ggml_nbytes(t);

            std::vector<uint8_t> tmp(nbytes);
            fseek(fp, (long)offset, SEEK_SET);
            if (fread(tmp.data(), 1, nbytes, fp) != nbytes) {
                fprintf(stderr, "voxtral: failed to read tensor '%s'\n", name);
                fclose(fp);
                voxtral_model_free(model);
                return nullptr;
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
        }
        fclose(fp);
    }

    // Map weight tensors
    model->enc_conv0_weight = get_tensor(ctx_meta, "enc.conv0.weight");
    model->enc_conv0_bias   = get_tensor(ctx_meta, "enc.conv0.bias");
    model->enc_conv1_weight = get_tensor(ctx_meta, "enc.conv1.weight");
    model->enc_conv1_bias   = get_tensor(ctx_meta, "enc.conv1.bias");
    model->enc_norm_weight  = get_tensor(ctx_meta, "enc.norm.weight");
    model->enc_norm_bias    = get_tensor_opt(ctx_meta, "enc.norm.bias");   // offline LayerNorm
    model->enc_pos_embedding = get_tensor_opt(ctx_meta, "enc.pos_embedding"); // offline Whisper sinusoids
    model->output_weight    = get_tensor_opt(ctx_meta, "output.weight");   // offline untied output

    model->enc_layers.resize(model->hp.enc_layers);
    for (int32_t i = 0; i < model->hp.enc_layers; i++) {
        char nm[256];
        auto & L = model->enc_layers[i];
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_norm.weight",i); L.attn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_q.weight",i);    L.attn_q_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_q.bias",i);      L.attn_q_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_k.weight",i);    L.attn_k_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_v.weight",i);    L.attn_v_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_v.bias",i);      L.attn_v_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_o.weight",i);    L.attn_o_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_o.bias",i);      L.attn_o_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_norm.weight",i);  L.ffn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w1.weight",i);    L.ffn_w1_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w2.weight",i);    L.ffn_w2_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w2.bias",i);      L.ffn_w2_bias   = get_tensor(ctx_meta,nm);
        if (model->hp.is_offline) {
            // Offline Whisper encoder: LayerNorm biases + GELU-MLP w1 bias (no SwiGLU w3).
            snprintf(nm,sizeof(nm),"enc.blk.%d.attn_norm.bias",i); L.attn_norm_bias = get_tensor_opt(ctx_meta,nm);
            snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_norm.bias",i);  L.ffn_norm_bias  = get_tensor_opt(ctx_meta,nm);
            snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w1.bias",i);    L.ffn_w1_bias    = get_tensor_opt(ctx_meta,nm);
        } else {
            snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w3.weight",i);  L.ffn_w3_weight  = get_tensor(ctx_meta,nm);
        }
    }

    model->adapter_0_weight = get_tensor(ctx_meta, "adapter.0.weight");
    model->adapter_2_weight = get_tensor(ctx_meta, "adapter.2.weight");

    model->tok_embeddings_weight = get_tensor(ctx_meta, "tok_embeddings.weight");
    model->dec_norm_weight       = get_tensor(ctx_meta, "norm.weight");

    model->dec_layers.resize(model->hp.dec_layers);
    for (int32_t i = 0; i < model->hp.dec_layers; i++) {
        char nm[256];
        auto & L = model->dec_layers[i];
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_norm.weight",i); L.attn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_q.weight",i);    L.attn_q_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_k.weight",i);    L.attn_k_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_v.weight",i);    L.attn_v_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_o.weight",i);    L.attn_o_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_norm.weight",i);  L.ffn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_w1.weight",i);    L.ffn_w1_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_w2.weight",i);    L.ffn_w2_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_w3.weight",i);    L.ffn_w3_weight = get_tensor(ctx_meta,nm);
        // Adaptive RMS-norm time-conditioning weights are realtime-only.
        L.ada0_weight = nullptr;
        L.ada2_weight = nullptr;
        if (model->hp.ada_t_cond) {
            snprintf(nm,sizeof(nm),"dec.blk.%d.ada0.weight",i);  L.ada0_weight   = get_tensor(ctx_meta,nm);
            snprintf(nm,sizeof(nm),"dec.blk.%d.ada2.weight",i);  L.ada2_weight   = get_tensor(ctx_meta,nm);
        }
    }

    model->mel_filters = get_tensor(ctx_meta, "audio.mel_filters");

    // Tokenizer metadata (Tekken)
    {
        const int64_t key_num_special = gguf_find_key(gguf_ctx, "voxtral.tokenizer.num_special_tokens");
        if (key_num_special >= 0) {
            model->tokenizer_num_special_tokens = gguf_get_val_i32(gguf_ctx, key_num_special);
        }

        const int64_t key_special = gguf_find_key(gguf_ctx, "voxtral.tokenizer.special_token_ranks");
        if (key_special >= 0 && gguf_get_kv_type(gguf_ctx, key_special) == GGUF_TYPE_ARRAY) {
            if (gguf_get_arr_type(gguf_ctx, key_special) == GGUF_TYPE_INT32) {
                const size_t n = gguf_get_arr_n(gguf_ctx, key_special);
                const int32_t * data = (const int32_t *) gguf_get_arr_data(gguf_ctx, key_special);
                if (data) {
                    for (size_t i = 0; i < n; ++i) {
                        model->tokenizer_special_ranks.insert(data[i]);
                    }
                }
            }
        }

        const int64_t key_vocab = gguf_find_key(gguf_ctx, "voxtral.tokenizer.vocab_token_bytes_b64");
        if (key_vocab >= 0 && gguf_get_kv_type(gguf_ctx, key_vocab) == GGUF_TYPE_ARRAY) {
            if (gguf_get_arr_type(gguf_ctx, key_vocab) == GGUF_TYPE_STRING) {
                const size_t n = gguf_get_arr_n(gguf_ctx, key_vocab);
                model->tokenizer_vocab_b64.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    const char * s = gguf_get_arr_str(gguf_ctx, key_vocab, i);
                    model->tokenizer_vocab_b64.emplace_back(s ? s : "");
                }
            }
        }
    }

    log_info(std::string("model arch: ") + (model->hp.is_offline ? "voxtral (offline)" : "voxtral_realtime") +
             " enc_causal=" + (model->hp.enc_causal ? "1" : "0") +
             " ada_t_cond=" + (model->hp.ada_t_cond ? "1" : "0"));
    log_info("model loaded: enc_layers=" + std::to_string(model->hp.enc_layers) +
             " dec_layers=" + std::to_string(model->hp.dec_layers) +
             " vocab=" + std::to_string(model->hp.vocab_size));

    if (model->buf_weights) {
        const double sz_mb = (double) ggml_backend_buffer_get_size(model->buf_weights) / 1e6;
        log_info("model weights: " + std::to_string(sz_mb) + " MB");
    }
    log_info("encoder: dim=" + std::to_string(model->hp.enc_dim) +
             " heads=" + std::to_string(model->hp.enc_heads) +
             " head_dim=" + std::to_string(model->hp.enc_head_dim) +
             " hidden=" + std::to_string(model->hp.enc_hidden));
    log_info("decoder: dim=" + std::to_string(model->hp.dec_dim) +
             " heads=" + std::to_string(model->hp.dec_heads) +
             " head_dim=" + std::to_string(model->hp.dec_head_dim) +
             " hidden=" + std::to_string(model->hp.dec_hidden) +
             " kv_heads=" + std::to_string(model->hp.dec_kv_heads));

    {
        char buf[128];
        snprintf(buf, sizeof(buf), "model load time: %.2f ms", elapsed_ms(t_load_start));
        log_info(std::string(buf));
    }

    return model;
}

void voxtral_model_free(voxtral_model * model) {
    if (!model) return;
    if (model->buf_weights) ggml_backend_buffer_free(model->buf_weights);
    if (model->backend_weights) ggml_backend_free(model->backend_weights);
    if (model->ctx_gguf)    ggml_free(model->ctx_gguf);
    if (model->gguf_ctx)    gguf_free(model->gguf_ctx);
    delete model;
}

// ============================================================================
// Context initialization
// ============================================================================

voxtral_context * voxtral_init_from_model(
    voxtral_model              * model,
    const voxtral_context_params & params)
{
    voxtral_context * ctx = new voxtral_context();
    ctx->model     = model;
    ctx->log_level = params.log_level;
    ctx->logger    = params.logger;
    ctx->profile.enabled = std::getenv("VOXTRAL_PROFILE") != nullptr;
    ctx->numerical_diagnostics = std::getenv("VOXTRAL_NUMERICAL_DIAGNOSTICS") != nullptr;
    ctx->capture_encoder_diagnostics = ctx->numerical_diagnostics;
    ctx->capture_adapter_diagnostics = ctx->numerical_diagnostics;
    voxtral_context_profile_reset_internal(ctx);
    if (params.n_threads > 0) {
        ctx->n_threads = params.n_threads;
    } else {
        const unsigned hw = std::thread::hardware_concurrency();
        ctx->n_threads = hw > 0 ? (int32_t) std::min(hw, 16u) : 4;
    }

    voxtral_gpu_backend gpu = params.gpu;
    if (gpu == voxtral_gpu_backend::none && model && model->weights_on_gpu) {
        gpu = model->gpu_type;
    }
    ctx->gpu_type = voxtral_gpu_backend::none;

    // GPU compute backend (registry): match the requested backend, or any GPU.
    if (gpu != voxtral_gpu_backend::none) {
        ggml_backend_dev_t gdev = find_gpu_device(gpu);
        if (gdev) {
            ctx->backend = ggml_backend_dev_init(gdev, nullptr);
        }
        if (ctx->backend) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(gdev);
            ctx->gpu_type = gpu_from_reg(reg ? ggml_backend_reg_name(reg) : "");
        } else {
            LOG_INFO(ctx, "no GPU backend available, using CPU");
        }
    }

    bool has_gpu = (ctx->gpu_type != voxtral_gpu_backend::none);

    if (!ctx->backend) {
        // CPU-only: the CPU backend is the compute backend.
        ctx->backend = init_cpu_backend();
        backend_set_threads(ctx->backend, ctx->n_threads);
        LOG_INFO(ctx, "backend: CPU with %d threads", ctx->n_threads);
    } else {
        // GPU compute + a CPU backend for fallback ops.
        ctx->backend_cpu = init_cpu_backend();
        backend_set_threads(ctx->backend_cpu, ctx->n_threads);
        const char * gpu_name = "GPU";
        if (ctx->gpu_type == voxtral_gpu_backend::cuda)   gpu_name = "CUDA";
        if (ctx->gpu_type == voxtral_gpu_backend::metal)  gpu_name = "METAL";
        if (ctx->gpu_type == voxtral_gpu_backend::vulkan) gpu_name = "VULKAN";
        LOG_INFO(ctx, "backend: %s (CPU fallback %d threads)", gpu_name, ctx->n_threads);
    }

    // Accelerator (BLAS/Accelerate) device for faster CPU matmuls, added to the
    // scheduler below — preserves CPU-path performance.
    if (ggml_backend_dev_t adev = find_accel_device()) {
        ctx->blas_backend = ggml_backend_dev_init(adev, nullptr);
        if (ctx->blas_backend) {
            backend_set_threads(ctx->blas_backend, ctx->n_threads);
            LOG_INFO(ctx, "BLAS backend enabled with %d threads", ctx->n_threads);
        }
    }

    // Allocate persistent tensors: encoder chunk output, decoder logits, KV cache
    {
        constexpr size_t n_tensors = 6;
        ggml_init_params p = {
            /*.mem_size  =*/ ggml_tensor_overhead() * n_tensors,
            /*.mem_buffer=*/ nullptr,
            /*.no_alloc  =*/ true,
        };
        ctx->ctx_persistent = ggml_init(p);

        // encoder_chunk_output: [enc_dim, MAX_ENC_CHUNK]
        ctx->encoder_chunk_output = ggml_new_tensor_2d(ctx->ctx_persistent, GGML_TYPE_F32,
            VOXTRAL_ENC_DIM, VOXTRAL_MAX_ENC_CHUNK);
        ggml_set_name(ctx->encoder_chunk_output, "encoder_chunk_output");

        // decoder_logits: [vocab_size]
        ctx->decoder_logits = ggml_new_tensor_1d(ctx->ctx_persistent, GGML_TYPE_F32,
            VOXTRAL_VOCAB_SIZE);
        ggml_set_name(ctx->decoder_logits, "decoder_logits");

        // decoder_argmax: [1] i32 — greedy token id computed on device, so the
        // hot decode loop reads back 4 bytes instead of the full vocab logits.
        ctx->decoder_argmax = ggml_new_tensor_1d(ctx->ctx_persistent, GGML_TYPE_I32, 1);
        ggml_set_name(ctx->decoder_argmax, "decoder_argmax");

        // A tiny persistent destination for the opt-in numerical suite. It is
        // never copied to by ordinary graphs and adds no F32 KV shadow.
        ctx->decoder_hidden_diagnostic = ggml_new_tensor_1d(
            ctx->ctx_persistent, GGML_TYPE_F32, VOXTRAL_DEC_DIM);
        ggml_set_name(ctx->decoder_hidden_diagnostic, "decoder_hidden_diagnostic");

        // KV cache: [kv_dim, dec_window, dec_layers] — layer count is model-dependent
        // (realtime 26, offline 30); window is the physical cache capacity for both.
        const int32_t kv_dim = ctx->model->hp.dec_kv_heads * ctx->model->hp.dec_head_dim;  // 1024
        const int32_t kv_layers = ctx->model->hp.dec_layers;
        const ggml_type decoder_kv_type = decoder_kv_type_from_env();
        ctx->kv_self_k = ggml_new_tensor_3d(ctx->ctx_persistent, decoder_kv_type,
            kv_dim, VOXTRAL_DEC_WINDOW, kv_layers);
        ggml_set_name(ctx->kv_self_k, "kv_self_k");

        ctx->kv_self_v = ggml_new_tensor_3d(ctx->ctx_persistent, decoder_kv_type,
            kv_dim, VOXTRAL_DEC_WINDOW, kv_layers);
        ggml_set_name(ctx->kv_self_v, "kv_self_v");

        ctx->buf_persistent = ggml_backend_alloc_ctx_tensors(ctx->ctx_persistent, ctx->backend);
        if (!ctx->buf_persistent) {
            fprintf(stderr, "voxtral: failed to allocate persistent buffer\n");
            voxtral_free(ctx);
            return nullptr;
        }

        // Zero persistent buffer (KV cache etc.)
        ggml_backend_buffer_clear(ctx->buf_persistent, 0);
    }

    // Session 7: device-resident incremental adapter/decoder rings, in their OWN
    // buffer so clear_kv_cache()'s whole-buffer clear never wipes live encoder
    // output or audio embeddings.
    {
        ggml_init_params p = { ggml_tensor_overhead() * 2, nullptr, /*.no_alloc=*/ true };
        ctx->ctx_rings = ggml_init(p);
        ctx->enc_out_ring = ggml_new_tensor_2d(ctx->ctx_rings, GGML_TYPE_F32,
            VOXTRAL_ENC_DIM, VOXTRAL_ENC_OUT_RING_CAP);
        ggml_set_name(ctx->enc_out_ring, "enc_out_ring");
        ctx->audio_emb_ring = ggml_new_tensor_2d(ctx->ctx_rings, GGML_TYPE_F32,
            ctx->model->hp.dec_dim, VOXTRAL_AEMB_RING_CAP);
        ggml_set_name(ctx->audio_emb_ring, "audio_emb_ring");
        ctx->buf_rings = ggml_backend_alloc_ctx_tensors(ctx->ctx_rings, ctx->backend);
        if (!ctx->buf_rings) {
            fprintf(stderr, "voxtral: failed to allocate incremental ring buffer\n");
            voxtral_free(ctx);
            return nullptr;
        }
        ggml_backend_buffer_clear(ctx->buf_rings, 0);
    }

    {
        const double chunk_mb = (double) ggml_nbytes(ctx->encoder_chunk_output) / 1e6;
        const double kv_mb  = (double) (ggml_nbytes(ctx->kv_self_k) + ggml_nbytes(ctx->kv_self_v)) / 1e6;
        LOG_INFO(ctx, "buffers: encoder_chunk=%.2f MB kv_cache=%.2f MB (%s)",
            chunk_mb, kv_mb, ctx->kv_self_k->type == GGML_TYPE_F16 ? "F16" : "F32");
    }

    // Schedulers — ggml requires the last backend to be CPU.
    // With GPU:    [GPU, BLAS?, CPU]
    // Without GPU: [BLAS?, CPU]
    ggml_backend_t backends[4];
    int n_backends = 0;
    if (has_gpu) {
        backends[n_backends++] = ctx->backend;           // GPU first
    }
    if (ctx->blas_backend) {
        backends[n_backends++] = ctx->blas_backend;      // BLAS before CPU
    }
    // CPU must be last
    ggml_backend_t cpu_be = has_gpu ? ctx->backend_cpu : ctx->backend;
    backends[n_backends++] = cpu_be;
    const bool op_offload = has_gpu;

    // The encoder scheduler runs both the batch encoder graph and the larger
    // per-layer KV graph (whose wrap-gather adds concat nodes per layer), so it is
    // sized generously to fit the KV graph's node+leaf count across a ring wrap.
    ctx->sched_encoder  = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE * 8, false, op_offload);
    ctx->sched_encoder_steady = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE * 8, false, op_offload);
    ctx->sched_adapter  = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, op_offload);
    ctx->sched_adapter_group = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, op_offload);
    ctx->sched_adapter_batch = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, op_offload);
    ctx->sched_dec_pre  = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, op_offload);
    ctx->sched_dec_step = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, op_offload);

    // Hann window (shared frontend implementation)
    voxtral_mel_build_hann_window(ctx->hann_window);

    // Mel filters (compute on CPU if not available from model, else load from GGUF)
    if (model->mel_filters) {
        constexpr int32_t n = VOXTRAL_MEL_N_FREQ * VOXTRAL_NUM_MEL_BINS;
        ctx->mel_filters_cpu.resize(n);
        ggml_backend_tensor_get(model->mel_filters, ctx->mel_filters_cpu.data(), 0, n * sizeof(float));
    } else {
        voxtral_mel_build_slaney_filters(ctx->mel_filters_cpu);
    }

    // Time embedding for t = N_DELAY_TOKENS
    compute_time_embedding(ctx->time_emb_cpu, (float)VOXTRAL_N_DELAY_TOKENS, VOXTRAL_DEC_DIM);

    if (const char * value = std::getenv("VOXTRAL_DECODER_KV_TEST_CAPACITY")) {
        const long parsed = std::strtol(value, nullptr, 10);
        if (parsed >= 64 && parsed <= VOXTRAL_DEC_WINDOW) {
            ctx->decoder_kv_configured_capacity = (int32_t) parsed;
            ctx->decoder_kv.capacity = parsed;
            LOG_WARN(ctx, "decoder KV: TEST-ONLY logical capacity=%ld (backing=%d)",
                     parsed, VOXTRAL_DEC_WINDOW);
        }
    }

    LOG_INFO(ctx, "context initialized");
    return ctx;
}

const float * voxtral_ctx_hann_window(const voxtral_context * ctx) {
    return (ctx && !ctx->hann_window.empty()) ? ctx->hann_window.data() : nullptr;
}

const float * voxtral_ctx_mel_filters(const voxtral_context * ctx) {
    return (ctx && !ctx->mel_filters_cpu.empty()) ? ctx->mel_filters_cpu.data() : nullptr;
}

void voxtral_free(voxtral_context * ctx) {
    if (!ctx) return;
    if (ctx->sched_encoder)  ggml_backend_sched_free(ctx->sched_encoder);
    if (ctx->sched_encoder_steady) ggml_backend_sched_free(ctx->sched_encoder_steady);
    if (ctx->sched_adapter)  ggml_backend_sched_free(ctx->sched_adapter);
    if (ctx->sched_adapter_group) ggml_backend_sched_free(ctx->sched_adapter_group);
    if (ctx->sched_adapter_batch) ggml_backend_sched_free(ctx->sched_adapter_batch);
    if (ctx->sched_dec_pre)  ggml_backend_sched_free(ctx->sched_dec_pre);
    if (ctx->sched_dec_step) ggml_backend_sched_free(ctx->sched_dec_step);
    if (ctx->decoder_step_graph_ctx) ggml_free(ctx->decoder_step_graph_ctx);
    if (ctx->adapter_group_graph.gctx) ggml_free(ctx->adapter_group_graph.gctx);
    if (ctx->adapter_batch_graph.gctx) ggml_free(ctx->adapter_batch_graph.gctx);
    if (ctx->encoder_steady_graph_ctx) ggml_free(ctx->encoder_steady_graph_ctx);
    if (ctx->buf_enc_full)   ggml_backend_buffer_free(ctx->buf_enc_full);
    if (ctx->ctx_enc_full)   ggml_free(ctx->ctx_enc_full);
    if (ctx->buf_dec_mem)    ggml_backend_buffer_free(ctx->buf_dec_mem);
    if (ctx->ctx_dec_mem)    ggml_free(ctx->ctx_dec_mem);
    if (ctx->buf_persistent) ggml_backend_buffer_free(ctx->buf_persistent);
    if (ctx->ctx_persistent) ggml_free(ctx->ctx_persistent);
    if (ctx->buf_rings)      ggml_backend_buffer_free(ctx->buf_rings);
    if (ctx->ctx_rings)      ggml_free(ctx->ctx_rings);
    if (ctx->blas_backend)   ggml_backend_free(ctx->blas_backend);
    if (ctx->backend_cpu)    ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)        ggml_backend_free(ctx->backend);
    delete ctx;
}

// ============================================================================
// KV cache helpers
// ============================================================================

static void clear_kv_cache(voxtral_context * ctx) {
    if (!ctx) return;
    // Logical invalidation only.  Every subsequent read is bounded by `used`, so
    // stale device slots are unreachable.  The backing tensors are cleared once
    // at allocation, never per stream and never at rollover.
    ctx->decoder_kv = voxtral_decoder_kv_ring{};
    ctx->decoder_kv.capacity = ctx->decoder_kv_configured_capacity;
    ctx->decoder_rollover = {};
    ctx->decoder_step_mask_valid = -1;
    ctx->diagnostic_first_hidden.clear();
    ctx->diagnostic_first_logits.clear();
    ctx->capture_decoder_diagnostics = ctx->numerical_diagnostics;
}

static int32_t decoder_kv_segments(int64_t absolute_start, int32_t len, int32_t capacity,
                                   voxtral_decoder_kv_seg out[2]) {
    if (len <= 0) return 0;
    const int32_t slot = (int32_t) (absolute_start % capacity);
    const int32_t first = std::min(len, capacity - slot);
    out[0] = { slot, first };
    if (first == len) return 1;
    out[1] = { 0, len - first };
    return 2;
}

bool voxtral_decoder_kv_plan(const voxtral_decoder_kv_ring & ring,
                             int64_t absolute_start, int32_t n_new,
                             voxtral_decoder_kv_plan_t & out) {
    out = voxtral_decoder_kv_plan_t{};
    if (ring.capacity <= 0 || ring.used < 0 || ring.used > ring.capacity ||
        ring.oldest_absolute_position < 0 || ring.next_absolute_position < 0 ||
        ring.oldest_absolute_position + ring.used != ring.next_absolute_position ||
        absolute_start != ring.next_absolute_position || n_new <= 0 ||
        n_new > ring.capacity) {
        return false;
    }

    out.absolute_start = absolute_start;
    out.n_new = n_new;
    out.write_nseg = decoder_kv_segments(absolute_start, n_new, (int32_t) ring.capacity,
                                         out.write_seg);
    out.next_after = absolute_start + n_new;
    out.used_after = std::min<int64_t>(ring.capacity, ring.used + n_new);
    out.oldest_after = out.next_after - out.used_after;
    out.evicted = std::max<int64_t>(0, ring.used + n_new - ring.capacity);
    out.read_nseg = decoder_kv_segments(out.oldest_after, (int32_t) out.used_after,
                                        (int32_t) ring.capacity, out.read_seg);

    // A physical wrap occurs when the appended absolute span crosses a capacity
    // cycle boundary, including the single-token write exactly at slot zero.
    const int64_t previous_cycle = absolute_start == 0 ? 0 : (absolute_start - 1) / ring.capacity;
    const int64_t newest_cycle = (out.next_after - 1) / ring.capacity;
    out.wrapped = newest_cycle > previous_cycle;
    return true;
}

static void decoder_kv_commit_plan(voxtral_context * ctx,
                                   const voxtral_decoder_kv_plan_t & plan) {
    auto & ring = ctx->decoder_kv;
    ring.used = plan.used_after;
    ring.oldest_absolute_position = plan.oldest_after;
    ring.next_absolute_position = plan.next_after;
    ring.evictions += plan.evicted;
    if (plan.wrapped) ++ring.wraps;
    // bytes_moved/full_buffer_moves deliberately remain zero: only the newly
    // projected K/V columns are written by the graph.
}

// ============================================================================
// Graph Building: Encoder
// ============================================================================


struct causal_conv1d_dims {
    int32_t pad_left = 0;
    int32_t pad_right = 0;
    int32_t padded_len = 0;
    int32_t out_len = 0;
};

causal_conv1d_dims compute_causal_conv1d_dims(int32_t in_len, int32_t kernel_size, int32_t stride) {
    causal_conv1d_dims out{};
    if (in_len <= 0 || kernel_size <= 0 || stride <= 0) {
        return out;
    }

    const int32_t padding_total = kernel_size - stride;
    const float n_frames = (static_cast<float>(in_len - kernel_size + padding_total) / static_cast<float>(stride)) + 1.0f;
    const int32_t target_length =
        (static_cast<int32_t>(std::ceil(n_frames)) - 1) * stride + (kernel_size - padding_total);
    const int32_t extra_padding = target_length - in_len;

    out.pad_left = padding_total;
    out.pad_right = std::max<int32_t>(0, extra_padding);
    out.padded_len = in_len + out.pad_left + out.pad_right;
    out.out_len = (out.padded_len - kernel_size) / stride + 1;
    return out;
}

// Compute the number of encoder tokens from mel frames (accounting for conv and truncation)
static int32_t mel_frames_to_enc_tokens(int32_t n_frames) {
    auto d0 = compute_causal_conv1d_dims(n_frames, 3, 1);  // conv0
    auto d1 = compute_causal_conv1d_dims(d0.out_len, 3, 2); // conv1 (stride 2)
    int32_t trunc = d1.out_len % VOXTRAL_DOWNSAMPLE_FACTOR;
    return d1.out_len - trunc;
}

// Model-free accessor (see voxtral-internal.h): encoder frames from Mel frames.
int32_t voxtral_enc_frames_for_mel_internal(int32_t mel_frames) {
    return mel_frames > 0 ? mel_frames_to_enc_tokens(mel_frames) : 0;
}

// Pre-compute total encoder tokens for a given mel frame count (for buffer allocation)
static int32_t compute_total_enc_tokens(int32_t total_mel_frames) {
    const int32_t mel_stride = VOXTRAL_ENC_CHUNK_MEL - VOXTRAL_ENC_CHUNK_OVERLAP * 2;
    int32_t total = 0;
    int32_t mel_offset = 0;
    bool first = true;

    while (mel_offset < total_mel_frames) {
        int32_t chunk_mel = std::min(VOXTRAL_ENC_CHUNK_MEL, total_mel_frames - mel_offset);
        int32_t chunk_tokens = mel_frames_to_enc_tokens(chunk_mel);
        int32_t skip = first ? 0 : VOXTRAL_ENC_CHUNK_OVERLAP;
        int32_t stride = chunk_tokens - skip;
        if (stride <= 0) break;
        total += stride;
        mel_offset += mel_stride;
        first = false;
    }
    return total;
}

// Allocate per-utterance encoder output buffer on device
static bool alloc_encoder_output(voxtral_context * ctx, int32_t n_tokens) {
    // Free previous allocation
    if (ctx->buf_enc_full) { ggml_backend_buffer_free(ctx->buf_enc_full); ctx->buf_enc_full = nullptr; }
    if (ctx->ctx_enc_full) { ggml_free(ctx->ctx_enc_full); ctx->ctx_enc_full = nullptr; }
    ctx->encoder_output = nullptr;

    ggml_init_params p = {
        /*.mem_size  =*/ ggml_tensor_overhead(),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ctx->ctx_enc_full = ggml_init(p);
    ctx->encoder_output = ggml_new_tensor_2d(ctx->ctx_enc_full, GGML_TYPE_F32,
        VOXTRAL_ENC_DIM, n_tokens);
    ggml_set_name(ctx->encoder_output, "encoder_output");
    ctx->buf_enc_full = ggml_backend_alloc_ctx_tensors(ctx->ctx_enc_full, ctx->backend);
    if (!ctx->buf_enc_full) return false;

    ctx->total_enc_tokens = n_tokens;
    return true;
}

// Allocate per-utterance decoder memory buffer on device
static bool alloc_decoder_memory(voxtral_context * ctx, int32_t dec_seq) {
    if (ctx->buf_dec_mem) { ggml_backend_buffer_free(ctx->buf_dec_mem); ctx->buf_dec_mem = nullptr; }
    if (ctx->ctx_dec_mem) { ggml_free(ctx->ctx_dec_mem); ctx->ctx_dec_mem = nullptr; }
    ctx->decoder_memory = nullptr;

    ggml_init_params p = {
        /*.mem_size  =*/ ggml_tensor_overhead(),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ctx->ctx_dec_mem = ggml_init(p);
    ctx->decoder_memory = ggml_new_tensor_2d(ctx->ctx_dec_mem, GGML_TYPE_F32,
        ctx->model->hp.dec_dim, dec_seq);
    ggml_set_name(ctx->decoder_memory, "decoder_memory");
    ctx->buf_dec_mem = ggml_backend_alloc_ctx_tensors(ctx->ctx_dec_mem, ctx->backend);
    if (!ctx->buf_dec_mem) return false;

    ctx->dec_seq_len = dec_seq;
    return true;
}

ggml_tensor * causal_conv1d_graph(
    ggml_context * ctx0,
    ggml_tensor * x,
    int32_t in_len,
    ggml_tensor * weight,
    ggml_tensor * bias,
    int32_t out_channels,
    int32_t kernel_size,
    int32_t stride,
    int32_t & out_len,
    bool symmetric = false) {
    out_len = 0;
    if (ctx0 == nullptr || x == nullptr || weight == nullptr || kernel_size <= 0 || stride <= 0) {
        return nullptr;
    }
    if (in_len <= 0 || out_channels <= 0) {
        return nullptr;
    }

    causal_conv1d_dims dims{};
    if (symmetric) {
        // PyTorch Conv1d(padding=p) with p=(kernel-1)/2 — used by the offline Whisper encoder.
        const int32_t p = (kernel_size - 1) / 2;
        dims.pad_left = p;
        dims.pad_right = p;
        dims.padded_len = in_len + 2 * p;
        dims.out_len = (dims.padded_len - kernel_size) / stride + 1;
    } else {
        dims = compute_causal_conv1d_dims(in_len, kernel_size, stride);
    }
    if (dims.out_len <= 0) {
        return nullptr;
    }

    ggml_tensor * x_pad = ggml_pad_ext(ctx0, x, dims.pad_left, dims.pad_right, 0, 0, 0, 0, 0, 0);
    if (x_pad == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = ggml_conv_1d(ctx0, weight, x_pad, stride, 0, 1);
    if (y == nullptr) {
        return nullptr;
    }

    if (bias != nullptr) {
        y = ggml_add(ctx0, y, ggml_reshape_3d(ctx0, bias, 1, out_channels, 1));
    }

    out_len = dims.out_len;
    return y;
}


void print_tensor_info(struct ggml_tensor * tensor) {
    printf("Tensor name: %s\n", tensor->name);
    printf("Tensor type: %s\n", ggml_type_name(tensor->type));
    printf("Number of dimensions: %d\n", ggml_n_dims(tensor));
    printf("Total elements: %" PRId64 "\n", ggml_nelements(tensor));
    printf("Shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
           tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
}

static void log_tensor_info(voxtral_context * ctx, const char * tag, struct ggml_tensor * t) {
    if (t == nullptr) {
        LOG_DBG(ctx, "%s: <null>", tag);
        return;
    }
    LOG_DBG(ctx, "%s: type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] nb=[%zu,%zu,%zu,%zu] n_dims=%d nbytes=%zu",
        tag,
        ggml_type_name(t->type),
        t->ne[0], t->ne[1], t->ne[2], t->ne[3],
        (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3],
        ggml_n_dims(t),
        (size_t) ggml_nbytes(t));
}

static void log_graph_info(voxtral_context * ctx, const char * name, struct ggml_cgraph * gf) {
    if (gf == nullptr) {
        return;
    }
    const int size  = ggml_graph_size(gf);
    const int nodes = ggml_graph_n_nodes(gf);
    LOG_INFO(ctx, "%s graph: size=%d nodes=%d", name, size, nodes);
}

// Build encoder graph that writes output into ctx->encoder_chunk_output
// Apply LayerNorm (offline Whisper: weight+bias) or RMSNorm (realtime: weight only),
// if a bias tensor is provided.
static ggml_tensor * enc_apply_norm(ggml_context * gctx, ggml_tensor * x,
                                    ggml_tensor * w, ggml_tensor * b, float eps, bool layernorm) {
    ggml_tensor * n = layernorm ? ggml_norm(gctx, x, eps) : ggml_rms_norm(gctx, x, eps);
    n = ggml_mul(gctx, n, w);
    if (b) n = ggml_add(gctx, n, b);
    return n;
}

static ggml_cgraph * build_encoder_graph(
    voxtral_context * ctx,
    ggml_context * gctx,
    const float * mel_data,   // [n_mel, n_frames] on CPU
    int32_t n_frames,
    int32_t * out_seq_len)    // output: encoder tokens produced by this chunk
{
    LOG_DBG(ctx, "Building encoder graph");
    voxtral_model * model = ctx->model;

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    // ggml_conv_1d expects input as [length, in_channels, batch]
    // mel_data is [n_mel, n_frames] on CPU; we transpose on upload.
    ggml_tensor * mel_input = ggml_new_tensor_3d(
        gctx, GGML_TYPE_F32, n_frames, VOXTRAL_NUM_MEL_BINS, 1);
    ggml_set_name(mel_input, "mel_input");

    // We need to set data after sched_alloc, mark as input
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, mel_input, ctx->backend);

    // Conv stem: mel is [n_frames, n_mel, 1], weights are [k, in_ch, out_ch]
    log_tensor_info(ctx, "enc.conv0.weight", model->enc_conv0_weight);
    log_tensor_info(ctx, "enc.conv1.weight", model->enc_conv1_weight);
    log_tensor_info(ctx, "mel_input", mel_input);

    const bool conv_sym = model->hp.is_offline;  // offline Whisper uses symmetric conv padding
    int32_t conv0_len = 0;
    ggml_tensor * conv0_out = causal_conv1d_graph(
        gctx, mel_input, n_frames,
        model->enc_conv0_weight, model->enc_conv0_bias,
        model->hp.enc_dim, 3, 1, conv0_len, conv_sym);
    if (conv0_out == nullptr) {
        LOG_ERR(ctx, "conv0_out is null");
        return gf;
    }
    log_tensor_info(ctx, "conv0_out(pre_act)", conv0_out);
    conv0_out = ggml_gelu_erf(gctx, conv0_out);

    int32_t conv_out_len = 0;
    ggml_tensor * conv1_out = causal_conv1d_graph(
        gctx, conv0_out, conv0_len,
        model->enc_conv1_weight, model->enc_conv1_bias,
        model->hp.enc_dim, 3, 2, conv_out_len, conv_sym);
    if (conv1_out == nullptr) {
        LOG_ERR(ctx, "conv1_out is null");
        return gf;
    }
    log_tensor_info(ctx, "conv1_out(pre_act)", conv1_out);
    conv1_out = ggml_gelu_erf(gctx, conv1_out);
    log_tensor_info(ctx, "conv1_out", conv1_out);

    // Transpose for transformer: [enc_dim, seq] -> [enc_dim, seq] (already correct for ggml)
    // In ggml, tensor is [ne0=enc_dim, ne1=seq], which means each "row" (token) has enc_dim elements
    // This is what we need for mul_mat: ggml_mul_mat(weight[out,in], x[in,seq]) -> [out,seq]

    // Left-truncate to multiple of downsample_factor (matching Python)
    const int32_t trunc = conv_out_len % model->hp.downsample_factor;
    ggml_tensor * x_len_first = conv1_out;
    int32_t seq_len = conv_out_len;
    if (trunc > 0) {
        // Skip first 'trunc' frames along length dimension (ne0)
        x_len_first = ggml_view_3d(gctx, conv1_out,
            conv_out_len - trunc, model->hp.enc_dim, 1,
            conv1_out->nb[1], conv1_out->nb[2],
            (size_t) trunc * conv1_out->nb[0]); // [len, enc_dim, 1]
        seq_len = conv_out_len - trunc;
    }
    LOG_DBG(ctx, "encoder conv: in_frames=%d conv0_len=%d conv1_len=%d trunc=%d seq_len=%d",
        n_frames, conv0_len, conv_out_len, trunc, seq_len);

    // Transpose to [enc_dim, seq_len] for transformer blocks
    ggml_tensor * x = ggml_permute(gctx, x_len_first, 1, 0, 2, 3); // [enc_dim, seq_len, 1]
    x = ggml_cont(gctx, x);
    x = ggml_reshape_2d(gctx, x, model->hp.enc_dim, seq_len);
    // Offline Whisper encoder: add fixed sinusoidal positional embeddings to the
    // conv output (it uses absolute positions, not RoPE).
    if (model->hp.is_offline && model->enc_pos_embedding) {
        ggml_tensor * pos = ggml_view_2d(gctx, model->enc_pos_embedding,
            model->hp.enc_dim, seq_len, model->enc_pos_embedding->nb[1], 0);
        x = ggml_add(gctx, x, pos);
    }
    log_tensor_info(ctx, "encoder_x", x);

    // Position tensor for RoPE: [seq_len] int32
    ggml_tensor * enc_positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(enc_positions, "enc_positions");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, enc_positions, ctx->backend);

    // Encoder attention mask (sliding causal window) — realtime only. The offline
    // Whisper encoder uses full bidirectional attention, so no mask is created.
    ggml_tensor * enc_attn_mask_f16 = nullptr;
    if (!model->hp.is_offline) {
        ggml_tensor * enc_attn_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, seq_len, seq_len);
        ggml_set_name(enc_attn_mask, "enc_attn_mask");
        ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, enc_attn_mask, ctx->backend);
        enc_attn_mask_f16 = ggml_cast(gctx, enc_attn_mask, GGML_TYPE_F16);
    }

    // Transformer layers
    const auto & hp = model->hp;
    const bool   ln = hp.is_offline;       // offline = LayerNorm; realtime = RMSNorm
    const int32_t e_heads    = hp.enc_heads;
    const int32_t e_kv_heads = hp.enc_kv_heads;
    const int32_t e_hd       = hp.enc_head_dim;
    for (int32_t i = 0; i < hp.enc_layers; i++) {
        auto & L = model->enc_layers[i];

        // Pre-attention norm (LayerNorm for offline, RMSNorm for realtime)
        ggml_tensor * residual = x; // [enc_dim, seq_len]
        ggml_tensor * x_norm = enc_apply_norm(gctx, x, L.attn_norm_weight, L.attn_norm_bias, hp.enc_norm_eps, ln);

        // Q, K, V projections
        ggml_tensor * q = ggml_mul_mat(gctx, L.attn_q_weight, x_norm); // [e_heads*hd, seq_len]
        q = ggml_add(gctx, q, L.attn_q_bias);

        ggml_tensor * k = ggml_mul_mat(gctx, L.attn_k_weight, x_norm); // [e_kv_heads*hd, seq_len] (no bias)

        ggml_tensor * v = ggml_mul_mat(gctx, L.attn_v_weight, x_norm); // [e_kv_heads*hd, seq_len]
        v = ggml_add(gctx, v, L.attn_v_bias);

        q = ggml_reshape_3d(gctx, q, e_hd, e_heads, seq_len);
        k = ggml_reshape_3d(gctx, k, e_hd, e_kv_heads, seq_len);

        // Realtime encoder uses RoPE; offline Whisper encoder uses absolute
        // sinusoidal positions added above, so no RoPE here.
        if (!hp.is_offline) {
            q = ggml_rope_ext(gctx, q, enc_positions, nullptr,
                e_hd, 0, 0, hp.enc_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(gctx, k, enc_positions, nullptr,
                e_hd, 0, 0, hp.enc_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }

        q = ggml_permute(gctx, q, 0, 2, 1, 3); // [hd, seq_len, e_heads]
        k = ggml_permute(gctx, k, 0, 2, 1, 3); // [hd, seq_len, e_kv_heads]

        v = ggml_reshape_3d(gctx, v, e_hd, e_kv_heads, seq_len);
        v = ggml_permute(gctx, v, 0, 2, 1, 3); // [hd, seq_len, e_kv_heads]

        const float scale = 1.0f / sqrtf((float)e_hd);

        // Mask: realtime passes the sliding-causal mask; offline passes none (full attention).
        ggml_tensor * mask = ln ? nullptr : enc_attn_mask_f16;
        ggml_tensor * attn_out = ggml_flash_attn_ext(gctx, q, k, v, mask, scale, 0.0f, 0.0f);
        attn_out = ggml_cont(gctx, attn_out);
        attn_out = ggml_reshape_2d(gctx, attn_out, e_heads * e_hd, seq_len);

        // Output projection + residual
        ggml_tensor * attn_proj = ggml_mul_mat(gctx, L.attn_o_weight, attn_out); // [enc_dim, seq_len]
        attn_proj = ggml_add(gctx, attn_proj, L.attn_o_bias);
        x = ggml_add(gctx, residual, attn_proj);

        // FFN
        residual = x;
        x_norm = enc_apply_norm(gctx, x, L.ffn_norm_weight, L.ffn_norm_bias, hp.enc_norm_eps, ln);

        ggml_tensor * ffn_out;
        if (hp.is_offline) {
            // Standard Whisper MLP: w2(gelu(w1 x + b1)) + b2
            ggml_tensor * h = ggml_mul_mat(gctx, L.ffn_w1_weight, x_norm); // [enc_hidden, seq_len]
            if (L.ffn_w1_bias) h = ggml_add(gctx, h, L.ffn_w1_bias);
            h = ggml_gelu_erf(gctx, h);
            ffn_out = ggml_mul_mat(gctx, L.ffn_w2_weight, h); // [enc_dim, seq_len]
        } else {
            // Realtime SwiGLU: silu(w1 x) * w3(x), then w2
            ggml_tensor * gate = ggml_mul_mat(gctx, L.ffn_w1_weight, x_norm);
            gate = ggml_silu(gctx, gate);
            ggml_tensor * up = ggml_mul_mat(gctx, L.ffn_w3_weight, x_norm);
            ffn_out = ggml_mul(gctx, gate, up);
            ffn_out = ggml_mul_mat(gctx, L.ffn_w2_weight, ffn_out); // [enc_dim, seq_len]
        }
        if (L.ffn_w2_bias) ffn_out = ggml_add(gctx, ffn_out, L.ffn_w2_bias);

        x = ggml_add(gctx, residual, ffn_out);
    }

    // Final norm
    x = enc_apply_norm(gctx, x, model->enc_norm_weight, model->enc_norm_bias, hp.enc_norm_eps, ln);

    // Copy result to encoder_chunk_output (per-chunk buffer, reused each chunk)
    ggml_tensor * enc_out_view = ggml_view_2d(gctx, ctx->encoder_chunk_output,
        VOXTRAL_ENC_DIM, seq_len,
        ctx->encoder_chunk_output->nb[1], 0); // [enc_dim, seq_len]
    ggml_tensor * cpy = ggml_cpy(gctx, x, enc_out_view);
    ggml_build_forward_expand(gf, cpy);

    if (out_seq_len) *out_seq_len = seq_len;

    return gf;
}

// ============================================================================
// Graph Building: Encoder per-layer KV-cache (Sessions 6.1/6.2)
// ----------------------------------------------------------------------------
// Produces ONLY the new enc frames [P, P+N) (P = plan.q_start, N = plan.n_new) and
// runs each through the 32 transformer layers exactly once. Old frames are never
// re-run: their K/V live in the persistent per-stream ring (ring_k/ring_v,
// [kv_dim, capacity, enc_layers]). Per layer: compute Q/K/V for the new frames,
// RoPE at ABSOLUTE positions, write post-RoPE K + V into the ring at the plan's
// physical slots, then attend the new queries over the causal window gathered from
// the ring (a single view when contiguous, a concat of two segments on wrap). The
// write cpys are expanded before the window read is built, so — exactly as the
// decoder KV cache — the read observes the just-written new frames (insertion-order
// dependency, honoured by the Vulkan buffer barriers). Realtime models only.
// ============================================================================
static ggml_cgraph * build_encoder_kv_graph(
    voxtral_context * ctx,
    ggml_tensor * ring_k,          // [kv_dim, capacity, enc_layers] FP16/F32, persistent
    ggml_tensor * ring_v,          // [kv_dim, capacity, enc_layers] FP16/F32, persistent
    ggml_context * gctx,
    const voxtral_enc_kv_plan_t & plan,
    int32_t physical_rows,         // fixed query rows for this physical block
    int64_t * out_materialized_frames)
{
    voxtral_model * model = ctx->model;
    const auto & hp = model->hp;
    const int32_t N        = plan.n_new;
    const int32_t P        = physical_rows;
    const int32_t q_offset = (int32_t) (plan.q_start % P);
    const int64_t block_start = plan.q_start - q_offset;
    const int64_t key_start = std::max<int64_t>(0, block_start - (int64_t) (VOXTRAL_ENC_WINDOW - 1));
    const int64_t key_end = block_start + P;
    const int32_t L        = (int32_t) (key_end - key_start);
    const int32_t e_heads    = hp.enc_heads;
    const int32_t e_kv_heads = hp.enc_kv_heads;
    const int32_t e_hd       = hp.enc_head_dim;
    const int32_t kv_dim     = e_kv_heads * e_hd;
    const int32_t conv_len = (int32_t) (plan.conv_mel_end - plan.conv_mel_start);

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 8, false);

    // --- Conv stem over the plan's Mel window, sliced to the new enc frames ------
    ggml_tensor * mel_input = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, conv_len, VOXTRAL_NUM_MEL_BINS, 1);
    ggml_set_name(mel_input, "mel_input");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, mel_input, ctx->backend);

    int32_t conv0_len = 0;
    ggml_tensor * conv0_out = causal_conv1d_graph(gctx, mel_input, conv_len,
        model->enc_conv0_weight, model->enc_conv0_bias, hp.enc_dim, 3, 1, conv0_len, /*symmetric=*/false);
    conv0_out = ggml_gelu_erf(gctx, conv0_out);
    int32_t conv1_len = 0;
    ggml_tensor * conv1_out = causal_conv1d_graph(gctx, conv0_out, conv0_len,
        model->enc_conv1_weight, model->enc_conv1_bias, hp.enc_dim, 3, 2, conv1_len, /*symmetric=*/false);
    conv1_out = ggml_gelu_erf(gctx, conv1_out);  // [conv1_len, enc_dim, 1]

    // Slice local conv1 frames [conv_slice_start, conv_slice_start+N) — the global
    // enc frames [q_start, q_start+N) — then transpose to [enc_dim, N].
    ggml_tensor * x_len_first = ggml_view_3d(gctx, conv1_out,
        N, hp.enc_dim, 1, conv1_out->nb[1], conv1_out->nb[2],
        (size_t) plan.conv_slice_start * conv1_out->nb[0]);
    ggml_tensor * x = ggml_cont(gctx, ggml_permute(gctx, x_len_first, 1, 0, 2, 3)); // [enc_dim, N, 1]
    x = ggml_reshape_2d(gctx, x, hp.enc_dim, N);

    // Run a fixed physical query shape. Real rows occupy their absolute slot in
    // the block; zero-padded dummy rows are carried through every transformer
    // layer but are never committed to KV or copied to the output.
    if (q_offset + N > P) {
        LOG_ERR(ctx, "encoder KV: logical batch crosses physical block (q=%lld N=%d P=%d)",
                (long long) plan.q_start, N, P);
        return nullptr;
    }
    if (q_offset != 0 || N != P) {
        x = ggml_pad_ext(gctx, x, 0, 0, q_offset, P - q_offset - N, 0, 0, 0, 0);
    }

    // --- Inputs: absolute RoPE positions + causal-window mask [L, P] -------------
    ggml_tensor * enc_positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, P);
    ggml_set_name(enc_positions, "enc_positions");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, enc_positions, ctx->backend);

    // Mask is [L, P]. Real query rows use the absolute causal-window predicate;
    // dummy rows are fully masked and are discarded after the transformer.
    ggml_tensor * enc_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, L, P);
    ggml_set_name(enc_mask, "enc_kv_mask");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, enc_mask, ctx->backend);

    const float scale = 1.0f / sqrtf((float) e_hd);
    const size_t k_layer_stride = ring_k->nb[2];
    const size_t v_layer_stride = ring_v->nb[2];
    const size_t k_col = ring_k->nb[1];   // kv_dim * storage element size
    const size_t v_col = ring_v->nb[1];

    for (int32_t i = 0; i < hp.enc_layers; i++) {
        auto & Lyr = model->enc_layers[i];

        ggml_tensor * residual = x; // [enc_dim, P]
        ggml_tensor * x_norm = enc_apply_norm(gctx, x, Lyr.attn_norm_weight, Lyr.attn_norm_bias,
                                              hp.enc_norm_eps, /*layernorm=*/false);

        ggml_tensor * q = ggml_add(gctx, ggml_mul_mat(gctx, Lyr.attn_q_weight, x_norm), Lyr.attn_q_bias);
        ggml_tensor * k = ggml_mul_mat(gctx, Lyr.attn_k_weight, x_norm);                    // no bias
        ggml_tensor * v = ggml_add(gctx, ggml_mul_mat(gctx, Lyr.attn_v_weight, x_norm), Lyr.attn_v_bias);

        q = ggml_reshape_3d(gctx, q, e_hd, e_heads, P);
        k = ggml_reshape_3d(gctx, k, e_hd, e_kv_heads, P);
        q = ggml_rope_ext(gctx, q, enc_positions, nullptr,
            e_hd, 0, 0, hp.enc_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(gctx, k, enc_positions, nullptr,
            e_hd, 0, 0, hp.enc_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        // Post-RoPE K + V flattened to [kv_dim, P]. Only the real subrange is
        // committed; dummy rows never enter the persistent ring.
        ggml_tensor * k_store = ggml_cont(gctx, ggml_reshape_2d(gctx, k, kv_dim, P));
        ggml_tensor * v_store = ggml_reshape_2d(gctx, v, kv_dim, P);   // V has no RoPE
        // Quantize each new column exactly once into the cache storage type.  The
        // same FP16 values feed current attention and the persistent write, so
        // old/new columns never mix precisions and no F32 shadow cache exists.
        if (ring_k->type != k_store->type) k_store = ggml_cast(gctx, k_store, ring_k->type);
        if (ring_v->type != v_store->type) v_store = ggml_cast(gctx, v_store, ring_v->type);

        // WRITE new K/V into the ring at the plan's physical slots (<=2 segments).
        // Expanded now so the window read below sees the new frames.
        int32_t src_off = 0;
        for (int32_t s = 0; s < plan.write_nseg; ++s) {
            const int32_t seg_slot = plan.write_seg[s].slot;
            const int32_t seg_len  = plan.write_seg[s].len;
            ggml_tensor * k_src = ggml_view_2d(gctx, k_store, kv_dim, seg_len, k_store->nb[1], (size_t) (q_offset + src_off) * k_store->nb[1]);
            ggml_tensor * v_src = ggml_view_2d(gctx, v_store, kv_dim, seg_len, v_store->nb[1], (size_t) (q_offset + src_off) * v_store->nb[1]);
            ggml_tensor * k_dst = ggml_view_2d(gctx, ring_k, kv_dim, seg_len, k_col, (size_t) i * k_layer_stride + (size_t) seg_slot * k_col);
            ggml_tensor * v_dst = ggml_view_2d(gctx, ring_v, kv_dim, seg_len, v_col, (size_t) i * v_layer_stride + (size_t) seg_slot * v_col);
            ggml_build_forward_expand(gf, ggml_cpy(gctx, k_src, k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(gctx, v_src, v_dst));
            src_off += seg_len;
        }

        // Describe the logical physical-block window [key_start, key_end).  The
        // low-query graph consumes these segments directly; the 32+ throughput
        // graph retains its fused contiguous-window implementation below.
        const int64_t valid_old_start = std::max<int64_t>(key_start,
            std::max<int64_t>(0, plan.q_start - (int64_t) (VOXTRAL_ENC_WINDOW - 1)));
        const int32_t prefix_len = (int32_t) (valid_old_start - key_start);
        const int32_t old_len = (int32_t) (plan.q_start - valid_old_start);
        const int32_t future_len = (int32_t) (key_end - (plan.q_start + N));
        ggml_tensor * k_real = ggml_view_2d(gctx, k_store, kv_dim, N, k_store->nb[1],
                                            (size_t) q_offset * k_store->nb[1]);
        ggml_tensor * v_real = ggml_view_2d(gctx, v_store, kv_dim, N, v_store->nb[1],
                                            (size_t) q_offset * v_store->nb[1]);
        ggml_tensor * q3 = ggml_permute(gctx, q, 0, 2, 1, 3);                               // [hd, P, heads]
        ggml_tensor * attn;
        if (encoder_kv_uses_segmented_attention(P)) {
            // Logical order is prefix-mask, old ring (one or two physical
            // views), new rows, future-mask.  Concatenate only QK scores; K/V
            // themselves remain fixed-size FP16 cache views.  The weighted V
            // reductions are summed after splitting the common softmax back at
            // the same segment boundaries.
            struct attention_segment {
                ggml_tensor * k = nullptr;
                ggml_tensor * v = nullptr;
                int32_t len = 0;
            };
            std::array<attention_segment, 5> segments;
            int32_t n_segments = 0;
            auto append_segment = [&](ggml_tensor * sk, ggml_tensor * sv, int32_t len) {
                if (len > 0) segments[(size_t) n_segments++] = { sk, sv, len };
            };

            if (prefix_len > 0) {
                GGML_ASSERT(prefix_len <= q_offset);
                append_segment(
                    ggml_view_2d(gctx, k_store, kv_dim, prefix_len, k_store->nb[1], 0),
                    ggml_view_2d(gctx, v_store, kv_dim, prefix_len, v_store->nb[1], 0),
                    prefix_len);
            }
            if (old_len > 0) {
                const int32_t cap = VOXTRAL_ENC_KV_CAP;
                const int32_t slot0 = (int32_t) (valid_old_start % cap);
                const int32_t len0 = std::min(old_len, cap - slot0);
                append_segment(
                    ggml_view_2d(gctx, ring_k, kv_dim, len0, k_col,
                        (size_t) i * k_layer_stride + (size_t) slot0 * k_col),
                    ggml_view_2d(gctx, ring_v, kv_dim, len0, v_col,
                        (size_t) i * v_layer_stride + (size_t) slot0 * v_col),
                    len0);
                const int32_t len1 = old_len - len0;
                if (len1 > 0) {
                    append_segment(
                        ggml_view_2d(gctx, ring_k, kv_dim, len1, k_col,
                            (size_t) i * k_layer_stride),
                        ggml_view_2d(gctx, ring_v, kv_dim, len1, v_col,
                            (size_t) i * v_layer_stride),
                        len1);
                }
            }
            append_segment(k_real, v_real, N);
            if (future_len > 0) {
                const int32_t future_col = q_offset + N;
                GGML_ASSERT(future_col + future_len <= P);
                append_segment(
                    ggml_view_2d(gctx, k_store, kv_dim, future_len, k_store->nb[1],
                        (size_t) future_col * k_store->nb[1]),
                    ggml_view_2d(gctx, v_store, kv_dim, future_len, v_store->nb[1],
                        (size_t) future_col * v_store->nb[1]),
                    future_len);
            }
            int32_t logical_len = 0;
            ggml_tensor * scores = nullptr;
            ggml_tensor * q_cont = ggml_cont(gctx, q3);
            for (int32_t s = 0; s < n_segments; ++s) {
                const auto & seg = segments[(size_t) s];
                ggml_tensor * k3 = ggml_permute(gctx,
                    ggml_reshape_3d(gctx, seg.k, e_hd, e_kv_heads, seg.len), 0, 2, 1, 3);
                ggml_tensor * part = ggml_mul_mat(gctx, k3, q_cont);
                scores = scores ? ggml_concat(gctx, scores, part, 0) : part;
                logical_len += seg.len;
            }
            GGML_ASSERT(logical_len == L && scores != nullptr);
            ggml_tensor * probs = ggml_soft_max_ext(gctx, scores, enc_mask, scale, 0.0f);

            ggml_tensor * weighted = nullptr;
            int32_t prob_offset = 0;
            for (int32_t s = 0; s < n_segments; ++s) {
                const auto & seg = segments[(size_t) s];
                ggml_tensor * weights = ggml_view_3d(gctx, probs, seg.len, P, e_heads,
                    probs->nb[1], probs->nb[2], (size_t) prob_offset * probs->nb[0]);
                ggml_tensor * v3 = ggml_permute(gctx,
                    ggml_reshape_3d(gctx, seg.v, e_hd, e_kv_heads, seg.len), 0, 2, 1, 3);
                ggml_tensor * v_t = ggml_cont(gctx, ggml_transpose(gctx, v3));
                ggml_tensor * part = ggml_mul_mat(gctx, v_t, weights);
                weighted = weighted ? ggml_add(gctx, weighted, part) : part;
                prob_offset += seg.len;
            }
            attn = ggml_permute(gctx, weighted, 0, 2, 1, 3);
            attn = ggml_reshape_2d(gctx, ggml_cont(gctx, attn), e_heads * e_hd, P);
        } else {
            // Throughput/reference family: preserve the proven fused flash graph.
            ggml_tensor * k_win = k_real;
            ggml_tensor * v_win = v_real;
            if (old_len > 0) {
                const int32_t cap = VOXTRAL_ENC_KV_CAP;
                const int32_t os0 = (int32_t) (valid_old_start % cap);
                const int32_t of0 = std::min(old_len, cap - os0);
                ggml_tensor * k_old;
                ggml_tensor * v_old;
                if (of0 >= old_len) {
                    k_old = ggml_cont(gctx, ggml_view_2d(gctx, ring_k, kv_dim, old_len, k_col, (size_t) i * k_layer_stride + (size_t) os0 * k_col));
                    v_old = ggml_cont(gctx, ggml_view_2d(gctx, ring_v, kv_dim, old_len, v_col, (size_t) i * v_layer_stride + (size_t) os0 * v_col));
                } else {
                    const int32_t l1 = old_len - of0;
                    ggml_tensor * k0 = ggml_cont(gctx, ggml_view_2d(gctx, ring_k, kv_dim, of0, k_col, (size_t) i * k_layer_stride + (size_t) os0 * k_col));
                    ggml_tensor * k1 = ggml_cont(gctx, ggml_view_2d(gctx, ring_k, kv_dim, l1, k_col, (size_t) i * k_layer_stride));
                    ggml_tensor * v0 = ggml_cont(gctx, ggml_view_2d(gctx, ring_v, kv_dim, of0, v_col, (size_t) i * v_layer_stride + (size_t) os0 * v_col));
                    ggml_tensor * v1 = ggml_cont(gctx, ggml_view_2d(gctx, ring_v, kv_dim, l1, v_col, (size_t) i * v_layer_stride));
                    k_old = ggml_concat(gctx, k0, k1, 1);
                    v_old = ggml_concat(gctx, v0, v1, 1);
                }
                k_win = ggml_concat(gctx, k_old, k_win, 1);
                v_win = ggml_concat(gctx, v_old, v_win, 1);
                if (i == 0 && out_materialized_frames) *out_materialized_frames += old_len;
            }
            if (ring_k->type == GGML_TYPE_F16) {
                // PAD is F32-only in the pinned backend. Fully-masked headroom
                // can use same-typed dummy block columns without changing the
                // fused attention result.
                if (prefix_len > 0) {
                    GGML_ASSERT(prefix_len <= q_offset);
                    k_win = ggml_concat(gctx,
                        ggml_view_2d(gctx, k_store, kv_dim, prefix_len, k_store->nb[1], 0),
                        k_win, 1);
                    v_win = ggml_concat(gctx,
                        ggml_view_2d(gctx, v_store, kv_dim, prefix_len, v_store->nb[1], 0),
                        v_win, 1);
                }
                if (future_len > 0) {
                    const int32_t future_col = q_offset + N;
                    k_win = ggml_concat(gctx, k_win,
                        ggml_view_2d(gctx, k_store, kv_dim, future_len, k_store->nb[1],
                            (size_t) future_col * k_store->nb[1]), 1);
                    v_win = ggml_concat(gctx, v_win,
                        ggml_view_2d(gctx, v_store, kv_dim, future_len, v_store->nb[1],
                            (size_t) future_col * v_store->nb[1]), 1);
                }
            } else {
                if (prefix_len > 0) {
                    k_win = ggml_pad_ext(gctx, k_win, 0, 0, prefix_len, 0, 0, 0, 0, 0);
                    v_win = ggml_pad_ext(gctx, v_win, 0, 0, prefix_len, 0, 0, 0, 0, 0);
                }
                if (future_len > 0) {
                    k_win = ggml_pad_ext(gctx, k_win, 0, 0, 0, future_len, 0, 0, 0, 0);
                    v_win = ggml_pad_ext(gctx, v_win, 0, 0, 0, future_len, 0, 0, 0, 0);
                }
            }
            ggml_tensor * k3 = ggml_permute(gctx,
                ggml_reshape_3d(gctx, k_win, e_hd, e_kv_heads, L), 0, 2, 1, 3);
            ggml_tensor * v3 = ggml_permute(gctx,
                ggml_reshape_3d(gctx, v_win, e_hd, e_kv_heads, L), 0, 2, 1, 3);
            ggml_tensor * mask_f16 = ggml_cast(gctx, enc_mask, GGML_TYPE_F16);
            attn = ggml_flash_attn_ext(gctx, q3, k3, v3, mask_f16, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            attn = ggml_reshape_2d(gctx, ggml_cont(gctx, attn), e_heads * e_hd, P);
        }

        ggml_tensor * attn_proj = ggml_add(gctx, ggml_mul_mat(gctx, Lyr.attn_o_weight, attn), Lyr.attn_o_bias);
        x = ggml_add(gctx, residual, attn_proj);

        // SwiGLU FFN (realtime).
        residual = x;
        x_norm = enc_apply_norm(gctx, x, Lyr.ffn_norm_weight, Lyr.ffn_norm_bias, hp.enc_norm_eps, false);
        ggml_tensor * gate = ggml_silu(gctx, ggml_mul_mat(gctx, Lyr.ffn_w1_weight, x_norm));
        ggml_tensor * up   = ggml_mul_mat(gctx, Lyr.ffn_w3_weight, x_norm);
        ggml_tensor * ffn  = ggml_mul_mat(gctx, Lyr.ffn_w2_weight, ggml_mul(gctx, gate, up));
        if (Lyr.ffn_w2_bias) ffn = ggml_add(gctx, ffn, Lyr.ffn_w2_bias);
        x = ggml_add(gctx, residual, ffn);
    }

    // Final norm + stage only the real subrange into encoder_chunk_output.
    x = enc_apply_norm(gctx, x, model->enc_norm_weight, model->enc_norm_bias, hp.enc_norm_eps, false);
    ggml_tensor * out_view = ggml_view_2d(gctx, ctx->encoder_chunk_output, VOXTRAL_ENC_DIM, N,
        ctx->encoder_chunk_output->nb[1], 0);
    ggml_tensor * real_x = ggml_view_2d(gctx, x, hp.enc_dim, N, x->nb[1],
                                        (size_t) q_offset * x->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, real_x, out_view));
    // The device-resident encoder-output ring is populated by a separate
    // device->device copy from encoder_chunk_output after this graph completes
    // (see copy_chunk_to_enc_out_ring). Folding a second cpy from `x` into this
    // graph is unsafe: ggml_backend_sched can reuse x's buffer after the first
    // consumer, corrupting the ring copy.
    return gf;
}

// Fixed 4-row production encoder graph. Absolute positions, physical KV slots,
// output-ring slots and the causal mask are inputs; graph topology is invariant
// for every steady 80 ms audio quantum. Attention reads the 878-slot physical
// ring directly and masks the exact 750-frame logical window, so no K/V gather,
// concat or rollover copy is required.
static ggml_cgraph * build_encoder_steady_graph(
    voxtral_context * ctx, ggml_tensor * ring_k, ggml_tensor * ring_v,
    ggml_context * gctx) {
    constexpr int32_t P = 4;
    constexpr int32_t conv_len = 2 * P + 4;
    voxtral_model * model = ctx->model;
    const auto & hp = model->hp;
    const int32_t e_heads = hp.enc_heads;
    const int32_t e_kv_heads = hp.enc_kv_heads;
    const int32_t e_hd = hp.enc_head_dim;
    const int32_t kv_dim = e_kv_heads * e_hd;
    const int32_t cap = VOXTRAL_ENC_KV_CAP;
    const float scale = 1.0f / sqrtf((float) e_hd);

    ggml_cgraph * gf = ggml_new_graph_custom(
        gctx, GGML_DEFAULT_GRAPH_SIZE * 8, false);
    ggml_tensor * mel_input = ggml_new_tensor_3d(
        gctx, GGML_TYPE_F32, conv_len, VOXTRAL_NUM_MEL_BINS, 1);
    ggml_set_name(mel_input, "mel_input");
    ggml_backend_sched_set_tensor_backend(
        ctx->sched_encoder_steady, mel_input, ctx->backend);
    int32_t conv0_len = 0;
    ggml_tensor * conv0 = causal_conv1d_graph(
        gctx, mel_input, conv_len, model->enc_conv0_weight,
        model->enc_conv0_bias, hp.enc_dim, 3, 1, conv0_len, false);
    conv0 = ggml_gelu_erf(gctx, conv0);
    int32_t conv1_len = 0;
    ggml_tensor * conv1 = causal_conv1d_graph(
        gctx, conv0, conv0_len, model->enc_conv1_weight,
        model->enc_conv1_bias, hp.enc_dim, 3, 2, conv1_len, false);
    conv1 = ggml_gelu_erf(gctx, conv1);
    ggml_tensor * x_local = ggml_view_3d(
        gctx, conv1, P, hp.enc_dim, 1, conv1->nb[1], conv1->nb[2],
        (size_t) 2 * conv1->nb[0]);
    ggml_tensor * x = ggml_cont(
        gctx, ggml_permute(gctx, x_local, 1, 0, 2, 3));
    x = ggml_reshape_2d(gctx, x, hp.enc_dim, P);

    ggml_tensor * positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, P);
    ggml_set_name(positions, "enc_positions");
    ggml_backend_sched_set_tensor_backend(
        ctx->sched_encoder_steady, positions, ctx->backend);
    ggml_tensor * kv_rows = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, P);
    ggml_set_name(kv_rows, "enc_kv_rows");
    ggml_backend_sched_set_tensor_backend(
        ctx->sched_encoder_steady, kv_rows, ctx->backend);
    ggml_tensor * output_rows = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, P);
    ggml_set_name(output_rows, "enc_output_rows");
    ggml_backend_sched_set_tensor_backend(
        ctx->sched_encoder_steady, output_rows, ctx->backend);
    ggml_tensor * mask = ggml_new_tensor_2d(
        gctx, GGML_TYPE_F32, cap, P);
    ggml_set_name(mask, "enc_kv_mask");
    ggml_backend_sched_set_tensor_backend(
        ctx->sched_encoder_steady, mask, ctx->backend);

    for (int32_t layer = 0; layer < hp.enc_layers; ++layer) {
        auto & L = model->enc_layers[layer];
        ggml_tensor * residual = x;
        ggml_tensor * norm = enc_apply_norm(
            gctx, x, L.attn_norm_weight, L.attn_norm_bias,
            hp.enc_norm_eps, false);
        ggml_tensor * q = ggml_add(
            gctx, ggml_mul_mat(gctx, L.attn_q_weight, norm), L.attn_q_bias);
        ggml_tensor * k = ggml_mul_mat(gctx, L.attn_k_weight, norm);
        ggml_tensor * v = ggml_add(
            gctx, ggml_mul_mat(gctx, L.attn_v_weight, norm), L.attn_v_bias);
        q = ggml_reshape_3d(gctx, q, e_hd, e_heads, P);
        k = ggml_reshape_3d(gctx, k, e_hd, e_kv_heads, P);
        q = ggml_rope_ext(gctx, q, positions, nullptr,
            e_hd, 0, 0, hp.enc_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(gctx, k, positions, nullptr,
            e_hd, 0, 0, hp.enc_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * k_new = ggml_cont(
            gctx, ggml_reshape_2d(gctx, k, kv_dim, P));
        ggml_tensor * v_new = ggml_reshape_2d(gctx, v, kv_dim, P);
        ggml_tensor * k_layer = ggml_view_2d(
            gctx, ring_k, kv_dim, cap, ring_k->nb[1],
            (size_t) layer * ring_k->nb[2]);
        ggml_tensor * v_layer = ggml_view_2d(
            gctx, ring_v, kv_dim, cap, ring_v->nb[1],
            (size_t) layer * ring_v->nb[2]);
        ggml_tensor * k_cache = ggml_set_rows(
            gctx, k_layer, k_new, kv_rows);
        ggml_tensor * v_cache = ggml_set_rows(
            gctx, v_layer, v_new, kv_rows);

        ggml_tensor * q3 = ggml_permute(gctx, q, 0, 2, 1, 3);
        ggml_tensor * k3 = ggml_permute(
            gctx, ggml_reshape_3d(
                gctx, k_cache, e_hd, e_kv_heads, cap), 0, 2, 1, 3);
        ggml_tensor * scores = ggml_mul_mat(gctx, k3, ggml_cont(gctx, q3));
        ggml_tensor * probs = ggml_soft_max_ext(
            gctx, scores, mask, scale, 0.0f);
        ggml_tensor * v3 = ggml_permute(
            gctx, ggml_reshape_3d(
                gctx, v_cache, e_hd, e_kv_heads, cap), 0, 2, 1, 3);
        ggml_tensor * weighted = ggml_mul_mat(
            gctx, ggml_cont(gctx, ggml_transpose(gctx, v3)), probs);
        ggml_tensor * attn = ggml_permute(gctx, weighted, 0, 2, 1, 3);
        attn = ggml_reshape_2d(
            gctx, ggml_cont(gctx, attn), e_heads * e_hd, P);
        ggml_tensor * projected = ggml_add(
            gctx, ggml_mul_mat(gctx, L.attn_o_weight, attn), L.attn_o_bias);
        x = ggml_add(gctx, residual, projected);

        residual = x;
        norm = enc_apply_norm(
            gctx, x, L.ffn_norm_weight, L.ffn_norm_bias,
            hp.enc_norm_eps, false);
        ggml_tensor * gate = ggml_silu(
            gctx, ggml_mul_mat(gctx, L.ffn_w1_weight, norm));
        ggml_tensor * up = ggml_mul_mat(gctx, L.ffn_w3_weight, norm);
        ggml_tensor * ffn = ggml_mul_mat(
            gctx, L.ffn_w2_weight, ggml_mul(gctx, gate, up));
        if (L.ffn_w2_bias) ffn = ggml_add(gctx, ffn, L.ffn_w2_bias);
        x = ggml_add(gctx, residual, ffn);
    }
    x = enc_apply_norm(
        gctx, x, model->enc_norm_weight, model->enc_norm_bias,
        hp.enc_norm_eps, false);
    ggml_tensor * stored = ggml_set_rows(
        gctx, ctx->enc_out_ring, x, output_rows);
    ggml_build_forward_expand(gf, stored);
    return gf;
}

// ============================================================================
// Graph Building: Adapter
// ============================================================================

static ggml_cgraph * build_adapter_graph(
    voxtral_context * ctx,
    ggml_context * gctx)
{
    voxtral_model * model = ctx->model;
    const int32_t enc_seq = ctx->enc_seq_used;
    const int32_t dec_seq = enc_seq / VOXTRAL_DOWNSAMPLE_FACTOR;

    ggml_cgraph * gf = ggml_new_graph(gctx);

    // Read encoder_output: [enc_dim, enc_seq]
    ggml_tensor * enc_out = ggml_view_2d(gctx, ctx->encoder_output,
        VOXTRAL_ENC_DIM, enc_seq,
        ctx->encoder_output->nb[1], 0); // [enc_dim, enc_seq]

    // Reshape for downsample: [enc_dim, enc_seq] -> [enc_dim * 4, enc_seq/4]
    ggml_tensor * x = ggml_reshape_2d(gctx, enc_out,
        VOXTRAL_ENC_DIM * VOXTRAL_DOWNSAMPLE_FACTOR, dec_seq); // [enc_dim*4, dec_seq]

    // Linear 0: [enc_dim*4, dec_seq] -> [dec_dim, dec_seq]
    x = ggml_mul_mat(gctx, model->adapter_0_weight, x); // [dec_dim, dec_seq]
    x = ggml_gelu_erf(gctx, x); // [dec_dim, dec_seq]

    // Linear 2: [dec_dim, dec_seq] -> [dec_dim, dec_seq]
    x = ggml_mul_mat(gctx, model->adapter_2_weight, x); // [dec_dim, dec_seq]

    // Copy to persistent decoder_memory
    ggml_tensor * dec_mem_view = ggml_view_2d(gctx, ctx->decoder_memory,
        ctx->model->hp.dec_dim, dec_seq,
        ctx->decoder_memory->nb[1], 0); // [dec_dim, dec_seq]
    ggml_tensor * cpy = ggml_cpy(gctx, x, dec_mem_view);
    ggml_build_forward_expand(gf, cpy);

    ctx->dec_seq_len = dec_seq;

    return gf;
}

// Session 7: device-resident incremental adapter over a contiguous run of
// `n_groups` complete encoder-output groups (DOWNSAMPLE_FACTOR frames each) that are
// resident in enc_out_ring starting at ring column `enc_col`, writing the resulting
// audio embeddings into audio_emb_ring starting at column `aemb_col`. The caller
// splits any run that would wrap either ring, so both views here are contiguous and
// the frame layout is byte-identical to the batch adapter's [enc_dim, enc_seq] ->
// [enc_dim*4, dec_seq] reshape. No host copy of encoder output or audio embeddings.
static ggml_cgraph * build_adapter_ring_graph(voxtral_context * ctx, ggml_context * gctx,
                                              int32_t enc_col, int32_t aemb_col, int32_t n_groups) {
    voxtral_model * model = ctx->model;
    ggml_cgraph * gf = ggml_new_graph(gctx);

    // enc_out_ring[:, enc_col : enc_col + 4*n_groups] — contiguous (no wrap here).
    // Materialize to a fresh contiguous tensor before the reshape: unlike the batch
    // adapter (which reshapes an offset-0 view of encoder_output), this view starts
    // at an arbitrary ring column, and reshaping a non-zero-offset view directly
    // into mul_mat is not reliable across backends.
    ggml_tensor * enc_out = ggml_cont(gctx, ggml_view_2d(gctx, ctx->enc_out_ring,
        VOXTRAL_ENC_DIM, n_groups * VOXTRAL_DOWNSAMPLE_FACTOR,
        ctx->enc_out_ring->nb[1], (size_t) enc_col * ctx->enc_out_ring->nb[1]));
    ggml_tensor * x = ggml_reshape_2d(gctx, enc_out,
        VOXTRAL_ENC_DIM * VOXTRAL_DOWNSAMPLE_FACTOR, n_groups); // [enc_dim*4, n_groups]

    x = ggml_mul_mat(gctx, model->adapter_0_weight, x); // [dec_dim, n_groups]
    x = ggml_gelu_erf(gctx, x);
    x = ggml_mul_mat(gctx, model->adapter_2_weight, x); // [dec_dim, n_groups]

    ggml_tensor * dst = ggml_view_2d(gctx, ctx->audio_emb_ring,
        model->hp.dec_dim, n_groups,
        ctx->audio_emb_ring->nb[1], (size_t) aemb_col * ctx->audio_emb_ring->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, dst));
    return gf;
}

static ggml_cgraph * build_adapter_ring_graph_reusable(
    voxtral_context * ctx, ggml_context * gctx, int32_t n_groups) {
    voxtral_model * model = ctx->model;
    ggml_cgraph * gf = ggml_new_graph(gctx);

    ggml_tensor * encoder_rows = ggml_new_tensor_1d(
        gctx, GGML_TYPE_I32,
        n_groups * VOXTRAL_DOWNSAMPLE_FACTOR);
    ggml_set_name(encoder_rows, "adapter_encoder_rows");
    ggml_backend_sched_set_tensor_backend(
        n_groups == 1 ? ctx->sched_adapter_group : ctx->sched_adapter_batch,
        encoder_rows, ctx->backend);
    ggml_tensor * audio_rows = ggml_new_tensor_1d(
        gctx, GGML_TYPE_I32, n_groups);
    ggml_set_name(audio_rows, "adapter_audio_rows");
    ggml_backend_sched_set_tensor_backend(
        n_groups == 1 ? ctx->sched_adapter_group : ctx->sched_adapter_batch,
        audio_rows, ctx->backend);

    ggml_tensor * enc_out = ggml_get_rows(
        gctx, ctx->enc_out_ring, encoder_rows);
    ggml_tensor * x = ggml_reshape_2d(
        gctx, enc_out,
        VOXTRAL_ENC_DIM * VOXTRAL_DOWNSAMPLE_FACTOR, n_groups);
    x = ggml_mul_mat(gctx, model->adapter_0_weight, x);
    x = ggml_gelu_erf(gctx, x);
    x = ggml_mul_mat(gctx, model->adapter_2_weight, x);
    ggml_tensor * stored = ggml_set_rows(
        gctx, ctx->audio_emb_ring, x, audio_rows);
    ggml_build_forward_expand(gf, stored);
    return gf;
}

// ============================================================================
// Graph Building: Decoder (common layer forward)
// ============================================================================

// Build one decoder layer. Returns updated hidden state.
// For prefill: n_tokens > 1, positions = [0..n_tokens-1]
// For step: n_tokens = 1
static ggml_tensor * build_decoder_layer(
    voxtral_context     * ctx,
    ggml_context * gctx,
    ggml_cgraph  * gf,
    ggml_tensor  * x,          // [dec_dim, n_tokens]
    ggml_tensor  * positions,  // [n_tokens] int32
    ggml_tensor  * time_emb,   // [dec_dim]
    int32_t layer_idx,
    int32_t n_tokens,
    const voxtral_decoder_kv_plan_t & kv_plan,
    ggml_tensor  * attn_mask,  // [n_kv, n_tokens] or nullptr
    ggml_tensor  * dynamic_kv_slot = nullptr) // [1] I32 for reusable step graph
{
    voxtral_model * model = ctx->model;
    const auto & hp = model->hp;
    auto & L = model->dec_layers[layer_idx];

    const int32_t head_dim   = hp.dec_head_dim;
    const int32_t n_heads    = hp.dec_heads;
    const int32_t n_kv_heads = hp.dec_kv_heads;
    const int32_t kv_dim     = n_kv_heads * head_dim;
    const float   norm_eps   = hp.dec_norm_eps;
    const float   dec_rope_theta = hp.dec_rope_theta;

    // Pre-attention RMS norm
    ggml_tensor * residual = x; // [dec_dim, n_tokens]
    ggml_tensor * x_norm = ggml_rms_norm(gctx, x, norm_eps); // [dec_dim, n_tokens]
    x_norm = ggml_mul(gctx, x_norm, L.attn_norm_weight); // [dec_dim, n_tokens]

    // Q, K, V (no bias in decoder)
    ggml_tensor * q = ggml_mul_mat(gctx, L.attn_q_weight, x_norm); // [dec_heads*head_dim, n_tokens]
    ggml_tensor * k = ggml_mul_mat(gctx, L.attn_k_weight, x_norm); // [kv_dim, n_tokens]
    ggml_tensor * v = ggml_mul_mat(gctx, L.attn_v_weight, x_norm); // [kv_dim, n_tokens]

    // Reshape for RoPE: [head_dim, n_heads, n_tokens]
    q = ggml_reshape_3d(gctx, q, head_dim, n_heads, n_tokens);
    k = ggml_reshape_3d(gctx, k, head_dim, n_kv_heads, n_tokens);

    // RoPE (interleaved, mode=0)
    q = ggml_rope_ext(gctx, q, positions, nullptr,
        head_dim, 0, 0, dec_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(gctx, k, positions, nullptr,
        head_dim, 0, 0, dec_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Flatten Q back: [head_dim, n_heads, n_tokens] -> [n_heads*head_dim, n_tokens]
    q = ggml_cont(gctx, ggml_reshape_2d(gctx, q, n_heads * head_dim, n_tokens));
    k = ggml_cont(gctx, ggml_reshape_2d(gctx, k, kv_dim, n_tokens));

    // Store only the newly projected K/V columns at absolute_position % capacity.
    // A write can straddle slot zero, so source and destination are split into at
    // most two views.  This is the only cache traffic at rollover.
    ggml_tensor * dynamic_k_layer = nullptr;
    ggml_tensor * dynamic_v_layer = nullptr;
    if (dynamic_kv_slot) {
        assert(n_tokens == 1);
        const int32_t capacity = ctx->decoder_kv_configured_capacity;
        ggml_tensor * k_layer_view = ggml_view_2d(
            gctx, ctx->kv_self_k, kv_dim, capacity, ctx->kv_self_k->nb[1],
            (size_t) layer_idx * ctx->kv_self_k->nb[2]);
        ggml_tensor * v_layer_view = ggml_view_2d(
            gctx, ctx->kv_self_v, kv_dim, capacity, ctx->kv_self_v->nb[1],
            (size_t) layer_idx * ctx->kv_self_v->nb[2]);
        // SET_ROWS converts the projected F32 row directly into the FP16 cache
        // and returns a dependency-carrying view of the full layer cache.
        dynamic_k_layer = ggml_set_rows(gctx, k_layer_view, k, dynamic_kv_slot);
        dynamic_v_layer = ggml_set_rows(gctx, v_layer_view, v, dynamic_kv_slot);
    } else {
        int32_t src_col = 0;
        for (int32_t s = 0; s < kv_plan.write_nseg; ++s) {
            const int32_t slot = kv_plan.write_seg[s].slot;
            const int32_t len  = kv_plan.write_seg[s].len;
            ggml_tensor * k_src = ggml_view_2d(gctx, k, kv_dim, len, k->nb[1],
                                                (size_t) src_col * k->nb[1]);
            ggml_tensor * v_src = ggml_view_2d(gctx, v, kv_dim, len, v->nb[1],
                                                (size_t) src_col * v->nb[1]);
            ggml_tensor * k_dst = ggml_view_2d(gctx, ctx->kv_self_k, kv_dim, len,
                ctx->kv_self_k->nb[1], (size_t) layer_idx * ctx->kv_self_k->nb[2] +
                (size_t) slot * ctx->kv_self_k->nb[1]);
            ggml_tensor * v_dst = ggml_view_2d(gctx, ctx->kv_self_v, kv_dim, len,
                ctx->kv_self_v->nb[1], (size_t) layer_idx * ctx->kv_self_v->nb[2] +
                (size_t) slot * ctx->kv_self_v->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(gctx, k_src, k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(gctx, v_src, v_dst));
            src_col += len;
        }
    }

    const int32_t n_kv = (int32_t) kv_plan.used_after;

    // Flash attention with GQA
    // Q: [n_heads*head_dim, n_tokens] -> [head_dim, n_heads, n_tokens] -> [head_dim, n_tokens, n_heads]
    ggml_tensor * q3 = ggml_reshape_3d(gctx, q, head_dim, n_heads, n_tokens);
    q3 = ggml_permute(gctx, q3, 0, 2, 1, 3); // [head_dim, n_tokens, n_heads]

    const float scale = 1.0f / sqrtf((float) head_dim);
    const size_t k_layer = (size_t) layer_idx * ctx->kv_self_k->nb[2];
    const size_t v_layer = (size_t) layer_idx * ctx->kv_self_v->nb[2];

    auto cache_view = [&](ggml_tensor * cache, size_t layer_offset,
                          const voxtral_decoder_kv_seg & seg) {
        return ggml_view_2d(gctx, cache, kv_dim, seg.len, cache->nb[1],
                            layer_offset + (size_t) seg.slot * cache->nb[1]);
    };
    auto key_3d = [&](const voxtral_decoder_kv_seg & seg) {
        ggml_tensor * view = cache_view(ctx->kv_self_k, k_layer, seg);
        return ggml_permute(gctx,
            ggml_reshape_3d(gctx, view, head_dim, n_kv_heads, seg.len), 0, 2, 1, 3);
    };
    auto value_3d = [&](const voxtral_decoder_kv_seg & seg) {
        ggml_tensor * view = cache_view(ctx->kv_self_v, v_layer, seg);
        return ggml_permute(gctx,
            ggml_reshape_3d(gctx, view, head_dim, n_kv_heads, seg.len), 0, 2, 1, 3);
    };

    ggml_tensor * attn_out = nullptr;
    const char * ring_attention_mode = std::getenv("VOXTRAL_DECODER_RING_ATTENTION");
    const bool logical_concat_oracle = ring_attention_mode &&
        std::strcmp(ring_attention_mode, "logical") == 0;
    const bool logical_manual_oracle = ring_attention_mode &&
        std::strcmp(ring_attention_mode, "manual") == 0;
    if (dynamic_kv_slot) {
        const int32_t capacity = ctx->decoder_kv_configured_capacity;
        ggml_tensor * k3 = ggml_permute(gctx,
            ggml_reshape_3d(gctx, dynamic_k_layer,
                            head_dim, n_kv_heads, capacity), 0, 2, 1, 3);
        ggml_tensor * v3 = ggml_permute(gctx,
            ggml_reshape_3d(gctx, dynamic_v_layer,
                            head_dim, n_kv_heads, capacity), 0, 2, 1, 3);
        ggml_tensor * attn_mask_f16 = attn_mask
            ? ggml_cast(gctx, attn_mask, GGML_TYPE_F16) : nullptr;
        attn_out = ggml_flash_attn_ext(gctx, q3, k3, v3, attn_mask_f16,
                                       scale, 0.0f, 0.0f);
        attn_out = ggml_reshape_2d(gctx, ggml_cont(gctx, attn_out),
                                   n_heads * head_dim, 1);
    } else if (kv_plan.read_nseg == 1) {
        // The common pre-wrap path is byte-for-byte the old fused attention: one
        // logically ordered contiguous cache view and the same reduction shape.
        ggml_tensor * k3 = key_3d(kv_plan.read_seg[0]);
        ggml_tensor * v3 = value_3d(kv_plan.read_seg[0]);
        ggml_tensor * attn_mask_f16 = attn_mask
            ? ggml_cast(gctx, attn_mask, GGML_TYPE_F16) : nullptr;
        attn_out = ggml_flash_attn_ext(gctx, q3, k3, v3, attn_mask_f16,
                                       scale, 0.0f, 0.0f);
        attn_out = ggml_reshape_2d(gctx, ggml_cont(gctx, attn_out),
                                   n_heads * head_dim, n_tokens);
    } else if (logical_concat_oracle) {
        // Test-only bounded reference: materialize the reduced-capacity logical
        // window and feed it through the SAME fused attention kernel. This is
        // intentionally unavailable at the 8192-slot production capacity; it
        // exists only to validate physical permutation/eviction semantics
        // without conflating ordering with a different manual attention kernel.
        assert(ctx->decoder_kv_configured_capacity < VOXTRAL_DEC_WINDOW &&
               kv_plan.read_nseg == 2 && n_tokens == 1 && attn_mask == nullptr);
        ggml_tensor * k_logical = ggml_concat(
            gctx,
            cache_view(ctx->kv_self_k, k_layer, kv_plan.read_seg[0]),
            cache_view(ctx->kv_self_k, k_layer, kv_plan.read_seg[1]), 1);
        ggml_tensor * v_logical = ggml_concat(
            gctx,
            cache_view(ctx->kv_self_v, v_layer, kv_plan.read_seg[0]),
            cache_view(ctx->kv_self_v, v_layer, kv_plan.read_seg[1]), 1);
        ggml_tensor * k3 = ggml_permute(gctx,
            ggml_reshape_3d(gctx, k_logical, head_dim, n_kv_heads, n_kv),
            0, 2, 1, 3);
        ggml_tensor * v3 = ggml_permute(gctx,
            ggml_reshape_3d(gctx, v_logical, head_dim, n_kv_heads, n_kv),
            0, 2, 1, 3);
        attn_out = ggml_flash_attn_ext(gctx, q3, k3, v3, nullptr,
                                       scale, 0.0f, 0.0f);
        attn_out = ggml_reshape_2d(gctx, ggml_cont(gctx, attn_out),
                                   n_heads * head_dim, 1);
    } else if (!logical_manual_oracle) {
        // Once full, the physical ring [0, capacity) contains exactly the same
        // (K,V) pairs as the logical [oldest, newest] window, only cyclically
        // permuted. Single-query attention has no position-dependent mask and is
        // permutation-equivariant when K and V are permuted together; absolute
        // RoPE is already embedded in every K. Therefore one full physical view
        // is wrap-aware without materialising or moving any cache bytes, and it
        // retains the fast fused Vulkan attention path. The test-only concat
        // branch above keeps oldest-to-newest order with the same fused kernel.
        assert(kv_plan.read_nseg == 2 && n_tokens == 1 && attn_mask == nullptr &&
               kv_plan.used_after == ctx->decoder_kv.capacity);
        const voxtral_decoder_kv_seg physical = {
            /*slot=*/0, /*len=*/(int32_t) kv_plan.used_after
        };
        ggml_tensor * k3 = key_3d(physical);
        ggml_tensor * v3 = value_3d(physical);
        attn_out = ggml_flash_attn_ext(gctx, q3, k3, v3, nullptr,
                                       scale, 0.0f, 0.0f);
        attn_out = ggml_reshape_2d(gctx, ggml_cont(gctx, attn_out),
                                   n_heads * head_dim, 1);
    } else {
        // Optional manual diagnostic: concatenate only attention SCORES
        // (kilobytes), never
        // the multi-gigabyte K/V cache.  Softmax sees [oldest..newest] in logical
        // order; its weights are split back across the two V views and reduced.
        // Decoder rollover occurs only on single-token steps, so no causal mask is
        // required here.  Absolute RoPE positions are already embedded in K/Q.
        assert(kv_plan.read_nseg == 2 && n_tokens == 1 && attn_mask == nullptr);
        ggml_tensor * q_cont = ggml_cont(gctx, q3);
        ggml_tensor * k0 = key_3d(kv_plan.read_seg[0]);
        ggml_tensor * k1 = key_3d(kv_plan.read_seg[1]);
        ggml_tensor * score0 = ggml_mul_mat(gctx, k0, q_cont);
        ggml_tensor * score1 = ggml_mul_mat(gctx, k1, q_cont);
        ggml_tensor * scores = ggml_concat(gctx, score0, score1, 0);
        ggml_tensor * probs = ggml_soft_max_ext(gctx, scores, nullptr, scale, 0.0f);

        const int32_t n0 = kv_plan.read_seg[0].len;
        const int32_t n1 = kv_plan.read_seg[1].len;
        ggml_tensor * p0 = ggml_view_3d(gctx, probs, n0, 1, n_heads,
                                        probs->nb[1], probs->nb[2], 0);
        ggml_tensor * p1 = ggml_view_3d(gctx, probs, n1, 1, n_heads,
                                        probs->nb[1], probs->nb[2],
                                        (size_t) n0 * probs->nb[0]);
        auto weighted_value = [&](const voxtral_decoder_kv_seg & seg, ggml_tensor * weights) {
            ggml_tensor * v3 = value_3d(seg);
            ggml_tensor * vt = ggml_cont(gctx, ggml_transpose(gctx, v3));
            return ggml_mul_mat(gctx, vt, weights); // [head_dim, 1, n_heads]
        };
        ggml_tensor * out0 = weighted_value(kv_plan.read_seg[0], p0);
        ggml_tensor * out1 = weighted_value(kv_plan.read_seg[1], p1);
        ggml_tensor * summed = ggml_add(gctx, out0, out1);
        summed = ggml_permute(gctx, summed, 0, 2, 1, 3);
        attn_out = ggml_reshape_2d(gctx, ggml_cont(gctx, summed),
                                   n_heads * head_dim, 1);
    }

    // Output projection + residual
    ggml_tensor * attn_proj = ggml_mul_mat(gctx, L.attn_o_weight, attn_out); // [dec_dim, n_tokens]
    x = ggml_add(gctx, residual, attn_proj); // [dec_dim, n_tokens]

    // Pre-FFN RMS norm
    residual = x; // [dec_dim, n_tokens]
    ggml_tensor * h_norm = ggml_rms_norm(gctx, x, norm_eps); // [dec_dim, n_tokens]
    h_norm = ggml_mul(gctx, h_norm, L.ffn_norm_weight); // [dec_dim, n_tokens]

    // Ada time conditioning: h_norm *= (1 + ada_mlp(time_emb)).
    // The offline Ministral decoder has no time conditioning just RMSNorm.
    if (model->hp.ada_t_cond && L.ada0_weight && L.ada2_weight && time_emb) {
        ggml_tensor * ada_hidden = ggml_mul_mat(gctx, L.ada0_weight, time_emb); // [ada_dim]
        ada_hidden = ggml_gelu_erf(gctx, ada_hidden); // [ada_dim]
        ggml_tensor * ada_scale = ggml_mul_mat(gctx, L.ada2_weight, ada_hidden); // [dec_dim]

        // h_norm * (1 + ada_scale) = h_norm + h_norm * ada_scale
        ggml_tensor * scaled = ggml_mul(gctx, h_norm, ada_scale); // [dec_dim, n_tokens]
        h_norm = ggml_add(gctx, h_norm, scaled); // [dec_dim, n_tokens]
    }

    // SwiGLU FFN
    ggml_tensor * gate = ggml_mul_mat(gctx, L.ffn_w1_weight, h_norm); // [dec_hidden, n_tokens]
    gate = ggml_silu(gctx, gate); // [dec_hidden, n_tokens]
    ggml_tensor * up = ggml_mul_mat(gctx, L.ffn_w3_weight, h_norm); // [dec_hidden, n_tokens]
    ggml_tensor * ffn_out = ggml_mul(gctx, gate, up); // [dec_hidden, n_tokens]
    ffn_out = ggml_mul_mat(gctx, L.ffn_w2_weight, ffn_out); // [dec_dim, n_tokens]

    x = ggml_add(gctx, residual, ffn_out); // [dec_dim, n_tokens]

    return x;
}

// ============================================================================
// Graph Building: Decoder Prefill
// ============================================================================

static ggml_cgraph * build_decoder_prefill_graph(
    voxtral_context     * ctx,
    ggml_context * gctx,
    int32_t               n_tokens)  // number of prompt tokens
{
    voxtral_model * model = ctx->model;
    voxtral_decoder_kv_plan_t kv_plan;
    if (!voxtral_decoder_kv_plan(ctx->decoder_kv,
                                 ctx->decoder_kv.next_absolute_position,
                                 n_tokens, kv_plan)) return nullptr;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    // Token IDs input: [n_tokens] int32
    ggml_tensor * token_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(token_ids, "token_ids");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, token_ids, ctx->backend);

    // Position indices: [n_tokens] int32
    ggml_tensor * positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, positions, ctx->backend);

    // Time embedding: [dec_dim]
    ggml_tensor * time_emb = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, VOXTRAL_DEC_DIM);
    ggml_set_name(time_emb, "time_emb");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, time_emb, ctx->backend);

    // Token embeddings: [dec_dim, n_tokens]
    ggml_tensor * tok_emb = ggml_get_rows(gctx, model->tok_embeddings_weight, token_ids); // [dec_dim, n_tokens]

    // Audio embeddings: linear decoder_memory (batch/finish path) or, for the
    // incremental decoder, the device-resident audio_emb_ring. Prefill only ever
    // runs once at the very start of a stream (positions [0, n_tokens)), before the
    // ring has wrapped, so a contiguous view at column 0 is valid for both.
    ggml_tensor * audio_src = ctx->dec_audio_src ? ctx->dec_audio_src : ctx->decoder_memory;
    ggml_tensor * audio_emb = ggml_view_2d(gctx, audio_src,
        VOXTRAL_DEC_DIM, n_tokens,
        audio_src->nb[1], 0); // [dec_dim, n_tokens]

    // Combined input: tok_emb + audio_emb
    ggml_tensor * x = ggml_add(gctx, tok_emb, audio_emb); // [dec_dim, n_tokens]

    // Causal mask for prefill: [n_tokens, n_tokens] additive mask
    // -inf for positions that should not attend
    ggml_tensor * causal_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, causal_mask, ctx->backend);

    // Decoder layers
    for (int32_t i = 0; i < model->hp.dec_layers; i++) {
        x = build_decoder_layer(ctx, gctx, gf, x, positions, time_emb,
            i, n_tokens, kv_plan, causal_mask);
    }

    // Final norm
    x = ggml_rms_norm(gctx, x, model->hp.dec_norm_eps); // [dec_dim, n_tokens]
    x = ggml_mul(gctx, x, model->dec_norm_weight); // [dec_dim, n_tokens]

    // Logits for last token only: extract last token -> matmul with embeddings
    ggml_tensor * last_hidden = ggml_view_1d(gctx, x, VOXTRAL_DEC_DIM,
        (n_tokens - 1) * x->nb[1]); // [dec_dim]

    ggml_tensor * logits = ggml_mul_mat(gctx, model->tok_embeddings_weight, last_hidden); // [vocab_size]

    // Copy logits to persistent
    ggml_build_forward_expand(gf, ggml_cpy(gctx, logits, ctx->decoder_logits));

    return gf;
}

// Emit logits = W @ last_hidden into the persistent decoder_logits tensor, plus an
// on-device greedy argmax into decoder_argmax (so the hot loop reads back 4 bytes).
// last_hidden is [dec_dim]; W is the tied (tok_embeddings) or untied output proj.
static void emit_logits_argmax(voxtral_context * ctx, ggml_context * gctx, ggml_cgraph * gf,
                               ggml_tensor * last_hidden, ggml_tensor * W) {
    ggml_tensor * logits = ggml_mul_mat(gctx, W, last_hidden); // [vocab]
    ggml_build_forward_expand(gf, ggml_cpy(gctx, logits, ctx->decoder_logits));
    ggml_tensor * amax = ggml_argmax(gctx, ggml_reshape_2d(gctx, logits, ctx->model->hp.vocab_size, 1));
    ggml_build_forward_expand(gf, ggml_cpy(gctx, amax, ctx->decoder_argmax));
}

// ============================================================================
// Graph Building: Decoder Step (single token)
// ============================================================================

static ggml_context * init_graph_ctx(std::vector<uint8_t> & buf,
                                     int32_t graph_mult);

static ggml_cgraph * build_decoder_step_graph(
    voxtral_context     * ctx,
    ggml_context * gctx,
    int32_t               position,    // absolute position
    int32_t               audio_pos)   // position in audio embeddings (may differ)
{
    voxtral_model * model = ctx->model;
    voxtral_decoder_kv_plan_t kv_plan;
    if (!voxtral_decoder_kv_plan(ctx->decoder_kv, position, 1, kv_plan)) return nullptr;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    // Token ID input: [1] int32
    ggml_tensor * token_id = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(token_id, "token_id");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, token_id, ctx->backend);

    // Position: [1] int32
    ggml_tensor * pos_tensor = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(pos_tensor, "position");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, pos_tensor, ctx->backend);

    // Time embedding: [dec_dim]
    ggml_tensor * time_emb = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, VOXTRAL_DEC_DIM);
    ggml_set_name(time_emb, "time_emb");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, time_emb, ctx->backend);

    // Token embedding: [dec_dim, 1]
    ggml_tensor * tok_emb = ggml_get_rows(gctx, model->tok_embeddings_weight, token_id); // [dec_dim, 1]

    // Audio embedding at audio_pos: linear decoder_memory (batch/finish path) or
    // the device-resident audio_emb_ring with modulo indexing (incremental decoder).
    ggml_tensor * audio_src = ctx->dec_audio_src ? ctx->dec_audio_src : ctx->decoder_memory;
    const int32_t audio_col = ctx->dec_audio_cap > 0 ? (audio_pos % ctx->dec_audio_cap) : audio_pos;
    ggml_tensor * audio_emb = ggml_view_2d(gctx, audio_src,
        VOXTRAL_DEC_DIM, 1,
        audio_src->nb[1],
        (size_t)audio_col * audio_src->nb[1]); // [dec_dim, 1]

    ggml_tensor * x = ggml_add(gctx, tok_emb, audio_emb); // [dec_dim, 1]

    // Decoder layers (no mask needed for single token - all KV positions are valid)
    for (int32_t i = 0; i < model->hp.dec_layers; i++) {
        x = build_decoder_layer(ctx, gctx, gf, x, pos_tensor, time_emb,
            i, 1, kv_plan, /*attn_mask=*/nullptr);
    }

    // Final norm
    x = ggml_rms_norm(gctx, x, model->hp.dec_norm_eps); // [dec_dim, 1]
    x = ggml_mul(gctx, x, model->dec_norm_weight); // [dec_dim, 1]

    // Logits (tied to token embeddings) + on-device argmax.
    ggml_tensor * x_flat = ggml_reshape_1d(gctx, x, VOXTRAL_DEC_DIM); // [dec_dim]
    if (ctx->capture_decoder_diagnostics) {
        ggml_build_forward_expand(gf,
            ggml_cpy(gctx, x_flat, ctx->decoder_hidden_diagnostic));
    }
    emit_logits_argmax(ctx, gctx, gf, x_flat, model->tok_embeddings_weight);

    return gf;
}

// Fixed-topology production step graph. Every dynamic address is supplied as a
// small input tensor, so the graph can be allocated once and submitted for the
// lifetime of the context. The fixed physical KV view is masked until the cache
// fills; after rollover it contains the exact sliding window as a cyclic
// permutation of paired K/V rows.
static ggml_cgraph * build_decoder_step_graph_reusable(
    voxtral_context * ctx, ggml_context * gctx, bool full_cache) {
    voxtral_model * model = ctx->model;
    const int32_t capacity = ctx->decoder_kv_configured_capacity;
    ggml_cgraph * gf = ggml_new_graph_custom(
        gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    ggml_tensor * token_id = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(token_id, "token_id");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, token_id, ctx->backend);
    ggml_tensor * position = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(position, "position");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, position, ctx->backend);
    ggml_tensor * time_emb = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, VOXTRAL_DEC_DIM);
    ggml_set_name(time_emb, "time_emb");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, time_emb, ctx->backend);
    ggml_tensor * kv_slot = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(kv_slot, "decoder_kv_slot");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, kv_slot, ctx->backend);
    ggml_tensor * audio_slot = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(audio_slot, "decoder_audio_slot");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, audio_slot, ctx->backend);
    ggml_tensor * kv_mask = nullptr;
    if (!full_cache) {
        kv_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, capacity, 1);
        ggml_set_name(kv_mask, "decoder_kv_mask");
        ggml_backend_sched_set_tensor_backend(
            ctx->sched_dec_step, kv_mask, ctx->backend);
    }

    ggml_tensor * tok_emb = ggml_get_rows(
        gctx, model->tok_embeddings_weight, token_id);
    ggml_tensor * audio_emb = ggml_get_rows(
        gctx, ctx->audio_emb_ring, audio_slot);
    ggml_tensor * x = ggml_add(gctx, tok_emb, audio_emb);

    // The dynamic-slot branch of build_decoder_layer does not consume physical
    // segments from the plan; retain a valid shape object for the shared API.
    voxtral_decoder_kv_plan_t static_plan{};
    static_plan.used_after = capacity;
    static_plan.read_nseg = 1;
    static_plan.read_seg[0] = {0, capacity};
    for (int32_t i = 0; i < model->hp.dec_layers; ++i) {
        x = build_decoder_layer(ctx, gctx, gf, x, position, time_emb,
            i, 1, static_plan, kv_mask, kv_slot);
    }

    x = ggml_rms_norm(gctx, x, model->hp.dec_norm_eps);
    x = ggml_mul(gctx, x, model->dec_norm_weight);
    ggml_tensor * x_flat = ggml_reshape_1d(gctx, x, VOXTRAL_DEC_DIM);
    if (ctx->capture_decoder_diagnostics) {
        ggml_build_forward_expand(gf,
            ggml_cpy(gctx, x_flat, ctx->decoder_hidden_diagnostic));
    }
    emit_logits_argmax(ctx, gctx, gf, x_flat, model->tok_embeddings_weight);
    return gf;
}

static bool ensure_decoder_step_graph(voxtral_context * ctx, bool full_cache) {
    if (ctx->decoder_step_graph && ctx->decoder_step_graph_allocated &&
        ctx->decoder_step_graph_full == full_cache) {
        return true;
    }

    // The growing cache needs a mask; the full sliding window must use the
    // unmasked Flash Attention topology used by the bounded physical/logical
    // reference graphs. Keeping an all-zero mask after the cache fills is not
    // numerically equivalent on Vulkan: it selects a different shader and can
    // change a boundary top-1 token. This is one bounded transition per stream,
    // never a rebuild per decoder step.
    ggml_backend_sched_reset(ctx->sched_dec_step);
    if (ctx->decoder_step_graph_ctx) {
        ggml_free(ctx->decoder_step_graph_ctx);
        ctx->decoder_step_graph_ctx = nullptr;
    }
    ctx->decoder_step_graph = nullptr;
    ctx->decoder_step_graph_allocated = false;
    ctx->decoder_step_graph_meta.clear();

    const auto build_start = std::chrono::steady_clock::now();
    ctx->decoder_step_graph_ctx = init_graph_ctx(ctx->decoder_step_graph_meta, 4);
    if (!ctx->decoder_step_graph_ctx) return false;
    ctx->decoder_step_graph = build_decoder_step_graph_reusable(
        ctx, ctx->decoder_step_graph_ctx, full_cache);
    if (!ctx->decoder_step_graph) return false;
    voxtral_context_profile_record_internal(
        ctx, voxtral_profile_stage::decoder_step_graph_build,
        elapsed_ms(build_start));

    ggml_backend_sched_reset(ctx->sched_dec_step);
    voxtral_context_profile_note_allocation_internal(
        ctx, voxtral_profile_stage::decoder_step_graph_execute);
    if (!ggml_backend_sched_alloc_graph(
            ctx->sched_dec_step, ctx->decoder_step_graph)) {
        LOG_ERR(ctx, "decoder step: reusable graph allocation failed");
        return false;
    }
    ctx->decoder_step_graph_allocated = true;
    ctx->decoder_step_graph_full = full_cache;
    if (!full_cache) {
        ctx->decoder_step_mask.assign(
            (size_t) ctx->decoder_kv_configured_capacity, -INFINITY);
    }
    ctx->decoder_step_mask_valid = -1;
    return true;
}

// ============================================================================
// Graph Building: Offline decode (Voxtral-Mini-3B-2507)
//
// The offline model is a standard audio-LLM: the prompt is a mixed sequence of
// text-token embeddings and audio embeddings (the adapter output substituted at
// the [AUDIO] placeholder positions), prefilled once; then text tokens are
// generated autoregressively until EOS. No per-frame audio indexing, no ada.
// ============================================================================

// Final RMS norm + (untied) output projection + argmax for the LAST token only.
static void offline_logits_tail(voxtral_context * ctx, ggml_context * gctx, ggml_cgraph * gf,
                                ggml_tensor * x /*[dec_dim, n_tokens]*/, int32_t n_tokens) {
    voxtral_model * m = ctx->model;
    ggml_tensor * last = ggml_view_1d(gctx, x, m->hp.dec_dim,
        (size_t)(n_tokens - 1) * x->nb[1]); // [dec_dim]
    last = ggml_rms_norm(gctx, last, m->hp.dec_norm_eps);
    last = ggml_mul(gctx, last, m->dec_norm_weight);
    // Offline model has an untied output projection; fall back to tied embeddings.
    ggml_tensor * W = m->output_weight ? m->output_weight : m->tok_embeddings_weight;
    emit_logits_argmax(ctx, gctx, gf, last, W);
}

// Prefill: prefix text tokens + n_audio audio embeddings + suffix text tokens.
static ggml_cgraph * build_offline_prefill_graph(
    voxtral_context * ctx, ggml_context * gctx,
    int32_t n_prefix, int32_t n_audio, int32_t n_suffix) {
    voxtral_model * model = ctx->model;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 8, false);
    const int32_t n_tokens = n_prefix + n_audio + n_suffix;
    voxtral_decoder_kv_plan_t kv_plan;
    if (!voxtral_decoder_kv_plan(ctx->decoder_kv,
                                 ctx->decoder_kv.next_absolute_position,
                                 n_tokens, kv_plan)) return nullptr;

    ggml_tensor * prefix_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_prefix);
    ggml_set_name(prefix_ids, "prefix_ids");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, prefix_ids, ctx->backend);
    ggml_tensor * suffix_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_suffix);
    ggml_set_name(suffix_ids, "suffix_ids");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, suffix_ids, ctx->backend);

    ggml_tensor * prefix_emb = ggml_get_rows(gctx, model->tok_embeddings_weight, prefix_ids); // [dec_dim, n_prefix]
    ggml_tensor * audio_emb = ggml_cont(gctx, ggml_view_2d(gctx, ctx->decoder_memory,
        model->hp.dec_dim, n_audio, ctx->decoder_memory->nb[1], 0)); // [dec_dim, n_audio]
    ggml_tensor * suffix_emb = ggml_get_rows(gctx, model->tok_embeddings_weight, suffix_ids); // [dec_dim, n_suffix]

    ggml_tensor * x = ggml_concat(gctx, prefix_emb, audio_emb, 1);
    x = ggml_concat(gctx, x, suffix_emb, 1); // [dec_dim, n_tokens]

    ggml_tensor * positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, positions, ctx->backend);

    ggml_tensor * causal_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, causal_mask, ctx->backend);

    for (int32_t i = 0; i < model->hp.dec_layers; i++) {
        x = build_decoder_layer(ctx, gctx, gf, x, positions, /*time_emb=*/nullptr,
            i, n_tokens, kv_plan, causal_mask);
    }
    offline_logits_tail(ctx, gctx, gf, x, n_tokens);
    return gf;
}

// Single-token autoregressive step (no audio embedding).
static ggml_cgraph * build_offline_step_graph(voxtral_context * ctx, ggml_context * gctx) {
    voxtral_model * model = ctx->model;
    voxtral_decoder_kv_plan_t kv_plan;
    if (!voxtral_decoder_kv_plan(ctx->decoder_kv,
                                 ctx->decoder_kv.next_absolute_position,
                                 1, kv_plan)) return nullptr;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    ggml_tensor * token_id = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(token_id, "token_id");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, token_id, ctx->backend);
    ggml_tensor * pos_tensor = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(pos_tensor, "position");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, pos_tensor, ctx->backend);

    ggml_tensor * x = ggml_get_rows(gctx, model->tok_embeddings_weight, token_id); // [dec_dim, 1]
    for (int32_t i = 0; i < model->hp.dec_layers; i++) {
        x = build_decoder_layer(ctx, gctx, gf, x, pos_tensor, /*time_emb=*/nullptr,
            i, 1, kv_plan, /*attn_mask=*/nullptr);
    }
    offline_logits_tail(ctx, gctx, gf, x, 1);
    return gf;
}

// ============================================================================
// Helper: set named input tensors in a graph
// ============================================================================

static ggml_tensor * find_tensor_in_graph(ggml_cgraph * gf, const char * name) {
    return ggml_graph_get_tensor(gf, name);
}

static void profile_tensor_set(voxtral_context * ctx, ggml_tensor * tensor,
                               const void * data, size_t offset, size_t bytes) {
    voxtral_context_profile_note_tensor_set_internal(ctx);
    ggml_backend_tensor_set(tensor, data, offset, bytes);
}

static void profile_tensor_get(voxtral_context * ctx, const ggml_tensor * tensor,
                               void * data, size_t offset, size_t bytes) {
    voxtral_context_profile_note_tensor_get_internal(ctx);
    ggml_backend_tensor_get(tensor, data, offset, bytes);
}

// Set a named graph input tensor if it exists (no-op if absent).
static void set_graph_input(voxtral_context * ctx, ggml_cgraph * gf,
                            const char * name, const void * data, size_t bytes) {
    if (ggml_tensor * t = find_tensor_in_graph(gf, name)) {
        profile_tensor_set(ctx, t, data, 0, bytes);
    }
}

// Allocate a no_alloc graph context backed by `buf` (resized as needed). The
// graph_mult scales the default node budget (step graphs use 4, larger prefill 8).
static ggml_context * init_graph_ctx(std::vector<uint8_t> & buf, int32_t graph_mult) {
    const size_t meta = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * graph_mult +
                        ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * graph_mult, false);
    if (buf.size() < meta) buf.resize(meta);
    ggml_init_params p = { meta, buf.data(), /*.no_alloc=*/ true };
    return ggml_init(p);
}

// Reset + allocate the graph on `sched`, run `set_inputs(gf)`, compute, then reset
// and free `gctx`. Readbacks target persistent tensors, so the caller reads them
// after this returns. Returns false (and frees gctx) on allocation failure.
template <typename SetInputs>
static bool run_graph(voxtral_context * ctx, ggml_backend_sched_t sched,
                      ggml_context * gctx, ggml_cgraph * gf,
                      SetInputs && set_inputs, const char * what,
                      voxtral_profile_stage execute_stage) {
    ggml_backend_sched_reset(sched);
    voxtral_context_profile_note_allocation_internal(ctx, execute_stage);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        LOG_ERR(ctx, "%s: failed to allocate graph", what);
        ggml_free(gctx);
        return false;
    }
    set_inputs(gf);
    const auto exec_start = std::chrono::steady_clock::now();
    voxtral_context_profile_note_submit_internal(ctx);
    const enum ggml_status status = ggml_backend_sched_graph_compute_async(sched, gf);
    const auto sync_start = std::chrono::steady_clock::now();
    ggml_backend_sched_synchronize(sched);
    voxtral_context_profile_record_internal(ctx, voxtral_profile_stage::backend_synchronize,
                                            elapsed_ms(sync_start));
    voxtral_context_profile_record_internal(ctx, execute_stage, elapsed_ms(exec_start));
    ggml_backend_sched_reset(sched);
    ggml_free(gctx);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR(ctx, "%s: graph compute failed (%d)", what, (int) status);
        return false;
    }
    return true;
}

// Fill an [n, n] lower-triangular causal mask (0 = attend, -inf = masked).
static void fill_causal_mask(std::vector<float> & mask, int32_t n) {
    mask.assign((size_t) n * n, 0.0f);
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = i + 1; j < n; ++j) {
            mask[(size_t) i * n + j] = -INFINITY;
        }
    }
}

// ============================================================================
// Run Encoder
// ============================================================================

// Run a single encoder chunk: build graph, set inputs, compute, return seq_len
static bool run_encoder_chunk(
    voxtral_context * ctx,
    const float * chunk_mel_data,  // [n_mel, chunk_mel_frames]
    int32_t chunk_mel_frames,
    int32_t rope_pos_offset,
    int32_t * out_seq_len)
{
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
    std::vector<uint8_t> meta_buf(meta_size);

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);

    int32_t chunk_seq_len = 0;
    ggml_cgraph * gf = build_encoder_graph(ctx, gctx, chunk_mel_data, chunk_mel_frames, &chunk_seq_len);

    ggml_backend_sched_reset(ctx->sched_encoder);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_encoder, gf)) {
        LOG_ERR(ctx, "encoder chunk: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    // Set mel input
    ggml_tensor * mel_t = find_tensor_in_graph(gf, "mel_input");
    if (mel_t) {
        const int64_t expected_ne0 = chunk_mel_frames;
        const int64_t expected_ne1 = VOXTRAL_NUM_MEL_BINS;
        if (mel_t->ne[0] == expected_ne0 && mel_t->ne[1] == expected_ne1) {
            ggml_backend_tensor_set(mel_t, chunk_mel_data, 0,
                (size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames * sizeof(float));
        } else if (mel_t->ne[0] == expected_ne1 && mel_t->ne[1] == expected_ne0) {
            std::vector<float> mel_tbuf((size_t) chunk_mel_frames * VOXTRAL_NUM_MEL_BINS);
            for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
                const float * src = chunk_mel_data + (size_t) m * chunk_mel_frames;
                for (int32_t f = 0; f < chunk_mel_frames; ++f) {
                    mel_tbuf[(size_t) m + (size_t) VOXTRAL_NUM_MEL_BINS * f] = src[f];
                }
            }
            ggml_backend_tensor_set(mel_t, mel_tbuf.data(), 0,
                (size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames * sizeof(float));
        } else {
            ggml_backend_tensor_set(mel_t, chunk_mel_data, 0,
                (size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames * sizeof(float));
        }
    }

    // Set positions with RoPE offset for absolute positions across chunks
    ggml_tensor * pos_t = find_tensor_in_graph(gf, "enc_positions");
    if (pos_t) {
        std::vector<int32_t> pos(chunk_seq_len);
        std::iota(pos.begin(), pos.end(), rope_pos_offset);
        ggml_backend_tensor_set(pos_t, pos.data(), 0, chunk_seq_len * sizeof(int32_t));
    }

    // Set encoder sliding causal mask (local to chunk)
    ggml_tensor * mask_t = find_tensor_in_graph(gf, "enc_attn_mask");
    if (mask_t) {
        std::vector<float> mask((size_t) chunk_seq_len * chunk_seq_len);
        for (int32_t q = 0; q < chunk_seq_len; ++q) {
            const int32_t min_kv = std::max<int32_t>(0, q - (VOXTRAL_ENC_WINDOW - 1));
            for (int32_t kv = 0; kv < chunk_seq_len; ++kv) {
                const bool allow = (kv <= q) && (kv >= min_kv);
                mask[(size_t) q * chunk_seq_len + kv] = allow ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));
    }

    // Compute
    ggml_backend_sched_graph_compute(ctx->sched_encoder, gf);
    ggml_backend_sched_reset(ctx->sched_encoder);
    ggml_free(gctx);

    if (out_seq_len) *out_seq_len = chunk_seq_len;
    return true;
}

// Process mel spectrogram in overlapping chunks, accumulating encoder output on device
// Encoder token count for the offline (symmetric-conv) Whisper encoder.
static int32_t mel_frames_to_enc_tokens_sym(int32_t n_frames) {
    const int32_t c0 = n_frames;            // conv0 k3 s1 pad1 -> same length
    const int32_t c1 = (c0 - 1) / 2 + 1;    // conv1 k3 s2 pad1 -> (n+2-3)/2 + 1
    return c1 - (c1 % VOXTRAL_DOWNSAMPLE_FACTOR);
}

// Offline encoder: single full-window pass (<=30s), non-causal, no chunk overlap.
static bool run_encoder_offline(voxtral_context * ctx, const float * mel_data, int32_t total_mel_frames) {
    int32_t est = mel_frames_to_enc_tokens_sym(total_mel_frames);
    if (est <= 0) { LOG_ERR(ctx, "encoder offline: audio too short"); return false; }
    if (est > VOXTRAL_MAX_ENC_CHUNK) est = VOXTRAL_MAX_ENC_CHUNK;
    if (!alloc_encoder_output(ctx, est)) {
        LOG_ERR(ctx, "encoder offline: failed to allocate output (%d tokens)", est);
        return false;
    }
    int32_t seq_len = 0;
    if (!run_encoder_chunk(ctx, mel_data, total_mel_frames, /*rope_pos_offset=*/0, &seq_len)) {
        return false;
    }
    if (seq_len > est) seq_len = est;
    const size_t bytes = (size_t) seq_len * VOXTRAL_ENC_DIM * sizeof(float);
    std::vector<uint8_t> tmp(bytes);
    ggml_backend_tensor_get(ctx->encoder_chunk_output, tmp.data(), 0, bytes);
    ggml_backend_tensor_set(ctx->encoder_output, tmp.data(), 0, bytes);
    // Trim to a multiple of the downsample factor for the adapter.
    ctx->enc_seq_used = (seq_len / VOXTRAL_DOWNSAMPLE_FACTOR) * VOXTRAL_DOWNSAMPLE_FACTOR;
    ctx->total_enc_tokens = ctx->enc_seq_used;
    LOG_INFO(ctx, "encoder offline: %d mel frames -> %d enc tokens", total_mel_frames, ctx->enc_seq_used);
    return true;
}

static bool run_encoder_chunked(voxtral_context * ctx, const float * mel_data, int32_t total_mel_frames) {
    const int32_t mel_overlap = VOXTRAL_ENC_CHUNK_OVERLAP * 2;  // mel frames of overlap (1500)
    const int32_t mel_stride = VOXTRAL_ENC_CHUNK_MEL - mel_overlap;  // 1500

    // Pre-compute total encoder tokens for allocation
    int32_t alloc_total = compute_total_enc_tokens(total_mel_frames);
    if (alloc_total <= 0) {
        LOG_ERR(ctx, "encoder: audio too short to produce encoder tokens");
        return false;
    }

    // Allocate encoder_output on device
    if (!alloc_encoder_output(ctx, alloc_total)) {
        LOG_ERR(ctx, "encoder: failed to allocate encoder output (%d tokens, %.2f MB)",
                alloc_total, (double) alloc_total * VOXTRAL_ENC_DIM * sizeof(float) / 1e6);
        return false;
    }

    LOG_INFO(ctx, "encoder chunked: %d mel frames, %d alloc enc tokens, mel_stride=%d",
             total_mel_frames, alloc_total, mel_stride);

    int32_t mel_offset = 0;
    int32_t enc_write_offset = 0;
    int32_t chunk_idx = 0;

    while (mel_offset < total_mel_frames) {
        int32_t chunk_mel_frames = std::min(VOXTRAL_ENC_CHUNK_MEL, total_mel_frames - mel_offset);

        // Pre-check: will this chunk contribute any new tokens?
        // This avoids building and running the full encoder graph for nothing.
        int32_t skip = (chunk_idx > 0) ? VOXTRAL_ENC_CHUNK_OVERLAP : 0;
        {
            int32_t expected_tokens = mel_frames_to_enc_tokens(chunk_mel_frames);
            if (expected_tokens - skip <= 0) {
                LOG_DBG(ctx, "encoder chunk %d: skipped (expected %d tokens, skip=%d)",
                        chunk_idx, expected_tokens, skip);
                break;
            }
        }

        // For single-chunk case (entire mel fits), use mel_data directly to avoid copy
        const float * chunk_mel_ptr = nullptr;
        std::vector<float> chunk_mel_buf;
        if (mel_offset == 0 && chunk_mel_frames == total_mel_frames) {
            // Single chunk — mel_data is already in [n_mel, total_frames] layout
            chunk_mel_ptr = mel_data;
        } else {
            // Multi-chunk — extract sub-range of frames for this chunk
            chunk_mel_buf.resize((size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames);
            for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; m++) {
                memcpy(chunk_mel_buf.data() + (size_t) m * chunk_mel_frames,
                       mel_data + (size_t) m * total_mel_frames + mel_offset,
                       chunk_mel_frames * sizeof(float));
            }
            chunk_mel_ptr = chunk_mel_buf.data();
        }

        int32_t rope_offset = enc_write_offset - skip;

        // Run encoder for this chunk
        int32_t chunk_seq_len = 0;
        if (!run_encoder_chunk(ctx, chunk_mel_ptr, chunk_mel_frames, rope_offset, &chunk_seq_len)) {
            LOG_ERR(ctx, "encoder chunk %d: failed", chunk_idx);
            return false;
        }

        int32_t stride = chunk_seq_len - skip;
        if (stride <= 0) {
            LOG_DBG(ctx, "encoder chunk %d: no new tokens (seq_len=%d, skip=%d), stopping",
                    chunk_idx, chunk_seq_len, skip);
            break;
        }

        // Clamp stride to not overflow pre-allocated buffer
        if (enc_write_offset + stride > alloc_total) {
            stride = alloc_total - enc_write_offset;
            if (stride <= 0) break;
        }

        LOG_INFO(ctx, "encoder chunk %d: mel[%d..%d) enc_tokens=%d skip=%d stride=%d rope_offset=%d",
                 chunk_idx, mel_offset, mel_offset + chunk_mel_frames,
                 chunk_seq_len, skip, stride, rope_offset);

        // Copy stride portion from encoder_chunk_output to encoder_output
        // Goes through CPU (device->CPU->device)
        {
            const size_t elem_bytes = VOXTRAL_ENC_DIM * sizeof(float);
            const size_t src_offset = (size_t) skip * elem_bytes;
            const size_t dst_offset = (size_t) enc_write_offset * elem_bytes;
            const size_t copy_bytes = (size_t) stride * elem_bytes;

            std::vector<uint8_t> tmp(copy_bytes);
            ggml_backend_tensor_get(ctx->encoder_chunk_output, tmp.data(), src_offset, copy_bytes);
            ggml_backend_tensor_set(ctx->encoder_output, tmp.data(), dst_offset, copy_bytes);
        }

        enc_write_offset += stride;
        mel_offset += mel_stride;
        chunk_idx++;
    }

    // Trim to multiple of downsample factor for adapter compatibility
    ctx->enc_seq_used = (enc_write_offset / VOXTRAL_DOWNSAMPLE_FACTOR) * VOXTRAL_DOWNSAMPLE_FACTOR;
    ctx->total_enc_tokens = ctx->enc_seq_used;

    LOG_INFO(ctx, "encoder done: %d chunks, enc_seq_used=%d (raw=%d)",
             chunk_idx, ctx->enc_seq_used, enc_write_offset);
    return true;
}

// ============================================================================
// Per-layer encoder KV-cache — model-free ring planner (Sessions 6.1/6.2)
// ----------------------------------------------------------------------------
// A single pure function derives every physical offset the KV graph needs for one
// execution over new enc frames [q_start, q_start+n_new): the causal window, the
// ring segments for the window READ and the new-frame WRITE, and the conv input
// window. Absolute positions drive RoPE + eviction; ring slots = pos % capacity.
// This is the only copy of the wrap/eviction arithmetic; the graph consumes it and
// tests/cpp/test_encoder_kv.cpp locks it without a backend.
// ============================================================================

int32_t voxtral_enc_kv_capacity_internal() { return VOXTRAL_ENC_KV_CAP; }
int32_t voxtral_enc_kv_max_new_internal()  { return VOXTRAL_ENC_KV_MAX_NEW; }
int32_t voxtral_enc_kv_window_internal()   { return VOXTRAL_ENC_WINDOW; }
int32_t voxtral_enc_kv_logical_batch_internal() { return encoder_kv_schedule_from_env().logical; }
int32_t voxtral_enc_kv_physical_rows_internal() { return encoder_kv_schedule_from_env().physical; }
int64_t voxtral_context_decoder_kv_bytes_internal(const voxtral_context * ctx) {
    if (!ctx || !ctx->kv_self_k || !ctx->kv_self_v) return 0;
    return (int64_t) ggml_nbytes(ctx->kv_self_k) + (int64_t) ggml_nbytes(ctx->kv_self_v);
}

bool voxtral_enc_kv_mask_allows(int64_t kv_abs, int64_t q_abs, int32_t window) {
    return kv_abs <= q_abs && kv_abs >= q_abs - (int64_t) (window - 1);
}

// Split the logical span [abs_start, abs_start+len) into <= 2 physical ring
// segments (in ascending-position order). Returns the segment count.
static int32_t enc_kv_ring_segments(int64_t abs_start, int32_t len, int32_t capacity,
                                    voxtral_enc_kv_seg out[2]) {
    if (len <= 0) return 0;
    const int32_t slot0 = (int32_t) (abs_start % capacity);
    const int32_t first = std::min(len, capacity - slot0);
    out[0] = { slot0, first };
    if (first >= len) return 1;
    out[1] = { 0, len - first };
    return 2;
}

bool voxtral_enc_kv_plan(int64_t q_start, int32_t n_new,
                         int32_t capacity, int32_t window,
                         voxtral_enc_kv_plan_t & out) {
    out = voxtral_enc_kv_plan_t{};
    if (n_new <= 0 || capacity <= 0 || window <= 0) return false;
    // capacity must hold the whole causal window of a batch (window-1 + n_new
    // frames) so the window is a single logical span and the write cannot clobber a
    // still-live frame. This bounds n_new <= capacity - window.
    if (n_new > capacity - window) return false;
    if (q_start < 0) return false;

    out.q_start = q_start;
    out.n_new   = n_new;

    out.win_start = std::max<int64_t>(0, q_start - (int64_t) (window - 1));
    out.win_end   = q_start + n_new;
    out.win_len   = (int32_t) (out.win_end - out.win_start);

    out.read_nseg  = enc_kv_ring_segments(out.win_start, out.win_len, capacity, out.read_seg);
    out.read_wraps = out.read_nseg > 1;

    out.write_nseg  = enc_kv_ring_segments(q_start, n_new, capacity, out.write_seg);
    out.write_wraps = out.write_nseg > 1;

    // Conv input: enc frame e depends on Mel[2e-3 .. 2e+1]; feed an even-aligned
    // window with >= 2 enc frames of real left context so the sliced frames are
    // past the fed window's front zero-pad (bit-exact), except at the true stream
    // start (q_start < 2) where the zero-pad is the intended behaviour.
    out.conv_mel_start   = std::max<int64_t>(0, 2 * (q_start - 2));
    out.conv_mel_end     = 2 * (q_start + n_new);
    out.conv_slice_start = (int32_t) (q_start - out.conv_mel_start / 2);

    // Frames evicted as the head advances to q_start+n_new. Retention is keyed on
    // the new head H': the cache logically keeps [max(0,H'-window), H') (<= window
    // frames), so evicted = the delta of that lower bound over this batch.
    const int64_t old_lo = std::max<int64_t>(0, q_start - (int64_t) window);
    const int64_t new_lo = std::max<int64_t>(0, out.win_end - (int64_t) window);
    out.evicted = new_lo - old_lo;
    return true;
}

// Enc frames realizable from `mel_frames` stable Mel frames WITHOUT the conv
// right-pad: enc frame e needs Mel[2e-3 .. 2e+1], so the largest fully-covered e is
// floor((mel-2)/2). At the padded finish (mel a multiple of 8) this equals the
// batch's enc_seq_used (mel/2), so the KV encoder emits exactly the batch frames;
// mid-stream it stops short of the pad-right frame the batch would only add at the
// true end. Monotonic non-decreasing in mel_frames.
static int64_t enc_frames_realizable(int64_t mel_frames) {
    return mel_frames < 2 ? 0 : (mel_frames - 2) / 2 + 1;
}

// ============================================================================
// Per-layer encoder KV-cache — device ring + per-stream runtime (Sessions 6.1/6.2)
// ----------------------------------------------------------------------------
// Owns the persistent device K/V ring (build_encoder_kv_graph consumes it), a
// bounded host Mel tail for the conv stem, and the accumulated encoder output.
// Each enc frame passes through the 32 transformer layers exactly once.
// ============================================================================
struct voxtral_encoder_kv_state {
    voxtral_context * ctx = nullptr;
    bool reusable_stream_graph = false;

    // Persistent device ring, [kv_dim, capacity, enc_layers]. Session 8.1 uses
    // F32 by default; explicit f16/f32 overrides drive the acceptance matrix.
    ggml_context        * ctx_ring = nullptr;
    ggml_backend_buffer_t buf_ring = nullptr;
    ggml_tensor         * ring_k   = nullptr;
    ggml_tensor         * ring_v   = nullptr;
    int64_t               ring_bytes = 0;

    int64_t emitted    = 0;   // absolute enc frames produced (== next q_start)
    int64_t stable_mel = 0;   // absolute stable Mel frames received

    // Bounded host Mel tail for the conv stem, frame-major ([n_mel] per frame).
    // Spans [tail_base, stable_mel); trimmed to the next batch's conv left context.
    std::vector<float> mel_tail;
    int64_t            tail_base = 0;

    std::vector<float>   enc_accum;  // host channel-major [enc_dim, emitted] (Stage 11 linear buffer)
    std::vector<float>   conv_cm;    // scratch: conv Mel window channel-major [n_mel, conv_len]
    std::vector<float>   mask_buf;   // scratch: causal window mask [physical_key_len * physical_rows]
    std::vector<int32_t> pos_buf;    // scratch: absolute RoPE positions [physical_rows]

    bool finalized = false;

    // Work / memory instrumentation (Stages 16/17).
    int64_t transformer_frames   = 0;   // enc frames through the transformer (each once)
    int64_t kv_appends           = 0;
    int64_t kv_evictions         = 0;
    int64_t kv_wraps             = 0;
    int64_t kv_materialized      = 0;
    int64_t graph_execs          = 0;
    int32_t max_new_per_exec     = 0;
    int32_t warmup_frames        = 0;
    int64_t frames_before_finish = 0;
    int64_t frames_at_finish     = 0;
    int64_t frames_at_finish_exec= 0;   // transformer frames evaluated inside finish()
    int64_t peak_logical         = 0;
    int64_t peak_mel_tail        = 0;
    int64_t peak_output          = 0;
    // Actual encoder-output host round-trip performed by this stream. In the
    // incremental production path the adapter reads encoder output from the
    // device ring, so this MUST stay 0 (the enc_accum D2H below is skipped);
    // the reference finish-only path keeps the host accumulation and counts it.
    int64_t enc_out_d2h_bytes    = 0;
    int64_t logical_frames_submitted = 0;
    int64_t physical_rows_evaluated = 0;
    int64_t padding_rows_evaluated = 0;

    // Timeline/latency collector. It is opt-in so ordinary production streams
    // retain only the bounded scheduler state and encoder output queue.
    bool telemetry = false;
    int64_t timeline_start_ns = 0;
    int64_t current_audio_samples = 0;
    int64_t left_pad_samples = 0;
    struct audio_arrival { int64_t sample_end = 0; int64_t wall_ns = 0; };
    struct mel_arrival { int64_t frame_end = 0; int64_t wall_ns = 0; };
    std::vector<audio_arrival> audio_arrivals;
    std::vector<mel_arrival> mel_arrivals;
    std::vector<double> residence_ms;
    std::vector<double> group_residence_ms;
    std::vector<double> compute_ms;
    double first_frame_absolute_ms = 0.0;
    double first_frame_residence_ms = 0.0;
    double first_group_absolute_ms = 0.0;
    double first_group_residence_ms = 0.0;
    double first_eight_absolute_ms = 0.0;
    double first_eight_residence_ms = 0.0;
    bool have_first_frame = false;
    bool have_first_real_frame = false;
    bool have_first_group = false;
    bool have_first_eight = false;
    double group_ready[4] = {0, 0, 0, 0};
    double group_required[4] = {0, 0, 0, 0};
    bool group_real[4] = {false, false, false, false};
    double eight_ready[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    double eight_required[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool eight_real[8] = {false, false, false, false, false, false, false, false};
    FILE * trace_file = nullptr;
};

static int64_t encoder_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double encoder_percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double x = p * (double) (values.size() - 1);
    const size_t lo = (size_t) x;
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (x - (double) lo);
}

static int64_t encoder_arrival_ns(const voxtral_encoder_kv_state * kv, int64_t sample_index) {
    if (!kv || !kv->telemetry || sample_index <= 0 || kv->audio_arrivals.empty()) {
        return kv ? kv->timeline_start_ns : 0;
    }
    const int64_t wanted = sample_index + 1; // sample index is inclusive
    for (const auto & a : kv->audio_arrivals) {
        if (a.sample_end >= wanted) return a.wall_ns;
    }
    return kv->audio_arrivals.back().wall_ns;
}

static int64_t encoder_mel_ready_ns(const voxtral_encoder_kv_state * kv, int64_t mel_frame) {
    if (!kv || !kv->telemetry || mel_frame < 0 || kv->mel_arrivals.empty()) {
        return kv ? kv->timeline_start_ns : 0;
    }
    const int64_t wanted = mel_frame + 1;
    for (const auto & a : kv->mel_arrivals) {
        if (a.frame_end >= wanted) return a.wall_ns;
    }
    return kv->mel_arrivals.back().wall_ns;
}

static void encoder_note_audio(voxtral_encoder_kv_state * kv, int64_t sample_count,
                               int64_t arrival_ns, int64_t mel_frames_ready,
                               int64_t mel_ready_ns, int64_t timeline_start_ns,
                               int64_t left_pad_samples) {
    if (!kv) return;
    kv->current_audio_samples = std::max<int64_t>(0, sample_count);
    kv->timeline_start_ns = timeline_start_ns;
    kv->left_pad_samples = left_pad_samples;
    if (!kv->telemetry) return;
    if (!kv->trace_file) {
        if (const char * path = std::getenv("VOXTRAL_ENCODER_TRACE_JSONL")) {
            kv->trace_file = std::fopen(path, "w");
        }
    }
    if (kv->audio_arrivals.empty() || sample_count > kv->audio_arrivals.back().sample_end) {
        kv->audio_arrivals.push_back({sample_count, arrival_ns});
    }
    if (mel_frames_ready > 0 &&
        (kv->mel_arrivals.empty() || mel_frames_ready > kv->mel_arrivals.back().frame_end)) {
        kv->mel_arrivals.push_back({mel_frames_ready, mel_ready_ns});
    }
}

static void encoder_record_outputs(voxtral_encoder_kv_state * kv,
                                   const voxtral_enc_kv_plan_t & plan,
                                   int64_t queued_ns, int64_t compute_start_ns,
                                   int64_t ready_ns) {
    if (!kv || !kv->telemetry) return;
    const double absolute_ms = kv->timeline_start_ns > 0
        ? (double) (ready_ns - kv->timeline_start_ns) / 1e6 : 0.0;
    for (int32_t i = 0; i < plan.n_new; ++i) {
        const int64_t frame = plan.q_start + i;
        const int64_t padded_dep = voxtral_mel_frame_required_sample(2 * frame + 1);
        const int64_t required_sample = padded_dep - kv->left_pad_samples;
        // A right-pad dependency beyond the actual utterance is not a realtime
        // frame: it is flushed after input ended and must not inflate residence
        // percentiles. Likewise, left-pad-only frames are warmup, not audio.
        const bool real_audio = required_sample > 0 && required_sample <= kv->current_audio_samples;
        const int64_t req_ns = real_audio
            ? encoder_arrival_ns(kv, std::min(required_sample, kv->current_audio_samples))
            : kv->timeline_start_ns;
        const double req_ms = kv->timeline_start_ns > 0
            ? (double) (req_ns - kv->timeline_start_ns) / 1e6 : 0.0;
        const double residence = std::max(0.0, absolute_ms - req_ms);
        if (kv->trace_file) {
            const int64_t mel_frame = 2 * frame + 1;
            const int64_t mel_ns = encoder_mel_ready_ns(kv, mel_frame);
            const double mel_ms = kv->timeline_start_ns > 0
                ? (double) (mel_ns - kv->timeline_start_ns) / 1e6 : 0.0;
            const double queued_ms = kv->timeline_start_ns > 0
                ? (double) (queued_ns - kv->timeline_start_ns) / 1e6 : 0.0;
            const double start_ms = kv->timeline_start_ns > 0
                ? (double) (compute_start_ns - kv->timeline_start_ns) / 1e6 : 0.0;
            std::fprintf(kv->trace_file,
                         "{\"encoderFrame\":%lld,\"requiredPcmSampleIndex\":%lld,"
                         "\"requiredAudioTimestampMs\":%.6f,\"melReadyMs\":%.6f,"
                         "\"encoderQueuedMs\":%.6f,\"encoderExecutionStartMs\":%.6f,"
                         "\"encoderReadyMs\":%.6f,\"residenceMs\":%.6f,\"realAudio\":%s}\n",
                         (long long) frame, (long long) required_sample,
                         (double) required_sample * 1000.0 / VOXTRAL_SAMPLE_RATE,
                         mel_ms, queued_ms, start_ms, absolute_ms,
                         real_audio ? residence : 0.0,
                         real_audio ? "true" : "false");
        }
        if (!kv->have_first_frame) kv->have_first_frame = true;
        if (real_audio) {
            if (!kv->have_first_real_frame) {
                kv->have_first_real_frame = true;
                // Export a matched pair for the first output that actually
                // depends on caller PCM. Left-pad-only encoder rows are useful
                // warmup work but are not a realtime first-output milestone.
                kv->first_frame_absolute_ms = absolute_ms;
                kv->first_frame_residence_ms = residence;
            }
            kv->residence_ms.push_back(residence);
        }

        const int32_t slot = (int32_t) (frame % 4);
        kv->group_ready[slot] = absolute_ms;
        kv->group_required[slot] = req_ms;
        kv->group_real[slot] = real_audio;
        if (slot == 3) {
            double ready = 0.0, required = 0.0;
            bool group_real = true;
            for (int j = 0; j < 4; ++j) {
                ready = std::max(ready, kv->group_ready[j]);
                required = std::max(required, kv->group_required[j]);
                group_real = group_real && kv->group_real[j];
            }
            const double group_residence = std::max(0.0, ready - required);
            if (group_real) {
                kv->group_residence_ms.push_back(group_residence);
                if (std::getenv("VOXTRAL_ENCODER_TELEMETRY_DEBUG") && (frame < 140 || frame % 32 == 3)) {
                    std::fprintf(stderr, "[ENC_LAT] frame=%lld abs=%.3f req=%.3f res=%.3f group=%.3f\n",
                                 (long long) frame, absolute_ms, required, residence, group_residence);
                }
                if (!kv->have_first_group) {
                    kv->have_first_group = true;
                    kv->first_group_absolute_ms = ready;
                    kv->first_group_residence_ms = group_residence;
                }
            }
        }

        const int32_t slot8 = (int32_t) (frame % 8);
        kv->eight_ready[slot8] = absolute_ms;
        kv->eight_required[slot8] = req_ms;
        kv->eight_real[slot8] = real_audio;
        if (slot8 == 7 && !kv->have_first_eight) {
            double ready = 0.0, required = 0.0;
            bool all_real = true;
            for (int j = 0; j < 8; ++j) {
                ready = std::max(ready, kv->eight_ready[j]);
                required = std::max(required, kv->eight_required[j]);
                all_real = all_real && kv->eight_real[j];
            }
            if (all_real) {
                kv->have_first_eight = true;
                kv->first_eight_absolute_ms = ready;
                kv->first_eight_residence_ms = std::max(0.0, ready - required);
            }
        }
    }
}

static bool encoder_kv_alloc(voxtral_encoder_kv_state * kv) {
    if (kv->ring_k) return true;
    voxtral_context * ctx = kv->ctx;
    kv->telemetry = std::getenv("VOXTRAL_ENCODER_TELEMETRY") != nullptr;
    const int32_t kv_dim = ctx->model->hp.enc_kv_heads * ctx->model->hp.enc_head_dim;
    const int32_t layers = ctx->model->hp.enc_layers;

    ggml_init_params p = { 2 * ggml_tensor_overhead(), nullptr, /*.no_alloc=*/true };
    kv->ctx_ring = ggml_init(p);
    if (!kv->ctx_ring) return false;
    const ggml_type encoder_kv_type = encoder_kv_type_from_env();
    kv->ring_k = ggml_new_tensor_3d(kv->ctx_ring, encoder_kv_type, kv_dim, VOXTRAL_ENC_KV_CAP, layers);
    kv->ring_v = ggml_new_tensor_3d(kv->ctx_ring, encoder_kv_type, kv_dim, VOXTRAL_ENC_KV_CAP, layers);
    ggml_set_name(kv->ring_k, "enc_kv_k");
    ggml_set_name(kv->ring_v, "enc_kv_v");
    kv->buf_ring = ggml_backend_alloc_ctx_tensors(kv->ctx_ring, ctx->backend);
    if (!kv->buf_ring) { ggml_free(kv->ctx_ring); kv->ctx_ring = nullptr; return false; }
    kv->ring_bytes = (int64_t) ggml_nbytes(kv->ring_k) + (int64_t) ggml_nbytes(kv->ring_v);
    ctx->encoder_kv_allocated_bytes = kv->ring_bytes;
    ctx->encoder_kv_storage_type = encoder_kv_type;
    // Flash attention may speculatively load masked physical columns.  A
    // one-time device clear keeps those loads finite before the ring has wrapped;
    // it is not repeated per graph and therefore does not grow with the stream.
    ggml_backend_buffer_clear(kv->buf_ring, 0);
    LOG_INFO(ctx, "encoder KV: ring %d x %d x %d %s, %.2f MB on device",
             kv_dim, VOXTRAL_ENC_KV_CAP, layers,
             kv->ring_k->type == GGML_TYPE_F16 ? "F16" : "F32",
             (double) kv->ring_bytes / 1e6);
    return true;
}

static void encoder_kv_free_ring(voxtral_encoder_kv_state * kv) {
    if (kv->ctx && kv->ctx->encoder_steady_graph_ctx) {
        ggml_backend_sched_reset(kv->ctx->sched_encoder_steady);
        ggml_free(kv->ctx->encoder_steady_graph_ctx);
        kv->ctx->encoder_steady_graph_ctx = nullptr;
        kv->ctx->encoder_steady_graph = nullptr;
        kv->ctx->encoder_steady_graph_allocated = false;
        kv->ctx->encoder_steady_graph_meta.clear();
        kv->ctx->encoder_steady_mask.clear();
    }
    if (kv->trace_file) { std::fclose(kv->trace_file); kv->trace_file = nullptr; }
    if (kv->buf_ring) { ggml_backend_buffer_free(kv->buf_ring); kv->buf_ring = nullptr; }
    if (kv->ctx_ring) { ggml_free(kv->ctx_ring); kv->ctx_ring = nullptr; }
    kv->ring_k = kv->ring_v = nullptr;
    kv->ring_bytes = 0;
    if (kv->ctx) {
        kv->ctx->encoder_kv_allocated_bytes = 0;
        kv->ctx->encoder_kv_storage_type = GGML_TYPE_COUNT;
    }
}

// Run one KV graph over the enc frames [q_start, q_start+n_new) described by `plan`
// and append the outputs of columns [emit_col, n_new) to enc_accum. Requires the
// Mel tail to cover [conv_mel_start, conv_mel_end). emit_col > 0 is the finish-tail
// backward extension: frames [q_start, q_start+emit_col) were already emitted (their
// K/V are re-written identically), so only the newer frames are appended.
// Session 7: append the N newest encoder frames — already resident on device in
// encoder_chunk_output[0, N) after the encoder graph — to the persistent
// encoder-output ring at absolute columns [q_start, q_start+N) modulo capacity
// (monotonic circular append; wraps into <=2 segments). A standalone device->device
// copy graph: no D2H/H2D, and the delicate attention graph is left untouched. (The
// ring lives in its own buffer, so clear_kv_cache never wipes it.)
static bool copy_chunk_to_enc_out_ring(voxtral_context * ctx, int64_t q_start, int32_t N) {
    if (!ctx->want_enc_out_ring || !ctx->enc_out_ring || N <= 0) return true;
    const auto build_start = std::chrono::steady_clock::now();
    const int32_t cap  = (int32_t) ctx->enc_out_ring->ne[1];
    const size_t  scol = ctx->encoder_chunk_output->nb[1];
    const size_t  dcol = ctx->enc_out_ring->nb[1];

    static thread_local std::vector<uint8_t> meta_buf;
    ggml_context * gctx = init_graph_ctx(meta_buf, 2);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    int32_t done = 0;
    while (done < N) {
        const int32_t slot = (int32_t) ((q_start + done) % cap);
        const int32_t seg  = std::min(N - done, cap - slot);
        ggml_tensor * src = ggml_view_2d(gctx, ctx->encoder_chunk_output, VOXTRAL_ENC_DIM, seg,
                                         scol, (size_t) done * scol);
        ggml_tensor * dst = ggml_view_2d(gctx, ctx->enc_out_ring, VOXTRAL_ENC_DIM, seg,
                                         dcol, (size_t) slot * dcol);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, src, dst));
        done += seg;
    }
    voxtral_context_profile_record_internal(ctx, voxtral_profile_stage::encoder_graph_build,
                                            elapsed_ms(build_start));
    return run_graph(ctx, ctx->sched_encoder, gctx, gf, [](ggml_cgraph *){},
                     "enc-out ring copy", voxtral_profile_stage::encoder_device_copy);
}

static bool ensure_encoder_steady_graph(voxtral_encoder_kv_state * kv) {
    voxtral_context * ctx = kv->ctx;
    if (ctx->encoder_steady_graph && ctx->encoder_steady_graph_allocated) return true;
    const auto build_start = std::chrono::steady_clock::now();
    ctx->encoder_steady_graph_ctx = init_graph_ctx(
        ctx->encoder_steady_graph_meta, 8);
    if (!ctx->encoder_steady_graph_ctx) return false;
    ctx->encoder_steady_graph = build_encoder_steady_graph(
        ctx, kv->ring_k, kv->ring_v, ctx->encoder_steady_graph_ctx);
    if (!ctx->encoder_steady_graph) return false;
    voxtral_context_profile_record_internal(
        ctx, voxtral_profile_stage::encoder_graph_build,
        elapsed_ms(build_start));
    ggml_backend_sched_reset(ctx->sched_encoder_steady);
    voxtral_context_profile_note_allocation_internal(
        ctx, voxtral_profile_stage::encoder_graph_execute);
    if (!ggml_backend_sched_alloc_graph(
            ctx->sched_encoder_steady, ctx->encoder_steady_graph)) {
        LOG_ERR(ctx, "encoder KV: reusable 4-row graph allocation failed");
        return false;
    }
    ctx->encoder_steady_graph_allocated = true;
    ctx->encoder_steady_mask.assign(
        (size_t) VOXTRAL_ENC_KV_CAP * 4, -INFINITY);
    return true;
}

static bool run_encoder_steady_batch(
    voxtral_encoder_kv_state * kv, const voxtral_enc_kv_plan_t & plan) {
    constexpr int32_t P = 4;
    constexpr int32_t conv_len = 2 * P + 4;
    if (plan.n_new != P) return false;
    voxtral_context * ctx = kv->ctx;
    if (!ensure_encoder_steady_graph(kv)) return false;

    // Fixed conv window [2*(q-2), 2*(q+P)); negative startup positions are the
    // same causal zeros used by the variable graph.
    kv->conv_cm.assign(
        (size_t) VOXTRAL_MEL_N_MEL * conv_len, 0.0f);
    const int64_t first_mel = 2 * (plan.q_start - 2);
    const int64_t tail_frames =
        (int64_t) kv->mel_tail.size() / VOXTRAL_MEL_N_MEL;
    for (int32_t i = 0; i < conv_len; ++i) {
        const int64_t absolute = first_mel + i;
        if (absolute < 0) continue;
        const int64_t rel = absolute - kv->tail_base;
        if (rel < 0 || rel >= tail_frames) {
            LOG_ERR(ctx, "encoder steady: Mel %lld outside tail [%lld,%lld)",
                    (long long) absolute, (long long) kv->tail_base,
                    (long long) (kv->tail_base + tail_frames));
            return false;
        }
        const float * src = kv->mel_tail.data() +
            (size_t) rel * VOXTRAL_NUM_MEL_BINS;
        for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
            kv->conv_cm[(size_t) m * conv_len + i] = src[m];
        }
    }

    std::fill(ctx->encoder_steady_mask.begin(),
              ctx->encoder_steady_mask.end(), -INFINITY);
    for (int32_t q = 0; q < P; ++q) {
        const int64_t absolute_q = plan.q_start + q;
        ctx->encoder_steady_positions[(size_t) q] = (int32_t) absolute_q;
        ctx->encoder_steady_kv_rows[(size_t) q] =
            (int32_t) (absolute_q % VOXTRAL_ENC_KV_CAP);
        ctx->encoder_steady_output_rows[(size_t) q] =
            (int32_t) (absolute_q % VOXTRAL_ENC_OUT_RING_CAP);
        const int64_t first_key = std::max<int64_t>(
            0, absolute_q - (VOXTRAL_ENC_WINDOW - 1));
        for (int64_t key = first_key; key <= absolute_q; ++key) {
            const int32_t slot = (int32_t) (key % VOXTRAL_ENC_KV_CAP);
            ctx->encoder_steady_mask[(size_t) q * VOXTRAL_ENC_KV_CAP + slot] = 0.0f;
        }
    }

    ggml_cgraph * gf = ctx->encoder_steady_graph;
    set_graph_input(ctx, gf, "mel_input", kv->conv_cm.data(),
                    kv->conv_cm.size() * sizeof(float));
    set_graph_input(ctx, gf, "enc_positions",
                    ctx->encoder_steady_positions.data(),
                    ctx->encoder_steady_positions.size() * sizeof(int32_t));
    set_graph_input(ctx, gf, "enc_kv_rows",
                    ctx->encoder_steady_kv_rows.data(),
                    ctx->encoder_steady_kv_rows.size() * sizeof(int32_t));
    set_graph_input(ctx, gf, "enc_output_rows",
                    ctx->encoder_steady_output_rows.data(),
                    ctx->encoder_steady_output_rows.size() * sizeof(int32_t));
    set_graph_input(ctx, gf, "enc_kv_mask",
                    ctx->encoder_steady_mask.data(),
                    ctx->encoder_steady_mask.size() * sizeof(float));

    const int64_t queued_ns = encoder_now_ns();
    const int64_t compute_start_ns = encoder_now_ns();
    const auto exec_start = std::chrono::steady_clock::now();
    voxtral_context_profile_note_submit_internal(ctx);
    const enum ggml_status status = ggml_backend_sched_graph_compute_async(
        ctx->sched_encoder_steady, gf);
    const auto sync_start = std::chrono::steady_clock::now();
    ggml_backend_sched_synchronize(ctx->sched_encoder_steady);
    voxtral_context_profile_record_internal(
        ctx, voxtral_profile_stage::backend_synchronize,
        elapsed_ms(sync_start));
    voxtral_context_profile_record_internal(
        ctx, voxtral_profile_stage::encoder_graph_execute,
        elapsed_ms(exec_start));
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR(ctx, "encoder KV: reusable graph compute failed (%d)", (int) status);
        return false;
    }

    if (ctx->capture_encoder_diagnostics) {
        // Numerical acceptance samples one real steady 4-row production block.
        // This is a bounded opt-in readback, never a resident F32 KV mirror.
        ctx->diagnostic_encoder_output.resize((size_t) P * VOXTRAL_ENC_DIM);
        for (int32_t q = 0; q < P; ++q) {
            const int32_t slot = ctx->encoder_steady_output_rows[(size_t) q];
            profile_tensor_get(ctx, ctx->enc_out_ring,
                ctx->diagnostic_encoder_output.data() + (size_t) q * VOXTRAL_ENC_DIM,
                (size_t) slot * ctx->enc_out_ring->nb[1],
                (size_t) VOXTRAL_ENC_DIM * sizeof(float));
        }
        ctx->capture_encoder_diagnostics = false;
    }

    const int64_t ready_ns = encoder_now_ns();
    if (kv->telemetry) {
        kv->compute_ms.push_back((double) (ready_ns - compute_start_ns) / 1e6);
        encoder_record_outputs(kv, plan, queued_ns, compute_start_ns, ready_ns);
    }
    kv->emitted = plan.q_start + P;
    kv->transformer_frames += P;
    kv->kv_appends += P;
    kv->kv_evictions += plan.evicted;
    if (plan.read_wraps || plan.write_wraps) ++kv->kv_wraps;
    ++kv->graph_execs;
    kv->logical_frames_submitted += P;
    kv->physical_rows_evaluated += P;
    kv->max_new_per_exec = std::max(kv->max_new_per_exec, P);
    const int64_t logical = kv->emitted -
        std::max<int64_t>(0, kv->emitted - VOXTRAL_ENC_WINDOW);
    kv->peak_logical = std::max(kv->peak_logical, logical);
    return true;
}

static bool run_encoder_kv_batch(voxtral_encoder_kv_state * kv, const voxtral_enc_kv_plan_t & plan,
                                 int32_t emit_col = 0,
                                 int32_t physical_override = 0) {
    voxtral_context * ctx = kv->ctx;
    const int32_t N       = plan.n_new;
    const int32_t P       = physical_override > 0 ? physical_override : encoder_kv_schedule_from_env().physical;
    const int32_t q_offset = (int32_t) (plan.q_start % P);
    const int64_t block_start = plan.q_start - q_offset;
    const int64_t key_start = std::max<int64_t>(0, block_start - (int64_t) (VOXTRAL_ENC_WINDOW - 1));
    const int64_t key_end = block_start + P;
    const int32_t L       = (int32_t) (key_end - key_start);
    const int32_t n_mel   = VOXTRAL_MEL_N_MEL;
    const int32_t conv_len = (int32_t) (plan.conv_mel_end - plan.conv_mel_start);

    if (kv->reusable_stream_graph && ctx->want_enc_out_ring &&
        emit_col == 0 && N == 4 && P == 4) {
        return run_encoder_steady_batch(kv, plan);
    }

    // Materialize the conv Mel window channel-major [n_mel, conv_len] from the tail.
    const int64_t tail_frames = (int64_t) (kv->mel_tail.size() / (size_t) n_mel);
    if (plan.conv_mel_start < kv->tail_base || plan.conv_mel_end > kv->tail_base + tail_frames) {
        LOG_ERR(ctx, "encoder KV: Mel tail [%lld,%lld) misses conv window [%lld,%lld)",
                (long long) kv->tail_base, (long long) (kv->tail_base + tail_frames),
                (long long) plan.conv_mel_start, (long long) plan.conv_mel_end);
        return false;
    }
    kv->conv_cm.assign((size_t) n_mel * conv_len, 0.0f);
    for (int32_t i = 0; i < conv_len; ++i) {
        const int64_t rel = (plan.conv_mel_start + i) - kv->tail_base;
        const float * src = kv->mel_tail.data() + (size_t) rel * n_mel;
        for (int32_t m = 0; m < n_mel; ++m) kv->conv_cm[(size_t) m * conv_len + i] = src[m];
    }

    const auto graph_build_start = std::chrono::steady_clock::now();
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 8 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 8, false);
    // Graph metadata is host scratch, not graph state. Retain its allocation
    // across microbatches so steady state does not call the host heap once per
    // encoder execution.
    static thread_local std::vector<uint8_t> meta_buf;
    if (meta_buf.size() < meta_size) meta_buf.resize(meta_size);
    ggml_init_params p = { meta_size, meta_buf.data(), /*.no_alloc=*/true };
    ggml_context * gctx = ggml_init(p);

    int64_t materialized = 0;
    ggml_cgraph * gf = build_encoder_kv_graph(ctx, kv->ring_k, kv->ring_v, gctx, plan, P, &materialized);
    if (!gf) {
        ggml_free(gctx);
        return false;
    }
    voxtral_context_profile_record_internal(ctx, voxtral_profile_stage::encoder_graph_build,
                                            elapsed_ms(graph_build_start));

    ggml_backend_sched_reset(ctx->sched_encoder);
    voxtral_context_profile_note_allocation_internal(ctx, voxtral_profile_stage::encoder_graph_execute);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_encoder, gf)) {
        LOG_ERR(ctx, "encoder KV: failed to allocate graph (N=%d P=%d L=%d)", N, P, L);
        ggml_free(gctx);
        return false;
    }

    if (ggml_tensor * t = find_tensor_in_graph(gf, "mel_input"))
        profile_tensor_set(ctx, t, kv->conv_cm.data(), 0, (size_t) n_mel * conv_len * sizeof(float));
    if (ggml_tensor * t = find_tensor_in_graph(gf, "enc_positions")) {
        kv->pos_buf.resize(P);
        for (int32_t i = 0; i < P; ++i) kv->pos_buf[i] = (int32_t) (block_start + i);
        profile_tensor_set(ctx, t, kv->pos_buf.data(), 0, (size_t) P * sizeof(int32_t));
    }
    if (ggml_tensor * t = find_tensor_in_graph(gf, "enc_kv_mask")) {
        kv->mask_buf.assign((size_t) L * P, -INFINITY);
        for (int32_t ql = 0; ql < P; ++ql) {
            const bool real_query = ql >= q_offset && ql < q_offset + N;
            const int64_t q_abs = block_start + ql;
            for (int32_t kl = 0; kl < L; ++kl) {
                const int64_t kv_abs = key_start + kl;
                if (real_query && voxtral_enc_kv_mask_allows(kv_abs, q_abs, VOXTRAL_ENC_WINDOW)) {
                    kv->mask_buf[(size_t) ql * L + kl] = 0.0f;
                } else if (!real_query && encoder_kv_uses_segmented_attention(P) && kl == 0) {
                    // The manual softmax has no defined value for an all -Inf
                    // row. Give every discarded dummy query one private
                    // numerical sink so the oracle stays finite. Dummy rows are
                    // still excluded from persistent KV and from encoder output;
                    // all real-query masks remain byte-for-byte unchanged.
                    kv->mask_buf[(size_t) ql * L + kl] = 0.0f;
                }
            }
        }
        profile_tensor_set(ctx, t, kv->mask_buf.data(), 0, kv->mask_buf.size() * sizeof(float));
    }

    const int64_t queued_ns = encoder_now_ns();
    const int64_t compute_start_ns = encoder_now_ns();
    const auto exec_start = std::chrono::steady_clock::now();
    voxtral_context_profile_note_submit_internal(ctx);
    const enum ggml_status compute_status =
        ggml_backend_sched_graph_compute_async(ctx->sched_encoder, gf);
    const auto sync_start = std::chrono::steady_clock::now();
    ggml_backend_sched_synchronize(ctx->sched_encoder);
    voxtral_context_profile_record_internal(ctx, voxtral_profile_stage::backend_synchronize,
                                            elapsed_ms(sync_start));
    voxtral_context_profile_record_internal(ctx, voxtral_profile_stage::encoder_graph_execute,
                                            elapsed_ms(exec_start));
    ggml_backend_sched_reset(ctx->sched_encoder);
    ggml_free(gctx);
    if (compute_status != GGML_STATUS_SUCCESS) {
        LOG_ERR(ctx, "encoder KV: graph compute failed (%d)", (int) compute_status);
        return false;
    }

    // Append the newly-emitted enc frames [emit_col, N) (channel-major) to the accum.
    // Session 7.1: the incremental production path (want_enc_out_ring) reads encoder
    // output from the device ring below and never needs the host accumulation, so the
    // D2H is skipped entirely (encoder_output_d2h_bytes stays 0 — the hard gate). The
    // reference finish-only mode keeps the accumulation and counts the real D2H bytes.
    const int32_t emit_n = N - emit_col;
    if (!ctx->want_enc_out_ring) {
        const size_t old = kv->enc_accum.size();
        kv->enc_accum.resize(old + (size_t) emit_n * VOXTRAL_ENC_DIM);
        ggml_backend_tensor_get(ctx->encoder_chunk_output, kv->enc_accum.data() + old,
                                (size_t) emit_col * VOXTRAL_ENC_DIM * sizeof(float),
                                (size_t) emit_n * VOXTRAL_ENC_DIM * sizeof(float));
        kv->enc_out_d2h_bytes += (int64_t) emit_n * VOXTRAL_ENC_DIM * (int64_t) sizeof(float);
    }
    // Session 7: mirror the same N frames into the device-resident encoder-output
    // ring (device->device from encoder_chunk_output; emit_col is always 0).
    if (!copy_chunk_to_enc_out_ring(ctx, plan.q_start, N)) {
        LOG_ERR(ctx, "encoder KV: enc-out ring copy failed");
        return false;
    }
    const int64_t ready_ns = encoder_now_ns();
    if (kv->telemetry) {
        kv->compute_ms.push_back((double) (ready_ns - compute_start_ns) / 1e6);
        encoder_record_outputs(kv, plan, queued_ns, compute_start_ns, ready_ns);
    }

    // Bookkeeping / metrics. The logical head advances to q_start+N (== the new head
    // for a normal batch; == target for the backward-extended finish tail).
    kv->emitted             = plan.q_start + N;
    kv->transformer_frames += N;
    kv->kv_appends         += N;
    kv->kv_evictions       += plan.evicted;
    if (plan.read_wraps || plan.write_wraps) kv->kv_wraps++;
    kv->kv_materialized    += materialized;
    kv->graph_execs++;
    kv->logical_frames_submitted += N;
    kv->physical_rows_evaluated += P;
    kv->padding_rows_evaluated += P - N;
    if (N > kv->max_new_per_exec) kv->max_new_per_exec = N;
    const int64_t logical = kv->emitted - std::max<int64_t>(0, kv->emitted - VOXTRAL_ENC_WINDOW);
    if (logical > kv->peak_logical) kv->peak_logical = logical;
    if ((int64_t) kv->enc_accum.size() / VOXTRAL_ENC_DIM > kv->peak_output)
        kv->peak_output = (int64_t) kv->enc_accum.size() / VOXTRAL_ENC_DIM;
    return true;
}

// Trim the Mel tail to the next microbatch's convolution left context (bounded:
// a couple of enc frames of Mel history beyond the running emit position).
static void encoder_kv_trim_tail(voxtral_encoder_kv_state * kv) {
    const int64_t keep_from = std::max<int64_t>(0, 2 * (kv->emitted - 2));
    if (keep_from > kv->tail_base) {
        const size_t drop = (size_t) (keep_from - kv->tail_base) * VOXTRAL_MEL_N_MEL;
        if (drop >= kv->mel_tail.size()) kv->mel_tail.clear();
        else kv->mel_tail.erase(kv->mel_tail.begin(), kv->mel_tail.begin() + (std::ptrdiff_t) drop);
        kv->tail_base = keep_from;
    }
    const int64_t tail_now = (int64_t) (kv->mel_tail.size() / (size_t) VOXTRAL_MEL_N_MEL);
    if (tail_now > kv->peak_mel_tail) kv->peak_mel_tail = tail_now;
}

// Run enc frames [q0, q0+n) through one KV graph and append their outputs.
static bool encoder_kv_run(voxtral_encoder_kv_state * kv, int64_t q0, int32_t n, bool final,
                           int32_t physical_override = 0) {
    voxtral_enc_kv_plan_t plan;
    if (!voxtral_enc_kv_plan(q0, n, VOXTRAL_ENC_KV_CAP, VOXTRAL_ENC_WINDOW, plan)) return false;
    if (plan.conv_mel_end > kv->stable_mel) return false;   // not enough Mel (caller guards)
    if (!run_encoder_kv_batch(kv, plan, /*emit_col=*/0, physical_override)) return false;
    if (final) { kv->frames_at_finish += n; kv->frames_at_finish_exec += n; }
    else       { kv->frames_before_finish += n; }
    encoder_kv_trim_tail(kv);
    return true;
}

// Emit realizable enc frames in adapter-aligned logical microbatches. The physical
// query block is fixed independently; a logical batch is placed at its absolute
// offset inside that block and never crosses the boundary. Thus feed chunks only
// decide when a complete logical batch becomes available, not the graph shape or
// the absolute reduction order of any real frame. During feed only full logical
// batches run; finish flushes the remainder.
static bool encoder_kv_drive(voxtral_encoder_kv_state * kv, bool final) {
    const encoder_kv_schedule schedule = encoder_kv_schedule_from_env();
    // The terminal right-padding is bounded, but it used to be fed through the
    // ordinary 4-row cadence and consequently launched about 17 independent
    // encoder graphs. A wider terminal graph preserves absolute positions and
    // the same causal mask while amortising submission/synchronisation overhead.
    // Keep a selector for the measured 4/8/16/32/128 acceptance matrix.
    // On RX 6600 the full 750-frame-window tail profile is non-monotonic. P8 is
    // the measured minimum for the fixed 72-frame right pad (about 420 ms),
    // versus about 467 ms at P4 and 727 ms at P16. Keep the full-window result,
    // rather than the misleading short-context result, as the default.
    int32_t finish_physical = 8;
    if (const char * value = std::getenv("VOXTRAL_ENCODER_FINISH_PHYSICAL")) {
        const int parsed = std::atoi(value);
        if (parsed == 4 || parsed == 8 || parsed == 16 || parsed == 32 || parsed == 128) {
            finish_physical = parsed;
        }
    }
    const int64_t target = enc_frames_realizable(kv->stable_mel);   // true global count
    const bool batch_terminal = final && kv->reusable_stream_graph &&
                                kv->ctx->want_enc_out_ring;
    while (kv->emitted < target) {
        const int64_t rem = target - kv->emitted;
        // The causal left pad makes an initial 128-frame block available before
        // ordinary audio-dependent frames. Run that block once at the stable
        // throughput shape to amortize Vulkan pipeline startup. The condition
        // is based only on the absolute Mel timeline, so every feed plan uses
        // the same graph sequence (and it never waits for 128 real audio
        // frames). The explicit throughput mode remains logical=physical=128
        // for all subsequent blocks as well.
        if (kv->emitted == 0 && !final && target < VOXTRAL_ENC_KV_MAX_NEW) {
            // Hold the padded startup prefix until the deterministic warmup
            // boundary is available; otherwise a very fine feed plan could
            // launch 4-row graphs before a coarser plan reaches this point.
            break;
        }
        if (kv->emitted == 0 && target >= VOXTRAL_ENC_KV_MAX_NEW) {
            kv->warmup_frames = VOXTRAL_ENC_KV_MAX_NEW;
            if (!encoder_kv_run(kv, 0, VOXTRAL_ENC_KV_MAX_NEW, final,
                                VOXTRAL_ENC_KV_MAX_NEW)) return false;
            continue;
        }
        if (!final && rem < schedule.logical) break;                // wait for a full logical batch
        const int32_t physical = batch_terminal ? finish_physical : schedule.physical;
        const int32_t logical  = batch_terminal ? finish_physical : schedule.logical;
        const int32_t q_offset = (int32_t) (kv->emitted % physical);
        const int32_t room = physical - q_offset;
        const int32_t n = (int32_t) std::min<int64_t>(
            std::min<int64_t>(logical, room), rem);
        if (n <= 0) return false;
        if (!encoder_kv_run(kv, kv->emitted, n, final, physical)) return false;
    }
    return true;
}

// ============================================================================
// Global batch encoder (Sessions 6.1/6.2) — the artifact-free source of truth.
// ----------------------------------------------------------------------------
// Runs the per-layer KV encoder over a full even-trimmed Mel in one shot, giving
// each enc frame a real 750-frame causal window (no chunk-boundary warmup
// truncation or conv zero-pad), producing the SAME tensor as the streaming KV
// encoder. Replaces run_encoder_chunked as the production batch/offline encoder;
// run_encoder_chunked is retained as the legacy/reference strategy. Fills
// ctx.encoder_output (device) and ctx.enc_seq_used exactly like run_encoder_chunked.
// ============================================================================
static bool run_encoder_global(voxtral_context * ctx, const float * mel_cm, int32_t n_frames) {
    if (n_frames <= 0) { LOG_ERR(ctx, "encoder global: audio too short"); return false; }

    voxtral_encoder_kv_state kv;
    kv.ctx = ctx;
    if (!encoder_kv_alloc(&kv)) { LOG_ERR(ctx, "encoder global: KV ring allocation failed"); return false; }

    // Push the whole Mel (channel-major [n_mel, n_frames] -> frame-major tail).
    const int32_t n_mel = VOXTRAL_MEL_N_MEL;
    kv.mel_tail.resize((size_t) n_frames * n_mel);
    for (int32_t f = 0; f < n_frames; ++f)
        for (int32_t m = 0; m < n_mel; ++m)
            kv.mel_tail[(size_t) f * n_mel + m] = mel_cm[(size_t) m * n_frames + f];
    kv.tail_base  = 0;
    kv.stable_mel = n_frames;

    bool ok = encoder_kv_drive(&kv, /*final=*/true);
    if (ok) {
        const int32_t emitted = (int32_t) kv.emitted;
        const int32_t used    = (emitted / VOXTRAL_DOWNSAMPLE_FACTOR) * VOXTRAL_DOWNSAMPLE_FACTOR;
        if (used <= 0) {
            LOG_ERR(ctx, "encoder global: fewer than %d enc frames (%d)", VOXTRAL_DOWNSAMPLE_FACTOR, emitted);
            ok = false;
        } else if (!alloc_encoder_output(ctx, used)) {
            LOG_ERR(ctx, "encoder global: failed to allocate encoder output (%d)", used);
            ok = false;
        } else {
            ggml_backend_tensor_set(ctx->encoder_output, kv.enc_accum.data(), 0,
                                    (size_t) used * VOXTRAL_ENC_DIM * sizeof(float));
            ctx->enc_seq_used     = used;
            ctx->total_enc_tokens = used;
            LOG_INFO(ctx, "encoder global: %d mel frames -> %d enc frames (%d used)",
                     n_frames, emitted, used);
        }
    }
    encoder_kv_free_ring(&kv);
    return ok;
}

// Production batch encoder default is the artifact-free global encoder; the legacy
// chunked bounded-window recompute is selected by VOXTRAL_ENCODER_STRATEGY=reference
// (kept for A/B, tensor parity and regression debugging — see docs).
static bool run_encoder_batch(voxtral_context * ctx, const float * mel_data, int32_t n_frames) {
    const char * s = getenv("VOXTRAL_ENCODER_STRATEGY");
    const bool legacy = s && (strcmp(s, "reference") == 0 || strcmp(s, "ref") == 0 ||
                              strcmp(s, "chunked") == 0 || strcmp(s, "bounded") == 0 ||
                              strcmp(s, "bounded-window-recompute") == 0);
    return legacy ? run_encoder_chunked(ctx, mel_data, n_frames)
                  : run_encoder_global(ctx, mel_data, n_frames);
}

// ============================================================================
// Legacy/reference incremental causal encoder — bounded-window recomputation that
// REPLAYS the exact run_encoder_chunked schedule, driven by stable Mel frames as
// they arrive during feed(). Reuses run_encoder_chunk() (and hence
// build_encoder_graph): batch and streaming share one transformer implementation.
//
// Strategy (see docs/architecture/streaming-runtime.md, session 6):
//   * chunk c covers Mel [c*mel_stride, c*mel_stride + CHUNK_MEL); mel_stride =
//     CHUNK_MEL - 2*OVERLAP = 1500. skip_c = (c==0?0:OVERLAP). enc-frame index ==
//     global conv1 index (the realtime padding forces left-trunc = 0).
//   * chunk 0 is emitted PROGRESSIVELY during feed (throttled, always trunc 0 by
//     running only whole-8 Mel prefixes); chunks c>=1 are emitted only when the
//     chunk's 3000-frame window is fully covered (a full, trunc-0 chunk identical
//     to batch); the last incomplete chunk is run once at finish() EXACTLY as
//     run_encoder_chunked's final iteration (build_encoder_graph's own trunc).
//   * RoPE is shift-invariant, so absolute positions are free; rope_offset mirrors
//     batch for clarity. Every emitted frame therefore matches batch bit-for-bit.
//   * work per chunk is bounded (<= CHUNK_MEL Mel); finish() runs at most the last
//     one/two chunks, never the whole Mel again.
// ============================================================================

static constexpr int32_t VOXTRAL_ENC_MEL_STRIDE =
    VOXTRAL_ENC_CHUNK_MEL - 2 * VOXTRAL_ENC_CHUNK_OVERLAP;   // 1500 Mel frames
// Chunk-0 progressive re-run granularity: emit newly-stable frames once this many
// new stable Mel frames have accumulated since the last chunk-0 run. Multiple of 8
// (one adapter group = 8 Mel), coarse enough to bound the (bounded, one-time)
// chunk-0 recomputation, fine enough that short single-chunk streams still emit
// encoder frames before finish().
static constexpr int32_t VOXTRAL_ENC_STREAM_EMIT_MEL = 256;

// Production default = per-layer KV-cache; reference = bounded-window recompute,
// retained for A/B, tensor parity and regression debugging. Selected once per
// stream via VOXTRAL_ENCODER_STRATEGY (kv | reference); not a public API contract.
enum class enc_strategy { kv, reference };

static enc_strategy encoder_strategy_from_env() {
    const char * s = getenv("VOXTRAL_ENCODER_STRATEGY");
    if (s && (strcmp(s, "reference") == 0 || strcmp(s, "ref") == 0 ||
              strcmp(s, "bounded") == 0 || strcmp(s, "bounded-window-recompute") == 0)) {
        return enc_strategy::reference;
    }
    return enc_strategy::kv;   // Session 6.2 production default
}

struct voxtral_encoder_stream {
    voxtral_context * ctx = nullptr;

    // Active strategy + the per-layer KV runtime (allocated lazily when kv).
    enc_strategy strategy = enc_strategy::kv;
    voxtral_encoder_kv_state * kv = nullptr;

    // --- reference (bounded-window recompute) state ---------------------------
    // Bounded Mel window, frame-major: relative frame r (absolute window_base + r)
    // occupies [r*n_mel, (r+1)*n_mel). window_base == cur_chunk*mel_stride, so the
    // retained span never exceeds ~CHUNK_MEL frames (plus one transient feed).
    std::vector<float> mel_window;
    int64_t window_base = 0;   // absolute Mel frame index of mel_window[0]
    int64_t stable_mel  = 0;   // absolute count of stable Mel frames received

    int32_t cur_chunk        = 0;   // chunk currently being filled / emitted
    int64_t emitted          = 0;   // enc frames emitted so far (== global conv1 index)
    int64_t last_run_mel_end = 0;   // absolute Mel end of the last chunk-0 progressive run
    bool    finalized        = false;

    // Accumulated encoder output, host, channel-major [enc_dim, emitted]: frame f
    // at [f*enc_dim, (f+1)*enc_dim). Uploaded once to ctx.encoder_output at finish.
    std::vector<float> enc_accum;

    // Scratch: one chunk's Mel transposed to channel-major [n_mel, run_len].
    std::vector<float> chunk_mel_cm;

    // Instrumentation (Stage 12/13).
    int64_t mel_frames_received       = 0;
    int64_t encoder_executions        = 0;
    int64_t encoder_input_frames      = 0;   // sum of Mel lengths run (recompute-inclusive)
    int64_t encoder_new_frames        = 0;   // == emitted at finish
    int32_t encoder_max_window        = 0;   // max chunk Mel length run
    int64_t encoder_peak_context      = 0;   // peak mel_window size, frames
    int64_t frames_before_finish      = 0;   // enc frames emitted during feed
    int64_t frames_at_finish          = 0;   // enc frames emitted by finish()
};

static void encoder_stream_update_peak(voxtral_encoder_stream * es) {
    const int64_t frames = (int64_t) (es->mel_window.size() / (size_t) VOXTRAL_MEL_N_MEL);
    if (frames > es->encoder_peak_context) es->encoder_peak_context = frames;
}

// Run one chunk over window[Lc .. Lc+run_len) and append the newly-emitted enc
// frames to enc_accum. `at_finish` routes the emitted count to the right counter.
static bool encoder_stream_run_chunk(voxtral_encoder_stream * es,
                                     int32_t run_len, bool at_finish) {
    voxtral_context * ctx = es->ctx;
    const int32_t n_mel = VOXTRAL_MEL_N_MEL;
    const int64_t Lc    = (int64_t) es->cur_chunk * VOXTRAL_ENC_MEL_STRIDE;
    const int64_t r0    = Lc - es->window_base;   // 0 by construction
    if (r0 < 0 || (r0 + run_len) * n_mel > (int64_t) es->mel_window.size()) {
        return false;   // window does not cover the requested range
    }

    // Transpose frame-major window slice -> channel-major [n_mel, run_len].
    es->chunk_mel_cm.resize((size_t) n_mel * run_len);
    for (int32_t m = 0; m < n_mel; ++m) {
        float * dst = es->chunk_mel_cm.data() + (size_t) m * run_len;
        for (int32_t i = 0; i < run_len; ++i) {
            dst[i] = es->mel_window[(size_t) (r0 + i) * n_mel + m];
        }
    }

    const int32_t rope_offset = es->cur_chunk * VOXTRAL_ENC_CHUNK_OVERLAP;
    int32_t seq_len = 0;
    if (!run_encoder_chunk(ctx, es->chunk_mel_cm.data(), run_len, rope_offset, &seq_len)) {
        return false;
    }
    es->encoder_executions++;
    es->encoder_input_frames += run_len;
    if (run_len > es->encoder_max_window) es->encoder_max_window = run_len;
    es->last_run_mel_end = Lc + run_len;

    // Emit local frames [src_local_start, seq_len): src_local_start is skip for a
    // fresh chunk c>=1, or the number already emitted for the progressive chunk 0.
    const int32_t skip = (es->cur_chunk == 0) ? 0 : VOXTRAL_ENC_CHUNK_OVERLAP;
    int64_t src_local_start = es->emitted - (int64_t) es->cur_chunk * VOXTRAL_ENC_CHUNK_OVERLAP;
    if (src_local_start < skip) src_local_start = skip;
    if ((int64_t) seq_len > src_local_start) {
        const int32_t n_new = (int32_t) ((int64_t) seq_len - src_local_start);
        const size_t old = es->enc_accum.size();
        es->enc_accum.resize(old + (size_t) n_new * VOXTRAL_ENC_DIM);
        ggml_backend_tensor_get(ctx->encoder_chunk_output,
                                es->enc_accum.data() + old,
                                (size_t) src_local_start * VOXTRAL_ENC_DIM * sizeof(float),
                                (size_t) n_new * VOXTRAL_ENC_DIM * sizeof(float));
        es->emitted = (int64_t) es->cur_chunk * VOXTRAL_ENC_CHUNK_OVERLAP + seq_len;
        es->encoder_new_frames = es->emitted;
        if (at_finish) es->frames_at_finish += n_new;
        else           es->frames_before_finish += n_new;
    }
    return true;
}

// Advance through as many chunks as the currently-stable Mel allows. `final` runs
// the last, incomplete chunk exactly as run_encoder_chunked's final iteration.
static bool encoder_stream_drive(voxtral_encoder_stream * es, bool final) {
    for (;;) {
        const int64_t Lc = (int64_t) es->cur_chunk * VOXTRAL_ENC_MEL_STRIDE;
        if (Lc >= es->stable_mel) break;
        const int64_t avail    = es->stable_mel - Lc;
        const bool    complete = (avail >= VOXTRAL_ENC_CHUNK_MEL);
        const int32_t skip     = (es->cur_chunk == 0) ? 0 : VOXTRAL_ENC_CHUNK_OVERLAP;

        int32_t run_len = 0;
        bool    advance = false;
        if (complete) {
            run_len = VOXTRAL_ENC_CHUNK_MEL;   // full, trunc-0 chunk (== batch)
            advance = true;
        } else if (final) {
            run_len = (int32_t) avail;         // last partial chunk, batch-exact
            advance = false;
        } else if (es->cur_chunk != 0) {
            break;                              // c>=1 emits only on completion
        } else {
            // chunk 0, incomplete, during feed: progressive whole-8 prefix, throttled.
            // The re-run interval grows with the prefix (max(EMIT_MEL, prefix/2)) so
            // the number of progressive runs is logarithmic and the total chunk-0
            // recomputation stays a bounded multiple of the chunk — while a small
            // floor still lets short single-chunk streams emit during feed.
            const int32_t m8 = (int32_t) ((avail / 8) * 8);
            if (m8 <= 0) break;
            const int64_t need = std::max<int64_t>(VOXTRAL_ENC_STREAM_EMIT_MEL, es->last_run_mel_end / 2);
            if ((int64_t) m8 - es->last_run_mel_end < need) break;
            run_len = m8;
            advance = false;
        }

        // Batch pre-check: a chunk that cannot produce more than `skip` enc frames
        // contributes nothing (mirrors run_encoder_chunked's early break).
        if (mel_frames_to_enc_tokens(run_len) - skip <= 0) break;

        if (!encoder_stream_run_chunk(es, run_len, final)) return false;

        if (advance) {
            es->cur_chunk++;
            const int64_t new_base = (int64_t) es->cur_chunk * VOXTRAL_ENC_MEL_STRIDE;
            if (new_base > es->window_base) {
                const size_t drop = (size_t) (new_base - es->window_base) * VOXTRAL_MEL_N_MEL;
                if (drop >= es->mel_window.size()) es->mel_window.clear();
                else es->mel_window.erase(es->mel_window.begin(),
                                          es->mel_window.begin() + (std::ptrdiff_t) drop);
                es->window_base = new_base;
            }
            continue;   // the next chunk may already be complete
        }
        break;
    }
    return true;
}

voxtral_encoder_stream * voxtral_encoder_stream_create(voxtral_context * ctx) {
    if (!ctx) return nullptr;
    auto * es = new (std::nothrow) voxtral_encoder_stream();
    if (!es) return nullptr;
    es->ctx = ctx;
    es->strategy = encoder_strategy_from_env();
    if (es->strategy == enc_strategy::kv) {
        es->kv = new (std::nothrow) voxtral_encoder_kv_state();
        if (!es->kv) { delete es; return nullptr; }
        es->kv->ctx = ctx;
    }
    return es;
}

void voxtral_encoder_stream_destroy(voxtral_encoder_stream * es) {
    if (es && es->kv) { encoder_kv_free_ring(es->kv); delete es->kv; es->kv = nullptr; }
    delete es;
}

static void encoder_kv_reset(voxtral_encoder_kv_state * kv) {
    // Logically invalidate the cache: metadata only, no device memset (unwritten
    // slots are never read). The device ring buffer is retained for cheap reuse.
    kv->emitted = 0;
    kv->stable_mel = 0;
    kv->mel_tail.clear();
    kv->tail_base = 0;
    kv->enc_accum.clear();
    kv->finalized = false;
    kv->transformer_frames = 0;
    kv->kv_appends = 0;
    kv->kv_evictions = 0;
    kv->kv_wraps = 0;
    kv->kv_materialized = 0;
    kv->graph_execs = 0;
    kv->max_new_per_exec = 0;
    kv->warmup_frames = 0;
    kv->frames_before_finish = 0;
    kv->frames_at_finish = 0;
    kv->frames_at_finish_exec = 0;
    kv->peak_logical = 0;
    kv->peak_mel_tail = 0;
    kv->peak_output = 0;
    kv->enc_out_d2h_bytes = 0;
    kv->logical_frames_submitted = 0;
    kv->physical_rows_evaluated = 0;
    kv->padding_rows_evaluated = 0;
    kv->timeline_start_ns = 0;
    kv->current_audio_samples = 0;
    kv->left_pad_samples = 0;
    kv->audio_arrivals.clear();
    kv->mel_arrivals.clear();
    if (kv->trace_file) { std::fclose(kv->trace_file); kv->trace_file = nullptr; }
    kv->residence_ms.clear();
    kv->group_residence_ms.clear();
    kv->compute_ms.clear();
    kv->first_frame_absolute_ms = 0.0;
    kv->first_frame_residence_ms = 0.0;
    kv->first_group_absolute_ms = 0.0;
    kv->first_group_residence_ms = 0.0;
    kv->first_eight_absolute_ms = 0.0;
    kv->first_eight_residence_ms = 0.0;
    kv->have_first_frame = false;
    kv->have_first_real_frame = false;
    kv->have_first_group = false;
    kv->have_first_eight = false;
    std::fill(std::begin(kv->group_ready), std::end(kv->group_ready), 0.0);
    std::fill(std::begin(kv->group_required), std::end(kv->group_required), 0.0);
    std::fill(std::begin(kv->group_real), std::end(kv->group_real), false);
    std::fill(std::begin(kv->eight_ready), std::end(kv->eight_ready), 0.0);
    std::fill(std::begin(kv->eight_required), std::end(kv->eight_required), 0.0);
    std::fill(std::begin(kv->eight_real), std::end(kv->eight_real), false);
}

void voxtral_encoder_stream_reset(voxtral_encoder_stream * es) {
    if (!es) return;
    if (es->kv) encoder_kv_reset(es->kv);
    es->mel_window.clear();
    es->window_base = 0;
    es->stable_mel  = 0;
    es->cur_chunk   = 0;
    es->emitted     = 0;
    es->last_run_mel_end = 0;
    es->finalized   = false;
    es->enc_accum.clear();
    es->chunk_mel_cm.clear();
    es->mel_frames_received  = 0;
    es->encoder_executions   = 0;
    es->encoder_input_frames = 0;
    es->encoder_new_frames   = 0;
    es->encoder_max_window   = 0;
    es->encoder_peak_context = 0;
    es->frames_before_finish = 0;
    es->frames_at_finish     = 0;
    // Keep vector capacities (and the device KV ring) for cheap reuse.
}

bool voxtral_encoder_stream_warmup(voxtral_encoder_stream * es) {
    if (!es || !es->kv || !es->ctx || !es->ctx->want_enc_out_ring) return false;

    // 156 encoder frames are the first point at which the production stream has
    // all 39 adapter groups needed for prompt prefill + its first decoder step.
    // They exercise the startup 128-row graph followed by the steady 4-row
    // segmented-attention graph. Scratch Mel is never exposed to stream events.
    constexpr int32_t warm_enc_frames =
        VOXTRAL_N_LEFT_PAD_TOKENS + VOXTRAL_N_DELAY_TOKENS + 1;
    static_assert(warm_enc_frames == 39, "decoder prompt geometry changed");
    constexpr int32_t warm_encoder_rows = warm_enc_frames * VOXTRAL_DOWNSAMPLE_FACTOR;
    constexpr int32_t warm_mel_frames = warm_encoder_rows * 2;

    const bool capture_encoder_saved = es->ctx->capture_encoder_diagnostics;
    const bool capture_adapter_saved = es->ctx->capture_adapter_diagnostics;
    const bool capture_decoder_saved = es->ctx->capture_decoder_diagnostics;
    std::vector<float> encoder_saved = std::move(es->ctx->diagnostic_encoder_output);
    std::vector<float> adapter_saved = std::move(es->ctx->diagnostic_adapter_output);
    std::vector<float> hidden_saved = std::move(es->ctx->diagnostic_first_hidden);
    std::vector<float> logits_saved = std::move(es->ctx->diagnostic_first_logits);
    es->ctx->capture_encoder_diagnostics = false;
    es->ctx->capture_adapter_diagnostics = false;
    auto restore_diagnostics = [&] {
        es->ctx->capture_encoder_diagnostics = capture_encoder_saved;
        es->ctx->capture_adapter_diagnostics = capture_adapter_saved;
        es->ctx->capture_decoder_diagnostics = capture_decoder_saved;
        es->ctx->diagnostic_encoder_output = std::move(encoder_saved);
        es->ctx->diagnostic_adapter_output = std::move(adapter_saved);
        es->ctx->diagnostic_first_hidden = std::move(hidden_saved);
        es->ctx->diagnostic_first_logits = std::move(logits_saved);
    };

    voxtral_encoder_stream_reset(es);
    voxtral_encoder_kv_state * kv = es->kv;
    kv->reusable_stream_graph = true;
    if (!encoder_kv_alloc(kv)) {
        restore_diagnostics();
        return false;
    }
    kv->mel_tail.assign((size_t) warm_mel_frames * VOXTRAL_MEL_N_MEL, 0.0f);
    kv->tail_base = 0;
    kv->stable_mel = warm_mel_frames;
    es->stable_mel = warm_mel_frames;
    es->mel_frames_received = warm_mel_frames;

    bool ok = encoder_kv_drive(kv, /*final=*/false) &&
              kv->emitted == warm_encoder_rows;
    if (ok) {
        ok = voxtral_ctx_adapter_commit(es->ctx, 0, warm_enc_frames) == warm_enc_frames;
    }

    // clear_kv_cache() enables the diagnostic copy node while the reusable
    // decoder graph is built. The captured warmup values are discarded below;
    // the real stream reuses that graph and records its own first step.
    es->ctx->capture_decoder_diagnostics = false;

    if (ok) {
        int32_t prompt[VOXTRAL_N_LEFT_PAD_TOKENS + VOXTRAL_N_DELAY_TOKENS + 1];
        prompt[0] = VOXTRAL_TOKEN_BOS;
        std::fill(prompt + 1, prompt + warm_enc_frames, VOXTRAL_TOKEN_STREAMING_PAD);
        voxtral_ctx_decoder_begin_incremental(es->ctx);
        ok = voxtral_ctx_decoder_prefill_incremental(es->ctx, prompt, warm_enc_frames - 1);
        int32_t ignored_token = 0;
        if (ok) {
            ok = voxtral_ctx_decoder_step_incremental(
                es->ctx, prompt[warm_enc_frames - 1], warm_enc_frames - 1,
                &ignored_token);
        }
        voxtral_ctx_decoder_reset_incremental(es->ctx);
    }

    restore_diagnostics();

    // Logical state is pristine; retained capacities, scheduler buffers and
    // Vulkan pipelines are deliberately kept. Warmup work is excluded from the
    // production profile counters and first-output timeline.
    voxtral_encoder_stream_reset(es);
    voxtral_context_profile_reset_internal(es->ctx);
    return ok;
}

// Push newly-stable, frame-major Mel frames [base_frame, base_frame+n) (each
// n_mel contiguous) and run whatever became available. base_frame must equal the
// running stable_mel (contiguous, in order).
bool voxtral_encoder_stream_push_mel(voxtral_encoder_stream * es,
                                     const float * frames, int64_t base_frame, int32_t n) {
    if (!es || es->finalized) return false;
    if (n <= 0) return true;
    if (!frames) return false;
    if (base_frame != es->stable_mel) return false;   // must be contiguous

    if (es->kv) {
        es->kv->reusable_stream_graph = es->ctx->want_enc_out_ring;
        if (!encoder_kv_alloc(es->kv)) return false;
        es->kv->mel_tail.insert(es->kv->mel_tail.end(), frames,
                                frames + (size_t) n * VOXTRAL_MEL_N_MEL);
        es->kv->stable_mel += n;
        es->stable_mel += n;                 // keep the shared counter in step
        es->mel_frames_received += n;
        return encoder_kv_drive(es->kv, /*final=*/false);
    }

    es->mel_window.insert(es->mel_window.end(), frames,
                          frames + (size_t) n * VOXTRAL_MEL_N_MEL);
    es->stable_mel += n;
    es->mel_frames_received += n;
    encoder_stream_update_peak(es);
    const bool ok = encoder_stream_drive(es, /*final=*/false);
    encoder_stream_update_peak(es);
    return ok;
}

// Terminal counterpart to push_mel(). The frontend exposes all right-padding
// frames at once, so preserve that boundary and let encoder_kv_drive(final=true)
// use a bounded tail graph instead of pretending every eight Mel frames arrived
// as a separate realtime feed. Each encoder frame is still evaluated once.
bool voxtral_encoder_stream_push_mel_final(voxtral_encoder_stream * es,
                                           const float * frames,
                                           int64_t base_frame,
                                           int32_t n) {
    if (!es || es->finalized) return false;
    if (n <= 0) return true;
    if (!frames || base_frame != es->stable_mel) return false;

    if (es->kv) {
        es->kv->reusable_stream_graph = es->ctx->want_enc_out_ring;
        if (!encoder_kv_alloc(es->kv)) return false;
        es->kv->mel_tail.insert(es->kv->mel_tail.end(), frames,
                                frames + (size_t) n * VOXTRAL_MEL_N_MEL);
        es->kv->stable_mel += n;
        es->stable_mel += n;
        es->mel_frames_received += n;
        return encoder_kv_drive(es->kv, /*final=*/true);
    }

    es->mel_window.insert(es->mel_window.end(), frames,
                          frames + (size_t) n * VOXTRAL_MEL_N_MEL);
    es->stable_mel += n;
    es->mel_frames_received += n;
    encoder_stream_update_peak(es);
    const bool ok = encoder_stream_drive(es, /*final=*/true);
    encoder_stream_update_peak(es);
    return ok;
}

void voxtral_encoder_stream_note_audio(voxtral_encoder_stream * es,
                                       int64_t sample_count, int64_t arrival_ns,
                                       int64_t mel_frames_ready, int64_t mel_ready_ns,
                                       int64_t timeline_start_ns,
                                       int64_t left_pad_samples) {
    if (!es || !es->kv) return;
    // Allocate the KV runtime before recording the first audio timestamp.  The
    // ring is otherwise created lazily by push_mel(), which used to let the
    // initial padded graph observe current_audio_samples without the matching
    // arrival map (making the first residence look like absolute latency).
    if (!es->kv->ring_k && !encoder_kv_alloc(es->kv)) return;
    encoder_note_audio(es->kv, sample_count, arrival_ns, mel_frames_ready, mel_ready_ns,
                       timeline_start_ns, left_pad_samples);
}

// Finalize: emit the remaining realizable frames, then the accumulated output is
// complete. For the KV path this evaluates only the final frontend-flush frames
// through the transformer (never a replay of the encoder prefix).
bool voxtral_encoder_stream_finish(voxtral_encoder_stream * es) {
    if (!es) return false;
    if (es->finalized) return true;

    if (es->kv) {
        const bool ok = encoder_kv_drive(es->kv, /*final=*/true);
        es->kv->finalized = true;
        es->finalized = true;
        es->emitted = es->kv->emitted;
        es->kv->mel_tail.clear();           // frames complete; tail no longer needed
        es->kv->tail_base = es->kv->stable_mel;
        return ok;
    }

    const bool ok = encoder_stream_drive(es, /*final=*/true);
    es->finalized = true;
    es->encoder_new_frames = es->emitted;
    es->mel_window.clear();
    es->window_base = es->stable_mel;
    return ok;
}

const float * voxtral_encoder_stream_output(const voxtral_encoder_stream * es) {
    if (!es) return nullptr;
    const std::vector<float> & accum = es->kv ? es->kv->enc_accum : es->enc_accum;
    return accum.empty() ? nullptr : accum.data();
}

int32_t voxtral_encoder_stream_output_frames(const voxtral_encoder_stream * es) {
    if (!es) return 0;
    return (int32_t) (es->kv ? es->kv->emitted : es->emitted);
}

bool voxtral_encoder_stream_uses_kv(const voxtral_encoder_stream * es) {
    return es && es->kv != nullptr;
}

voxtral_encoder_metrics voxtral_encoder_stream_metrics(const voxtral_encoder_stream * es) {
    voxtral_encoder_metrics m{};
    if (!es) return m;

    if (es->kv) {
        const voxtral_encoder_kv_state * kv = es->kv;
        m.strategy                     = "per-layer-kv";
        m.melFramesReceived            = es->mel_frames_received;
        m.encoderExecutions            = kv->graph_execs;
        m.encoderNewFramesProduced     = kv->emitted;
        m.encoderContextFramesRetained = (int64_t) (kv->mel_tail.size() / (size_t) VOXTRAL_MEL_N_MEL);
        m.encoderPeakContextFrames     = kv->peak_mel_tail;
        m.encoderFramesBeforeFinish    = kv->frames_before_finish;
        m.encoderFramesFlushedAtFinish = kv->frames_at_finish;
        m.encoderOutputFrames          = kv->emitted;
        m.encoderStateBytes            = (int64_t) (kv->mel_tail.capacity() * sizeof(float) +
                                                    kv->conv_cm.capacity() * sizeof(float) +
                                                    kv->mask_buf.capacity() * sizeof(float));
        m.encoderOutputAccumulatedBytes= (int64_t) (kv->enc_accum.size() * sizeof(float));
        m.encoderOutputD2hBytes        = kv->enc_out_d2h_bytes;
        m.finalized                    = kv->finalized;
        // Ideal work: each unique enc frame runs the transformer exactly once.
        m.encoderUniqueFrames             = kv->emitted;
        m.encoderTransformerFramesComputed= kv->transformer_frames;
        m.encoderFrameLayerEvaluations    = kv->transformer_frames * (int64_t) es->ctx->model->hp.enc_layers;
        m.encoderInputFramesProcessed     = kv->transformer_frames;   // transformer frames (unit: enc frames)
        m.encoderFramesRecomputed         = kv->transformer_frames - kv->emitted;   // 0 in the ideal case
        m.encoderKvAppends                = kv->kv_appends;
        m.encoderKvEvictions              = kv->kv_evictions;
        m.encoderKvWraps                  = kv->kv_wraps;
        m.encoderKvMaterializedFrames     = kv->kv_materialized;
        m.encoderGraphExecutions          = kv->graph_execs;
        m.encoderMaxNewFramesPerExecution = kv->max_new_per_exec;
        m.encoderWarmupFrames             = kv->warmup_frames;
        m.encoderMaxWindowFrames          = kv->max_new_per_exec;
        m.encoderFramesComputedDuringFinish= kv->frames_at_finish_exec;
        m.encoderKvAllocatedBytes         = kv->ring_bytes;
        m.encoderKvLogicalFrames          = kv->emitted - std::max<int64_t>(0, kv->emitted - VOXTRAL_ENC_WINDOW);
        m.encoderKvPeakLogicalFrames      = kv->peak_logical;
        m.encoderKvCapacityFrames         = VOXTRAL_ENC_KV_CAP;
        m.encoderKvElementSize            = (int32_t) ggml_type_size(kv->ring_k->type);
        m.encoderMelRetainedBytes         = (int64_t) (kv->mel_tail.size() * sizeof(float));
        m.encoderMelRetainedFrames        = (int64_t) (kv->mel_tail.size() / (size_t) VOXTRAL_MEL_N_MEL);
        m.encoderMelPeakRetainedFrames    = kv->peak_mel_tail;
        m.encoderOutputQueuedFrames       = kv->emitted;
        m.encoderOutputPeakQueuedFrames   = kv->peak_output;
        const encoder_kv_schedule schedule = encoder_kv_schedule_from_env();
        m.encoderScheduler                 = "static";
        m.encoderLogicalBatchFrames       = schedule.logical;
        m.encoderPhysicalQueryRows        = schedule.physical;
        m.encoderLogicalFramesSubmitted   = kv->logical_frames_submitted;
        m.encoderPhysicalQueryRowsEvaluated = kv->physical_rows_evaluated;
        m.encoderPaddingRowsEvaluated     = kv->padding_rows_evaluated;
        if (kv->telemetry) {
            m.encoderFirstFrameAbsoluteMs  = kv->first_frame_absolute_ms;
            m.encoderFirstFrameResidenceMs = kv->first_frame_residence_ms;
            m.firstAdapterGroupAbsoluteMs  = kv->first_group_absolute_ms;
            m.firstAdapterGroupResidenceMs = kv->first_group_residence_ms;
            m.firstEightFrameGroupAbsoluteMs = kv->first_eight_absolute_ms;
            m.firstEightFrameGroupResidenceMs = kv->first_eight_residence_ms;
            m.encoderResidenceP50Ms        = encoder_percentile(kv->residence_ms, 0.50);
            m.encoderResidenceP95Ms        = encoder_percentile(kv->residence_ms, 0.95);
            m.encoderResidenceP99Ms        = encoder_percentile(kv->residence_ms, 0.99);
            m.encoderResidenceMaxMs        = kv->residence_ms.empty() ? 0.0 : *std::max_element(kv->residence_ms.begin(), kv->residence_ms.end());
            m.adapterGroupResidenceP50Ms   = encoder_percentile(kv->group_residence_ms, 0.50);
            m.adapterGroupResidenceP95Ms   = encoder_percentile(kv->group_residence_ms, 0.95);
            m.adapterGroupResidenceP99Ms   = encoder_percentile(kv->group_residence_ms, 0.99);
            m.adapterGroupResidenceMaxMs   = kv->group_residence_ms.empty() ? 0.0 : *std::max_element(kv->group_residence_ms.begin(), kv->group_residence_ms.end());
            m.encoderComputeP50Ms          = encoder_percentile(kv->compute_ms, 0.50);
            m.encoderComputeP95Ms          = encoder_percentile(kv->compute_ms, 0.95);
            m.encoderComputeP99Ms          = encoder_percentile(kv->compute_ms, 0.99);
            m.encoderComputeMaxMs          = kv->compute_ms.empty() ? 0.0 : *std::max_element(kv->compute_ms.begin(), kv->compute_ms.end());
            m.encoderComputeWarmMaxMs      = kv->compute_ms.size() > 1
                ? *std::max_element(kv->compute_ms.begin() + 1, kv->compute_ms.end())
                : m.encoderComputeMaxMs;
        }
        return m;
    }

    m.strategy                    = "bounded-window-recompute";
    m.melFramesReceived           = es->mel_frames_received;
    m.encoderExecutions           = es->encoder_executions;
    m.encoderInputFramesProcessed = es->encoder_input_frames;
    m.encoderNewFramesProduced    = es->emitted;
    m.encoderFramesRecomputed     = es->encoder_input_frames - es->mel_frames_received;
    m.encoderMaxWindowFrames      = es->encoder_max_window;
    m.encoderContextFramesRetained= (int64_t) (es->mel_window.size() / (size_t) VOXTRAL_MEL_N_MEL);
    m.encoderPeakContextFrames    = es->encoder_peak_context;
    m.encoderFramesBeforeFinish   = es->frames_before_finish;
    m.encoderFramesFlushedAtFinish= es->frames_at_finish;
    m.encoderOutputFrames         = es->emitted;
    m.encoderStateBytes           = (int64_t) (es->mel_window.capacity() * sizeof(float) +
                                               es->chunk_mel_cm.capacity() * sizeof(float));
    m.encoderOutputAccumulatedBytes = (int64_t) (es->enc_accum.size() * sizeof(float));
    // The reference/chunked path always D2Hs its output into the host accumulation.
    m.encoderOutputD2hBytes       = (int64_t) (es->enc_accum.size() * sizeof(float));
    m.curChunk                    = es->cur_chunk;
    m.finalized                   = es->finalized;
    m.encoderUniqueFrames             = es->emitted;
    m.encoderTransformerFramesComputed= es->encoder_input_frames;   // reference recomputes overlap
    m.encoderKvCapacityFrames         = 0;
    return m;
}

// ============================================================================
// Run Adapter
// ============================================================================

static bool run_adapter(voxtral_context * ctx) {
    const int32_t enc_seq = ctx->enc_seq_used;
    const int32_t dec_seq = enc_seq / VOXTRAL_DOWNSAMPLE_FACTOR;
    if (ctx->numerical_diagnostics) ctx->diagnostic_adapter_output.clear();

    LOG_INFO(ctx, "running adapter: enc_seq=%d -> dec_seq=%d", enc_seq, dec_seq);

    // Allocate decoder_memory for this utterance
    if (!alloc_decoder_memory(ctx, dec_seq)) {
        LOG_ERR(ctx, "adapter: failed to allocate decoder memory (%d tokens, %.2f MB)",
                dec_seq, (double) dec_seq * VOXTRAL_DEC_DIM * sizeof(float) / 1e6);
        return false;
    }

    const auto build_start = std::chrono::steady_clock::now();
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
    std::vector<uint8_t> meta_buf(meta_size);

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);

    ggml_cgraph * gf = build_adapter_graph(ctx, gctx);
    voxtral_context_profile_record_internal(ctx, voxtral_profile_stage::adapter_graph_build,
                                            elapsed_ms(build_start));
    log_graph_info(ctx, "adapter", gf);

    ggml_backend_sched_reset(ctx->sched_adapter);
    voxtral_context_profile_note_allocation_internal(ctx, voxtral_profile_stage::adapter_graph_execute);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_adapter, gf)) {
        LOG_ERR(ctx, "adapter: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    const auto exec_start = std::chrono::steady_clock::now();
    voxtral_context_profile_note_submit_internal(ctx);
    const enum ggml_status status = ggml_backend_sched_graph_compute_async(ctx->sched_adapter, gf);
    const auto sync_start = std::chrono::steady_clock::now();
    ggml_backend_sched_synchronize(ctx->sched_adapter);
    voxtral_context_profile_record_internal(ctx, voxtral_profile_stage::backend_synchronize,
                                            elapsed_ms(sync_start));
    voxtral_context_profile_record_internal(ctx, voxtral_profile_stage::adapter_graph_execute,
                                            elapsed_ms(exec_start));
    ggml_backend_sched_reset(ctx->sched_adapter);
    ggml_free(gctx);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR(ctx, "adapter: graph compute failed (%d)", (int) status);
        return false;
    }

    if (ctx->numerical_diagnostics) {
        ctx->diagnostic_adapter_output.resize((size_t) ctx->dec_seq_len * VOXTRAL_DEC_DIM);
        profile_tensor_get(ctx, ctx->decoder_memory,
            ctx->diagnostic_adapter_output.data(), 0,
            ctx->diagnostic_adapter_output.size() * sizeof(float));
    }

    LOG_INFO(ctx, "adapter done: dec_seq_len=%d (%.2f MB on device)",
             ctx->dec_seq_len,
             (double) ggml_nbytes(ctx->decoder_memory) / 1e6);
    return true;
}

// ============================================================================
// Run Decoder Prefill
// ============================================================================

static bool run_decoder_prefill(
    voxtral_context * ctx,
    const int32_t   * token_ids,
    int32_t           n_tokens,
    float           * logits_out)  // [vocab_size]
{
    LOG_INFO(ctx, "decoder prefill: %d tokens", n_tokens);

    if (n_tokens > VOXTRAL_DEC_WINDOW) {
        LOG_ERR(ctx, "decoder prefill: n_tokens=%d exceeds DEC_WINDOW=%d", n_tokens, VOXTRAL_DEC_WINDOW);
        return false;
    }
    voxtral_decoder_kv_plan_t kv_plan;
    if (!voxtral_decoder_kv_plan(ctx->decoder_kv,
                                 ctx->decoder_kv.next_absolute_position,
                                 n_tokens, kv_plan)) {
        LOG_ERR(ctx, "decoder prefill: invalid KV append start=%lld n=%d",
                (long long) ctx->decoder_kv.next_absolute_position, n_tokens);
        return false;
    }

    const auto build_start = std::chrono::steady_clock::now();
    static thread_local std::vector<uint8_t> meta_buf;
    ggml_context * gctx = init_graph_ctx(meta_buf, 4);
    ggml_cgraph * gf = build_decoder_prefill_graph(ctx, gctx, n_tokens);
    if (!gf) { ggml_free(gctx); return false; }
    voxtral_context_profile_record_internal(ctx, voxtral_profile_stage::decoder_prefill_graph_build,
                                            elapsed_ms(build_start));
    log_graph_info(ctx, "decoder prefill", gf);

    if (!run_graph(ctx, ctx->sched_dec_pre, gctx, gf, [&](ggml_cgraph * g) {
        set_graph_input(ctx, g, "token_ids", token_ids, n_tokens * sizeof(int32_t));
        std::vector<int32_t> pos(n_tokens);
        for (int32_t i = 0; i < n_tokens; ++i) {
            pos[(size_t) i] = (int32_t) (kv_plan.absolute_start + i);
        }
        set_graph_input(ctx, g, "positions", pos.data(), n_tokens * sizeof(int32_t));
        set_graph_input(ctx, g, "time_emb", ctx->time_emb_cpu.data(), VOXTRAL_DEC_DIM * sizeof(float));
        std::vector<float> mask;
        fill_causal_mask(mask, n_tokens);
        set_graph_input(ctx, g, "causal_mask", mask.data(), mask.size() * sizeof(float));
    }, "decoder prefill", voxtral_profile_stage::decoder_prefill_graph_execute)) {
        return false;
    }

    if (logits_out) {
        profile_tensor_get(ctx, ctx->decoder_logits, logits_out, 0, VOXTRAL_VOCAB_SIZE * sizeof(float));
    }
    decoder_kv_commit_plan(ctx, kv_plan);

    LOG_INFO(ctx, "decoder prefill done");
    return true;
}

// ============================================================================
// Run Decoder Step
// ============================================================================

static bool run_decoder_step(
    voxtral_context * ctx,
    int32_t           token_id,
    int32_t           position,     // absolute position in decoder sequence
    int32_t           audio_pos,    // position in adapter output for audio embedding
    float           * logits_out,   // [vocab_size] — optional, full logits readback
    int32_t         * token_out)    // optional — greedy argmax token (cheap readback)
{
    voxtral_decoder_kv_plan_t kv_plan;
    if (!voxtral_decoder_kv_plan(ctx->decoder_kv, position, 1, kv_plan)) {
        LOG_ERR(ctx, "decoder step: invalid absolute position %d (next=%lld)",
                position, (long long) ctx->decoder_kv.next_absolute_position);
        return false;
    }

    const auto rollover_step_start = std::chrono::steady_clock::now();
    const char * step_graph_mode = std::getenv("VOXTRAL_DECODER_STEP_GRAPH");
    const bool force_dynamic_graph = step_graph_mode &&
        (std::strcmp(step_graph_mode, "dynamic") == 0 ||
         std::strcmp(step_graph_mode, "oracle") == 0);
    if (ctx->dec_audio_cap > 0 && !force_dynamic_graph) {
        // Incremental production owns the device-resident audio ring and can
        // reuse one fixed-topology graph.  The finish-only oracle instead owns
        // a linear decoder_memory tensor; it must keep the dynamic view below
        // or this graph would silently read unrelated audio-ring columns.
        const bool full_cache =
            kv_plan.used_after == ctx->decoder_kv_configured_capacity;
        if (!ensure_decoder_step_graph(ctx, full_cache)) return false;
        ggml_cgraph * gf = ctx->decoder_step_graph;

        const int32_t kv_slot =
            position % ctx->decoder_kv_configured_capacity;
        const int32_t audio_slot = audio_pos % ctx->dec_audio_cap;
        set_graph_input(ctx, gf, "token_id", &token_id, sizeof(token_id));
        set_graph_input(ctx, gf, "position", &position, sizeof(position));
        set_graph_input(ctx, gf, "time_emb", ctx->time_emb_cpu.data(),
                        VOXTRAL_DEC_DIM * sizeof(float));
        set_graph_input(ctx, gf, "decoder_kv_slot", &kv_slot, sizeof(kv_slot));
        set_graph_input(ctx, gf, "decoder_audio_slot", &audio_slot, sizeof(audio_slot));

        const int32_t valid = (int32_t) kv_plan.used_after;
        if (!full_cache && ctx->decoder_step_mask_valid != valid) {
            std::fill(ctx->decoder_step_mask.begin(),
                      ctx->decoder_step_mask.end(), -INFINITY);
            std::fill(ctx->decoder_step_mask.begin(),
                      ctx->decoder_step_mask.begin() + valid, 0.0f);
            set_graph_input(ctx, gf, "decoder_kv_mask",
                            ctx->decoder_step_mask.data(),
                            ctx->decoder_step_mask.size() * sizeof(float));
            ctx->decoder_step_mask_valid = valid;
        }

        const auto exec_start = std::chrono::steady_clock::now();
        voxtral_context_profile_note_submit_internal(ctx);
        const enum ggml_status status =
            ggml_backend_sched_graph_compute_async(ctx->sched_dec_step, gf);
        const auto sync_start = std::chrono::steady_clock::now();
        ggml_backend_sched_synchronize(ctx->sched_dec_step);
        voxtral_context_profile_record_internal(
            ctx, voxtral_profile_stage::backend_synchronize,
            elapsed_ms(sync_start));
        voxtral_context_profile_record_internal(
            ctx, voxtral_profile_stage::decoder_step_graph_execute,
            elapsed_ms(exec_start));
        if (status != GGML_STATUS_SUCCESS) {
            LOG_ERR(ctx, "decoder step: reusable graph compute failed (%d)",
                    (int) status);
            return false;
        }
    } else {
        // Reference/offline execution addresses the linear adapter output by a
        // graph view.  This path is intentionally not the production steady
        // graph and may rebuild per step; it preserves the accepted oracle.
        const auto build_start = std::chrono::steady_clock::now();
        static thread_local std::vector<uint8_t> step_meta_buf;
        ggml_context * gctx = init_graph_ctx(step_meta_buf, 4);
        ggml_cgraph * gf = build_decoder_step_graph(ctx, gctx, position, audio_pos);
        if (!gf) {
            ggml_free(gctx);
            return false;
        }
        voxtral_context_profile_record_internal(
            ctx, voxtral_profile_stage::decoder_step_graph_build,
            elapsed_ms(build_start));
        if (!run_graph(ctx, ctx->sched_dec_step, gctx, gf,
                       [&](ggml_cgraph * graph) {
                           set_graph_input(ctx, graph, "token_id", &token_id,
                                           sizeof(token_id));
                           set_graph_input(ctx, graph, "position", &position,
                                           sizeof(position));
                           set_graph_input(ctx, graph, "time_emb",
                                           ctx->time_emb_cpu.data(),
                                           VOXTRAL_DEC_DIM * sizeof(float));
                       }, "decoder step",
                       voxtral_profile_stage::decoder_step_graph_execute)) {
            return false;
        }
    }
    const double rollover_step_ms = elapsed_ms(rollover_step_start);
    auto & rollover = ctx->decoder_rollover;
    if (rollover.first_wrap_position < 0) {
        if (kv_plan.wrapped) {
            rollover.first_wrap_position = position;
            rollover.pre_wrap_p99_ms = fixed_percentile(
                rollover.recent, rollover.recent_count, 0.99);
            rollover.wrap_step_ms = rollover_step_ms;
        } else {
            rollover.recent[rollover.recent_next] = rollover_step_ms;
            rollover.recent_next =
                (rollover.recent_next + 1) % rollover.recent.size();
            rollover.recent_count = std::min(
                rollover.recent_count + 1, rollover.recent.size());
        }
    } else if (position > rollover.first_wrap_position &&
               rollover.post_count < rollover.post.size()) {
        rollover.post[rollover.post_count++] = rollover_step_ms;
    }

    // Read back the greedy token (4 bytes) and/or full logits if requested.
    // Argmax is a node in the decoder graph, so its count is exact but its GPU
    // duration is inseparable from decoder-step execution without a Vulkan
    // timestamp-query build.  Keep total=0 and report that containment explicitly.
    voxtral_context_profile_record_internal(ctx, voxtral_profile_stage::argmax, 0.0);
    if (token_out) {
        const auto read_start = std::chrono::steady_clock::now();
        profile_tensor_get(ctx, ctx->decoder_argmax, token_out, 0, sizeof(int32_t));
        voxtral_context_profile_record_internal(ctx, voxtral_profile_stage::token_readback,
                                                elapsed_ms(read_start));
    }
    if (logits_out) {
        profile_tensor_get(ctx, ctx->decoder_logits, logits_out, 0, VOXTRAL_VOCAB_SIZE * sizeof(float));
    }
    if (ctx->capture_decoder_diagnostics) {
        ctx->diagnostic_first_hidden.resize(VOXTRAL_DEC_DIM);
        ctx->diagnostic_first_logits.resize(VOXTRAL_VOCAB_SIZE);
        profile_tensor_get(ctx, ctx->decoder_hidden_diagnostic,
            ctx->diagnostic_first_hidden.data(), 0, VOXTRAL_DEC_DIM * sizeof(float));
        profile_tensor_get(ctx, ctx->decoder_logits,
            ctx->diagnostic_first_logits.data(), 0, VOXTRAL_VOCAB_SIZE * sizeof(float));
        ctx->capture_decoder_diagnostics = false;
    }

    decoder_kv_commit_plan(ctx, kv_plan);
    return true;
}

const std::vector<float> & voxtral_context_diagnostic_first_hidden_internal(
    const voxtral_context * ctx) {
    static const std::vector<float> empty;
    return ctx ? ctx->diagnostic_first_hidden : empty;
}

const std::vector<float> & voxtral_context_diagnostic_first_logits_internal(
    const voxtral_context * ctx) {
    static const std::vector<float> empty;
    return ctx ? ctx->diagnostic_first_logits : empty;
}

const std::vector<float> & voxtral_context_diagnostic_adapter_output_internal(
    const voxtral_context * ctx) {
    static const std::vector<float> empty;
    return ctx ? ctx->diagnostic_adapter_output : empty;
}

const std::vector<float> & voxtral_context_diagnostic_encoder_output_internal(
    const voxtral_context * ctx) {
    static const std::vector<float> empty;
    return ctx ? ctx->diagnostic_encoder_output : empty;
}

// ============================================================================
// Session 7: device-resident incremental adapter + decoder entry points.
// These drive the SAME graphs as the batch/finish path (build_adapter_ring_graph,
// build_decoder_prefill_graph, build_decoder_step_graph) but over the persistent
// device rings, so during feed() the adapter and decoder advance without ever
// copying encoder output or audio embeddings across the device boundary. Only a
// 4-byte token id is read back per decoder step.
// ============================================================================

static bool ensure_adapter_graph(voxtral_context * ctx, int32_t groups) {
    voxtral_adapter_graph_cache & cache = groups == 1
        ? ctx->adapter_group_graph : ctx->adapter_batch_graph;
    ggml_backend_sched_t sched = groups == 1
        ? ctx->sched_adapter_group : ctx->sched_adapter_batch;
    if (cache.graph && cache.allocated) return true;
    cache.groups = groups;
    const auto build_start = std::chrono::steady_clock::now();
    cache.gctx = init_graph_ctx(cache.meta, 2);
    if (!cache.gctx) return false;
    cache.graph = build_adapter_ring_graph_reusable(ctx, cache.gctx, groups);
    if (!cache.graph) return false;
    voxtral_context_profile_record_internal(
        ctx, voxtral_profile_stage::adapter_graph_build,
        elapsed_ms(build_start));
    ggml_backend_sched_reset(sched);
    voxtral_context_profile_note_allocation_internal(
        ctx, voxtral_profile_stage::adapter_graph_execute);
    if (!ggml_backend_sched_alloc_graph(sched, cache.graph)) {
        LOG_ERR(ctx, "adapter: reusable %d-group graph allocation failed", groups);
        return false;
    }
    cache.allocated = true;
    cache.encoder_rows.resize(
        (size_t) groups * VOXTRAL_DOWNSAMPLE_FACTOR);
    cache.audio_rows.resize((size_t) groups);
    return true;
}

int32_t voxtral_ctx_adapter_commit(voxtral_context * ctx, int64_t group_start, int32_t n_groups) {
    if (!ctx || !ctx->enc_out_ring || !ctx->audio_emb_ring || n_groups <= 0) return -1;
    const int32_t enc_cap = (int32_t) ctx->enc_out_ring->ne[1];
    const int32_t aemb_cap = (int32_t) ctx->audio_emb_ring->ne[1];
    int32_t done = 0;
    while (done < n_groups) {
        // The startup prefix is exactly 32 groups. Everything else uses the
        // one-group steady graph; arbitrary bounded tails are split without
        // creating another graph family.
        const int32_t run = n_groups - done >= 32 ? 32 : 1;
        if (!ensure_adapter_graph(ctx, run)) return -1;
        voxtral_adapter_graph_cache & cache = run == 1
            ? ctx->adapter_group_graph : ctx->adapter_batch_graph;
        ggml_backend_sched_t sched = run == 1
            ? ctx->sched_adapter_group : ctx->sched_adapter_batch;

        for (int32_t i = 0; i < run; ++i) {
            const int64_t group = group_start + done + i;
            cache.audio_rows[(size_t) i] = (int32_t) (group % aemb_cap);
            for (int32_t j = 0; j < VOXTRAL_DOWNSAMPLE_FACTOR; ++j) {
                cache.encoder_rows[(size_t) i * VOXTRAL_DOWNSAMPLE_FACTOR + j] =
                    (int32_t) ((group * VOXTRAL_DOWNSAMPLE_FACTOR + j) % enc_cap);
            }
        }
        set_graph_input(ctx, cache.graph, "adapter_encoder_rows",
                        cache.encoder_rows.data(),
                        cache.encoder_rows.size() * sizeof(int32_t));
        set_graph_input(ctx, cache.graph, "adapter_audio_rows",
                        cache.audio_rows.data(),
                        cache.audio_rows.size() * sizeof(int32_t));

        const auto exec_start = std::chrono::steady_clock::now();
        voxtral_context_profile_note_submit_internal(ctx);
        const enum ggml_status status =
            ggml_backend_sched_graph_compute_async(sched, cache.graph);
        const auto sync_start = std::chrono::steady_clock::now();
        ggml_backend_sched_synchronize(sched);
        voxtral_context_profile_record_internal(
            ctx, voxtral_profile_stage::backend_synchronize,
            elapsed_ms(sync_start));
        voxtral_context_profile_record_internal(
            ctx, voxtral_profile_stage::adapter_graph_execute,
            elapsed_ms(exec_start));
        if (status != GGML_STATUS_SUCCESS) {
            LOG_ERR(ctx, "adapter: reusable graph compute failed (%d)", (int) status);
            return -1;
        }
        if (ctx->capture_adapter_diagnostics) {
            // Capture one production adapter vector after the real reusable
            // graph. It is sufficient for an F16/F32 tensor delta and bounded
            // independently of utterance length.
            ctx->diagnostic_adapter_output.resize(VOXTRAL_DEC_DIM);
            profile_tensor_get(ctx, ctx->audio_emb_ring,
                ctx->diagnostic_adapter_output.data(),
                (size_t) cache.audio_rows[0] * ctx->audio_emb_ring->nb[1],
                (size_t) VOXTRAL_DEC_DIM * sizeof(float));
            ctx->capture_adapter_diagnostics = false;
        }
        done += run;
    }
    return n_groups;
}

// Prepare the persistent decoder for an incremental stream: clear its KV cache and
// route the realtime decoder graphs to read audio embeddings from audio_emb_ring.
void voxtral_ctx_decoder_begin_incremental(voxtral_context * ctx) {
    if (!ctx) return;
    clear_kv_cache(ctx);
    ctx->dec_audio_src = ctx->audio_emb_ring;
    ctx->dec_audio_cap = VOXTRAL_AEMB_RING_CAP;
}

// One incremental decoder prefill over the prompt tokens (positions [0, n_tokens)).
// Runs exactly once per stream; logits are not read back.
bool voxtral_ctx_decoder_prefill_incremental(voxtral_context * ctx, const int32_t * token_ids, int32_t n_tokens) {
    if (!ctx || n_tokens <= 0) return false;
    return run_decoder_prefill(ctx, token_ids, n_tokens, /*logits_out=*/nullptr);
}

// One incremental decoder step at absolute `position`, reading the audio embedding
// at audio_pos == position from audio_emb_ring. Returns the greedy argmax token in
// *token_out via a 4-byte device readback; no full-logits copy.
bool voxtral_ctx_decoder_step_incremental(voxtral_context * ctx, int32_t token_id, int32_t position, int32_t * token_out) {
    if (!ctx) return false;
    return run_decoder_step(ctx, token_id, position, /*audio_pos=*/position, /*logits_out=*/nullptr, token_out);
}

// Detach the incremental audio source (restores batch/finish behavior) and reset KV.
void voxtral_ctx_decoder_reset_incremental(voxtral_context * ctx) {
    if (!ctx) return;
    ctx->dec_audio_src = nullptr;
    ctx->dec_audio_cap = 0;
    clear_kv_cache(ctx);
}

void voxtral_ctx_set_enc_out_ring_active(voxtral_context * ctx, bool active) {
    if (ctx) ctx->want_enc_out_ring = active;
}

int32_t voxtral_ctx_enc_out_ring_frames(const voxtral_context * ctx) {
    return (ctx && ctx->enc_out_ring) ? (int32_t) ctx->enc_out_ring->ne[1] : 0;
}
int32_t voxtral_ctx_aemb_ring_frames(const voxtral_context * ctx) {
    return (ctx && ctx->audio_emb_ring) ? (int32_t) ctx->audio_emb_ring->ne[1] : 0;
}

static bool read_realtime_ring_rows(const ggml_tensor * tensor,
                                    int64_t absolute_start, int32_t rows,
                                    int32_t width, float * dst) {
    if (!tensor || !dst || absolute_start < 0 || rows <= 0 || width <= 0 ||
        tensor->type != GGML_TYPE_F32) return false;
    const int32_t capacity = (int32_t) tensor->ne[1];
    const size_t row_bytes = (size_t) width * sizeof(float);
    if (rows > capacity || tensor->ne[0] != width ||
        tensor->nb[0] != sizeof(float) || tensor->nb[1] != row_bytes) {
        return false;
    }
    int32_t done = 0;
    while (done < rows) {
        const int32_t slot =
            (int32_t) ((absolute_start + done) % capacity);
        const int32_t count = std::min(rows - done, capacity - slot);
        ggml_backend_tensor_get(
            tensor,
            dst + (size_t) done * width,
            (size_t) slot * tensor->nb[1],
            (size_t) count * row_bytes);
        done += count;
    }
    return true;
}

bool voxtral_ctx_read_enc_out_ring_internal(
    const voxtral_context * ctx, int64_t absolute_start, int32_t rows, float * dst) {
    return ctx && read_realtime_ring_rows(
        ctx->enc_out_ring, absolute_start, rows, VOXTRAL_ENC_DIM, dst);
}

bool voxtral_ctx_read_aemb_ring_internal(
    const voxtral_context * ctx, int64_t absolute_start, int32_t rows, float * dst) {
    return ctx && read_realtime_ring_rows(
        ctx->audio_emb_ring, absolute_start, rows, VOXTRAL_DEC_DIM, dst);
}

// Offline prefill: prefix tokens + audio embeddings + suffix tokens in one graph.
static bool run_offline_prefill(voxtral_context * ctx,
    const int32_t * prefix_ids, int32_t n_prefix,
    int32_t n_audio,
    const int32_t * suffix_ids, int32_t n_suffix,
    int32_t * token_out, float * logits_out) {
    const int32_t n_tokens = n_prefix + n_audio + n_suffix;
    voxtral_decoder_kv_plan_t kv_plan;
    if (!voxtral_decoder_kv_plan(ctx->decoder_kv,
                                 ctx->decoder_kv.next_absolute_position,
                                 n_tokens, kv_plan)) return false;
    static thread_local std::vector<uint8_t> meta_buf;
    ggml_context * gctx = init_graph_ctx(meta_buf, 8);
    ggml_cgraph * gf = build_offline_prefill_graph(ctx, gctx, n_prefix, n_audio, n_suffix);
    if (!gf) { ggml_free(gctx); return false; }

    if (!run_graph(ctx, ctx->sched_dec_pre, gctx, gf, [&](ggml_cgraph * g) {
        set_graph_input(ctx, g, "prefix_ids", prefix_ids, n_prefix * sizeof(int32_t));
        set_graph_input(ctx, g, "suffix_ids", suffix_ids, n_suffix * sizeof(int32_t));
        std::vector<int32_t> pos(n_tokens);
        for (int32_t i = 0; i < n_tokens; ++i) {
            pos[(size_t) i] = (int32_t) (kv_plan.absolute_start + i);
        }
        set_graph_input(ctx, g, "positions", pos.data(), n_tokens * sizeof(int32_t));
        std::vector<float> mask;
        fill_causal_mask(mask, n_tokens);
        set_graph_input(ctx, g, "causal_mask", mask.data(), mask.size() * sizeof(float));
    }, "offline prefill", voxtral_profile_stage::decoder_prefill_graph_execute)) {
        return false;
    }

    if (token_out)  profile_tensor_get(ctx, ctx->decoder_argmax, token_out, 0, sizeof(int32_t));
    if (logits_out) profile_tensor_get(ctx, ctx->decoder_logits, logits_out, 0, VOXTRAL_VOCAB_SIZE * sizeof(float));
    decoder_kv_commit_plan(ctx, kv_plan);
    return true;
}

// Offline single-token step.
static bool run_offline_step(voxtral_context * ctx, int32_t token_id, int32_t position, int32_t * token_out) {
    voxtral_decoder_kv_plan_t kv_plan;
    if (!voxtral_decoder_kv_plan(ctx->decoder_kv, position, 1, kv_plan)) return false;
    static thread_local std::vector<uint8_t> step_meta_buf;
    ggml_context * gctx = init_graph_ctx(step_meta_buf, 4);
    ggml_cgraph * gf = build_offline_step_graph(ctx, gctx);
    if (!gf) { ggml_free(gctx); return false; }

    if (!run_graph(ctx, ctx->sched_dec_step, gctx, gf, [&](ggml_cgraph * g) {
        set_graph_input(ctx, g, "token_id", &token_id, sizeof(int32_t));
        set_graph_input(ctx, g, "position", &position, sizeof(int32_t));
    }, "offline step", voxtral_profile_stage::decoder_step_graph_execute)) {
        return false;
    }

    if (token_out) profile_tensor_get(ctx, ctx->decoder_argmax, token_out, 0, sizeof(int32_t));
    decoder_kv_commit_plan(ctx, kv_plan);
    return true;
}

// ============================================================================
// High-level: Transcribe (offline Voxtral-Mini-3B-2507)
// ============================================================================

static constexpr int32_t VOXTRAL_OFFLINE_WINDOW_SEC = 30;
static constexpr int32_t VOXTRAL_OFFLINE_MAX_DECODE = 448;

static void compute_mel_even(voxtral_context & ctx, const float * samples, int32_t n_samples,
                             std::vector<float> & mel_data, int32_t & n_frames) {
    const int32_t max_frames = n_samples / VOXTRAL_HOP_LENGTH + 1;
    mel_data.assign((size_t) VOXTRAL_NUM_MEL_BINS * max_frames, 0.0f);
    n_frames = 0;
    voxtral_mel_compute_batch(samples, n_samples, ctx.mel_filters_cpu.data(),
        ctx.hann_window.data(), mel_data.data(), &n_frames);
    if (n_frames % 2 != 0) {
        for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; m++)
            memmove(mel_data.data() + (size_t) m * (n_frames - 1),
                    mel_data.data() + (size_t) m * n_frames + 1,
                    (size_t) (n_frames - 1) * sizeof(float));
        n_frames -= 1;
    }
}

static bool transcribe_offline_window(
    voxtral_context & ctx, const float * wav, int32_t n_wav,
    int32_t max_tokens, float * first_logits_or_null,
    std::vector<int32_t> & out_tokens)
{
    const int32_t win = VOXTRAL_OFFLINE_WINDOW_SEC * VOXTRAL_SAMPLE_RATE;
    std::vector<float> padded((size_t) win, 0.0f);
    const int32_t ncopy = std::min(n_wav, win);
    memcpy(padded.data(), wav, (size_t) ncopy * sizeof(float));

    int32_t n_frames = 0;
    std::vector<float> mel_data;
    compute_mel_even(ctx, padded.data(), win, mel_data, n_frames);

    if (!run_encoder_offline(&ctx, mel_data.data(), n_frames)) return false;
    if (!run_adapter(&ctx)) return false;

    const auto & hp = ctx.model->hp;
    const int32_t n_audio = ctx.dec_seq_len;
    // Prompt: <s>[INST][BEGIN_AUDIO] [AUDIO]xN [/INST] lang:en [TRANSCRIBE]
    // The middle three suffix ids are the tekken ranks for the text "lang:en"
    // (language is currently fixed to English).
    static const int32_t kLangEn[3] = { 9909, 1058, 1262 };
    const int32_t prefix[3] = { hp.tok_bos, hp.tok_inst, hp.tok_begin_audio };
    const int32_t suffix[5] = { hp.tok_inst_end, kLangEn[0], kLangEn[1], kLangEn[2], hp.tok_transcribe };
    const int32_t L = 3 + n_audio + 5;

    clear_kv_cache(&ctx);
    int32_t token = 0;
    if (!run_offline_prefill(&ctx, prefix, 3, n_audio, suffix, 5, &token, first_logits_or_null)) return false;
    if (token != VOXTRAL_TOKEN_EOS) out_tokens.push_back(token);

    const int32_t cap = (max_tokens > 0) ? max_tokens : VOXTRAL_OFFLINE_MAX_DECODE;
    for (int32_t i = 0; i < cap && token != VOXTRAL_TOKEN_EOS; ++i) {
        if (!run_offline_step(&ctx, token, L + i, &token)) return false;
        if (token == VOXTRAL_TOKEN_EOS) break;
        out_tokens.push_back(token);
    }
    return true;
}

static std::string trim_copy(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static bool voxtral_transcribe_offline(
    voxtral_context & ctx,
    const float     * audio,
    int32_t           n_samples,
    int32_t           max_tokens,
    voxtral_result  & result)
{
    const int32_t win = VOXTRAL_OFFLINE_WINDOW_SEC * VOXTRAL_SAMPLE_RATE;
    int32_t n_windows = (n_samples + win - 1) / win;
    if (n_windows < 1) n_windows = 1;
    LOG_INFO(&ctx, "offline: %d samples (%.1fs) -> %d window(s) of %ds",
        n_samples, (float) n_samples / VOXTRAL_SAMPLE_RATE, n_windows, VOXTRAL_OFFLINE_WINDOW_SEC);

    auto t_all = std::chrono::steady_clock::now();
    std::string full;
    int32_t total_steps = 0;
    for (int32_t w = 0; w < n_windows; ++w) {
        const int32_t start = w * win;
        const int32_t len = std::min(win, n_samples - start);
        if (len <= 0) break;

        std::vector<int32_t> wtoks;
        // Capture the first window's prefill logits for diagnostics (--dump-logits).
        float * flog = nullptr;
        if (w == 0) {
            result.first_step_logits.resize(VOXTRAL_VOCAB_SIZE);
            flog = result.first_step_logits.data();
        }
        auto tw = std::chrono::steady_clock::now();
        if (!transcribe_offline_window(ctx, audio + start, len, max_tokens, flog, wtoks)) return false;
        total_steps += (int32_t) wtoks.size();

        std::string wtext = trim_copy(decode_tokens(*ctx.model, wtoks));
        LOG_INFO(&ctx, "window %d/%d: %.1f-%.1fs -> %d tokens, %.0f ms",
            w + 1, n_windows, (float) start / VOXTRAL_SAMPLE_RATE,
            (float) (start + len) / VOXTRAL_SAMPLE_RATE, (int) wtoks.size(), elapsed_ms(tw));

        result.tokens.insert(result.tokens.end(), wtoks.begin(), wtoks.end());
        if (!wtext.empty()) {
            if (!full.empty()) full += " ";
            full += wtext;
        }
    }
    result.text = full;
    const double ms = elapsed_ms(t_all);
    LOG_INFO(&ctx, "offline total: %d windows, %d tokens, %.0f ms (RTF %.3f)",
        n_windows, total_steps, ms, ms / 1000.0 / ((double) n_samples / VOXTRAL_SAMPLE_RATE));
    return true;
}

// ============================================================================
// High-level: Transcribe
// ============================================================================

// Adapter + decoder + detokenize, over an encoder output already resident in
// ctx.encoder_output with ctx.enc_seq_used set (a multiple of downsample_factor).
// This is the second half of the realtime Mel->text path, factored out so both
// the batch encoder (run_encoder_chunked) and the streaming incremental encoder
// (which accumulates the identical encoder output during feed) share exactly one
// adapter/decoder implementation. See voxtral_transcribe_encoder_output_internal.
static bool run_adapter_and_decode_realtime(
    voxtral_context & ctx,
    int32_t           max_tokens,
    voxtral_result  & result)
{
    // Run adapter
    auto t_adapter = std::chrono::steady_clock::now();
    if (!run_adapter(&ctx)) {
        return false;
    }
    LOG_INFO(&ctx, "adapter time: %.1f ms", elapsed_ms(t_adapter));

    const int32_t n_audio = ctx.dec_seq_len;

    // Build prompt tokens: [BOS] + [STREAMING_PAD] * (N_LEFT_PAD_TOKENS + N_DELAY_TOKENS)
    std::vector<int32_t> prompt_ids;
    prompt_ids.push_back(VOXTRAL_TOKEN_BOS);
    for (int32_t i = 0; i < VOXTRAL_N_LEFT_PAD_TOKENS + VOXTRAL_N_DELAY_TOKENS; i++) {
        prompt_ids.push_back(VOXTRAL_TOKEN_STREAMING_PAD);
    }
    const int32_t L = (int32_t)prompt_ids.size();  // 39

    LOG_INFO(&ctx, "prompt: %d tokens, audio_tokens: %d", L, n_audio);

    if (L > n_audio) {
        LOG_ERR(&ctx, "prompt length %d exceeds audio tokens %d", L, n_audio);
        return false;
    }

    // Reset KV cache
    clear_kv_cache(&ctx);

    // Decoder prefill
    auto t_prefill = std::chrono::steady_clock::now();
    std::vector<float> logits(VOXTRAL_VOCAB_SIZE);
    if (L > 1) {
        if (!run_decoder_prefill(&ctx, prompt_ids.data(), L - 1, logits.data())) {
            return false;
        }
    }

    // One step with last prefix token
    // Request full logits here so we can store first_step_logits for diagnostics.
    int32_t token = 0;
    if (!run_decoder_step(&ctx, prompt_ids[L - 1], L - 1, L - 1, logits.data(), &token)) {
        return false;
    }
    LOG_INFO(&ctx, "prefill time: %.1f ms", elapsed_ms(t_prefill));

    // Store first step logits
    result.first_step_logits = logits;
    result.tokens.push_back(token);

    LOG_INFO(&ctx, "first token: %d", token);

    // Autoregressive decoding
    //
    // The realtime model emits one token per audio frame. To transcribe the
    // whole file we must decode every audio position and only stop on EOS (or
    // the optional max_tokens cap, where <= 0 means until end of audio).
    //
    // we deliberately do NOT early-stop on a run of consecutive pad
    // tokens. A pad run only means a pause in speech, which is indistinguishable
    // from end-of-audio without lookahead, so a pad-based early stop truncates
    // longform transcripts at the first pause. Trailing pads are filtered out
    // during detokenization, so decoding to the end costs only the (bounded)
    // trailing-silence frames.
    const bool unlimited = (max_tokens <= 0);
    auto t_decode = std::chrono::steady_clock::now();
    for (int32_t pos = L; pos < n_audio && (unlimited || (int32_t)result.tokens.size() < max_tokens); pos++) {
        if (token == VOXTRAL_TOKEN_EOS) break;

        // Hot path: read back only the on-device greedy argmax token (4 bytes),
        // not the full 131072-float logits vector.
        if (!run_decoder_step(&ctx, token, pos, pos, /*logits_out=*/nullptr, &token)) {
            return false;
        }

        result.tokens.push_back(token);
    }
    LOG_INFO(&ctx, "decode time: %.1f ms (%d steps, %.1f ms/step)",
        elapsed_ms(t_decode), (int)result.tokens.size() - 1,
        result.tokens.size() > 1 ? elapsed_ms(t_decode) / (result.tokens.size() - 1) : 0.0);

    // Remove trailing EOS
    if (!result.tokens.empty() && result.tokens.back() == VOXTRAL_TOKEN_EOS) {
        result.tokens.pop_back();
    }

    LOG_INFO(&ctx, "generated %d tokens", (int)result.tokens.size());

    // Decode tokens to text (Tekken vocab from GGUF metadata)
    result.text = decode_tokens(*ctx.model, result.tokens);

    return true;
}

// Realtime Mel -> text. Shared by the batch path and the streaming finish():
// runs encoder / adapter / decoder over a precomputed even-trimmed, channel-major
// [n_mel, n_frames] Mel matrix. Encoder / adapter / decoder are defined exactly
// once and never duplicated across the batch and streaming entry points.
bool voxtral_transcribe_mel_internal(
    voxtral_context & ctx,
    const float     * mel,
    int32_t           n_frames,
    int32_t           max_tokens,
    voxtral_result  & result)
{
    result.text.clear();
    result.tokens.clear();
    result.first_step_logits.clear();

    if (ctx.model->hp.is_offline) {
        LOG_ERR(&ctx, "voxtral_transcribe_mel_internal: realtime-only path");
        return false;
    }
    if (mel == nullptr || n_frames <= 0) {
        LOG_ERR(&ctx, "voxtral_transcribe_mel_internal: empty mel");
        return false;
    }

    // Run encoder (global sliding-window by default; legacy chunked via strategy env)
    auto t_encoder = std::chrono::steady_clock::now();
    if (!run_encoder_batch(&ctx, mel, n_frames)) {
        return false;
    }
    LOG_INFO(&ctx, "encoder time: %.1f ms", elapsed_ms(t_encoder));

    return run_adapter_and_decode_realtime(ctx, max_tokens, result);
}

// Realtime encoder-output -> text. Uploads a host-side, channel-major
// [enc_dim, enc_frames] encoder output (exactly the tensor run_encoder_chunked
// leaves in ctx.encoder_output) into the device buffer, trims to a multiple of
// the downsample factor, and runs the shared adapter/decoder tail. The streaming
// finish() uses this with the encoder output it accumulated incrementally during
// feed, so it never re-runs the encoder over the full Mel.
bool voxtral_transcribe_encoder_output_internal(
    voxtral_context & ctx,
    const float     * enc_output,
    int32_t           enc_frames,
    int32_t           max_tokens,
    voxtral_result  & result)
{
    result.text.clear();
    result.tokens.clear();
    result.first_step_logits.clear();

    if (ctx.model->hp.is_offline) {
        LOG_ERR(&ctx, "voxtral_transcribe_encoder_output_internal: realtime-only path");
        return false;
    }
    if (enc_output == nullptr || enc_frames <= 0) {
        LOG_ERR(&ctx, "voxtral_transcribe_encoder_output_internal: empty encoder output");
        return false;
    }

    // Trim to a multiple of the downsample factor (adapter groups of 4), matching
    // run_encoder_chunked's final ctx.enc_seq_used = (n/4)*4.
    const int32_t used = (enc_frames / ctx.model->hp.downsample_factor) * ctx.model->hp.downsample_factor;
    if (used <= 0) {
        LOG_ERR(&ctx, "voxtral_transcribe_encoder_output_internal: fewer than %d encoder frames (%d)",
                ctx.model->hp.downsample_factor, enc_frames);
        return false;
    }

    if (!alloc_encoder_output(&ctx, used)) {
        LOG_ERR(&ctx, "voxtral_transcribe_encoder_output_internal: failed to allocate encoder output (%d)", used);
        return false;
    }
    ggml_backend_tensor_set(ctx.encoder_output, enc_output, 0,
                            (size_t) used * VOXTRAL_ENC_DIM * sizeof(float));
    ctx.enc_seq_used     = used;
    ctx.total_enc_tokens = used;

    return run_adapter_and_decode_realtime(ctx, max_tokens, result);
}

// Encoder-only batch pass: runs run_encoder_chunked over an even-trimmed Mel and
// copies the resulting ctx.encoder_output into a host, channel-major
// [enc_dim, enc_seq_used] buffer. This is the canonical reference the streaming
// incremental encoder is compared against (batch-vs-incremental tensor parity).
bool voxtral_encode_mel_batch_internal(
    voxtral_context & ctx,
    const float     * mel,
    int32_t           n_frames,
    std::vector<float> & out_enc,
    int32_t         & out_frames)
{
    out_enc.clear();
    out_frames = 0;
    if (ctx.model->hp.is_offline) {
        LOG_ERR(&ctx, "voxtral_encode_mel_batch_internal: realtime-only path");
        return false;
    }
    if (mel == nullptr || n_frames <= 0) {
        return false;
    }
    if (!run_encoder_batch(&ctx, mel, n_frames)) {
        return false;
    }
    const int32_t frames = ctx.enc_seq_used;
    if (frames <= 0) {
        return false;
    }
    out_enc.resize((size_t) frames * VOXTRAL_ENC_DIM);
    ggml_backend_tensor_get(ctx.encoder_output, out_enc.data(), 0,
                            (size_t) frames * VOXTRAL_ENC_DIM * sizeof(float));
    out_frames = frames;
    return true;
}

static bool voxtral_transcribe_from_audio(
    voxtral_context & ctx,
    const float     * audio,
    int32_t           n_samples,
    int32_t           max_tokens,
    voxtral_result  & result,
    bool              log_audio)
{
    result.text.clear();
    result.tokens.clear();
    result.first_step_logits.clear();

    if (audio == nullptr || n_samples <= 0) {
        LOG_ERR(&ctx, "audio input is empty");
        return false;
    }

    if (log_audio) {
        LOG_INFO(&ctx, "audio input: %d samples (%.1f s)", n_samples,
            (float)n_samples / VOXTRAL_SAMPLE_RATE);
    }

    // Offline: different preprocessing, prompt and decode.
    if (ctx.model->hp.is_offline) {
        return voxtral_transcribe_offline(ctx, audio, n_samples, max_tokens, result);
    }

    // Streaming padding (matching Python pad_audio_streaming)
    constexpr int32_t mult_of   = VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK;   // 1280
    const int32_t n_raw     = n_samples;
    const int32_t align_pad = (mult_of - (n_raw % mult_of)) % mult_of;
    const int32_t right_pad = align_pad + VOXTRAL_N_RIGHT_PAD_TOKENS * mult_of;
    constexpr int32_t left_pad  = VOXTRAL_N_LEFT_PAD_TOKENS * mult_of;

    std::vector<float> padded(left_pad + n_raw + right_pad, 0.0f);
    memcpy(padded.data() + left_pad, audio, n_raw * sizeof(float));

    LOG_INFO(&ctx, "padded audio: %d samples (left=%d, right=%d)", (int)padded.size(), left_pad, right_pad);

    // Compute mel spectrogram (truncated to an even frame count for conv stride 2)
    int32_t n_frames = 0;
    std::vector<float> mel_data;
    compute_mel_even(ctx, padded.data(), (int32_t) padded.size(), mel_data, n_frames);
    LOG_INFO(&ctx, "mel spectrogram: %d frames", n_frames);

    // Shared Mel -> text path (encoder / adapter / decoder / detokenize).
    return voxtral_transcribe_mel_internal(ctx, mel_data.data(), n_frames, max_tokens, result);
}

bool voxtral_transcribe_audio(
    voxtral_context   & ctx,
    const std::vector<float> & audio,
    int32_t             max_tokens,
    voxtral_result    & result)
{
    return voxtral_transcribe_from_audio(
        ctx, audio.data(), (int32_t) audio.size(), max_tokens, result, true);
}

bool voxtral_transcribe_file(
    voxtral_context   & ctx,
    const std::string & audio_path,
    int32_t             max_tokens,
    voxtral_result    & result)
{
    std::vector<float> audio;
    if (!load_wav_file(audio_path, audio)) {
        LOG_ERR(&ctx, "failed to load WAV: %s", audio_path.c_str());
        return false;
    }
    LOG_INFO(&ctx, "audio loaded: %d samples (%.1f s)", (int)audio.size(),
        (float)audio.size() / VOXTRAL_SAMPLE_RATE);

    return voxtral_transcribe_from_audio(
        ctx, audio.data(), (int32_t) audio.size(), max_tokens, result, false);
}
