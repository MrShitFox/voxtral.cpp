#ifndef VOXTRAL_H
#define VOXTRAL_H

/**
 * @file voxtral.h
 * Stable, allocation-safe C ABI for Voxtral model and inference contexts.
 *
 * The declarations in this file are valid in C11, C17, C++17 and C++20.
 * Handles are opaque and must only be created and destroyed through this API.
 */

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(VOXTRAL_BUILD_SHARED)
#    define VOXTRAL_API __declspec(dllexport)
#  elif defined(VOXTRAL_USE_SHARED)
#    define VOXTRAL_API __declspec(dllimport)
#  else
#    define VOXTRAL_API
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define VOXTRAL_API __attribute__((visibility("default")))
#else
#  define VOXTRAL_API
#endif

#define VOXTRAL_API_VERSION_MAJOR 1u
#define VOXTRAL_API_VERSION_MINOR 0u
#define VOXTRAL_API_VERSION_PATCH 0u
#define VOXTRAL_API_VERSION                                                   \
    ((VOXTRAL_API_VERSION_MAJOR << 24u) |                                     \
     (VOXTRAL_API_VERSION_MINOR << 16u) |                                     \
      VOXTRAL_API_VERSION_PATCH)

/* Model constants retained for source compatibility and public capability use. */
#define VOXTRAL_ENC_DIM         1280
#define VOXTRAL_ENC_LAYERS      32
#define VOXTRAL_ENC_HEADS       32
#define VOXTRAL_ENC_HEAD_DIM    64
#define VOXTRAL_ENC_HIDDEN      5120
#define VOXTRAL_ENC_KV_HEADS    32
#define VOXTRAL_ENC_WINDOW      750
#define VOXTRAL_ENC_NORM_EPS    1e-5f
#define VOXTRAL_ENC_ROPE_THETA  1000000.0f

#define VOXTRAL_DEC_DIM         3072
#define VOXTRAL_DEC_LAYERS      26
#define VOXTRAL_DEC_HEADS       32
#define VOXTRAL_DEC_HEAD_DIM    128
#define VOXTRAL_DEC_HIDDEN      9216
#define VOXTRAL_DEC_KV_HEADS    8
#define VOXTRAL_DEC_WINDOW      8192
#define VOXTRAL_DEC_NORM_EPS    1e-5f
#define VOXTRAL_DEC_ROPE_THETA  1000000.0f
#define VOXTRAL_VOCAB_SIZE      131072

#define VOXTRAL_SAMPLE_RATE         16000
#define VOXTRAL_FRAME_RATE          12.5f
#define VOXTRAL_NUM_MEL_BINS        128
#define VOXTRAL_HOP_LENGTH          160
#define VOXTRAL_WINDOW_SIZE         400
#define VOXTRAL_GLOBAL_LOG_MEL_MAX  1.5f
#define VOXTRAL_DOWNSAMPLE_FACTOR   4
#define VOXTRAL_ADA_NORM_DIM        32

#define VOXTRAL_N_LEFT_PAD_TOKENS       32
#define VOXTRAL_TRANSCRIPTION_DELAY_MS  480
#define VOXTRAL_N_DELAY_TOKENS          6
#define VOXTRAL_N_RIGHT_PAD_TOKENS      17
#define VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK 1280

#define VOXTRAL_TOKEN_BOS           1
#define VOXTRAL_TOKEN_EOS           2
#define VOXTRAL_TOKEN_STREAMING_PAD 32
#define VOXTRAL_TOKEN_BEGIN_AUDIO   25
#define VOXTRAL_TOKEN_AUDIO         24

#ifdef __cplusplus
extern "C" {
#endif

/** Immutable model weights and metadata. The layout is private. */
typedef struct voxtral_model voxtral_model;

/** Mutable backend allocations and inference state. The layout is private. */
typedef struct voxtral_context voxtral_context;

/** One transcription lifecycle. Defined by <voxtral-stream.h>. */
typedef struct voxtral_stream voxtral_stream;

/**
 * Stable status values. Zero is always success. New values are append-only
 * within API major version 1.
 */
typedef enum voxtral_status {
    VOXTRAL_STATUS_OK = 0,
    VOXTRAL_STATUS_INVALID_ARGUMENT = 1,
    VOXTRAL_STATUS_INVALID_STATE = 2,
    VOXTRAL_STATUS_OUT_OF_MEMORY = 3,
    VOXTRAL_STATUS_QUEUE_FULL = 4,
    VOXTRAL_STATUS_NOT_READY = 5,
    VOXTRAL_STATUS_CANCELLED = 6,
    VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT = 7,
    VOXTRAL_STATUS_MODEL_ERROR = 8,
    VOXTRAL_STATUS_BACKEND_ERROR = 9,
    VOXTRAL_STATUS_INTERNAL_ERROR = 10,
    VOXTRAL_STATUS_BUFFER_TOO_SMALL = 11
} voxtral_status;

/** Backend preference used while loading a model. */
typedef enum voxtral_backend {
    VOXTRAL_BACKEND_CPU = 0,
    VOXTRAL_BACKEND_AUTO = 1,
    VOXTRAL_BACKEND_CUDA = 2,
    VOXTRAL_BACKEND_METAL = 3,
    VOXTRAL_BACKEND_VULKAN = 4
} voxtral_backend;

/** Library log threshold. Session 11 does not expose a callback ABI. */
typedef enum voxtral_log_level_c {
    VOXTRAL_LOG_ERROR = 0,
    VOXTRAL_LOG_WARN = 1,
    VOXTRAL_LOG_INFO = 2,
    VOXTRAL_LOG_DEBUG = 3
} voxtral_log_level_c;

/**
 * Caller-owned structured error copy.
 *
 * Set struct_size and api_version before passing this structure to a query.
 * The returned message is UTF-8, null-terminated and never requires free().
 */
typedef struct voxtral_error_info {
    uint32_t struct_size;
    uint32_t api_version;
    voxtral_status status;
    int32_t backend_code;
    char message[256];
} voxtral_error_info;

/** Versioned model-loading parameters. Unknown trailing fields are ignored. */
typedef struct voxtral_model_params {
    uint32_t struct_size;
    uint32_t api_version;
    voxtral_backend backend;
    voxtral_log_level_c log_level;
    uint32_t flags;
    uint32_t reserved;
} voxtral_model_params;

/** Versioned context-creation parameters. Unknown trailing fields are ignored. */
typedef struct voxtral_context_config {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t n_threads;
    uint32_t flags;
} voxtral_context_config;

/**
 * Returns the runtime library version as immutable static UTF-8.
 *
 * Ownership: the caller must not modify or free the returned pointer.
 * Thread safety: safe to call concurrently.
 */
VOXTRAL_API const char * voxtral_version_string(void);

/**
 * Returns VOXTRAL_API_VERSION packed as major<<24 | minor<<16 | patch.
 *
 * Thread safety: safe to call concurrently.
 */
VOXTRAL_API uint32_t voxtral_api_version(void);

/**
 * Returns an immutable static diagnostic name for a status.
 *
 * The string is intended for logs and does not replace structured status
 * handling. Unknown numeric values return "unknown".
 * Thread safety: safe to call concurrently.
 */
VOXTRAL_API const char * voxtral_status_string(voxtral_status status);

/** Returns a fully initialized v1 model-parameter value. */
VOXTRAL_API voxtral_model_params voxtral_model_default_params(void);

/** Returns a fully initialized v1 context-configuration value. */
VOXTRAL_API voxtral_context_config voxtral_context_default_config(void);

/**
 * Loads immutable model weights from a GGUF file.
 *
 * @param path       Null-terminated UTF-8 filesystem path.
 * @param params     Versioned parameters, or NULL for defaults.
 * @param out_model  Receives a new handle on success; set to NULL on failure.
 *
 * Ownership: the caller owns the returned model and destroys it with
 * voxtral_model_destroy(). A model must outlive every context created from it.
 * Thread safety: serialize model creation/destruction with related factories.
 */
VOXTRAL_API voxtral_status voxtral_model_load(
    const char * path,
    const voxtral_model_params * params,
    voxtral_model ** out_model);

/**
 * Destroys a caller-owned model. Passing NULL is safe.
 *
 * Preconditions: no context created from the model may remain alive.
 * Thread safety: no concurrent use of the model is permitted during destroy.
 */
VOXTRAL_API void voxtral_model_destroy(voxtral_model * model);

/**
 * Creates a reusable mutable inference context from immutable model weights.
 *
 * @param model        Valid model that will outlive the context.
 * @param config       Versioned configuration, or NULL for defaults.
 * @param out_context  Receives a new handle on success; NULL on failure.
 *
 * Ownership: the caller owns the returned context. The context borrows model.
 * Thread safety: a context is not safe for concurrent inference.
 */
VOXTRAL_API voxtral_status voxtral_context_create(
    voxtral_model * model,
    const voxtral_context_config * config,
    voxtral_context ** out_context);

/**
 * Destroys a caller-owned context. Passing NULL is safe.
 *
 * Preconditions: every stream borrowing the context has been destroyed.
 * This function never destroys the borrowed model.
 * Thread safety: no concurrent use of the context is permitted during destroy.
 */
VOXTRAL_API void voxtral_context_destroy(voxtral_context * context);

/**
 * Copies the most recent public-API error associated with a context.
 *
 * @return VOXTRAL_STATUS_OK when the copy succeeds, including when the stored
 *         status itself is VOXTRAL_STATUS_OK.
 * @return VOXTRAL_STATUS_INVALID_ARGUMENT for invalid pointers or structure
 *         metadata.
 *
 * Ownership: out_error is caller-owned. No returned pointer can dangle.
 * Thread safety: externally serialize this query with context operations.
 */
VOXTRAL_API voxtral_status voxtral_context_get_last_error(
    const voxtral_context * context,
    voxtral_error_info * out_error);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VOXTRAL_H */
