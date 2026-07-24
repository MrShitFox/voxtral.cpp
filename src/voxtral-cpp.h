#ifndef VOXTRAL_CPP_INTERNAL_COMPAT_H
#define VOXTRAL_CPP_INTERNAL_COMPAT_H

/*
 * Legacy C++ surface used by the CLI and internal acceptance harness.
 *
 * This is deliberately private. Stable external consumers include
 * <voxtral.h> and <voxtral-stream.h>, which expose only the C ABI.
 */

#include "voxtral.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

enum class voxtral_log_level : int {
    error = 0,
    warn  = 1,
    info  = 2,
    debug = 3,
};

enum class voxtral_gpu_backend : int {
    none = 0,
    auto_detect,
    cuda,
    metal,
    vulkan,
};

using voxtral_log_callback =
    std::function<void(voxtral_log_level, const std::string &)>;

struct voxtral_context_params {
    int32_t              n_threads  = 0;
    voxtral_log_level    log_level  = voxtral_log_level::info;
    voxtral_log_callback logger     = nullptr;
    voxtral_gpu_backend  gpu        = voxtral_gpu_backend::none;
};

struct voxtral_result {
    std::string          text;
    std::vector<int32_t> tokens;
    std::vector<float>   first_step_logits;
};

voxtral_model * voxtral_model_load_from_file(
    const std::string    & path,
    voxtral_log_callback   logger = nullptr,
    voxtral_gpu_backend    gpu = voxtral_gpu_backend::none);

void voxtral_model_free(voxtral_model * model);

voxtral_context * voxtral_init_from_model(
    voxtral_model                * model,
    const voxtral_context_params & params);

void voxtral_free(voxtral_context * ctx);

bool voxtral_transcribe_file(
    voxtral_context   & ctx,
    const std::string & audio_path,
    int32_t             max_tokens,
    voxtral_result    & result);

bool voxtral_transcribe_audio(
    voxtral_context          & ctx,
    const std::vector<float> & audio,
    int32_t                    max_tokens,
    voxtral_result           & result);

#endif /* VOXTRAL_CPP_INTERNAL_COMPAT_H */
