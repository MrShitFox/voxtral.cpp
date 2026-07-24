#include <voxtral-stream.h>

#include "voxtral-cpp.h"
#include "voxtral-internal.h"
#include "voxtral-stream-internal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace {

constexpr uint32_t kVersionedHeaderSize = sizeof(uint32_t) * 2u;
constexpr uint32_t kMaxVersionedStructSize = 1024u * 1024u;
constexpr uint32_t kDefaultEventQueueCapacity =
    VOXTRAL_EVENT_QUEUE_CAPACITY_DEFAULT;
constexpr size_t kEventV1StructSize =
    offsetof(voxtral_event, text) + VOXTRAL_EVENT_TEXT_CAPACITY;
static_assert(kTerminalEventHeadroom == VOXTRAL_EVENT_TERMINAL_HEADROOM,
              "public and internal terminal event bounds must match");
static_assert(VOXTRAL_STREAM_MAX_AUDIO_SAMPLES == UINT64_MAX,
              "public sample ceiling must match its uint64_t counter");

struct versioned_header {
    uint32_t struct_size;
    uint32_t api_version;
};

bool api_version_compatible(uint32_t version) {
    return (version >> 24u) == VOXTRAL_API_VERSION_MAJOR;
}

bool read_header(const void * ptr, versioned_header & out) {
    if (!ptr) return false;
    std::memcpy(&out, ptr, sizeof(out));
    return out.struct_size >= kVersionedHeaderSize &&
           out.struct_size <= kMaxVersionedStructSize &&
           api_version_compatible(out.api_version);
}

bool field_available(uint32_t struct_size, size_t offset, size_t field_size) {
    return offset <= struct_size && field_size <= struct_size - offset;
}

template <typename T, typename Field>
void read_field_if_available(
    const void * source,
    uint32_t source_size,
    size_t offset,
    Field T::* member,
    T & destination)
{
    if (!field_available(source_size, offset, sizeof(Field))) return;
    Field value;
    std::memcpy(&value, static_cast<const unsigned char *>(source) + offset,
                sizeof(value));
    destination.*member = value;
}

template <typename T>
bool write_versioned_output(
    T * destination,
    const T & value,
    uint32_t minimum_size = kVersionedHeaderSize)
{
    versioned_header header{};
    if (!read_header(destination, header) || header.struct_size < minimum_size) {
        return false;
    }
    const size_t copy_size = std::min<size_t>(header.struct_size, sizeof(T));
    std::memcpy(destination, &value, copy_size);
    return true;
}

voxtral_model_params normalize_model_params(
    const voxtral_model_params * input,
    bool & ok)
{
    voxtral_model_params out = voxtral_model_default_params();
    ok = true;
    if (!input) return out;
    versioned_header header{};
    if (!read_header(input, header)) {
        ok = false;
        return out;
    }
    read_field_if_available(input, header.struct_size,
        offsetof(voxtral_model_params, backend),
        &voxtral_model_params::backend, out);
    read_field_if_available(input, header.struct_size,
        offsetof(voxtral_model_params, log_level),
        &voxtral_model_params::log_level, out);
    read_field_if_available(input, header.struct_size,
        offsetof(voxtral_model_params, flags),
        &voxtral_model_params::flags, out);
    read_field_if_available(input, header.struct_size,
        offsetof(voxtral_model_params, reserved),
        &voxtral_model_params::reserved, out);
    out.struct_size = sizeof(out);
    out.api_version = VOXTRAL_API_VERSION;
    return out;
}

voxtral_context_config normalize_context_config(
    const voxtral_context_config * input,
    bool & ok)
{
    voxtral_context_config out = voxtral_context_default_config();
    ok = true;
    if (!input) return out;
    versioned_header header{};
    if (!read_header(input, header)) {
        ok = false;
        return out;
    }
    read_field_if_available(input, header.struct_size,
        offsetof(voxtral_context_config, n_threads),
        &voxtral_context_config::n_threads, out);
    read_field_if_available(input, header.struct_size,
        offsetof(voxtral_context_config, flags),
        &voxtral_context_config::flags, out);
    out.struct_size = sizeof(out);
    out.api_version = VOXTRAL_API_VERSION;
    return out;
}

voxtral_audio_config normalize_audio_config(
    const voxtral_audio_config * input,
    bool & ok)
{
    voxtral_audio_config out = voxtral_audio_default_config();
    versioned_header header{};
    if (!read_header(input, header)) {
        ok = false;
        return out;
    }
    read_field_if_available(input, header.struct_size,
        offsetof(voxtral_audio_config, sample_rate),
        &voxtral_audio_config::sample_rate, out);
    read_field_if_available(input, header.struct_size,
        offsetof(voxtral_audio_config, channels),
        &voxtral_audio_config::channels, out);
    read_field_if_available(input, header.struct_size,
        offsetof(voxtral_audio_config, format),
        &voxtral_audio_config::format, out);
    read_field_if_available(input, header.struct_size,
        offsetof(voxtral_audio_config, reserved),
        &voxtral_audio_config::reserved, out);
    out.struct_size = sizeof(out);
    out.api_version = VOXTRAL_API_VERSION;
    return out;
}

voxtral_stream_params normalize_stream_params(
    const voxtral_stream_params * input,
    bool & ok)
{
    voxtral_stream_params out = voxtral_stream_default_params();
    ok = true;
    versioned_header header{};
    if (!read_header(input, header)) {
        ok = false;
        return out;
    }
    if (field_available(header.struct_size,
                        offsetof(voxtral_stream_params, audio),
                        sizeof(voxtral_audio_config))) {
        bool audio_ok = true;
        voxtral_audio_config raw_audio{};
        std::memcpy(&raw_audio,
                    reinterpret_cast<const unsigned char *>(input) +
                        offsetof(voxtral_stream_params, audio),
                    sizeof(raw_audio));
        out.audio = normalize_audio_config(&raw_audio, audio_ok);
        if (!audio_ok) ok = false;
    }
    read_field_if_available(input, header.struct_size,
        offsetof(voxtral_stream_params, event_queue_capacity),
        &voxtral_stream_params::event_queue_capacity, out);
    read_field_if_available(input, header.struct_size,
        offsetof(voxtral_stream_params, flags),
        &voxtral_stream_params::flags, out);
    read_field_if_available(input, header.struct_size,
        offsetof(voxtral_stream_params, reserved),
        &voxtral_stream_params::reserved, out);
    out.struct_size = sizeof(out);
    out.api_version = VOXTRAL_API_VERSION;
    return out;
}

voxtral_gpu_backend map_backend(voxtral_backend backend, bool & ok) {
    ok = true;
    switch (backend) {
        case VOXTRAL_BACKEND_CPU:    return voxtral_gpu_backend::none;
        case VOXTRAL_BACKEND_AUTO:   return voxtral_gpu_backend::auto_detect;
        case VOXTRAL_BACKEND_CUDA:   return voxtral_gpu_backend::cuda;
        case VOXTRAL_BACKEND_METAL:  return voxtral_gpu_backend::metal;
        case VOXTRAL_BACKEND_VULKAN: return voxtral_gpu_backend::vulkan;
    }
    ok = false;
    return voxtral_gpu_backend::none;
}

voxtral_log_level map_log_level(voxtral_log_level_c level, bool & ok) {
    ok = true;
    switch (level) {
        case VOXTRAL_LOG_ERROR: return voxtral_log_level::error;
        case VOXTRAL_LOG_WARN:  return voxtral_log_level::warn;
        case VOXTRAL_LOG_INFO:  return voxtral_log_level::info;
        case VOXTRAL_LOG_DEBUG: return voxtral_log_level::debug;
    }
    ok = false;
    return voxtral_log_level::error;
}

voxtral_status map_status(voxtral_status_internal status) {
    switch (status) {
        case voxtral_status_internal::ok:
            return VOXTRAL_STATUS_OK;
        case voxtral_status_internal::invalid_argument:
            return VOXTRAL_STATUS_INVALID_ARGUMENT;
        case voxtral_status_internal::invalid_state:
        case voxtral_status_internal::busy:
            return VOXTRAL_STATUS_INVALID_STATE;
        case voxtral_status_internal::unsupported_audio_format:
            return VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT;
        case voxtral_status_internal::cancelled:
            return VOXTRAL_STATUS_CANCELLED;
        case voxtral_status_internal::backend_error:
            return VOXTRAL_STATUS_BACKEND_ERROR;
        case voxtral_status_internal::out_of_memory:
            return VOXTRAL_STATUS_OUT_OF_MEMORY;
        case voxtral_status_internal::limit_exceeded:
            return VOXTRAL_STATUS_QUEUE_FULL;
        case voxtral_status_internal::internal_error:
            return VOXTRAL_STATUS_INTERNAL_ERROR;
    }
    return VOXTRAL_STATUS_INTERNAL_ERROR;
}

voxtral_status map_feed_status(
    const voxtral_stream * stream,
    voxtral_status_internal status)
{
    if (status != voxtral_status_internal::limit_exceeded) {
        return map_status(status);
    }
    return voxtral_stream_last_feed_status(stream) ==
                   voxtral_stream_feed_status::queue_full
        ? VOXTRAL_STATUS_QUEUE_FULL
        : VOXTRAL_STATUS_INVALID_ARGUMENT;
}

void set_context_error(
    voxtral_context * context,
    voxtral_status status,
    const char * message,
    int32_t backend_code = 0) noexcept
{
    voxtral_context_set_public_error_internal(
        context, status, backend_code, message ? message : "");
}

void set_stream_error_noexcept(
    voxtral_stream * stream,
    voxtral_status_internal status,
    const char * message) noexcept
{
    if (!stream) return;
    try {
        set_error(stream, status, message ? std::string(message) : std::string());
    } catch (...) {
        // The return status still communicates the failure if formatting the
        // optional diagnostic message itself runs out of memory.
    }
}

void fail_stream_noexcept(
    voxtral_stream * stream,
    voxtral_status_internal status,
    const char * message) noexcept
{
    if (!stream) return;
    try {
        stream->events.finalizing_flush = true;
        set_error(stream, status, message ? std::string(message) : std::string());
        emit_error(stream, status, stream->lifecycle.last_error);
    } catch (...) {
        // A status and terminal state remain observable even if constructing
        // the optional ERROR payload itself fails under memory pressure.
        set_stream_error_noexcept(stream, status, message);
    }
    stream->lifecycle.state = voxtral_stream_state_internal::failed;
}

size_t utf8_prefix_at_most(const std::string & text, size_t limit) {
    size_t i = 0;
    size_t stable = 0;
    const size_t end = std::min(text.size(), limit);
    while (i < end) {
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        size_t width = 0;
        if (lead <= 0x7f) width = 1;
        else if (lead >= 0xc2 && lead <= 0xdf) width = 2;
        else if (lead >= 0xe0 && lead <= 0xef) width = 3;
        else if (lead >= 0xf0 && lead <= 0xf4) width = 4;
        else break;
        if (i + width > end || i + width > text.size()) break;
        bool valid = true;
        for (size_t j = 1; j < width; ++j) {
            const unsigned char continuation =
                static_cast<unsigned char>(text[i + j]);
            if ((continuation & 0xc0u) != 0x80u) {
                valid = false;
                break;
            }
        }
        if (!valid) break;
        if (width == 3) {
            const unsigned char second =
                static_cast<unsigned char>(text[i + 1]);
            if ((lead == 0xe0 && second < 0xa0) ||
                (lead == 0xed && second >= 0xa0)) break;
        }
        if (width == 4) {
            const unsigned char second =
                static_cast<unsigned char>(text[i + 1]);
            if ((lead == 0xf0 && second < 0x90) ||
                (lead == 0xf4 && second >= 0x90)) break;
        }
        i += width;
        stable = i;
    }
    return stable;
}

void copy_event_text(
    voxtral_event & destination,
    const std::string & source,
    size_t stable_bytes)
{
    const size_t available = std::min(stable_bytes, source.size());
    const size_t limit = std::min<size_t>(
        available, VOXTRAL_EVENT_TEXT_CAPACITY - 1u);
    const size_t copy_bytes = utf8_prefix_at_most(source, limit);
    if (copy_bytes > 0) {
        std::memcpy(destination.text, source.data(), copy_bytes);
    }
    destination.text[copy_bytes] = '\0';
    destination.text_length = copy_bytes;
    if (copy_bytes < available) {
        destination.flags |= VOXTRAL_EVENT_FLAG_TEXT_TRUNCATED;
    }
}

voxtral_status event_status(const voxtral_stream_event_internal & event) {
    if (event.type == voxtral_stream_event_type_internal::cancelled) {
        return VOXTRAL_STATUS_CANCELLED;
    }
    if (event.type != voxtral_stream_event_type_internal::error) {
        return VOXTRAL_STATUS_OK;
    }
    if (event.error_code < static_cast<int32_t>(voxtral_status_internal::ok) ||
        event.error_code > static_cast<int32_t>(
            voxtral_status_internal::internal_error)) {
        return VOXTRAL_STATUS_INTERNAL_ERROR;
    }
    return map_status(static_cast<voxtral_status_internal>(event.error_code));
}

voxtral_event_type event_type(
    voxtral_stream_event_type_internal type)
{
    switch (type) {
        case voxtral_stream_event_type_internal::token:
            return VOXTRAL_EVENT_TOKEN;
        case voxtral_stream_event_type_internal::partial_text:
            return VOXTRAL_EVENT_PARTIAL_TEXT;
        case voxtral_stream_event_type_internal::final_text:
            return VOXTRAL_EVENT_FINAL_TEXT;
        case voxtral_stream_event_type_internal::error:
        case voxtral_stream_event_type_internal::cancelled:
            return VOXTRAL_EVENT_ERROR;
        case voxtral_stream_event_type_internal::completed:
            return VOXTRAL_EVENT_COMPLETED;
    }
    return VOXTRAL_EVENT_NONE;
}

voxtral_stream_state map_state(voxtral_stream_state_internal state) {
    switch (state) {
        case voxtral_stream_state_internal::created:
            return VOXTRAL_STREAM_CREATED;
        case voxtral_stream_state_internal::running:
            return VOXTRAL_STREAM_ACTIVE;
        case voxtral_stream_state_internal::finishing:
            return VOXTRAL_STREAM_FINISHING;
        case voxtral_stream_state_internal::completed:
            return VOXTRAL_STREAM_COMPLETED;
        case voxtral_stream_state_internal::cancelled:
            return VOXTRAL_STREAM_CANCELLED;
        case voxtral_stream_state_internal::failed:
            return VOXTRAL_STREAM_ERROR;
    }
    return VOXTRAL_STREAM_ERROR;
}

void copy_error_message(char (&destination)[256], const std::string & source) {
    const size_t limit = sizeof(destination) - 1u;
    const size_t copy_bytes = utf8_prefix_at_most(source, limit);
    if (copy_bytes > 0) {
        std::memcpy(destination, source.data(), copy_bytes);
    }
    destination[copy_bytes] = '\0';
}

voxtral_error_info make_error(
    voxtral_status status,
    int32_t backend_code,
    const std::string & message)
{
    voxtral_error_info out{};
    out.struct_size = sizeof(out);
    out.api_version = VOXTRAL_API_VERSION;
    out.status = status;
    out.backend_code = backend_code;
    copy_error_message(out.message, message);
    return out;
}

} // namespace

extern "C" {

const char * voxtral_version_string(void) {
    return "1.0.0";
}

uint32_t voxtral_api_version(void) {
    return VOXTRAL_API_VERSION;
}

const char * voxtral_status_string(voxtral_status status) {
    switch (status) {
        case VOXTRAL_STATUS_OK:                       return "ok";
        case VOXTRAL_STATUS_INVALID_ARGUMENT:         return "invalid_argument";
        case VOXTRAL_STATUS_INVALID_STATE:            return "invalid_state";
        case VOXTRAL_STATUS_OUT_OF_MEMORY:             return "out_of_memory";
        case VOXTRAL_STATUS_QUEUE_FULL:                return "queue_full";
        case VOXTRAL_STATUS_NOT_READY:                 return "not_ready";
        case VOXTRAL_STATUS_CANCELLED:                 return "cancelled";
        case VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT:  return "unsupported_audio_format";
        case VOXTRAL_STATUS_MODEL_ERROR:               return "model_error";
        case VOXTRAL_STATUS_BACKEND_ERROR:             return "backend_error";
        case VOXTRAL_STATUS_INTERNAL_ERROR:            return "internal_error";
        case VOXTRAL_STATUS_BUFFER_TOO_SMALL:          return "buffer_too_small";
    }
    return "unknown";
}

voxtral_model_params voxtral_model_default_params(void) {
    voxtral_model_params out{};
    out.struct_size = sizeof(out);
    out.api_version = VOXTRAL_API_VERSION;
    out.backend = VOXTRAL_BACKEND_AUTO;
    out.log_level = VOXTRAL_LOG_ERROR;
    return out;
}

voxtral_context_config voxtral_context_default_config(void) {
    voxtral_context_config out{};
    out.struct_size = sizeof(out);
    out.api_version = VOXTRAL_API_VERSION;
    out.n_threads = 0;
    return out;
}

voxtral_audio_config voxtral_audio_default_config(void) {
    voxtral_audio_config out{};
    out.struct_size = sizeof(out);
    out.api_version = VOXTRAL_API_VERSION;
    out.sample_rate = VOXTRAL_SAMPLE_RATE;
    out.channels = 1;
    out.format = VOXTRAL_AUDIO_PCM_S16LE;
    return out;
}

voxtral_stream_params voxtral_stream_default_params(void) {
    voxtral_stream_params out{};
    out.struct_size = sizeof(out);
    out.api_version = VOXTRAL_API_VERSION;
    out.audio = voxtral_audio_default_config();
    out.event_queue_capacity = kDefaultEventQueueCapacity;
    return out;
}

voxtral_status voxtral_model_load(
    const char * path,
    const voxtral_model_params * params,
    voxtral_model ** out_model)
{
    if (out_model) *out_model = nullptr;
    if (!path || path[0] == '\0' || !out_model) {
        return VOXTRAL_STATUS_INVALID_ARGUMENT;
    }
    bool params_ok = true;
    const voxtral_model_params normalized =
        normalize_model_params(params, params_ok);
    if (!params_ok || normalized.flags != 0 || normalized.reserved != 0) {
        return VOXTRAL_STATUS_INVALID_ARGUMENT;
    }
    bool backend_ok = true;
    bool log_ok = true;
    const voxtral_gpu_backend backend =
        map_backend(normalized.backend, backend_ok);
    const voxtral_log_level threshold =
        map_log_level(normalized.log_level, log_ok);
    if (!backend_ok || !log_ok) return VOXTRAL_STATUS_INVALID_ARGUMENT;

    try {
        voxtral_log_callback logger =
            [threshold](voxtral_log_level level, const std::string & message) {
                if (static_cast<int>(level) <= static_cast<int>(threshold)) {
                    std::fprintf(stderr, "voxtral: %s\n", message.c_str());
                }
            };
        voxtral_model * model =
            voxtral_model_load_from_file(path, std::move(logger), backend);
        if (!model) return VOXTRAL_STATUS_MODEL_ERROR;
        *out_model = model;
        return VOXTRAL_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return VOXTRAL_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return VOXTRAL_STATUS_MODEL_ERROR;
    }
}

void voxtral_model_destroy(voxtral_model * model) {
    try {
        voxtral_model_free(model);
    } catch (...) {
        /* C boundary: destructors must never propagate. */
    }
}

voxtral_status voxtral_context_create(
    voxtral_model * model,
    const voxtral_context_config * config,
    voxtral_context ** out_context)
{
    if (out_context) *out_context = nullptr;
    if (!model || !out_context) return VOXTRAL_STATUS_INVALID_ARGUMENT;
    bool config_ok = true;
    const voxtral_context_config normalized =
        normalize_context_config(config, config_ok);
    if (!config_ok || normalized.n_threads < 0 || normalized.flags != 0) {
        return VOXTRAL_STATUS_INVALID_ARGUMENT;
    }
    try {
        voxtral_context_params internal{};
        internal.n_threads = normalized.n_threads;
        internal.log_level = voxtral_log_level::error;
        internal.gpu = voxtral_gpu_backend::none;
        voxtral_context * context = voxtral_init_from_model(model, internal);
        if (!context) return VOXTRAL_STATUS_BACKEND_ERROR;
        set_context_error(context, VOXTRAL_STATUS_OK, "");
        *out_context = context;
        return VOXTRAL_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return VOXTRAL_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return VOXTRAL_STATUS_BACKEND_ERROR;
    }
}

void voxtral_context_destroy(voxtral_context * context) {
    if (!context) return;
    try {
        if (voxtral_context_has_public_stream_internal(context)) {
            set_context_error(context, VOXTRAL_STATUS_INVALID_STATE,
                              "cannot destroy context while a stream is alive");
            return;
        }
        voxtral_free(context);
    } catch (...) {
        /* C boundary: destructors must never propagate. */
    }
}

voxtral_status voxtral_context_get_last_error(
    const voxtral_context * context,
    voxtral_error_info * out_error)
{
    if (!context || !out_error) return VOXTRAL_STATUS_INVALID_ARGUMENT;
    const voxtral_error_info value = make_error(
        voxtral_context_public_last_status_internal(context),
        voxtral_context_public_backend_code_internal(context),
        voxtral_context_public_last_error_internal(context));
    return write_versioned_output(out_error, value)
        ? VOXTRAL_STATUS_OK
        : VOXTRAL_STATUS_INVALID_ARGUMENT;
}

voxtral_status voxtral_stream_create(
    voxtral_context * context,
    const voxtral_stream_params * params,
    voxtral_stream ** out_stream)
{
    if (out_stream) *out_stream = nullptr;
    if (!context || !params || !out_stream) {
        if (context) {
            set_context_error(context, VOXTRAL_STATUS_INVALID_ARGUMENT,
                              "stream create requires context, params and out_stream");
        }
        return VOXTRAL_STATUS_INVALID_ARGUMENT;
    }
    bool params_ok = true;
    const voxtral_stream_params normalized =
        normalize_stream_params(params, params_ok);
    if (!params_ok || normalized.flags != 0 || normalized.reserved != 0 ||
        normalized.audio.reserved != 0 ||
        normalized.event_queue_capacity == 0 ||
        normalized.event_queue_capacity > VOXTRAL_EVENT_QUEUE_CAPACITY_MAX) {
        set_context_error(context, VOXTRAL_STATUS_INVALID_ARGUMENT,
                          "invalid stream parameter structure or queue capacity");
        return VOXTRAL_STATUS_INVALID_ARGUMENT;
    }
    if (normalized.audio.sample_rate != VOXTRAL_SAMPLE_RATE ||
        normalized.audio.channels != 1 ||
        normalized.audio.format != VOXTRAL_AUDIO_PCM_S16LE) {
        set_context_error(context, VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT,
                          "only mono 16000 Hz signed PCM16LE is supported");
        return VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT;
    }
    if (!voxtral_context_supports_incremental_internal(context)) {
        set_context_error(
            context, VOXTRAL_STATUS_MODEL_ERROR,
            "the loaded model does not support realtime incremental streaming");
        return VOXTRAL_STATUS_MODEL_ERROR;
    }
    if (!voxtral_context_try_acquire_public_stream_internal(context)) {
        set_context_error(context, VOXTRAL_STATUS_INVALID_STATE,
                          "one live stream per context is supported in API v1");
        return VOXTRAL_STATUS_INVALID_STATE;
    }

    try {
        voxtral_stream_params_internal internal{};
        internal.sample_rate = static_cast<int32_t>(
            normalized.audio.sample_rate);
        internal.channels = static_cast<int32_t>(normalized.audio.channels);
        voxtral_stream * stream =
            voxtral_stream_create_from_context_internal(context, internal);
        if (!stream) {
            voxtral_context_release_public_stream_internal(context);
            set_context_error(context, VOXTRAL_STATUS_OUT_OF_MEMORY,
                              "failed to allocate stream state");
            return VOXTRAL_STATUS_OUT_OF_MEMORY;
        }
        if (!voxtral_stream_set_event_capacity_internal(
                stream, normalized.event_queue_capacity)) {
            voxtral_stream_destroy_internal(stream);
            voxtral_context_release_public_stream_internal(context);
            set_context_error(context, VOXTRAL_STATUS_INVALID_ARGUMENT,
                              "invalid event queue capacity");
            return VOXTRAL_STATUS_INVALID_ARGUMENT;
        }
        set_context_error(context, VOXTRAL_STATUS_OK, "");
        *out_stream = stream;
        return VOXTRAL_STATUS_OK;
    } catch (const std::bad_alloc &) {
        voxtral_context_release_public_stream_internal(context);
        set_context_error(context, VOXTRAL_STATUS_OUT_OF_MEMORY,
                          "stream creation allocation failed");
        return VOXTRAL_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        voxtral_context_release_public_stream_internal(context);
        set_context_error(context, VOXTRAL_STATUS_INTERNAL_ERROR,
                          "stream creation threw an internal exception");
        return VOXTRAL_STATUS_INTERNAL_ERROR;
    }
}

void voxtral_stream_destroy(voxtral_stream * stream) {
    if (!stream) return;
    voxtral_context * context = nullptr;
    try {
        context = const_cast<voxtral_context *>(
            static_cast<const voxtral_context *>(
                voxtral_stream_context_ptr(stream)));
        voxtral_stream_destroy_internal(stream);
    } catch (...) {
        /* Continue to release the one-stream lease below. */
    }
    try {
        voxtral_context_release_public_stream_internal(context);
    } catch (...) {
        /* C boundary: destruction and lease release never propagate. */
    }
}

voxtral_status voxtral_stream_feed_pcm16(
    voxtral_stream * stream,
    const int16_t * samples,
    size_t sample_count,
    size_t * samples_consumed)
{
    if (samples_consumed) *samples_consumed = 0;
    if (!stream) return VOXTRAL_STATUS_INVALID_ARGUMENT;
    const uint64_t before = voxtral_stream_samples_received(stream);
    try {
        const voxtral_status_internal result =
            voxtral_stream_feed_pcm16_internal(stream, samples, sample_count);
        const uint64_t after = voxtral_stream_samples_received(stream);
        if (samples_consumed && after >= before) {
            const uint64_t accepted = after - before;
            *samples_consumed = accepted > std::numeric_limits<size_t>::max()
                ? std::numeric_limits<size_t>::max()
                : static_cast<size_t>(accepted);
        }
        if (result == voxtral_status_internal::backend_error ||
            result == voxtral_status_internal::out_of_memory ||
            result == voxtral_status_internal::internal_error) {
            fail_stream_noexcept(
                stream, result, voxtral_stream_last_error(stream).c_str());
        }
        return map_feed_status(stream, result);
    } catch (const std::bad_alloc &) {
        fail_stream_noexcept(
            stream, voxtral_status_internal::out_of_memory,
            "feed allocation threw");
        return VOXTRAL_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        fail_stream_noexcept(
            stream, voxtral_status_internal::internal_error,
            "feed threw an internal exception");
        return VOXTRAL_STATUS_INTERNAL_ERROR;
    }
}

voxtral_status voxtral_stream_poll_event(
    voxtral_stream * stream,
    voxtral_event * out_event)
{
    if (!stream || !out_event) return VOXTRAL_STATUS_INVALID_ARGUMENT;
    versioned_header header{};
    if (!read_header(out_event, header) ||
        header.struct_size < kEventV1StructSize) {
        return VOXTRAL_STATUS_INVALID_ARGUMENT;
    }
    try {
        voxtral_stream_event_internal source;
        if (!voxtral_stream_poll_event_internal(stream, source)) {
            return VOXTRAL_STATUS_NOT_READY;
        }
        voxtral_event value{};
        value.struct_size = sizeof(value);
        value.api_version = VOXTRAL_API_VERSION;
        value.type = event_type(source.type);
        value.sequence = ++stream->events.public_poll_sequence;
        value.token_id = source.type == voxtral_stream_event_type_internal::token
            ? source.token : -1;
        value.status = event_status(source);
        if (source.special) {
            value.flags |= VOXTRAL_EVENT_FLAG_SPECIAL_TOKEN;
        }
        if (source.type == voxtral_stream_event_type_internal::partial_text ||
            source.type == voxtral_stream_event_type_internal::final_text) {
            value.flags |= VOXTRAL_EVENT_FLAG_TEXT_STABLE;
        }
        const double end_ms = std::max(0.0, source.t_audio_ms);
        value.audio_end_ms = static_cast<uint64_t>(end_ms);
        if (source.type == voxtral_stream_event_type_internal::token ||
            source.type == voxtral_stream_event_type_internal::partial_text) {
            value.audio_start_ms = value.audio_end_ms >= 80u
                ? value.audio_end_ms - 80u : 0u;
        }
        size_t stable_bytes = source.text.size();
        if (source.type == voxtral_stream_event_type_internal::partial_text) {
            stable_bytes = source.stable_prefix_bytes;
        }
        if (source.type == voxtral_stream_event_type_internal::cancelled &&
            source.text.empty()) {
            source.text = "cancelled";
            stable_bytes = source.text.size();
        }
        copy_event_text(value, source.text, stable_bytes);
        std::memcpy(
            out_event, &value,
            std::min<size_t>(header.struct_size, sizeof(value)));
        return VOXTRAL_STATUS_OK;
    } catch (const std::bad_alloc &) {
        fail_stream_noexcept(
            stream, voxtral_status_internal::out_of_memory,
            "event copy allocation threw");
        return VOXTRAL_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        fail_stream_noexcept(
            stream, voxtral_status_internal::internal_error,
            "event poll threw an internal exception");
        return VOXTRAL_STATUS_INTERNAL_ERROR;
    }
}

voxtral_status voxtral_stream_finish(voxtral_stream * stream) {
    if (!stream) return VOXTRAL_STATUS_INVALID_ARGUMENT;
    try {
        return map_status(voxtral_stream_finish_internal(stream));
    } catch (const std::bad_alloc &) {
        fail_stream_noexcept(
            stream, voxtral_status_internal::out_of_memory,
            "finish allocation threw");
        return VOXTRAL_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        fail_stream_noexcept(
            stream, voxtral_status_internal::internal_error,
            "finish threw an internal exception");
        return VOXTRAL_STATUS_INTERNAL_ERROR;
    }
}

voxtral_status voxtral_stream_cancel(voxtral_stream * stream) {
    if (!stream) return VOXTRAL_STATUS_INVALID_ARGUMENT;
    try {
        return map_status(voxtral_stream_cancel_internal(stream));
    } catch (const std::bad_alloc &) {
        fail_stream_noexcept(
            stream, voxtral_status_internal::out_of_memory,
            "cancel allocation threw");
        return VOXTRAL_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        fail_stream_noexcept(
            stream, voxtral_status_internal::internal_error,
            "cancel threw an internal exception");
        return VOXTRAL_STATUS_INTERNAL_ERROR;
    }
}

voxtral_status voxtral_stream_reset(voxtral_stream * stream) {
    if (!stream) return VOXTRAL_STATUS_INVALID_ARGUMENT;
    try {
        const voxtral_stream_state_internal state =
            voxtral_stream_get_state_internal(stream);
        if (state == voxtral_stream_state_internal::running ||
            state == voxtral_stream_state_internal::finishing) {
            set_stream_error_noexcept(
                stream, voxtral_status_internal::invalid_state,
                "reset requires a created or terminal stream");
            return VOXTRAL_STATUS_INVALID_STATE;
        }
        return map_status(voxtral_stream_reset_internal(stream));
    } catch (const std::bad_alloc &) {
        fail_stream_noexcept(
            stream, voxtral_status_internal::out_of_memory,
            "reset allocation threw");
        return VOXTRAL_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        fail_stream_noexcept(
            stream, voxtral_status_internal::internal_error,
            "reset threw an internal exception");
        return VOXTRAL_STATUS_INTERNAL_ERROR;
    }
}

voxtral_stream_state voxtral_stream_get_state(
    const voxtral_stream * stream)
{
    return stream
        ? map_state(voxtral_stream_get_state_internal(stream))
        : VOXTRAL_STREAM_ERROR;
}

voxtral_status voxtral_stream_get_last_error(
    const voxtral_stream * stream,
    voxtral_error_info * out_error)
{
    if (!stream || !out_error) return VOXTRAL_STATUS_INVALID_ARGUMENT;
    const voxtral_status_internal stored = voxtral_stream_last_status(stream);
    const voxtral_status public_status =
        stored == voxtral_status_internal::limit_exceeded
        ? (voxtral_stream_last_feed_status(stream) ==
                   voxtral_stream_feed_status::queue_full
               ? VOXTRAL_STATUS_QUEUE_FULL
               : VOXTRAL_STATUS_INVALID_ARGUMENT)
        : map_status(stored);
    const voxtral_error_info value = make_error(
        public_status,
        0,
        voxtral_stream_last_error(stream));
    return write_versioned_output(out_error, value)
        ? VOXTRAL_STATUS_OK
        : VOXTRAL_STATUS_INVALID_ARGUMENT;
}

voxtral_status voxtral_stream_get_final_text(
    const voxtral_stream * stream,
    char * buffer,
    size_t capacity,
    size_t * required)
{
    if (required) *required = 0;
    if (!stream || (!buffer && capacity != 0)) {
        return VOXTRAL_STATUS_INVALID_ARGUMENT;
    }
    const voxtral_stream_state state = voxtral_stream_get_state(stream);
    if (state == VOXTRAL_STREAM_CREATED ||
        state == VOXTRAL_STREAM_ACTIVE ||
        state == VOXTRAL_STREAM_FINISHING) {
        return VOXTRAL_STATUS_NOT_READY;
    }
    if (state != VOXTRAL_STREAM_COMPLETED) {
        return VOXTRAL_STATUS_INVALID_STATE;
    }
    const std::string & text = voxtral_stream_transcript(stream);
    if (text.size() == std::numeric_limits<size_t>::max()) {
        return VOXTRAL_STATUS_INTERNAL_ERROR;
    }
    const size_t needed = text.size() + 1u;
    if (required) *required = needed;
    if (!buffer && capacity == 0) return VOXTRAL_STATUS_OK;
    if (!buffer || capacity < needed) {
        if (buffer && capacity > 0) buffer[0] = '\0';
        return VOXTRAL_STATUS_BUFFER_TOO_SMALL;
    }
    if (!text.empty()) std::memcpy(buffer, text.data(), text.size());
    buffer[text.size()] = '\0';
    return VOXTRAL_STATUS_OK;
}

voxtral_status voxtral_stream_get_metrics(
    const voxtral_stream * stream,
    voxtral_stream_metrics * out_metrics)
{
    if (!stream || !out_metrics) return VOXTRAL_STATUS_INVALID_ARGUMENT;
    try {
        voxtral_stream_metrics value{};
        value.struct_size = sizeof(value);
        value.api_version = VOXTRAL_API_VERSION;
        value.audio_samples_accepted =
            voxtral_stream_samples_received(stream);
        value.audio_duration_ms =
            static_cast<uint64_t>(std::max(0.0, voxtral_stream_audio_ms(stream)));
        value.encoder_frames = static_cast<uint64_t>(
            std::max<int64_t>(0, voxtral_stream_encoder_frames(stream)));
        value.adapter_groups = static_cast<uint64_t>(
            std::max<int64_t>(0, voxtral_stream_adapter_groups_committed(stream)));
        value.decoder_steps = static_cast<uint64_t>(
            std::max<int64_t>(0, voxtral_stream_decoder_steps(stream)));
        value.token_events = voxtral_stream_token_events(stream);
        value.partial_events = voxtral_stream_partial_events(stream);

        const auto * context = static_cast<const voxtral_context *>(
            voxtral_stream_context_ptr(stream));
        const voxtral_public_context_metrics_internal context_metrics =
            voxtral_context_public_metrics_internal(context);
        value.decoder_kv_wraps = static_cast<uint64_t>(
            std::max<int64_t>(0, context_metrics.decoder_kv_wraps));
        value.decoder_kv_evictions = static_cast<uint64_t>(
            std::max<int64_t>(0, context_metrics.decoder_kv_evictions));
        value.decoder_kv_bytes_moved = static_cast<uint64_t>(
            std::max<int64_t>(0, context_metrics.decoder_kv_bytes_moved));
        if (context_metrics.profile_enabled && value.audio_duration_ms > 0) {
            value.pipeline_rtf = context_metrics.total_pipeline_compute_ms /
                                 static_cast<double>(value.audio_duration_ms);
        }
        const voxtral_public_backlog_metrics_internal backlog =
            voxtral_stream_public_backlog_internal(stream);
        value.backlog_final_ms = backlog.final_ms;
        value.backlog_slope_ms_per_s = backlog.slope_ms_per_s;
        return write_versioned_output(out_metrics, value)
            ? VOXTRAL_STATUS_OK
            : VOXTRAL_STATUS_INVALID_ARGUMENT;
    } catch (const std::bad_alloc &) {
        return VOXTRAL_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return VOXTRAL_STATUS_INTERNAL_ERROR;
    }
}

voxtral_status voxtral_context_get_capabilities(
    const voxtral_context * context,
    voxtral_capabilities * out_capabilities)
{
    if (!context || !out_capabilities) {
        return VOXTRAL_STATUS_INVALID_ARGUMENT;
    }
    voxtral_capabilities value{};
    value.struct_size = sizeof(value);
    value.api_version = VOXTRAL_API_VERSION;
    value.sample_rate = VOXTRAL_SAMPLE_RATE;
    value.channels = 1;
    value.audio_format = VOXTRAL_AUDIO_PCM_S16LE;
    const uint32_t streaming =
        voxtral_context_supports_incremental_internal(context) ? 1u : 0u;
    value.supports_incremental = streaming;
    value.supports_reset = streaming;
    value.max_active_streams_per_context = streaming;
    return write_versioned_output(out_capabilities, value)
        ? VOXTRAL_STATUS_OK
        : VOXTRAL_STATUS_INVALID_ARGUMENT;
}

} /* extern "C" */
