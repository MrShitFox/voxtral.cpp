#include <voxtral.h>
#include <voxtral-stream.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(expr)                                                            \
    do {                                                                       \
        if (!(expr)) {                                                         \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, #expr);                                \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

typedef struct future_model_params {
    voxtral_model_params current;
    uint64_t future_field;
} future_model_params;

static void test_version_and_status(void) {
    static const char * const expected[] = {
        "ok",
        "invalid_argument",
        "invalid_state",
        "out_of_memory",
        "queue_full",
        "not_ready",
        "cancelled",
        "unsupported_audio_format",
        "model_error",
        "backend_error",
        "internal_error",
        "buffer_too_small",
    };
    CHECK(VOXTRAL_API_VERSION == 0x01000000u);
    CHECK(voxtral_api_version() == VOXTRAL_API_VERSION);
    CHECK(strcmp(voxtral_version_string(), "1.0.0") == 0);
    for (int value = VOXTRAL_STATUS_OK;
         value <= VOXTRAL_STATUS_BUFFER_TOO_SMALL;
         ++value) {
        CHECK(strcmp(voxtral_status_string((voxtral_status) value),
                     expected[value]) == 0);
    }
    CHECK(strcmp(voxtral_status_string((voxtral_status) 9999), "unknown") == 0);
}

static void test_defaults(void) {
    const voxtral_model_params model = voxtral_model_default_params();
    CHECK(model.struct_size == sizeof(model));
    CHECK(model.api_version == VOXTRAL_API_VERSION);
    CHECK(model.backend == VOXTRAL_BACKEND_AUTO);
    CHECK(model.flags == 0);

    const voxtral_context_config context =
        voxtral_context_default_config();
    CHECK(context.struct_size == sizeof(context));
    CHECK(context.api_version == VOXTRAL_API_VERSION);
    CHECK(context.n_threads == 0);
    CHECK(context.flags == 0);

    const voxtral_audio_config audio = voxtral_audio_default_config();
    CHECK(audio.struct_size == sizeof(audio));
    CHECK(audio.api_version == VOXTRAL_API_VERSION);
    CHECK(audio.sample_rate == 16000);
    CHECK(audio.channels == 1);
    CHECK(audio.format == VOXTRAL_AUDIO_PCM_S16LE);

    const voxtral_stream_params stream = voxtral_stream_default_params();
    CHECK(stream.struct_size == sizeof(stream));
    CHECK(stream.api_version == VOXTRAL_API_VERSION);
    CHECK(stream.audio.struct_size == sizeof(stream.audio));
    CHECK(stream.audio.api_version == VOXTRAL_API_VERSION);
    CHECK(stream.event_queue_capacity == 4096);
    CHECK(stream.flags == 0);
}

static void test_versioned_input_validation(void) {
    voxtral_model * model = (voxtral_model *) (uintptr_t) 1u;

    voxtral_model_params invalid = voxtral_model_default_params();
    invalid.struct_size = sizeof(uint32_t);
    CHECK(voxtral_model_load("/definitely/missing.gguf", &invalid, &model) ==
          VOXTRAL_STATUS_INVALID_ARGUMENT);
    CHECK(model == NULL);

    invalid = voxtral_model_default_params();
    invalid.api_version = 0x02000000u;
    CHECK(voxtral_model_load("/definitely/missing.gguf", &invalid, &model) ==
          VOXTRAL_STATUS_INVALID_ARGUMENT);
    CHECK(model == NULL);

    /* Header-only v1 prefix: unavailable fields take their documented defaults. */
    invalid = voxtral_model_default_params();
    invalid.struct_size = sizeof(uint32_t) * 2u;
    CHECK(voxtral_model_load("/definitely/missing.gguf", &invalid, &model) ==
          VOXTRAL_STATUS_MODEL_ERROR);
    CHECK(model == NULL);

    /* A future trailing field is ignored and never read as a current field. */
    future_model_params future;
    memset(&future, 0xa5, sizeof(future));
    future.current = voxtral_model_default_params();
    future.current.struct_size = sizeof(future);
    CHECK(voxtral_model_load("/definitely/missing.gguf", &future.current,
                             &model) == VOXTRAL_STATUS_MODEL_ERROR);
    CHECK(model == NULL);
}

static void test_null_contracts(void) {
    voxtral_model * model = (voxtral_model *) (uintptr_t) 1u;
    CHECK(voxtral_model_load(NULL, NULL, &model) ==
          VOXTRAL_STATUS_INVALID_ARGUMENT);
    CHECK(model == NULL);
    CHECK(voxtral_model_load("x", NULL, NULL) ==
          VOXTRAL_STATUS_INVALID_ARGUMENT);

    voxtral_context * context = (voxtral_context *) (uintptr_t) 1u;
    CHECK(voxtral_context_create(NULL, NULL, &context) ==
          VOXTRAL_STATUS_INVALID_ARGUMENT);
    CHECK(context == NULL);
    CHECK(voxtral_context_create(NULL, NULL, NULL) ==
          VOXTRAL_STATUS_INVALID_ARGUMENT);

    voxtral_stream * stream = (voxtral_stream *) (uintptr_t) 1u;
    voxtral_stream_params params = voxtral_stream_default_params();
    CHECK(voxtral_stream_create(NULL, &params, &stream) ==
          VOXTRAL_STATUS_INVALID_ARGUMENT);
    CHECK(stream == NULL);

    size_t consumed = 123;
    CHECK(voxtral_stream_feed_pcm16(NULL, NULL, 0, &consumed) ==
          VOXTRAL_STATUS_INVALID_ARGUMENT);
    CHECK(consumed == 0);
    CHECK(voxtral_stream_finish(NULL) == VOXTRAL_STATUS_INVALID_ARGUMENT);
    CHECK(voxtral_stream_cancel(NULL) == VOXTRAL_STATUS_INVALID_ARGUMENT);
    CHECK(voxtral_stream_reset(NULL) == VOXTRAL_STATUS_INVALID_ARGUMENT);
    CHECK(voxtral_stream_get_state(NULL) == VOXTRAL_STREAM_ERROR);

    voxtral_event event;
    memset(&event, 0, sizeof(event));
    event.struct_size = sizeof(event);
    event.api_version = VOXTRAL_API_VERSION;
    CHECK(voxtral_stream_poll_event(NULL, &event) ==
          VOXTRAL_STATUS_INVALID_ARGUMENT);

    voxtral_model_destroy(NULL);
    voxtral_context_destroy(NULL);
    voxtral_stream_destroy(NULL);
}

int main(void) {
    _Static_assert(sizeof(voxtral_status) == sizeof(int),
                   "P/Invoke status ABI expects a C int enum");
    _Static_assert(sizeof(voxtral_backend) == sizeof(int),
                   "P/Invoke backend ABI expects a C int enum");
    _Static_assert(sizeof(voxtral_audio_format) == sizeof(int),
                   "P/Invoke audio ABI expects a C int enum");
    _Static_assert(sizeof(voxtral_stream_state) == sizeof(int),
                   "P/Invoke state ABI expects a C int enum");
    _Static_assert(sizeof(voxtral_event_type) == sizeof(int),
                   "P/Invoke event ABI expects a C int enum");
    _Static_assert(offsetof(voxtral_error_info, message) == 16,
                   "voxtral_error_info v1 layout changed");
    _Static_assert(sizeof(voxtral_error_info) == 272,
                   "voxtral_error_info v1 size changed");
    _Static_assert(sizeof(voxtral_model_params) == 24,
                   "voxtral_model_params v1 size changed");
    _Static_assert(sizeof(voxtral_context_config) == 16,
                   "voxtral_context_config v1 size changed");
    _Static_assert(sizeof(voxtral_audio_config) == 24,
                   "voxtral_audio_config v1 size changed");
    _Static_assert(sizeof(voxtral_stream_params) == 48,
                   "voxtral_stream_params v1 size changed");
    _Static_assert(offsetof(voxtral_event, text) ==
                       56 + sizeof(size_t),
                   "voxtral_event v1 layout changed");
#if SIZE_MAX == UINT64_MAX
    _Static_assert(sizeof(voxtral_event) == 4160,
                   "64-bit voxtral_event v1 size changed");
#elif SIZE_MAX == UINT32_MAX
    _Static_assert(sizeof(voxtral_event) == 4156,
                   "32-bit voxtral_event v1 size changed");
#endif
    _Static_assert(sizeof(voxtral_stream_metrics) == 112,
                   "voxtral_stream_metrics v1 size changed");
    _Static_assert(sizeof(voxtral_capabilities) == 32,
                   "voxtral_capabilities v1 size changed");
    _Static_assert(VOXTRAL_EVENT_TEXT_CAPACITY == 4096,
                   "event text ABI changed");
    _Static_assert(VOXTRAL_EVENT_TERMINAL_HEADROOM == 64,
                   "terminal queue bound changed");
    _Static_assert(VOXTRAL_STREAM_MAX_AUDIO_SAMPLES == UINT64_C(57600000),
                   "stream duration bound changed");

    test_version_and_status();
    test_defaults();
    test_versioned_input_validation();
    test_null_contracts();

    if (failures != 0) {
        fprintf(stderr, "public C API unit failures: %d\n", failures);
        return 1;
    }
    puts("public C API unit: PASS");
    return 0;
}
