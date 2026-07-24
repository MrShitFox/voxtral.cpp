#include <voxtral.h>
#include <voxtral-stream.h>

#include <type_traits>

static_assert(std::is_standard_layout<voxtral_event>::value,
              "public event must remain standard-layout");
static_assert(std::is_trivially_copyable<voxtral_event>::value,
              "public event must remain trivially copyable");
static_assert(sizeof(voxtral_status) == sizeof(int),
              "public status ABI must remain a C int enum");
static_assert(sizeof(voxtral_audio_config) == 24,
              "public audio ABI changed");
static_assert(sizeof(voxtral_model_params) == 24,
              "public model params ABI changed");
static_assert(sizeof(voxtral_context_config) == 16,
              "public context config ABI changed");
static_assert(sizeof(voxtral_stream_params) == 48,
              "public stream params ABI changed");
static_assert(sizeof(voxtral_error_info) == 272,
              "public error ABI changed");
static_assert(sizeof(voxtral_stream_metrics) == 112,
              "public metrics ABI changed");
static_assert(sizeof(voxtral_capabilities) == 32,
              "public capabilities ABI changed");
#if SIZE_MAX == UINT64_MAX
static_assert(sizeof(voxtral_event) == 4160,
              "64-bit public event ABI changed");
#endif

int main() {
    voxtral_model * model = nullptr;
    voxtral_context * context = nullptr;
    voxtral_stream * stream = nullptr;
    voxtral_stream_params params{};
    (void) model;
    (void) context;
    (void) stream;
    (void) params;
    return VOXTRAL_STATUS_OK;
}
