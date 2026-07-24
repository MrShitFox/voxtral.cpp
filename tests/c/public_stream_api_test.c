#include <voxtral.h>
#include <voxtral-stream.h>

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__)
#include <glob.h>
#endif

typedef struct pcm_data {
    int16_t * samples;
    size_t count;
} pcm_data;

typedef struct token_list {
    int32_t * values;
    size_t count;
    size_t capacity;
} token_list;

typedef struct char_list {
    char * values;
    size_t count;
    size_t capacity;
} char_list;

typedef struct event_summary {
    token_list tokens;
    char_list signature;
    char * last_partial;
    char * final_event_text;
    uint64_t last_sequence;
    uint64_t token_events;
    uint64_t partial_events;
    uint64_t final_events;
    uint64_t completed_events;
    uint64_t cancelled_events;
    uint64_t error_events;
    uint64_t truncated_events;
    int ordering_ok;
    int partial_monotonic;
    int value_ok;
    int completed_seen;
    int event_after_completed;
} event_summary;

typedef struct run_result {
    token_list tokens;
    char * event_signature;
    char * transcript;
    voxtral_stream_metrics metrics;
    uint64_t samples_offered;
    uint64_t samples_consumed;
    uint64_t samples_retried;
    uint64_t queue_full_returns;
    uint64_t token_events;
    uint64_t partial_events;
    uint64_t final_events;
    uint64_t completed_events;
    uint64_t truncated_events;
    int event_ordering_ok;
    int partial_monotonic;
    int event_value_ok;
    int terminal_ordering_ok;
    int state_completed;
    int final_query_ok;
    int finish_idempotent;
    int feed_after_finish_rejected;
    int structured_stream_error_ok;
    int queue_error_ok;
    double feed_call_ms_total;
    uint64_t feed_calls;
} run_result;

typedef struct options {
    const char * model_path;
    const char * audio_path;
    size_t chunk_samples;
    uint32_t event_capacity;
    uint32_t iterations;
    uint32_t random_seed;
    uint32_t synthetic_seconds;
    int delayed_consumer;
    int lifecycle;
    int random_chunks;
    int paced;
} options;

typedef struct future_stream_params {
    voxtral_stream_params current;
    uint64_t future_field;
} future_stream_params;

typedef struct future_event {
    voxtral_event current;
    uint64_t future_field;
} future_event;

typedef struct destroy_summary {
    int null_safe;
    int created;
    int active;
    int cancelled;
    int completed;
    int under_backpressure;
    int finish_under_backpressure;
    int cancel_under_backpressure;
    int lease_released;
} destroy_summary;

static uint16_t read_u16le(const unsigned char * p) {
    return (uint16_t) p[0] | ((uint16_t) p[1] << 8u);
}

static uint32_t read_u32le(const unsigned char * p) {
    return (uint32_t) p[0] |
           ((uint32_t) p[1] << 8u) |
           ((uint32_t) p[2] << 16u) |
           ((uint32_t) p[3] << 24u);
}

static int read_exact(FILE * file, void * data, size_t size) {
    return size == 0 || fread(data, 1, size, file) == size;
}

static int load_pcm16_wav(const char * path, pcm_data * out) {
    unsigned char riff[12];
    FILE * file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "failed to open WAV %s: %s\n", path, strerror(errno));
        return 0;
    }
    if (!read_exact(file, riff, sizeof(riff)) ||
        memcmp(riff, "RIFF", 4) != 0 ||
        memcmp(riff + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "not a RIFF/WAVE file: %s\n", path);
        fclose(file);
        return 0;
    }

    int have_fmt = 0;
    int have_data = 0;
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    unsigned char * bytes = NULL;
    size_t byte_count = 0;

    while (!have_data) {
        unsigned char header[8];
        if (!read_exact(file, header, sizeof(header))) break;
        const uint32_t chunk_size = read_u32le(header + 4);
        if (memcmp(header, "fmt ", 4) == 0) {
            unsigned char fmt[16];
            if (chunk_size < sizeof(fmt) ||
                !read_exact(file, fmt, sizeof(fmt))) {
                break;
            }
            audio_format = read_u16le(fmt);
            channels = read_u16le(fmt + 2);
            sample_rate = read_u32le(fmt + 4);
            bits_per_sample = read_u16le(fmt + 14);
            if (chunk_size > sizeof(fmt) &&
                fseek(file, (long) (chunk_size - sizeof(fmt)), SEEK_CUR) != 0) {
                break;
            }
            have_fmt = 1;
        } else if (memcmp(header, "data", 4) == 0) {
            if (chunk_size == 0 || (chunk_size & 1u) != 0) break;
            bytes = (unsigned char *) malloc(chunk_size);
            if (!bytes || !read_exact(file, bytes, chunk_size)) break;
            byte_count = chunk_size;
            have_data = 1;
        } else if (fseek(file, (long) chunk_size, SEEK_CUR) != 0) {
            break;
        }
        if ((chunk_size & 1u) != 0 && fseek(file, 1, SEEK_CUR) != 0) break;
    }
    fclose(file);

    if (!have_fmt || !have_data || audio_format != 1 || channels != 1 ||
        sample_rate != 16000 || bits_per_sample != 16) {
        fprintf(stderr,
                "WAV must be PCM16 mono 16000 Hz (fmt=%u channels=%u rate=%u bits=%u)\n",
                (unsigned) audio_format, (unsigned) channels,
                (unsigned) sample_rate, (unsigned) bits_per_sample);
        free(bytes);
        return 0;
    }

    const size_t sample_count = byte_count / 2u;
    int16_t * samples = (int16_t *) malloc(sample_count * sizeof(int16_t));
    if (!samples) {
        free(bytes);
        return 0;
    }
    for (size_t i = 0; i < sample_count; ++i) {
        samples[i] = (int16_t) read_u16le(bytes + i * 2u);
    }
    free(bytes);
    out->samples = samples;
    out->count = sample_count;
    return 1;
}

static int make_synthetic_pcm(uint32_t seconds, pcm_data * out) {
    if (seconds == 0 || seconds > SIZE_MAX / 16000u ||
        (size_t) seconds * 16000u > SIZE_MAX / sizeof(int16_t)) {
        return 0;
    }
    const size_t count = (size_t) seconds * 16000u;
    int16_t * samples = (int16_t *) malloc(count * sizeof(int16_t));
    if (!samples) return 0;
    uint32_t random_state = 0x1234567u;
    for (size_t i = 0; i < count; ++i) {
        random_state ^= random_state << 13u;
        random_state ^= random_state >> 17u;
        random_state ^= random_state << 5u;
        const double t = (double) i / 16000.0;
        const double signal =
            0.06 * sin(2.0 * 3.14159265 * 180.0 * t) +
            0.04 * sin(2.0 * 3.14159265 * 320.0 * t);
        const double noise =
            ((double) (random_state & 0xffffu) / 65535.0 - 0.5) * 0.01;
        double value = signal + noise;
        if (value < -1.0) value = -1.0;
        if (value > 1.0) value = 1.0;
        samples[i] = (int16_t) lround(value * 32000.0);
    }
    out->samples = samples;
    out->count = count;
    return 1;
}

static int tokens_push(token_list * list, int32_t value) {
    if (list->count == list->capacity) {
        const size_t next = list->capacity == 0 ? 128u : list->capacity * 2u;
        if (next < list->capacity ||
            next > SIZE_MAX / sizeof(int32_t)) return 0;
        int32_t * values = (int32_t *) realloc(
            list->values, next * sizeof(int32_t));
        if (!values) return 0;
        list->values = values;
        list->capacity = next;
    }
    list->values[list->count++] = value;
    return 1;
}

static int tokens_equal(const token_list * a, const token_list * b) {
    return a->count == b->count &&
           (a->count == 0 ||
            memcmp(a->values, b->values,
                   a->count * sizeof(int32_t)) == 0);
}

static int chars_push(char_list * list, char value) {
    if (list->count + 1u >= list->capacity) {
        const size_t next = list->capacity == 0 ? 128u : list->capacity * 2u;
        if (next <= list->capacity) return 0;
        char * values = (char *) realloc(list->values, next);
        if (!values) return 0;
        list->values = values;
        list->capacity = next;
    }
    list->values[list->count++] = value;
    list->values[list->count] = '\0';
    return 1;
}

static char * duplicate_text(const char * text) {
    const size_t size = strlen(text) + 1u;
    char * copy = (char *) malloc(size);
    if (copy) memcpy(copy, text, size);
    return copy;
}

static int is_prefix(const char * prefix, const char * text) {
    const size_t prefix_size = strlen(prefix);
    const size_t text_size = strlen(text);
    return prefix_size <= text_size &&
           memcmp(prefix, text, prefix_size) == 0;
}

static void event_summary_init(event_summary * summary) {
    memset(summary, 0, sizeof(*summary));
    summary->ordering_ok = 1;
    summary->partial_monotonic = 1;
    summary->value_ok = 1;
}

static void event_summary_free(event_summary * summary) {
    free(summary->tokens.values);
    free(summary->signature.values);
    free(summary->last_partial);
    free(summary->final_event_text);
    memset(summary, 0, sizeof(*summary));
}

static int handle_event(event_summary * summary, const voxtral_event * event) {
    if (event->text_length >= VOXTRAL_EVENT_TEXT_CAPACITY ||
        event->text[event->text_length] != '\0' ||
        strlen(event->text) != event->text_length) {
        summary->value_ok = 0;
    }
    if ((event->flags & VOXTRAL_EVENT_FLAG_TEXT_TRUNCATED) != 0) {
        ++summary->truncated_events;
    }
    if ((event->type == VOXTRAL_EVENT_PARTIAL_TEXT ||
         event->type == VOXTRAL_EVENT_FINAL_TEXT) &&
        (event->flags & VOXTRAL_EVENT_FLAG_TEXT_STABLE) == 0) {
        summary->value_ok = 0;
    }
    if (event->sequence <= summary->last_sequence) {
        summary->ordering_ok = 0;
    }
    summary->last_sequence = event->sequence;
    if (summary->completed_seen) summary->event_after_completed = 1;

    switch (event->type) {
        case VOXTRAL_EVENT_TOKEN:
            if (!chars_push(&summary->signature, 'T')) return 0;
            ++summary->token_events;
            if (!tokens_push(&summary->tokens, event->token_id)) return 0;
            break;
        case VOXTRAL_EVENT_PARTIAL_TEXT: {
            if (!chars_push(&summary->signature, 'P')) return 0;
            ++summary->partial_events;
            if (summary->last_partial &&
                !is_prefix(summary->last_partial, event->text)) {
                summary->partial_monotonic = 0;
            }
            char * next = duplicate_text(event->text);
            if (!next) return 0;
            free(summary->last_partial);
            summary->last_partial = next;
            break;
        }
        case VOXTRAL_EVENT_FINAL_TEXT:
            if (!chars_push(&summary->signature, 'F')) return 0;
            ++summary->final_events;
            free(summary->final_event_text);
            summary->final_event_text = duplicate_text(event->text);
            if (!summary->final_event_text) return 0;
            break;
        case VOXTRAL_EVENT_COMPLETED:
            if (!chars_push(&summary->signature, 'C')) return 0;
            ++summary->completed_events;
            summary->completed_seen = 1;
            break;
        case VOXTRAL_EVENT_ERROR:
            if (!chars_push(&summary->signature, 'E')) return 0;
            if (event->status == VOXTRAL_STATUS_CANCELLED) {
                ++summary->cancelled_events;
            } else {
                ++summary->error_events;
            }
            break;
        case VOXTRAL_EVENT_NONE:
            summary->ordering_ok = 0;
            break;
    }
    return 1;
}

static int drain_events(voxtral_stream * stream, event_summary * summary) {
    for (;;) {
        future_event event;
        memset(&event, 0, sizeof(event));
        event.current.struct_size = sizeof(event);
        event.current.api_version = VOXTRAL_API_VERSION;
        event.future_field = UINT64_C(0xa5a5a5a5a5a5a5a5);
        const voxtral_status status =
            voxtral_stream_poll_event(stream, &event.current);
        if (status == VOXTRAL_STATUS_NOT_READY) return 1;
        if (status != VOXTRAL_STATUS_OK) {
            fprintf(stderr, "poll_event failed: %s\n",
                    voxtral_status_string(status));
            return 0;
        }
        if (event.future_field != UINT64_C(0xa5a5a5a5a5a5a5a5) ||
            !handle_event(summary, &event.current)) {
            return 0;
        }
    }
}

static double monotonic_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double) ts.tv_sec * 1000.0 + (double) ts.tv_nsec / 1e6;
}

static void sleep_until_ms(double target_ms) {
    for (;;) {
        const double remaining_ms = target_ms - monotonic_ms();
        if (remaining_ms <= 0.0) return;
        struct timespec request;
        request.tv_sec = (time_t) (remaining_ms / 1000.0);
        request.tv_nsec =
            (long) ((remaining_ms - (double) request.tv_sec * 1000.0) * 1e6);
        if (request.tv_nsec < 0) request.tv_nsec = 0;
        if (nanosleep(&request, NULL) == 0 || errno != EINTR) return;
    }
}

static uint32_t next_random(uint32_t * state) {
    uint32_t value = *state;
    value ^= value << 13u;
    value ^= value >> 17u;
    value ^= value << 5u;
    *state = value;
    return value;
}

static int queue_error_is_structured(voxtral_stream * stream) {
    voxtral_error_info error;
    memset(&error, 0, sizeof(error));
    error.struct_size = sizeof(error);
    error.api_version = VOXTRAL_API_VERSION;
    return voxtral_stream_get_last_error(stream, &error) ==
               VOXTRAL_STATUS_OK &&
           error.status == VOXTRAL_STATUS_QUEUE_FULL &&
           error.message[0] != '\0';
}

static int feed_all(
    voxtral_stream * stream,
    const pcm_data * pcm,
    const options * opts,
    event_summary * events,
    run_result * result)
{
    size_t offset = 0;
    uint32_t random_state =
        opts->random_seed == 0 ? 0x6d2b79f5u : opts->random_seed;
    const double pacing_origin_ms = monotonic_ms();
    result->samples_offered = pcm->count;
    result->queue_error_ok = 1;
    while (offset < pcm->count) {
        size_t chunk_samples = opts->chunk_samples;
        if (opts->random_chunks) {
            chunk_samples =
                1u + (size_t) (next_random(&random_state) % chunk_samples);
        }
        const size_t requested = chunk_samples < pcm->count - offset
            ? chunk_samples : pcm->count - offset;
        size_t remaining = requested;
        while (remaining > 0) {
            size_t consumed = 0;
            const double before = monotonic_ms();
            const voxtral_status status = voxtral_stream_feed_pcm16(
                stream, pcm->samples + offset, remaining, &consumed);
            result->feed_call_ms_total += monotonic_ms() - before;
            ++result->feed_calls;
            if (consumed > remaining) return 0;
            result->samples_consumed += consumed;
            offset += consumed;
            remaining -= consumed;

            if (status == VOXTRAL_STATUS_QUEUE_FULL) {
                result->queue_error_ok =
                    result->queue_error_ok &&
                    queue_error_is_structured(stream);
                ++result->queue_full_returns;
                if (!drain_events(stream, events)) return 0;
                if (remaining > 0) result->samples_retried += remaining;
                continue;
            }
            if (status != VOXTRAL_STATUS_OK || consumed != remaining + consumed) {
                fprintf(stderr, "feed failed: %s consumed=%zu requested=%zu\n",
                        voxtral_status_string(status), consumed,
                        remaining + consumed);
                return 0;
            }
        }
        if (!opts->delayed_consumer && !drain_events(stream, events)) return 0;
        if (opts->paced) {
            const double audio_elapsed_ms =
                (double) offset * 1000.0 / 16000.0;
            sleep_until_ms(pacing_origin_ms + audio_elapsed_ms);
        }
    }

    /* Resume a decoder that accepted the final audio while its event queue was full. */
    for (;;) {
        size_t consumed = 99;
        const double before = monotonic_ms();
        const voxtral_status status =
            voxtral_stream_feed_pcm16(stream, NULL, 0, &consumed);
        result->feed_call_ms_total += monotonic_ms() - before;
        ++result->feed_calls;
        if (consumed != 0) return 0;
        if (status == VOXTRAL_STATUS_QUEUE_FULL) {
            result->queue_error_ok =
                result->queue_error_ok &&
                queue_error_is_structured(stream);
            ++result->queue_full_returns;
            if (!drain_events(stream, events)) return 0;
            continue;
        }
        if (status != VOXTRAL_STATUS_OK) {
            fprintf(stderr, "zero-feed resume failed: %s\n",
                    voxtral_status_string(status));
            return 0;
        }
        break;
    }
    return drain_events(stream, events);
}

static int copy_final_text(voxtral_stream * stream, char ** out) {
    size_t required = 0;
    voxtral_status status =
        voxtral_stream_get_final_text(stream, NULL, 0, &required);
    if (status != VOXTRAL_STATUS_OK || required == 0) return 0;

    char tiny[1] = {'x'};
    size_t tiny_required = 0;
    status = voxtral_stream_get_final_text(
        stream, tiny, sizeof(tiny), &tiny_required);
    if (required > 1 &&
        (status != VOXTRAL_STATUS_BUFFER_TOO_SMALL ||
         tiny[0] != '\0' || tiny_required != required)) return 0;

    char * text = (char *) malloc(required);
    if (!text) return 0;
    size_t copied_required = 0;
    status = voxtral_stream_get_final_text(
        stream, text, required, &copied_required);
    if (status != VOXTRAL_STATUS_OK || copied_required != required ||
        text[required - 1u] != '\0') {
        free(text);
        return 0;
    }
    *out = text;
    return 1;
}

static void run_result_free(run_result * result) {
    free(result->tokens.values);
    free(result->event_signature);
    free(result->transcript);
    memset(result, 0, sizeof(*result));
}

static int run_once(
    voxtral_stream * stream,
    const pcm_data * pcm,
    const options * opts,
    run_result * result)
{
    memset(result, 0, sizeof(*result));
    event_summary events;
    event_summary_init(&events);

    if (!feed_all(stream, pcm, opts, &events, result)) {
        voxtral_error_info error;
        memset(&error, 0, sizeof(error));
        error.struct_size = sizeof(error);
        error.api_version = VOXTRAL_API_VERSION;
        (void) voxtral_stream_get_last_error(stream, &error);
        fprintf(stderr, "feed_all failed: stored=%s message=%s\n",
                voxtral_status_string(error.status), error.message);
        event_summary_free(&events);
        return 0;
    }

    const voxtral_status finish1 = voxtral_stream_finish(stream);
    const voxtral_status finish2 = finish1 == VOXTRAL_STATUS_OK
        ? voxtral_stream_finish(stream)
        : finish1;
    result->finish_idempotent =
        finish1 == VOXTRAL_STATUS_OK && finish2 == VOXTRAL_STATUS_OK;
    if (finish1 != VOXTRAL_STATUS_OK || !drain_events(stream, &events)) {
        voxtral_error_info error;
        memset(&error, 0, sizeof(error));
        error.struct_size = sizeof(error);
        error.api_version = VOXTRAL_API_VERSION;
        (void) voxtral_stream_get_last_error(stream, &error);
        fprintf(stderr, "finish/drain failed: first=%s second=%s\n",
                voxtral_status_string(finish1),
                voxtral_status_string(finish2));
        fprintf(stderr, "finish stored error: %s (%s)\n",
                voxtral_status_string(error.status), error.message);
        event_summary_free(&events);
        return 0;
    }

    int16_t sample = 0;
    size_t consumed = 77;
    result->feed_after_finish_rejected =
        voxtral_stream_feed_pcm16(stream, &sample, 1, &consumed) ==
            VOXTRAL_STATUS_INVALID_STATE &&
        consumed == 0;
    voxtral_error_info stream_error;
    memset(&stream_error, 0, sizeof(stream_error));
    stream_error.struct_size = sizeof(stream_error);
    stream_error.api_version = VOXTRAL_API_VERSION;
    result->structured_stream_error_ok =
        voxtral_stream_get_last_error(stream, &stream_error) ==
            VOXTRAL_STATUS_OK &&
        stream_error.status == VOXTRAL_STATUS_INVALID_STATE &&
        stream_error.message[0] != '\0';

    result->state_completed =
        voxtral_stream_get_state(stream) == VOXTRAL_STREAM_COMPLETED;
    result->final_query_ok = copy_final_text(stream, &result->transcript);
    result->event_ordering_ok =
        events.ordering_ok && !events.event_after_completed;
    result->partial_monotonic = events.partial_monotonic;
    result->event_value_ok = events.value_ok;
    result->terminal_ordering_ok =
        events.final_events == 1 && events.completed_events == 1 &&
        events.error_events == 0 && events.cancelled_events == 0;
    result->token_events = events.token_events;
    result->partial_events = events.partial_events;
    result->final_events = events.final_events;
    result->completed_events = events.completed_events;
    result->truncated_events = events.truncated_events;
    result->tokens = events.tokens;
    events.tokens.values = NULL;
    events.tokens.count = events.tokens.capacity = 0;
    result->event_signature = events.signature.values;
    events.signature.values = NULL;
    events.signature.count = events.signature.capacity = 0;

    result->metrics.struct_size = sizeof(result->metrics);
    result->metrics.api_version = VOXTRAL_API_VERSION;
    if (voxtral_stream_get_metrics(stream, &result->metrics) !=
        VOXTRAL_STATUS_OK) {
        fprintf(stderr, "metrics query failed\n");
        event_summary_free(&events);
        return 0;
    }
    if (result->metrics.token_events != result->token_events ||
        result->metrics.audio_samples_accepted != pcm->count) {
        fprintf(stderr,
                "metrics mismatch: metric tokens=%" PRIu64
                " polled=%" PRIu64 " metric samples=%" PRIu64
                " pcm=%zu\n",
                result->metrics.token_events, result->token_events,
                result->metrics.audio_samples_accepted, pcm->count);
        event_summary_free(&events);
        return 0;
    }
    if (events.final_event_text &&
        !is_prefix(events.final_event_text, result->transcript)) {
        fprintf(stderr, "final event is not a prefix of final query\n");
        event_summary_free(&events);
        return 0;
    }
    event_summary_free(&events);
    const int ok = result->finish_idempotent &&
           result->feed_after_finish_rejected &&
           result->structured_stream_error_ok &&
           result->queue_error_ok &&
           result->state_completed &&
           result->final_query_ok &&
           result->event_ordering_ok &&
           result->partial_monotonic &&
           result->event_value_ok &&
           result->terminal_ordering_ok &&
           result->samples_consumed == result->samples_offered;
    if (!ok) {
        fprintf(stderr,
                "run checks failed: finish=%d feedAfter=%d error=%d "
                "queueError=%d state=%d final=%d "
                "ordering=%d partial=%d value=%d terminal=%d consumed=%" PRIu64
                " offered=%" PRIu64 "\n",
                result->finish_idempotent,
                result->feed_after_finish_rejected,
                result->structured_stream_error_ok,
                result->queue_error_ok,
                result->state_completed,
                result->final_query_ok,
                result->event_ordering_ok,
                result->partial_monotonic,
                result->event_value_ok,
                result->terminal_ordering_ok,
                result->samples_consumed,
                result->samples_offered);
    }
    return ok;
}

static uint64_t current_rss_kib(void) {
#if defined(__linux__)
    FILE * file = fopen("/proc/self/status", "r");
    char line[256];
    if (!file) return 0;
    while (fgets(line, sizeof(line), file)) {
        uint64_t value = 0;
        if (sscanf(line, "VmRSS: %" SCNu64 " kB", &value) == 1) {
            fclose(file);
            return value;
        }
    }
    fclose(file);
#endif
    return 0;
}

static uint64_t current_vram_bytes(void) {
#if defined(__linux__)
    glob_t matches;
    memset(&matches, 0, sizeof(matches));
    if (glob("/sys/class/drm/card*/device/mem_info_vram_used",
             0, NULL, &matches) != 0) {
        globfree(&matches);
        return 0;
    }
    uint64_t total = 0;
    for (size_t i = 0; i < matches.gl_pathc; ++i) {
        FILE * file = fopen(matches.gl_pathv[i], "r");
        uint64_t value = 0;
        if (file && fscanf(file, "%" SCNu64, &value) == 1) total += value;
        if (file) fclose(file);
    }
    globfree(&matches);
    return total;
#else
    return 0;
#endif
}

static uint64_t range_u64(const uint64_t * values, size_t count) {
    if (count == 0) return 0;
    uint64_t low = values[0];
    uint64_t high = values[0];
    for (size_t i = 1; i < count; ++i) {
        if (values[i] < low) low = values[i];
        if (values[i] > high) high = values[i];
    }
    return high - low;
}

static int reset_is_pristine(voxtral_stream * stream) {
    if (voxtral_stream_reset(stream) != VOXTRAL_STATUS_OK ||
        voxtral_stream_get_state(stream) != VOXTRAL_STREAM_CREATED) return 0;
    voxtral_event event;
    memset(&event, 0, sizeof(event));
    event.struct_size = sizeof(event);
    event.api_version = VOXTRAL_API_VERSION;
    if (voxtral_stream_poll_event(stream, &event) !=
        VOXTRAL_STATUS_NOT_READY) return 0;
    size_t required = 99;
    if (voxtral_stream_get_final_text(stream, NULL, 0, &required) !=
            VOXTRAL_STATUS_NOT_READY ||
        required != 0) return 0;
    voxtral_stream_metrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    metrics.struct_size = sizeof(metrics);
    metrics.api_version = VOXTRAL_API_VERSION;
    return voxtral_stream_get_metrics(stream, &metrics) ==
               VOXTRAL_STATUS_OK &&
           metrics.audio_samples_accepted == 0 &&
           metrics.encoder_frames == 0 &&
           metrics.adapter_groups == 0 &&
           metrics.decoder_steps == 0 &&
           metrics.token_events == 0 &&
           metrics.partial_events == 0 &&
           metrics.decoder_kv_wraps == 0 &&
           metrics.decoder_kv_evictions == 0 &&
           metrics.decoder_kv_bytes_moved == 0;
}

static int test_stream_parameter_validation(voxtral_context * context) {
    voxtral_stream * stream = (voxtral_stream *) (uintptr_t) 1u;
    voxtral_stream_params params = voxtral_stream_default_params();

    params.struct_size = sizeof(uint32_t);
    if (voxtral_stream_create(context, &params, &stream) !=
            VOXTRAL_STATUS_INVALID_ARGUMENT ||
        stream != NULL) return 0;

    params = voxtral_stream_default_params();
    params.api_version = 0x02000000u;
    stream = (voxtral_stream *) (uintptr_t) 1u;
    if (voxtral_stream_create(context, &params, &stream) !=
            VOXTRAL_STATUS_INVALID_ARGUMENT ||
        stream != NULL) return 0;

    params = voxtral_stream_default_params();
    params.audio.struct_size = sizeof(uint32_t);
    if (voxtral_stream_create(context, &params, &stream) !=
            VOXTRAL_STATUS_INVALID_ARGUMENT ||
        stream != NULL) return 0;

    params = voxtral_stream_default_params();
    params.audio.api_version = 0x02000000u;
    if (voxtral_stream_create(context, &params, &stream) !=
            VOXTRAL_STATUS_INVALID_ARGUMENT ||
        stream != NULL) return 0;

    params = voxtral_stream_default_params();
    params.audio.sample_rate = 8000;
    if (voxtral_stream_create(context, &params, &stream) !=
            VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT ||
        stream != NULL) return 0;
    voxtral_error_info unsupported_error;
    memset(&unsupported_error, 0, sizeof(unsupported_error));
    unsupported_error.struct_size = sizeof(unsupported_error);
    unsupported_error.api_version = VOXTRAL_API_VERSION;
    if (voxtral_context_get_last_error(context, &unsupported_error) !=
            VOXTRAL_STATUS_OK ||
        unsupported_error.status !=
            VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT ||
        unsupported_error.message[0] == '\0') return 0;

    params = voxtral_stream_default_params();
    params.audio.channels = 2;
    if (voxtral_stream_create(context, &params, &stream) !=
            VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT ||
        stream != NULL) return 0;

    params = voxtral_stream_default_params();
    params.audio.format = (voxtral_audio_format) 99;
    if (voxtral_stream_create(context, &params, &stream) !=
            VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT ||
        stream != NULL) return 0;

    params = voxtral_stream_default_params();
    params.event_queue_capacity = 0;
    if (voxtral_stream_create(context, &params, &stream) !=
            VOXTRAL_STATUS_INVALID_ARGUMENT ||
        stream != NULL) return 0;

    params = voxtral_stream_default_params();
    params.event_queue_capacity =
        VOXTRAL_EVENT_QUEUE_CAPACITY_MAX + 1u;
    if (voxtral_stream_create(context, &params, &stream) !=
            VOXTRAL_STATUS_INVALID_ARGUMENT ||
        stream != NULL) return 0;

    params = voxtral_stream_default_params();
    params.flags = 1u;
    if (voxtral_stream_create(context, &params, &stream) !=
            VOXTRAL_STATUS_INVALID_ARGUMENT ||
        stream != NULL) return 0;

    /* A v1 header-only prefix receives all current defaults. */
    params = voxtral_stream_default_params();
    params.struct_size = sizeof(uint32_t) * 2u;
    if (voxtral_stream_create(context, &params, &stream) !=
            VOXTRAL_STATUS_OK ||
        !stream) return 0;
    voxtral_stream_destroy(stream);
    stream = NULL;

    /* A future caller's unknown trailing field is ignored. */
    future_stream_params future;
    memset(&future, 0x5a, sizeof(future));
    future.current = voxtral_stream_default_params();
    future.current.struct_size = sizeof(future);
    if (voxtral_stream_create(context, &future.current, &stream) !=
            VOXTRAL_STATUS_OK ||
        !stream) return 0;
    voxtral_stream_destroy(stream);

    voxtral_error_info error;
    memset(&error, 0, sizeof(error));
    error.struct_size = sizeof(error);
    error.api_version = VOXTRAL_API_VERSION;
    return voxtral_context_get_last_error(context, &error) ==
               VOXTRAL_STATUS_OK &&
           error.status == VOXTRAL_STATUS_OK;
}

static int create_default_stream(
    voxtral_context * context,
    uint32_t event_capacity,
    voxtral_stream ** out_stream)
{
    voxtral_stream_params params = voxtral_stream_default_params();
    params.event_queue_capacity = event_capacity;
    return voxtral_stream_create(context, &params, out_stream) ==
               VOXTRAL_STATUS_OK &&
           *out_stream != NULL;
}

static int test_destroy_states(
    voxtral_context * context,
    const pcm_data * pcm,
    destroy_summary * summary)
{
    memset(summary, 0, sizeof(*summary));
    voxtral_stream_destroy(NULL);
    summary->null_safe = 1;

    voxtral_stream * stream = NULL;
    if (!create_default_stream(context, 4096, &stream)) return 0;
    voxtral_stream_destroy(stream);
    summary->created = 1;

    stream = NULL;
    if (!create_default_stream(context, 4096, &stream)) return 0;
    size_t consumed = 0;
    const size_t active_samples = pcm->count < 1280u ? pcm->count : 1280u;
    if (voxtral_stream_feed_pcm16(
            stream, pcm->samples, active_samples, &consumed) !=
            VOXTRAL_STATUS_OK ||
        consumed != active_samples) {
        voxtral_stream_destroy(stream);
        return 0;
    }
    voxtral_stream_destroy(stream);
    summary->active = 1;

    stream = NULL;
    if (!create_default_stream(context, 4096, &stream)) return 0;
    if (voxtral_stream_cancel(stream) != VOXTRAL_STATUS_OK) {
        voxtral_stream_destroy(stream);
        return 0;
    }
    voxtral_stream_destroy(stream);
    summary->cancelled = 1;

    stream = NULL;
    if (!create_default_stream(context, 4096, &stream)) return 0;
    if (voxtral_stream_finish(stream) != VOXTRAL_STATUS_OK ||
        voxtral_stream_get_state(stream) != VOXTRAL_STREAM_COMPLETED) {
        voxtral_stream_destroy(stream);
        return 0;
    }
    voxtral_stream_destroy(stream);
    summary->completed = 1;

    stream = NULL;
    if (!create_default_stream(context, 8, &stream)) return 0;
    size_t offset = 0;
    int queue_full = 0;
    while (offset < pcm->count && !queue_full) {
        const size_t requested =
            pcm->count - offset < 1280u ? pcm->count - offset : 1280u;
        consumed = 0;
        const voxtral_status status = voxtral_stream_feed_pcm16(
            stream, pcm->samples + offset, requested, &consumed);
        if (consumed > requested) {
            voxtral_stream_destroy(stream);
            return 0;
        }
        offset += consumed;
        if (status == VOXTRAL_STATUS_QUEUE_FULL) {
            queue_full = 1;
        } else if (status != VOXTRAL_STATUS_OK || consumed != requested) {
            voxtral_stream_destroy(stream);
            return 0;
        }
    }
    summary->under_backpressure = queue_full;
    summary->finish_under_backpressure =
        queue_full &&
        voxtral_stream_finish(stream) == VOXTRAL_STATUS_QUEUE_FULL &&
        voxtral_stream_get_state(stream) == VOXTRAL_STREAM_ACTIVE;
    voxtral_stream_destroy(stream);

    stream = NULL;
    if (!create_default_stream(context, 8, &stream)) return 0;
    offset = 0;
    queue_full = 0;
    while (offset < pcm->count && !queue_full) {
        const size_t requested =
            pcm->count - offset < 1280u ? pcm->count - offset : 1280u;
        consumed = 0;
        const voxtral_status status = voxtral_stream_feed_pcm16(
            stream, pcm->samples + offset, requested, &consumed);
        if (consumed > requested) {
            voxtral_stream_destroy(stream);
            return 0;
        }
        offset += consumed;
        if (status == VOXTRAL_STATUS_QUEUE_FULL) {
            queue_full = 1;
        } else if (status != VOXTRAL_STATUS_OK || consumed != requested) {
            voxtral_stream_destroy(stream);
            return 0;
        }
    }
    event_summary cancelled_events;
    event_summary_init(&cancelled_events);
    summary->cancel_under_backpressure =
        queue_full &&
        voxtral_stream_cancel(stream) == VOXTRAL_STATUS_OK &&
        voxtral_stream_get_state(stream) == VOXTRAL_STREAM_CANCELLED &&
        drain_events(stream, &cancelled_events) &&
        cancelled_events.cancelled_events == 1 &&
        cancelled_events.final_events == 0 &&
        cancelled_events.completed_events == 0;
    event_summary_free(&cancelled_events);
    voxtral_stream_destroy(stream);

    stream = NULL;
    if (!create_default_stream(context, 4096, &stream)) return 0;
    voxtral_stream_destroy(stream);
    summary->lease_released = 1;
    return summary->null_safe && summary->created && summary->active &&
           summary->cancelled && summary->completed &&
           summary->under_backpressure &&
           summary->finish_under_backpressure &&
           summary->cancel_under_backpressure &&
           summary->lease_released;
}

static int run_lifecycle_edges(
    voxtral_context * context,
    voxtral_stream * stream,
    const pcm_data * pcm,
    const voxtral_stream_params * params,
    int * second_stream_rejected,
    int * cancel_idempotent,
    int * duration_limit_structured,
    int * reset_active_rejected,
    int * reset_created,
    int * no_cancel_final)
{
    voxtral_stream * second = (voxtral_stream *) (uintptr_t) 1u;
    *second_stream_rejected =
        voxtral_stream_create(context, params, &second) ==
            VOXTRAL_STATUS_INVALID_STATE &&
        second == NULL;

    if (!reset_is_pristine(stream)) return 0;
    int16_t sample = 0;
    size_t consumed = 99;
    const voxtral_status duration_status = voxtral_stream_feed_pcm16(
        stream, &sample,
        (size_t) VOXTRAL_STREAM_MAX_AUDIO_SAMPLES + 1u,
        &consumed);
    voxtral_error_info duration_error;
    memset(&duration_error, 0, sizeof(duration_error));
    duration_error.struct_size = sizeof(duration_error);
    duration_error.api_version = VOXTRAL_API_VERSION;
    *duration_limit_structured =
        duration_status == VOXTRAL_STATUS_INVALID_ARGUMENT &&
        consumed == 0 &&
        voxtral_stream_get_state(stream) == VOXTRAL_STREAM_CREATED &&
        voxtral_stream_get_last_error(stream, &duration_error) ==
            VOXTRAL_STATUS_OK &&
        duration_error.status == VOXTRAL_STATUS_INVALID_ARGUMENT &&
        duration_error.message[0] != '\0';

    consumed = 0;
    const size_t n = pcm->count < 1280u ? pcm->count : 1280u;
    if (voxtral_stream_feed_pcm16(
            stream, pcm->samples, n, &consumed) != VOXTRAL_STATUS_OK ||
        consumed != n) return 0;
    *reset_active_rejected =
        voxtral_stream_reset(stream) == VOXTRAL_STATUS_INVALID_STATE &&
        voxtral_stream_get_state(stream) == VOXTRAL_STREAM_ACTIVE;
    const voxtral_status cancel1 = voxtral_stream_cancel(stream);
    const voxtral_status cancel2 = voxtral_stream_cancel(stream);
    *cancel_idempotent =
        cancel1 == VOXTRAL_STATUS_OK && cancel2 == VOXTRAL_STATUS_OK &&
        voxtral_stream_get_state(stream) == VOXTRAL_STREAM_CANCELLED;

    consumed = 99;
    if (voxtral_stream_feed_pcm16(stream, &sample, 1, &consumed) !=
            VOXTRAL_STATUS_INVALID_STATE ||
        consumed != 0 ||
        voxtral_stream_finish(stream) != VOXTRAL_STATUS_OK) return 0;

    event_summary events;
    event_summary_init(&events);
    if (!drain_events(stream, &events)) {
        event_summary_free(&events);
        return 0;
    }
    *no_cancel_final =
        events.cancelled_events == 1 &&
        events.final_events == 0 &&
        events.completed_events == 0;
    event_summary_free(&events);

    if (!reset_is_pristine(stream)) return 0;
    *reset_created =
        voxtral_stream_reset(stream) == VOXTRAL_STATUS_OK &&
        voxtral_stream_get_state(stream) == VOXTRAL_STREAM_CREATED;
    return *second_stream_rejected && *cancel_idempotent &&
           *duration_limit_structured &&
           *reset_active_rejected &&
           *reset_created && *no_cancel_final;
}

static void json_string(const char * text) {
    putchar('"');
    for (const unsigned char * p = (const unsigned char *) text; *p; ++p) {
        switch (*p) {
            case '"':  fputs("\\\"", stdout); break;
            case '\\': fputs("\\\\", stdout); break;
            case '\b': fputs("\\b", stdout); break;
            case '\f': fputs("\\f", stdout); break;
            case '\n': fputs("\\n", stdout); break;
            case '\r': fputs("\\r", stdout); break;
            case '\t': fputs("\\t", stdout); break;
            default:
                if (*p < 0x20u) printf("\\u%04x", (unsigned) *p);
                else putchar((int) *p);
                break;
        }
    }
    putchar('"');
}

static int parse_u32(const char * value, uint32_t * out) {
    char * end = NULL;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (!value[0] || !end || *end != '\0' || parsed > UINT32_MAX) return 0;
    *out = (uint32_t) parsed;
    return 1;
}

static int parse_size(const char * value, size_t * out) {
    char * end = NULL;
    const unsigned long long parsed = strtoull(value, &end, 10);
    if (!value[0] || !end || *end != '\0' || parsed == 0 ||
        parsed > SIZE_MAX) return 0;
    *out = (size_t) parsed;
    return 1;
}

static int parse_options(int argc, char ** argv, options * opts) {
    memset(opts, 0, sizeof(*opts));
    opts->chunk_samples = 1280;
    opts->event_capacity = 4096;
    opts->iterations = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            opts->model_path = argv[++i];
        } else if (strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            opts->audio_path = argv[++i];
        } else if (strcmp(argv[i], "--chunk-samples") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &opts->chunk_samples)) return 0;
        } else if (strcmp(argv[i], "--event-capacity") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &opts->event_capacity)) return 0;
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &opts->iterations) ||
                opts->iterations == 0) return 0;
        } else if (strcmp(argv[i], "--random-seed") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &opts->random_seed)) return 0;
            opts->random_chunks = 1;
        } else if (strcmp(argv[i], "--synthetic-seconds") == 0 &&
                   i + 1 < argc) {
            if (!parse_u32(argv[++i], &opts->synthetic_seconds) ||
                opts->synthetic_seconds == 0) return 0;
        } else if (strcmp(argv[i], "--delayed-consumer") == 0) {
            opts->delayed_consumer = 1;
        } else if (strcmp(argv[i], "--lifecycle") == 0) {
            opts->lifecycle = 1;
        } else if (strcmp(argv[i], "--paced") == 0) {
            opts->paced = 1;
        } else {
            return 0;
        }
    }
    return opts->model_path &&
           ((opts->audio_path != NULL) != (opts->synthetic_seconds != 0));
}

int main(int argc, char ** argv) {
    options opts;
    if (!parse_options(argc, argv, &opts)) {
        fprintf(stderr,
                "usage: %s --model FILE --audio PCM16.wav "
                "[--chunk-samples N] [--event-capacity N] "
                "[--iterations N] [--random-seed N] [--paced] "
                "[--delayed-consumer] [--lifecycle]\n"
                "       %s --model FILE --synthetic-seconds N "
                "[the same options]\n",
                argv[0],
                argv[0]);
        return 2;
    }

    pcm_data pcm = {0};
    if (opts.audio_path) {
        if (!load_pcm16_wav(opts.audio_path, &pcm)) return 2;
    } else if (!make_synthetic_pcm(opts.synthetic_seconds, &pcm)) {
        fprintf(stderr, "failed to allocate synthetic PCM\n");
        return 2;
    }

    voxtral_model * model = NULL;
    voxtral_model_params model_params = voxtral_model_default_params();
    model_params.backend = VOXTRAL_BACKEND_VULKAN;
    voxtral_status status =
        voxtral_model_load(opts.model_path, &model_params, &model);
    if (status != VOXTRAL_STATUS_OK) {
        fprintf(stderr, "model load failed: %s\n",
                voxtral_status_string(status));
        free(pcm.samples);
        return 1;
    }

    voxtral_context * context = NULL;
    voxtral_context_config context_config =
        voxtral_context_default_config();
    status = voxtral_context_create(model, &context_config, &context);
    if (status != VOXTRAL_STATUS_OK) {
        fprintf(stderr, "context create failed: %s\n",
                voxtral_status_string(status));
        voxtral_model_destroy(model);
        free(pcm.samples);
        return 1;
    }

    voxtral_capabilities caps;
    memset(&caps, 0, sizeof(caps));
    caps.struct_size = sizeof(caps);
    caps.api_version = VOXTRAL_API_VERSION;
    const int capabilities_ok =
        voxtral_context_get_capabilities(context, &caps) ==
            VOXTRAL_STATUS_OK &&
        caps.sample_rate == 16000 && caps.channels == 1 &&
        caps.audio_format == VOXTRAL_AUDIO_PCM_S16LE &&
        caps.supports_incremental == 1 &&
        caps.supports_reset == 1 &&
        caps.max_active_streams_per_context == 1;
    const int parameter_validation_ok =
        test_stream_parameter_validation(context);
    destroy_summary destroys;
    const int destroy_states_ok =
        test_destroy_states(context, &pcm, &destroys);

    voxtral_stream_params stream_params = voxtral_stream_default_params();
    stream_params.event_queue_capacity = opts.event_capacity;
    voxtral_stream * stream = NULL;
    status = voxtral_stream_create(context, &stream_params, &stream);
    if (status != VOXTRAL_STATUS_OK) {
        voxtral_error_info error;
        memset(&error, 0, sizeof(error));
        error.struct_size = sizeof(error);
        error.api_version = VOXTRAL_API_VERSION;
        (void) voxtral_context_get_last_error(context, &error);
        fprintf(stderr, "stream create failed: %s (%s)\n",
                voxtral_status_string(status), error.message);
        voxtral_context_destroy(context);
        voxtral_model_destroy(model);
        free(pcm.samples);
        return 1;
    }

    run_result reference;
    memset(&reference, 0, sizeof(reference));
    int all_runs_ok = 1;
    int token_consistency = 1;
    int transcript_consistency = 1;
    int reset_pristine = 1;
    uint64_t total_offered = 0;
    uint64_t total_consumed = 0;
    uint64_t total_retried = 0;
    uint64_t total_queue_full = 0;
    double total_feed_ms = 0.0;
    uint64_t total_feed_calls = 0;
    const size_t tail_window =
        opts.iterations < 20u ? opts.iterations : 20u;
    uint64_t * rss_tail = (uint64_t *) calloc(
        tail_window ? tail_window : 1u, sizeof(uint64_t));
    uint64_t * vram_tail = (uint64_t *) calloc(
        tail_window ? tail_window : 1u, sizeof(uint64_t));
    if (!rss_tail || !vram_tail) all_runs_ok = 0;

    for (uint32_t iteration = 0;
         all_runs_ok && iteration < opts.iterations;
         ++iteration) {
        run_result current;
        if (!run_once(stream, &pcm, &opts, &current)) {
            all_runs_ok = 0;
            break;
        }
        total_offered += current.samples_offered;
        total_consumed += current.samples_consumed;
        total_retried += current.samples_retried;
        total_queue_full += current.queue_full_returns;
        total_feed_ms += current.feed_call_ms_total;
        total_feed_calls += current.feed_calls;

        if (iteration == 0) {
            reference = current;
            memset(&current, 0, sizeof(current));
        } else {
            token_consistency =
                token_consistency &&
                tokens_equal(&reference.tokens, &current.tokens);
            transcript_consistency =
                transcript_consistency &&
                strcmp(reference.transcript, current.transcript) == 0;
            reference.metrics = current.metrics;
            run_result_free(&current);
        }

        if (iteration + 1u < opts.iterations) {
            reset_pristine = reset_pristine && reset_is_pristine(stream);
        }
        if (tail_window != 0 &&
            iteration >= opts.iterations - (uint32_t) tail_window) {
            const size_t index =
                iteration - (opts.iterations - (uint32_t) tail_window);
            rss_tail[index] = current_rss_kib();
            vram_tail[index] = current_vram_bytes();
        }
    }

    int second_stream_rejected = 0;
    int cancel_idempotent = 0;
    int duration_limit_structured = 0;
    int reset_active_rejected = 0;
    int reset_created = 0;
    int no_cancel_final = 0;
    int lifecycle_ok = 1;
    if (all_runs_ok && opts.lifecycle) {
        lifecycle_ok = run_lifecycle_edges(
            context, stream, &pcm, &stream_params,
            &second_stream_rejected, &cancel_idempotent,
            &duration_limit_structured,
            &reset_active_rejected,
            &reset_created, &no_cancel_final);
    }

    const uint64_t rss_tail_range = range_u64(rss_tail, tail_window);
    const uint64_t vram_tail_range = range_u64(vram_tail, tail_window);
    free(rss_tail);
    free(vram_tail);

    const int pass =
        all_runs_ok && capabilities_ok && parameter_validation_ok &&
        destroy_states_ok &&
        token_consistency &&
        transcript_consistency && reset_pristine && lifecycle_ok &&
        total_offered == total_consumed &&
        (!opts.delayed_consumer || total_queue_full > 0);

    printf("{\"ok\":%s", pass ? "true" : "false");
    printf(",\"iterations\":%u", opts.iterations);
    printf(",\"eventQueueCapacity\":%u", opts.event_capacity);
    printf(",\"delayedConsumer\":%s",
           opts.delayed_consumer ? "true" : "false");
    printf(",\"paced\":%s", opts.paced ? "true" : "false");
    printf(",\"randomSeed\":%u",
           opts.random_chunks ? opts.random_seed : 0u);
    printf(",\"samplesOffered\":%" PRIu64, total_offered);
    printf(",\"samplesConsumed\":%" PRIu64, total_consumed);
    printf(",\"samplesRetried\":%" PRIu64, total_retried);
    printf(",\"queueFullReturns\":%" PRIu64, total_queue_full);
    printf(",\"tokens\":[");
    for (size_t i = 0; i < reference.tokens.count; ++i) {
        if (i != 0) putchar(',');
        printf("%" PRId32, reference.tokens.values[i]);
    }
    printf("]");
    printf(",\"tokenCount\":%zu", reference.tokens.count);
    printf(",\"transcript\":");
    json_string(reference.transcript ? reference.transcript : "");
    printf(",\"eventSignature\":");
    json_string(reference.event_signature ? reference.event_signature : "");
    printf(",\"eventOrderingOk\":%s",
           reference.event_ordering_ok ? "true" : "false");
    printf(",\"partialMonotonic\":%s",
           reference.partial_monotonic ? "true" : "false");
    printf(",\"eventValueOk\":%s",
           reference.event_value_ok ? "true" : "false");
    printf(",\"terminalOrderingOk\":%s",
           reference.terminal_ordering_ok ? "true" : "false");
    printf(",\"finishIdempotent\":%s",
           reference.finish_idempotent ? "true" : "false");
    printf(",\"feedAfterFinishRejected\":%s",
           reference.feed_after_finish_rejected ? "true" : "false");
    printf(",\"structuredStreamErrorOk\":%s",
           reference.structured_stream_error_ok ? "true" : "false");
    printf(",\"queueErrorOk\":%s",
           reference.queue_error_ok ? "true" : "false");
    printf(",\"tokenEvents\":%" PRIu64, reference.token_events);
    printf(",\"partialEvents\":%" PRIu64, reference.partial_events);
    printf(",\"finalEvents\":%" PRIu64, reference.final_events);
    printf(",\"completedEvents\":%" PRIu64, reference.completed_events);
    printf(",\"truncatedEvents\":%" PRIu64, reference.truncated_events);
    printf(",\"tokenConsistency\":%s",
           token_consistency ? "true" : "false");
    printf(",\"transcriptConsistency\":%s",
           transcript_consistency ? "true" : "false");
    printf(",\"resetPristine\":%s",
           reset_pristine ? "true" : "false");
    printf(",\"capabilitiesOk\":%s",
           capabilities_ok ? "true" : "false");
    printf(",\"parameterValidationOk\":%s",
           parameter_validation_ok ? "true" : "false");
    printf(",\"destroy\":{\"nullSafe\":%s",
           destroys.null_safe ? "true" : "false");
    printf(",\"created\":%s", destroys.created ? "true" : "false");
    printf(",\"active\":%s", destroys.active ? "true" : "false");
    printf(",\"cancelled\":%s", destroys.cancelled ? "true" : "false");
    printf(",\"completed\":%s", destroys.completed ? "true" : "false");
    printf(",\"underBackpressure\":%s",
           destroys.under_backpressure ? "true" : "false");
    printf(",\"finishUnderBackpressure\":%s",
           destroys.finish_under_backpressure ? "true" : "false");
    printf(",\"cancelUnderBackpressure\":%s",
           destroys.cancel_under_backpressure ? "true" : "false");
    printf(",\"leaseReleased\":%s}",
           destroys.lease_released ? "true" : "false");
    printf(",\"lifecycle\":{\"secondStreamRejected\":%s",
           second_stream_rejected ? "true" : "false");
    printf(",\"cancelIdempotent\":%s",
           cancel_idempotent ? "true" : "false");
    printf(",\"durationLimitStructured\":%s",
           duration_limit_structured ? "true" : "false");
    printf(",\"resetActiveRejected\":%s",
           reset_active_rejected ? "true" : "false");
    printf(",\"resetFromCreated\":%s",
           reset_created ? "true" : "false");
    printf(",\"cancelEmitsNoFinal\":%s}",
           no_cancel_final ? "true" : "false");
    printf(",\"metrics\":{\"audioSamplesAccepted\":%" PRIu64,
           reference.metrics.audio_samples_accepted);
    printf(",\"audioDurationMs\":%" PRIu64,
           reference.metrics.audio_duration_ms);
    printf(",\"encoderFrames\":%" PRIu64,
           reference.metrics.encoder_frames);
    printf(",\"adapterGroups\":%" PRIu64,
           reference.metrics.adapter_groups);
    printf(",\"decoderSteps\":%" PRIu64,
           reference.metrics.decoder_steps);
    printf(",\"decoderKvWraps\":%" PRIu64,
           reference.metrics.decoder_kv_wraps);
    printf(",\"decoderKvEvictions\":%" PRIu64,
           reference.metrics.decoder_kv_evictions);
    printf(",\"decoderKvBytesMoved\":%" PRIu64,
           reference.metrics.decoder_kv_bytes_moved);
    printf(",\"pipelineRtf\":%.9f",
           reference.metrics.pipeline_rtf);
    printf(",\"backlogFinalMs\":%.9f",
           reference.metrics.backlog_final_ms);
    printf(",\"backlogSlopeMsPerS\":%.9f}",
           reference.metrics.backlog_slope_ms_per_s);
    printf(",\"feedCallMeanMs\":%.9f",
           total_feed_calls ? total_feed_ms / (double) total_feed_calls : 0.0);
    printf(",\"rssTailRangeKiB\":%" PRIu64, rss_tail_range);
    printf(",\"vramTailRangeBytes\":%" PRIu64, vram_tail_range);
    printf("}\n");

    run_result_free(&reference);
    voxtral_stream_destroy(stream);
    voxtral_context_destroy(context);
    voxtral_model_destroy(model);
    free(pcm.samples);
    return pass ? 0 : 1;
}
