#include <voxtral.h>
#include <voxtral-stream.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SAMPLE_RATE = 16000,
    CHUNK_SAMPLES = 1280
};

typedef struct pcm_audio {
    int16_t * samples;
    size_t count;
} pcm_audio;

static uint16_t u16le(const unsigned char * p) {
    return (uint16_t) p[0] | ((uint16_t) p[1] << 8u);
}

static uint32_t u32le(const unsigned char * p) {
    return (uint32_t) p[0] |
           ((uint32_t) p[1] << 8u) |
           ((uint32_t) p[2] << 16u) |
           ((uint32_t) p[3] << 24u);
}

/* Minimal RIFF reader for PCM16/mono/16 kHz example input. */
static int load_wav(const char * path, pcm_audio * out) {
    FILE * file = fopen(path, "rb");
    unsigned char riff[12];
    unsigned char * bytes = NULL;
    size_t byte_count = 0;
    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    int have_fmt = 0;
    if (!file || fread(riff, 1, sizeof(riff), file) != sizeof(riff) ||
        memcmp(riff, "RIFF", 4) != 0 ||
        memcmp(riff + 8, "WAVE", 4) != 0) {
        if (file) fclose(file);
        return 0;
    }
    for (;;) {
        unsigned char header[8];
        if (fread(header, 1, sizeof(header), file) != sizeof(header)) break;
        const uint32_t size = u32le(header + 4);
        if (memcmp(header, "fmt ", 4) == 0 && size >= 16) {
            unsigned char fmt[16];
            if (fread(fmt, 1, sizeof(fmt), file) != sizeof(fmt)) break;
            format = u16le(fmt);
            channels = u16le(fmt + 2);
            rate = u32le(fmt + 4);
            bits = u16le(fmt + 14);
            if (size > 16 && fseek(file, (long) (size - 16), SEEK_CUR) != 0)
                break;
            have_fmt = 1;
        } else if (memcmp(header, "data", 4) == 0) {
            if (size == 0 || (size & 1u) != 0) break;
            bytes = (unsigned char *) malloc(size);
            if (!bytes) break;
            if (fread(bytes, 1, size, file) != size) {
                free(bytes);
                bytes = NULL;
                break;
            }
            byte_count = size;
            break;
        } else if (fseek(file, (long) size, SEEK_CUR) != 0) {
            break;
        }
        if ((size & 1u) != 0 && fseek(file, 1, SEEK_CUR) != 0) break;
    }
    fclose(file);
    if (!have_fmt || !bytes || format != 1 || channels != 1 ||
        rate != SAMPLE_RATE || bits != 16) {
        free(bytes);
        return 0;
    }
    out->count = byte_count / 2u;
    out->samples = (int16_t *) malloc(out->count * sizeof(int16_t));
    if (!out->samples) {
        free(bytes);
        return 0;
    }
    for (size_t i = 0; i < out->count; ++i) {
        out->samples[i] = (int16_t) u16le(bytes + 2u * i);
    }
    free(bytes);
    return 1;
}

static int poll_all(voxtral_stream * stream, int * completed) {
    for (;;) {
        voxtral_event event = {0};
        event.struct_size = sizeof(event);
        event.api_version = VOXTRAL_API_VERSION;
        const voxtral_status status =
            voxtral_stream_poll_event(stream, &event);
        if (status == VOXTRAL_STATUS_NOT_READY) return 1;
        if (status != VOXTRAL_STATUS_OK) return 0;
        if (event.type == VOXTRAL_EVENT_PARTIAL_TEXT) {
            printf("\r%s", event.text);
            fflush(stdout);
        } else if (event.type == VOXTRAL_EVENT_FINAL_TEXT) {
            printf("\r%s\n", event.text);
        } else if (event.type == VOXTRAL_EVENT_ERROR) {
            fprintf(stderr, "stream event: %s: %s\n",
                    voxtral_status_string(event.status), event.text);
        } else if (event.type == VOXTRAL_EVENT_COMPLETED) {
            *completed = 1;
        }
    }
}

static voxtral_status drain_backpressure(
    voxtral_stream * stream,
    int * completed)
{
    for (;;) {
        if (!poll_all(stream, completed)) {
            return VOXTRAL_STATUS_INTERNAL_ERROR;
        }
        size_t consumed = 1;
        const voxtral_status status =
            voxtral_stream_feed_pcm16(stream, NULL, 0, &consumed);
        if (consumed != 0) return VOXTRAL_STATUS_INTERNAL_ERROR;
        if (status != VOXTRAL_STATUS_QUEUE_FULL) return status;
    }
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf AUDIO.wav\n", argv[0]);
        return 2;
    }

    pcm_audio audio = {0};
    if (!load_wav(argv[2], &audio)) {
        fprintf(stderr, "audio must be PCM16 mono %d Hz WAV\n", SAMPLE_RATE);
        return 2;
    }

    voxtral_model * model = NULL;
    voxtral_model_params model_params = voxtral_model_default_params();
    voxtral_status status =
        voxtral_model_load(argv[1], &model_params, &model);
    if (status != VOXTRAL_STATUS_OK) {
        fprintf(stderr, "model load: %s\n", voxtral_status_string(status));
        free(audio.samples);
        return 1;
    }

    voxtral_context * context = NULL;
    voxtral_context_config context_config =
        voxtral_context_default_config();
    status = voxtral_context_create(model, &context_config, &context);
    if (status != VOXTRAL_STATUS_OK) {
        fprintf(stderr, "context create: %s\n", voxtral_status_string(status));
        voxtral_model_destroy(model);
        free(audio.samples);
        return 1;
    }

    voxtral_stream * stream = NULL;
    voxtral_stream_params stream_params = voxtral_stream_default_params();
    status = voxtral_stream_create(context, &stream_params, &stream);
    if (status != VOXTRAL_STATUS_OK) {
        fprintf(stderr, "stream create: %s\n", voxtral_status_string(status));
        voxtral_context_destroy(context);
        voxtral_model_destroy(model);
        free(audio.samples);
        return 1;
    }

    int completed = 0;
    size_t offset = 0;
    while (offset < audio.count) {
        const size_t count = audio.count - offset < CHUNK_SAMPLES
            ? audio.count - offset : CHUNK_SAMPLES;
        size_t consumed = 0;
        status = voxtral_stream_feed_pcm16(
            stream, audio.samples + offset, count, &consumed);
        offset += consumed;
        if (status == VOXTRAL_STATUS_QUEUE_FULL) {
            status = drain_backpressure(stream, &completed);
            if (status != VOXTRAL_STATUS_OK) break;
            continue; /* retry only the unconsumed remainder */
        }
        if (status != VOXTRAL_STATUS_OK || consumed != count ||
            !poll_all(stream, &completed)) {
            break;
        }
    }

    if (offset == audio.count) {
        for (;;) {
            status = voxtral_stream_finish(stream);
            if (status != VOXTRAL_STATUS_QUEUE_FULL) break;
            status = drain_backpressure(stream, &completed);
            if (status != VOXTRAL_STATUS_OK) break;
        }
    }
    if (status == VOXTRAL_STATUS_OK) (void) poll_all(stream, &completed);

    size_t final_size = 0;
    char * final_text = NULL;
    if (status == VOXTRAL_STATUS_OK &&
        voxtral_stream_get_final_text(stream, NULL, 0, &final_size) ==
            VOXTRAL_STATUS_OK) {
        final_text = (char *) malloc(final_size);
        if (final_text &&
            voxtral_stream_get_final_text(
                stream, final_text, final_size, &final_size) ==
                VOXTRAL_STATUS_OK) {
            printf("final: %s\n", final_text);
        }
    }

    if (status != VOXTRAL_STATUS_OK || !completed) {
        voxtral_error_info error = {0};
        error.struct_size = sizeof(error);
        error.api_version = VOXTRAL_API_VERSION;
        (void) voxtral_stream_get_last_error(stream, &error);
        fprintf(stderr, "stream failed: %s: %s\n",
                voxtral_status_string(status), error.message);
    }

    free(final_text);
    voxtral_stream_destroy(stream);
    voxtral_context_destroy(context);
    voxtral_model_destroy(model);
    free(audio.samples);
    return status == VOXTRAL_STATUS_OK && completed ? 0 : 1;
}
