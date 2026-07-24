#include "batch-transcription.h"

#include <algorithm>
#include <exception>

namespace voxtral::server {
namespace {

bool drain_events(
    EngineStream & stream,
    BatchTranscriptionResult & result,
    const std::shared_ptr<Cancellation> & cancellation)
{
    for (;;) {
        if (cancellation->requested()) {
            (void) stream.cancel();
            result.cancelled = true;
            result.status = VOXTRAL_STATUS_CANCELLED;
            return false;
        }
        voxtral_event event{};
        const voxtral_status status = stream.poll(event);
        if (status == VOXTRAL_STATUS_NOT_READY) {
            return true;
        }
        if (status != VOXTRAL_STATUS_OK) {
            result.status = status;
            result.detail = stream.last_error();
            return false;
        }
        if (event.type == VOXTRAL_EVENT_TOKEN) {
            ++result.token_count;
        } else if (event.type == VOXTRAL_EVENT_ERROR) {
            result.status = event.status;
            result.detail.assign(event.text, event.text_length);
            result.cancelled =
                event.status == VOXTRAL_STATUS_CANCELLED;
            return false;
        }
    }
}

bool resume_after_backpressure(
    EngineStream & stream,
    BatchTranscriptionResult & result,
    const std::shared_ptr<Cancellation> & cancellation)
{
    for (;;) {
        if (!drain_events(stream, result, cancellation)) {
            return false;
        }
        std::size_t consumed = 1;
        const voxtral_status status = stream.feed(nullptr, 0, &consumed);
        if (consumed != 0) {
            result.status = VOXTRAL_STATUS_INTERNAL_ERROR;
            result.detail = "zero-length resume consumed audio";
            return false;
        }
        if (status == VOXTRAL_STATUS_QUEUE_FULL) {
            continue;
        }
        if (status != VOXTRAL_STATUS_OK) {
            result.status = status;
            result.detail = stream.last_error();
            return false;
        }
        return drain_events(stream, result, cancellation);
    }
}

} // namespace

BatchTranscriptionResult transcribe_batch(
    Engine & engine,
    const std::vector<std::int16_t> & samples,
    const std::shared_ptr<Cancellation> & cancellation)
{
    BatchTranscriptionResult result;
    try {
        auto stream = engine.create_stream();
        constexpr std::size_t kBatchChunkSamples = 16000;
        std::size_t offset = 0;
        while (offset < samples.size()) {
            if (cancellation->requested()) {
                (void) stream->cancel();
                result.cancelled = true;
                result.status = VOXTRAL_STATUS_CANCELLED;
                return result;
            }
            const std::size_t count =
                std::min(kBatchChunkSamples, samples.size() - offset);
            std::size_t chunk_offset = 0;
            while (chunk_offset < count) {
                std::size_t consumed = 0;
                const voxtral_status status = stream->feed(
                    samples.data() + offset + chunk_offset,
                    count - chunk_offset,
                    &consumed);
                chunk_offset += consumed;
                if (!drain_events(*stream, result, cancellation)) {
                    return result;
                }
                if (status == VOXTRAL_STATUS_QUEUE_FULL) {
                    if (!resume_after_backpressure(
                            *stream, result, cancellation)) {
                        return result;
                    }
                    continue;
                }
                if (status != VOXTRAL_STATUS_OK ||
                    (consumed == 0 && chunk_offset < count)) {
                    result.status =
                        status == VOXTRAL_STATUS_OK
                            ? VOXTRAL_STATUS_INTERNAL_ERROR
                            : status;
                    result.detail = stream->last_error();
                    return result;
                }
            }
            offset += count;
        }

        for (;;) {
            if (cancellation->requested()) {
                (void) stream->cancel();
                result.cancelled = true;
                result.status = VOXTRAL_STATUS_CANCELLED;
                return result;
            }
            const voxtral_status status = stream->finish();
            if (status == VOXTRAL_STATUS_QUEUE_FULL) {
                if (!resume_after_backpressure(
                        *stream, result, cancellation)) {
                    return result;
                }
                continue;
            }
            if (status != VOXTRAL_STATUS_OK) {
                result.status = status;
                result.detail = stream->last_error();
                return result;
            }
            break;
        }
        if (!drain_events(*stream, result, cancellation)) {
            return result;
        }
        if (stream->state() != VOXTRAL_STREAM_COMPLETED) {
            result.status = VOXTRAL_STATUS_INTERNAL_ERROR;
            result.detail = "public stream did not reach COMPLETED";
            return result;
        }
        result.text = stream->final_text();
        result.metrics = stream->metrics();
        result.ok = true;
        result.status = VOXTRAL_STATUS_OK;
        return result;
    } catch (const EngineError & error) {
        result.status = error.status();
        result.detail = error.what();
    } catch (const std::bad_alloc &) {
        result.status = VOXTRAL_STATUS_OUT_OF_MEMORY;
        result.detail = "server allocation failed";
    } catch (const std::exception & error) {
        result.status = VOXTRAL_STATUS_INTERNAL_ERROR;
        result.detail = error.what();
    }
    return result;
}

} // namespace voxtral::server
