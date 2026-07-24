#ifndef VOXTRAL_SERVER_BATCH_TRANSCRIPTION_H
#define VOXTRAL_SERVER_BATCH_TRANSCRIPTION_H

#include "engine.h"
#include "gpu-lease.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace voxtral::server {

struct BatchTranscriptionResult {
    bool ok = false;
    bool cancelled = false;
    voxtral_status status = VOXTRAL_STATUS_INTERNAL_ERROR;
    std::string text;
    std::string detail;
    voxtral_stream_metrics metrics{};
    std::uint64_t token_count = 0;
};

BatchTranscriptionResult transcribe_batch(
    Engine & engine,
    const std::vector<std::int16_t> & samples,
    const std::shared_ptr<Cancellation> & cancellation);

} // namespace voxtral::server

#endif
