#ifndef VOXTRAL_SERVER_WAV_READER_H
#define VOXTRAL_SERVER_WAV_READER_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace voxtral::server {

struct AudioData {
    std::vector<std::int16_t> samples;
    std::uint64_t duration_ms = 0;
};

struct AudioParseResult {
    bool ok = false;
    AudioData audio;
    std::string code;
    std::string message;
};

AudioParseResult parse_pcm16_wav(std::string_view body);
AudioParseResult parse_raw_pcm16(
    std::string_view body,
    std::string_view format,
    std::string_view sample_rate,
    std::string_view channels);

} // namespace voxtral::server

#endif
