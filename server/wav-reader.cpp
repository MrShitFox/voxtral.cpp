#include "wav-reader.h"

#include <cstring>
#include <limits>

namespace voxtral::server {
namespace {

std::uint16_t read_u16le(const unsigned char * bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8u);
}

std::uint32_t read_u32le(const unsigned char * bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8u) |
           (static_cast<std::uint32_t>(bytes[2]) << 16u) |
           (static_cast<std::uint32_t>(bytes[3]) << 24u);
}

AudioParseResult failure(std::string code, std::string message) {
    AudioParseResult result;
    result.code = std::move(code);
    result.message = std::move(message);
    return result;
}

AudioParseResult make_audio(const unsigned char * bytes, std::size_t byte_count) {
    if (byte_count == 0 || (byte_count & 1u) != 0) {
        return failure(
            "invalid_argument",
            "PCM audio must contain a non-zero even number of bytes.");
    }
    if (byte_count / 2u > std::numeric_limits<std::size_t>::max() /
            sizeof(std::int16_t)) {
        return failure("invalid_argument", "PCM audio size overflows.");
    }

    AudioParseResult result;
    result.ok = true;
    result.audio.samples.resize(byte_count / 2u);
    for (std::size_t i = 0; i < result.audio.samples.size(); ++i) {
        result.audio.samples[i] = static_cast<std::int16_t>(
            read_u16le(bytes + i * 2u));
    }
    result.audio.duration_ms =
        static_cast<std::uint64_t>(result.audio.samples.size()) * 1000u /
        16000u;
    return result;
}

} // namespace

AudioParseResult parse_pcm16_wav(std::string_view body) {
    const auto * bytes =
        reinterpret_cast<const unsigned char *>(body.data());
    const std::size_t body_size = body.size();
    if (body_size < 12 ||
        std::memcmp(bytes, "RIFF", 4) != 0 ||
        std::memcmp(bytes + 8, "WAVE", 4) != 0) {
        return failure(
            "invalid_argument",
            "The request body is not a valid RIFF/WAVE file.");
    }

    const std::uint64_t riff_size = read_u32le(bytes + 4);
    const std::uint64_t riff_end_u64 = riff_size + 8u;
    if (riff_end_u64 < 12u || riff_end_u64 > body_size ||
        riff_end_u64 > std::numeric_limits<std::size_t>::max()) {
        return failure("invalid_argument", "The WAV RIFF size is invalid.");
    }
    const std::size_t riff_end = static_cast<std::size_t>(riff_end_u64);

    bool have_fmt = false;
    bool have_data = false;
    std::uint16_t audio_format = 0;
    std::uint16_t channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint32_t byte_rate = 0;
    std::uint16_t block_align = 0;
    std::uint16_t bits_per_sample = 0;
    const unsigned char * data = nullptr;
    std::size_t data_size = 0;

    std::size_t offset = 12;
    while (offset < riff_end) {
        if (riff_end - offset < 8) {
            return failure("invalid_argument", "The WAV chunk header is truncated.");
        }
        const unsigned char * header = bytes + offset;
        const std::uint64_t chunk_size = read_u32le(header + 4);
        offset += 8;
        if (chunk_size > riff_end - offset) {
            return failure("invalid_argument", "The WAV chunk size is invalid.");
        }
        const std::size_t size = static_cast<std::size_t>(chunk_size);

        if (std::memcmp(header, "fmt ", 4) == 0 && !have_fmt) {
            if (size < 16) {
                return failure("invalid_argument", "The WAV fmt chunk is too small.");
            }
            audio_format = read_u16le(bytes + offset);
            channels = read_u16le(bytes + offset + 2);
            sample_rate = read_u32le(bytes + offset + 4);
            byte_rate = read_u32le(bytes + offset + 8);
            block_align = read_u16le(bytes + offset + 12);
            bits_per_sample = read_u16le(bytes + offset + 14);
            have_fmt = true;
        } else if (std::memcmp(header, "data", 4) == 0 && !have_data) {
            data = bytes + offset;
            data_size = size;
            have_data = true;
        }

        offset += size;
        if ((size & 1u) != 0) {
            if (offset == riff_end) {
                return failure(
                    "invalid_argument",
                    "The WAV chunk padding byte is missing.");
            }
            ++offset;
        }
    }

    if (!have_fmt || !have_data) {
        return failure(
            "invalid_argument",
            "The WAV file must contain fmt and data chunks.");
    }
    if (audio_format != 1) {
        return failure(
            "unsupported_audio_format",
            "Only uncompressed integer PCM WAV is supported.");
    }
    if (channels != 1 || sample_rate != 16000 || bits_per_sample != 16 ||
        block_align != 2 || byte_rate != 32000) {
        return failure(
            "unsupported_audio_format",
            "WAV audio must be signed PCM16 little-endian, mono, 16000 Hz.");
    }
    return make_audio(data, data_size);
}

AudioParseResult parse_raw_pcm16(
    std::string_view body,
    std::string_view format,
    std::string_view sample_rate,
    std::string_view channels)
{
    if (format != "pcm_s16le" || sample_rate != "16000" || channels != "1") {
        return failure(
            "unsupported_audio_format",
            "Raw audio requires pcm_s16le, 16000 Hz, and one channel.");
    }
    return make_audio(
        reinterpret_cast<const unsigned char *>(body.data()), body.size());
}

} // namespace voxtral::server
