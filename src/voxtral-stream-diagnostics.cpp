// ============================================================================
// Streaming diagnostic output hashing — implementation.
//
// Opt-in (VOXTRAL_CAPTURE_OUTPUT_SHA) chunk-plan-invariant SHA-256 of the
// encoder / adapter device-ring output. New rows are read back in bounded
// slices and hashed immediately, so no utterance-sized tensor is retained and
// the default path performs zero diagnostic device->host traffic. See
// voxtral-stream-diagnostics.h (Sha256) and voxtral-stream-internal.h.
// ============================================================================

#include "voxtral-stream-internal.h"

bool capture_new_encoder_output_sha(voxtral_stream * s) {
    if (!s || !s->diagnostics.capture_output_sha || !s->decoder.incremental || !s->frontend.enc || !s->ctx) {
        return true;
    }
    const int64_t available = voxtral_encoder_stream_output_frames(s->frontend.enc);
    const int64_t count64 = available - s->diagnostics.encoder_sha_rows;
    if (count64 <= 0) return true;
    const int32_t capacity = voxtral_ctx_enc_out_ring_frames(s->ctx);
    if (count64 > capacity || count64 > INT32_MAX) {
        set_error(s, voxtral_status::backend_error,
                  "encoder SHA capture fell behind bounded output ring");
        return false;
    }
    const int32_t count = (int32_t) count64;
    s->diagnostics.output_sha_scratch.resize((size_t) count * VOXTRAL_ENC_DIM);
    if (!voxtral_ctx_read_enc_out_ring_internal(
            s->ctx, s->diagnostics.encoder_sha_rows, count,
            s->diagnostics.output_sha_scratch.data())) {
        set_error(s, voxtral_status::backend_error,
                  "encoder SHA diagnostic readback failed");
        return false;
    }
    const size_t bytes = s->diagnostics.output_sha_scratch.size() * sizeof(float);
    s->diagnostics.encoder_output_sha.update(s->diagnostics.output_sha_scratch.data(), bytes);
    s->diagnostics.output_sha_d2h_bytes += (int64_t) bytes;
    s->diagnostics.encoder_sha_rows = available;
    return true;
}

bool capture_new_adapter_output_sha(voxtral_stream * s,
                                    int64_t start, int32_t count) {
    if (!s || !s->diagnostics.capture_output_sha || !s->decoder.incremental || !s->ctx ||
        count <= 0) return true;
    const int32_t capacity = voxtral_ctx_aemb_ring_frames(s->ctx);
    if (count > capacity || start != s->diagnostics.adapter_sha_rows) {
        set_error(s, voxtral_status::backend_error,
                  "adapter SHA capture lost monotonic ring position");
        return false;
    }
    s->diagnostics.output_sha_scratch.resize((size_t) count * VOXTRAL_DEC_DIM);
    if (!voxtral_ctx_read_aemb_ring_internal(
            s->ctx, start, count, s->diagnostics.output_sha_scratch.data())) {
        set_error(s, voxtral_status::backend_error,
                  "adapter SHA diagnostic readback failed");
        return false;
    }
    const size_t bytes = s->diagnostics.output_sha_scratch.size() * sizeof(float);
    s->diagnostics.adapter_output_sha.update(s->diagnostics.output_sha_scratch.data(), bytes);
    s->diagnostics.output_sha_d2h_bytes += (int64_t) bytes;
    s->diagnostics.adapter_sha_rows += count;
    return true;
}

// ---- introspection accessors ----
std::string voxtral_stream_pcm_sha256(const voxtral_stream * s) {
    return s ? s->diagnostics.pcm_sha.hex() : Sha256{}.hex();
}
std::string voxtral_stream_encoder_output_sha256(const voxtral_stream * s) {
    return s ? s->diagnostics.encoder_output_sha.hex() : Sha256{}.hex();
}
std::string voxtral_stream_adapter_output_sha256(const voxtral_stream * s) {
    return s ? s->diagnostics.adapter_output_sha.hex() : Sha256{}.hex();
}
int64_t voxtral_stream_encoder_output_sha_rows(const voxtral_stream * s) {
    return s ? s->diagnostics.encoder_sha_rows : 0;
}
int64_t voxtral_stream_adapter_output_sha_rows(const voxtral_stream * s) {
    return s ? s->diagnostics.adapter_sha_rows : 0;
}
int64_t voxtral_stream_output_sha_d2h_bytes(const voxtral_stream * s) {
    return s ? s->diagnostics.output_sha_d2h_bytes : 0;
}
