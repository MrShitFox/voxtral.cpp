// ============================================================================
// Streaming adapter + decoder scheduler — implementation.
//
// The incremental production path: one scheduler pass (pump_incremental) commits
// every ready adapter group into the audio-embedding ring, runs every ready
// decoder step, and emits TOKEN + PARTIAL_TEXT while audio is fed. Also owns the
// realtime decoder prompt and the append-only incremental detokenizer. No public
// lifecycle entry points live here. See voxtral-stream-internal.h.
// ============================================================================

#include "voxtral-stream-internal.h"

#include <algorithm>
#include <cctype>

// Prompt for the realtime decoder: [BOS] + STREAMING_PAD * (left-pad + delay).
// Exactly the tokens run_adapter_and_decode_realtime prefills, so the incremental
// decoder produces a byte-identical token stream.
static void build_prompt_ids(voxtral_stream * s) {
    if (!s->decoder.prompt_ids.empty()) return;
    s->decoder.prompt_ids.push_back(VOXTRAL_TOKEN_BOS);
    for (int32_t i = 0; i < VOXTRAL_N_LEFT_PAD_TOKENS + VOXTRAL_N_DELAY_TOKENS; ++i) {
        s->decoder.prompt_ids.push_back(VOXTRAL_TOKEN_STREAMING_PAD);
    }
}
// Length of the longest prefix of `text` that ends on a UTF-8 code-point boundary
// (i.e. excludes a trailing incomplete multi-byte sequence). Detokenization is
// append-only, so everything up to this point is permanently stable.
static size_t utf8_stable_prefix(const std::string & text) {
    size_t i = text.size();
    while (i > 0 && (static_cast<unsigned char>(text[i - 1]) & 0xC0) == 0x80) --i; // skip continuations
    if (i == 0) return text.size();
    const unsigned char lead = static_cast<unsigned char>(text[i - 1]);
    size_t need;
    if      ((lead & 0x80) == 0x00) need = 1;
    else if ((lead & 0xE0) == 0xC0) need = 2;
    else if ((lead & 0xF0) == 0xE0) need = 3;
    else if ((lead & 0xF8) == 0xF0) need = 4;
    else                            need = 1;   // stray continuation / invalid lead
    // The final code point starts at i-1; it is complete iff all its bytes are present.
    return (i - 1) + need <= text.size() ? text.size() : (i - 1);
}
// Append one non-special token's bytes to the incremental transcript and refresh
// the UTF-8-safe stable prefix. Byte-identical to decode_tokens() by construction.
static void append_token_to_partial(voxtral_stream * s, int32_t token) {
    const std::string & piece = voxtral_token_piece_internal(s->model, token);
    if (!piece.empty()) s->decoder.partial_text.append(piece);
    s->decoder.partial_stable_bytes = utf8_stable_prefix(s->decoder.partial_text);
}
static bool token_piece_has_lexical_content(const voxtral_stream * s, int32_t token) {
    if (!s || !s->model) return false;
    const std::string & piece = voxtral_token_piece_internal(s->model, token);
    return std::any_of(piece.begin(), piece.end(), [](unsigned char c) {
        return c >= 0x80 || std::isalnum(c) != 0;
    });
}
// One scheduler pass: commit every ready adapter group, then run every ready
// decoder step, emitting TOKEN + PARTIAL_TEXT. Called after each feed slice and at
// finish. All three stages stay bounded because they are drained here every slice.
voxtral_status pump_incremental(voxtral_stream * s) {
    if (!s->decoder.incremental || !s->frontend.enc || !s->ctx) return voxtral_status::ok;
    build_prompt_ids(s);

    const int32_t aemb_cap    = voxtral_ctx_aemb_ring_frames(s->ctx);
    const int64_t enc_frames  = voxtral_encoder_stream_output_frames(s->frontend.enc);
    const int64_t avail_groups = enc_frames / VOXTRAL_DOWNSAMPLE_FACTOR;

    // 1. Adapter: commit newly-complete groups, throttled so the audio-embedding
    //    ring never overwrites an embedding the decoder has not consumed yet.
    const int64_t consumed_floor = s->decoder.decoder_prefill_done ? s->decoder.decoder_position : 0;
    const int64_t committable    = std::min<int64_t>(avail_groups, consumed_floor + aemb_cap);
    if (committable > s->decoder.adapter_groups_committed) {
        const int32_t n = (int32_t) (committable - s->decoder.adapter_groups_committed);
        if (voxtral_ctx_adapter_commit(s->ctx, s->decoder.adapter_groups_committed, n) < 0) {
            set_error(s, voxtral_status::backend_error, "incremental adapter commit failed");
            return voxtral_status::backend_error;
        }
        if (!capture_new_adapter_output_sha(
                s, s->decoder.adapter_groups_committed, n)) {
            return voxtral_status::backend_error;
        }
        note_group_eligibility(s, s->decoder.adapter_groups_committed, n, aemb_cap);
        if (s->telemetry.first_adapter_commit_ms == 0.0) s->telemetry.first_adapter_commit_ms = stream_elapsed_ms(s);
        s->decoder.adapter_groups_committed = committable;
        s->decoder.adapter_commit_calls++;
    }

    // 2. Decoder: prefill once over the prompt (needs positions [0, L-1)), then step
    //    one token per available audio position (audio_pos == position).
    const int32_t L = (int32_t) s->decoder.prompt_ids.size();   // 39
    if (!s->decoder.decoder_prefill_done && s->decoder.adapter_groups_committed >= (L - 1)) {
        voxtral_ctx_decoder_begin_incremental(s->ctx);
        if (!voxtral_ctx_decoder_prefill_incremental(s->ctx, s->decoder.prompt_ids.data(), L - 1)) {
            set_error(s, voxtral_status::backend_error, "incremental decoder prefill failed");
            return voxtral_status::backend_error;
        }
        s->decoder.decoder_prefill_done = true;
        s->decoder.decoder_position     = L - 1;                // first step is at position L-1
        s->decoder.decoder_prev_token   = s->decoder.prompt_ids[L - 1];
    }

    // Deliver a token that a decoder step already produced (its KV is committed)
    // and advance the bookkeeping. Returns false when the mandatory TOKEN event
    // does not fit the bounded queue: the token stays stashed, the position does
    // not advance, and the step is never re-run — that is the backpressure point.
    auto commit_pending = [&]() -> bool {
        if (!s->decoder.pending.valid) return true;
        context_profile_scope profile(s->ctx, voxtral_profile_stage::event_processing);
        if (!emit_token_event(s, s->decoder.pending.token, s->decoder.pending.position, s->decoder.pending.special)) {
            s->events.decoder_backpressured = true;
            return false;
        }
        const int32_t tok      = s->decoder.pending.token;
        const int64_t position = s->decoder.pending.position;
        const bool is_special  = s->decoder.pending.special;
        const bool is_lexical  =
            !is_special && token_piece_has_lexical_content(s, tok);
        s->decoder.pending.valid = false;
        if (tok == VOXTRAL_TOKEN_EOS) {          // terminal: matches finish-path trailing-EOS drop
            s->decoder.decoder_eos = true;
            return true;
        }
        // Token history (drives FINAL_TEXT) + incremental transcript.
        s->decoder.tokens.push_back(tok);
        append_token_to_partial(s, tok);
        emit_partial_text_event(s, position);    // lossy snapshot; never backpressures
        s->decoder.decoder_tokens_emitted++;
        if (is_lexical && s->telemetry.first_token_ms == 0.0) {
            s->telemetry.first_token_ms = stream_elapsed_ms(s);
            s->telemetry.first_token_eligibility_ms = group_eligibility_ms(s, position);
            if (s->telemetry.first_token_eligibility_ms >= 0.0) {
                s->telemetry.first_token_overhead_ms =
                    s->telemetry.first_token_ms - s->telemetry.first_token_eligibility_ms;
            }
        }
        if (is_lexical && !s->decoder.partial_text.empty() &&
            s->telemetry.first_visible_text_ms == 0.0) {
            s->telemetry.first_visible_text_ms = stream_elapsed_ms(s);
            s->telemetry.first_partial_eligibility_ms = group_eligibility_ms(s, position);
            if (s->telemetry.first_partial_eligibility_ms >= 0.0) {
                s->telemetry.first_partial_overhead_ms =
                    s->telemetry.first_visible_text_ms - s->telemetry.first_partial_eligibility_ms;
            }
        }
        s->decoder.decoder_prev_token = tok;
        s->decoder.decoder_position++;
        return true;
    };

    // Resume: flush a token stashed by a previous backpressured pass before doing
    // any new work. Still full → remain backpressured (caller drains and retries).
    if (s->decoder.decoder_prefill_done) {
        if (!commit_pending()) return voxtral_status::limit_exceeded;
        s->events.decoder_backpressured = false;
    }

    if (s->decoder.decoder_prefill_done && !s->decoder.decoder_eos) {
        const int64_t last_pos = s->decoder.adapter_groups_committed - 1;   // last committed audio pos
        const bool unlimited   = (s->params.max_tokens <= 0);
        while (s->decoder.decoder_position <= last_pos && !s->decoder.decoder_eos &&
               (unlimited || s->decoder.decoder_tokens_emitted < (int64_t) s->params.max_tokens)) {
            int32_t tok = 0;
            if (!voxtral_ctx_decoder_step_incremental(
                    s->ctx, s->decoder.decoder_prev_token, (int32_t) s->decoder.decoder_position, &tok)) {
                set_error(s, voxtral_status::backend_error, "incremental decoder step failed");
                return voxtral_status::backend_error;
            }
            s->decoder.decoder_steps++;
            s->telemetry.token_id_d2h_bytes += (int64_t) sizeof(int32_t);   // 4-byte argmax readback
            if (s->telemetry.first_decoder_step_ms == 0.0) {
                s->telemetry.first_decoder_step_ms = stream_elapsed_ms(s);
                s->telemetry.first_decoder_step_eligibility_ms =
                    group_eligibility_ms(s, s->decoder.decoder_position);
                if (s->telemetry.first_decoder_step_eligibility_ms >= 0.0) {
                    s->telemetry.first_decoder_step_overhead_ms =
                        s->telemetry.first_decoder_step_ms -
                        s->telemetry.first_decoder_step_eligibility_ms;
                }
            }

            s->decoder.pending.valid    = true;
            s->decoder.pending.token    = tok;
            s->decoder.pending.position = s->decoder.decoder_position;
            s->decoder.pending.special  = (tok == VOXTRAL_TOKEN_EOS || tok == VOXTRAL_TOKEN_BOS ||
                                   tok == VOXTRAL_TOKEN_STREAMING_PAD);
            if (!commit_pending()) return voxtral_status::limit_exceeded;
        }
    }
    return voxtral_status::ok;
}

// ---- introspection accessors ----
int64_t voxtral_stream_decoder_kv_allocated_bytes(const voxtral_stream * s) {
    return s ? voxtral_context_decoder_kv_bytes_internal(s->ctx) : 0;
}
// --- Incremental adapter + decoder introspection ---------------------------
bool voxtral_stream_uses_incremental_decode(const voxtral_stream * s) {
    return s ? s->decoder.incremental : false;
}
int64_t voxtral_stream_adapter_groups_committed(const voxtral_stream * s) {
    return s ? s->decoder.adapter_groups_committed : 0;
}
int64_t voxtral_stream_adapter_commit_calls(const voxtral_stream * s) {
    return s ? s->decoder.adapter_commit_calls : 0;
}
int64_t voxtral_stream_decoder_steps(const voxtral_stream * s) {
    return s ? s->decoder.decoder_steps : 0;
}
int64_t voxtral_stream_decoder_tokens_emitted(const voxtral_stream * s) {
    return s ? s->decoder.decoder_tokens_emitted : 0;
}
int64_t voxtral_stream_decoder_position(const voxtral_stream * s) {
    return s ? s->decoder.decoder_position : 0;
}
bool voxtral_stream_decoder_prefill_complete(const voxtral_stream * s) {
    return s ? s->decoder.decoder_prefill_done : false;
}
int64_t voxtral_stream_tokens_before_finish(const voxtral_stream * s) {
    return s ? s->decoder.tokens_before_finish : 0;
}
int64_t voxtral_stream_tokens_flushed_at_finish(const voxtral_stream * s) {
    return s ? (s->decoder.decoder_tokens_emitted - s->decoder.tokens_before_finish) : 0;
}
int64_t voxtral_stream_adapter_input_d2h_bytes(const voxtral_stream * s) {
    (void) s; return 0;   // adapter reads the encoder-output ring on-device
}
int64_t voxtral_stream_adapter_output_d2h_bytes(const voxtral_stream * s) {
    (void) s; return 0;   // adapter writes the audio-embedding ring on-device
}
int64_t voxtral_stream_logits_d2h_bytes(const voxtral_stream * s) {
    (void) s; return 0;   // steps read back only the argmax token; prefill logits skipped
}
int64_t voxtral_stream_token_id_d2h_bytes(const voxtral_stream * s) {
    return s ? s->telemetry.token_id_d2h_bytes : 0;
}
uint64_t voxtral_stream_partial_text_revision(const voxtral_stream * s) {
    return s ? s->decoder.partial_revision : 0;
}
// Active decoder path. "incremental" is the production default;
// "reference" is the finish-only oracle (env override, or the coupled fallback
// when the reference encoder is selected). Meaningful once the encoder is created
// (first feed); reflects the env choice before then.
const char * voxtral_stream_decoder_mode(const voxtral_stream * s) {
    return (s && s->decoder.incremental) ? "incremental" : "reference";
}
