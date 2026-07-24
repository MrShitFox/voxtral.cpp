// ============================================================================
// Internal streaming session runtime — implementation.
//
// PCM/STFT/log-Mel, the causal per-layer-KV encoder, and the device-resident
// adapter + decoder all run incrementally during feed(); finish() only processes
// the bounded remaining tail. This translation unit holds the public lifecycle
// entry points, feed orchestration and the state machine; the per-subsystem work
// lives in voxtral-stream-{frontend,decoder,events,telemetry,diagnostics}.cpp.
//
// See src/voxtral-stream.h, src/voxtral-stream-internal.h and
// docs/architecture/streaming-runtime.md.
// ============================================================================

#include "voxtral-stream.h"
#include "voxtral-internal.h"   // voxtral_transcribe_mel_internal (Mel -> text path)
#include "voxtral-stream-internal.h"  // voxtral_stream + shared streaming internals

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------------------------------------------
// Bounds
// ----------------------------------------------------------------------------
namespace {

// ----------------------------------------------------------------------------
// Context factory / free — indirected through file-local pointers so the
// model-free unit tests can substitute them (see the test seam). Production
// always uses the real factory.
// ----------------------------------------------------------------------------
voxtral_stream_context_factory_fn g_context_factory = &voxtral_init_from_model;
voxtral_stream_context_free_fn    g_context_free    = &voxtral_free;

} // namespace

// ----------------------------------------------------------------------------
// Small helpers
// ----------------------------------------------------------------------------
namespace {

bool feed_allowed(voxtral_stream_state_internal st) {
    return st == voxtral_stream_state_internal::created || st == voxtral_stream_state_internal::running;
}

// ============================================================================
// Feed orchestration (incremental Mel -> encoder -> adapter -> decoder).
// ============================================================================

// Common core for both feed variants: guard reentrancy, validate state/args,
// size guards, then append via the caller-provided converter. `convert(dst)`
// must append exactly `count` floats to `dst`.
template <typename Convert>
voxtral_status_internal feed_common(voxtral_stream * s, const void * ptr, size_t count, Convert convert) {
    if (!s) return voxtral_status_internal::invalid_argument;

    op_guard guard(s);
    if (!guard.engage()) {
        // Concurrent/reentrant misuse. Do not mutate shared state (the returned
        // status is enough); see the threading contract.
        return voxtral_status_internal::busy;
    }

    if (!feed_allowed(s->lifecycle.state)) {
        set_error(s, voxtral_status_internal::invalid_state,
                  std::string("feed not allowed in state ") + voxtral_stream_state_name(s->lifecycle.state));
        return voxtral_status_internal::invalid_state;
    }

    // Backpressure resume: a prior feed stalled the decoder because the
    // event queue filled. Flush the stashed token + drain pending steps before
    // accepting anything new. If the queue is still full, reject this feed WITHOUT
    // consuming its audio (atomic; samples_received unchanged) so the caller drains
    // and retries the same buffer. A zero-length feed thus doubles as a drain pump.
    if (s->decoder.incremental && s->events.decoder_backpressured) {
        const voxtral_status_internal rp = pump_incremental(s);
        if (rp == voxtral_status_internal::limit_exceeded) {
            s->lifecycle.feed_calls++;
            return voxtral_status_internal::limit_exceeded;   // still backpressured; drain + retry
        }
        if (rp != voxtral_status_internal::ok) {
            return rp;
        }
        // Resumed with room; fall through (the zero-length early return or the
        // normal audio path below counts this feed call exactly once).
    }

    if (ptr == nullptr && count != 0) {
        set_error(s, voxtral_status_internal::invalid_argument, "null samples with non-zero count");
        return voxtral_status_internal::invalid_argument;
    }

    // Zero-length feed: successful no-op. Does not change state or audio position.
    if (count == 0) {
        clear_error(s);
        s->lifecycle.feed_calls++;
        return voxtral_status_internal::ok;
    }

    if (static_cast<uint64_t>(count) > kMaxFeedSamples) {
        set_error(s, voxtral_status_internal::invalid_argument, "feed exceeds per-call sample limit");
        return voxtral_status_internal::invalid_argument;
    }

    const uint64_t received = s->lifecycle.total_samples_received;
    // Overflow-safe cumulative admission bound, not an allocation failure.
    // Production inference retains only bounded frontend/encoder state, and
    // decoder rollover is handled by the fixed circular KV.
    if (count > s->params.max_total_samples ||
        received > s->params.max_total_samples - static_cast<uint64_t>(count)) {
        set_error(s, voxtral_status_internal::limit_exceeded,
                  "stream duration exceeds max_total_samples admission limit");
        return voxtral_status_internal::limit_exceeded;
    }

    // The payload became available when this accepted feed entered the
    // pipeline. Include frontend creation and PCM conversion in residence;
    // timestamping after conversion would make the metric systematically low.
    const int64_t arrival_ns = stream_now_ns();
    if (s->telemetry.timeline_start_ns == 0) s->telemetry.timeline_start_ns = arrival_ns;
    s->telemetry.current_audio_availability_ms =
        (double) (arrival_ns - s->telemetry.timeline_start_ns) / 1e6;
    const bool previous_partial_mode = s->events.aggressive_partial_coalescing;
    const uint64_t estimated_token_events =
        ((uint64_t) count + VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK - 1) /
        VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK +
        VOXTRAL_N_RIGHT_PAD_TOKENS + 2;
    s->events.aggressive_partial_coalescing =
        s->events.queue.size() + 2 * estimated_token_events > s->events.max_events &&
        s->events.queue.size() + estimated_token_events + 1 <= s->events.max_events;
    struct partial_mode_guard {
        voxtral_stream * stream;
        bool previous;
        ~partial_mode_guard() {
            stream->events.aggressive_partial_coalescing = previous;
        }
    } restore_partial_mode{s, previous_partial_mode};
    context_profile_scope pipeline_profile(s->ctx, voxtral_profile_stage::pipeline_feed);

    // Lazily bring up the incremental Mel frontend (inference streams). Done
    // before any state mutation so a failure leaves the stream untouched.
    if (!ensure_frontend(s)) {
        return voxtral_status_internal::out_of_memory;
    }

    // Convert into a reusable scratch buffer first, so a rejected payload never
    // mutates any persistent state.
    s->frontend.feed_scratch.clear();
    try {
        s->frontend.feed_scratch.reserve(count);
    } catch (const std::exception & e) {
        set_error(s, voxtral_status_internal::out_of_memory, std::string("pcm reserve failed: ") + e.what());
        return voxtral_status_internal::out_of_memory;
    }
    if (!convert(s->frontend.feed_scratch)) {
        // Converter rejected the payload (e.g. non-finite float); nothing changed.
        s->frontend.feed_scratch.clear();
        return s->lifecycle.last_status;
    }

    // Rolling SHA-256 of the canonical PCM (chunk-invariant; no retention needed).
    s->diagnostics.pcm_sha.update(s->frontend.feed_scratch.data(), s->frontend.feed_scratch.size() * sizeof(float));

    bool backpressured = false;   // event queue filled during this feed's decoder pump
    if (s->frontend.mel_fe) {
        // Inference stream: stream samples through the incremental Mel frontend.
        // No full PCM is retained — only the frontend's bounded rolling tail.
        ensure_left_pad(s);
        if (s->telemetry.first_mel_absolute_ms == 0.0 && voxtral_mel_frontend_frame_count(s->frontend.mel_fe) > 0) {
            s->telemetry.first_mel_absolute_ms = (double) (stream_now_ns() - s->telemetry.timeline_start_ns) / 1e6;
        }
        if (!ensure_encoder(s)) {
            s->frontend.feed_scratch.clear();
            return s->lifecycle.last_status;
        }
        // Bound frontend/encoder transient state for a large compute-only feed;
        // realtime callers normally enter this loop once (80/160 ms chunk).
        constexpr size_t kAudioDrainSlice = 16'000;
        for (size_t audio_off = 0; audio_off < s->frontend.feed_scratch.size(); audio_off += kAudioDrainSlice) {
            const size_t audio_n = std::min(kAudioDrainSlice, s->frontend.feed_scratch.size() - audio_off);
            const int64_t mel_start_ns = stream_now_ns();
            const bool mel_ok = voxtral_mel_frontend_feed(
                s->frontend.mel_fe, s->frontend.feed_scratch.data() + audio_off, audio_n);
            voxtral_context_profile_record_internal(
                s->ctx, voxtral_profile_stage::mel_compute,
                (double) (stream_now_ns() - mel_start_ns) / 1e6);
            if (!mel_ok) {
                set_error(s, voxtral_status_internal::backend_error, "incremental Mel frontend rejected feed");
                s->frontend.feed_scratch.clear();
                return voxtral_status_internal::backend_error;
            }
            // Drive the incremental causal encoder over the newly-stable Mel
            // frames, so output is produced during feed rather than finish().
            voxtral_encoder_stream_note_audio(
                s->frontend.enc,
                (int64_t) (received + (uint64_t) count),
                arrival_ns,
                voxtral_mel_frontend_frame_count(s->frontend.mel_fe),
                stream_now_ns(),
                s->telemetry.timeline_start_ns,
                kStreamLeftPad);
            if (!drain_mel_to_encoder(s)) {
                s->frontend.feed_scratch.clear();
                return voxtral_status_internal::backend_error;
            }
            // Advance the device-resident adapter + decoder during feed,
            // draining every slice so the encoder-output / audio-embedding rings stay
            // bounded. TOKEN and PARTIAL_TEXT are emitted here, not at finish.
            if (s->decoder.incremental && !backpressured) {
                const voxtral_status_internal ps = pump_incremental(s);
                if (ps == voxtral_status_internal::limit_exceeded) {
                    // Event queue full: the decoder is stalled with a stashed token.
                    // Keep consuming this feed's audio into the (bounded) Mel/encoder
                    // so nothing is lost, but stop advancing the decoder until the
                    // caller drains. feed returns queue_full below. For realtime
                    // (single-slice) feeds this is the whole feed → fully atomic.
                    backpressured = true;
                } else if (ps != voxtral_status_internal::ok) {
                    s->frontend.feed_scratch.clear();
                    return ps;
                }
            }
        }
    } else {
        // Lifecycle-only stream (no context): retain the full canonical PCM as
        // before (used by model-free tests; never runs inference).
        try {
            s->frontend.pcm.insert(s->frontend.pcm.end(), s->frontend.feed_scratch.begin(), s->frontend.feed_scratch.end());
        } catch (const std::exception & e) {
            set_error(s, voxtral_status_internal::out_of_memory, std::string("pcm append failed: ") + e.what());
            s->frontend.feed_scratch.clear();
            return voxtral_status_internal::out_of_memory;
        }
    }
    s->frontend.feed_scratch.clear();

    s->lifecycle.total_samples_received += static_cast<uint64_t>(count);
    s->lifecycle.feed_calls++;
    sample_stage_backlogs(s);
    if (s->lifecycle.state == voxtral_stream_state_internal::created) {
        s->lifecycle.state = voxtral_stream_state_internal::running;
    }
    if (backpressured) {
        // Audio was accepted (Mel/encoder consumed it); the decoder has output
        // pending because the event queue is full. Surface explicit backpressure —
        // the caller drains the event queue and feeds again (a zero-length feed
        // suffices) to resume. last_error carries the reason; do not clear it.
        return voxtral_status_internal::limit_exceeded;
    }
    clear_error(s);
    return voxtral_status_internal::ok;
}

} // namespace

void voxtral_stream_set_timeline_start_internal(voxtral_stream * stream,
                                                int64_t timeline_start_ns) {
    if (!stream || timeline_start_ns <= 0) return;
    // This is an internal/test seam. It is meaningful only before the first
    // feed; a running stream keeps the original anchor for all later chunks.
    if (stream->telemetry.timeline_start_ns == 0) stream->telemetry.timeline_start_ns = timeline_start_ns;
}

// ============================================================================
// Params validation
// ============================================================================
voxtral_status_internal voxtral_stream_params_check(const voxtral_stream_params_internal & params) {
    if (params.sample_rate != VOXTRAL_SAMPLE_RATE || params.channels != 1) {
        return voxtral_status_internal::unsupported_audio_format;
    }
    if (params.max_total_samples == 0) {
        return voxtral_status_internal::invalid_argument;
    }
    return voxtral_status_internal::ok;
}

// ============================================================================
// Lifecycle
// ============================================================================
static voxtral_stream * allocate_stream_state(
    voxtral_model * model,
    const voxtral_stream_params_internal & params) {
    if (voxtral_stream_params_check(params) != voxtral_status_internal::ok) {
        return nullptr;
    }
    auto * s = new (std::nothrow) voxtral_stream();
    if (!s) return nullptr;
    s->params = params;
    s->model  = model;

    // The device-resident incremental adapter + decoder is the
    // production default. Only the explicit reference oracle disables it; any
    // other value (including the legacy explicit "incremental") keeps the
    // incremental path. No silent incremental→reference fallback: a reference run
    // must be asked for. (ensure_encoder additionally couples the decoder to the
    // encoder strategy — the incremental decoder needs the KV encoder ring.)
    s->decoder.incremental = true;
    if (const char * mode = std::getenv("VOXTRAL_STREAM_DECODER")) {
        const std::string m = mode;
        if (m == "reference" || m == "finish-only" || m == "finish_only" || m == "oracle") {
            s->decoder.incremental = false;
        }
    }
    if (const char * capture = std::getenv("VOXTRAL_CAPTURE_OUTPUT_SHA")) {
        const std::string value = capture;
        s->diagnostics.capture_output_sha =
            value == "1" || value == "true" || value == "yes";
    }
    return s;
}

voxtral_stream * voxtral_stream_create_internal(
    voxtral_model                * model,
    const voxtral_context_params & ctx_params,
    const voxtral_stream_params_internal & params)
{
    voxtral_stream * s = allocate_stream_state(model, params);
    if (!s) return nullptr;

    if (model != nullptr) {
        // Preferred path: the stream creates and owns its own mutable execution
        // context from the shared, immutable model. One context per stream.
        voxtral_context * ctx = g_context_factory(model, ctx_params);
        if (!ctx) {
            // Surface the failure as a queryable status/error rather than a bare
            // nullptr. The caller still destroys the stream; it owns no context.
            s->lifecycle.state = voxtral_stream_state_internal::failed;
            set_error(s, voxtral_status_internal::backend_error,
                      "failed to create per-stream execution context from model");
            return s;
        }
        s->ctx          = ctx;
        s->owns_context = true;
        // The incremental Mel frontend is created lazily on the first feed (see
        // ensure_frontend): the model-free ownership tests substitute an opaque
        // sentinel context via the test seam and never feed, so create() must not
        // dereference the returned context here.
    }
    // model == nullptr: lifecycle-only stream, no owned context / frontend.
    return s;
}

voxtral_stream * voxtral_stream_create_from_context_internal(
    voxtral_context * ctx,
    const voxtral_stream_params_internal & params)
{
    if (!ctx) return nullptr;
    voxtral_model * model = voxtral_context_model_internal(ctx);
    if (!model) return nullptr;
    voxtral_stream * s = allocate_stream_state(model, params);
    if (!s) return nullptr;
    s->ctx = ctx;
    s->owns_context = false;
    // A context is reusable after the preceding public stream is destroyed.
    // Start a new borrowed lifecycle with the same pristine context/KV/profile
    // state as reset(), even if its predecessor ended under backpressure.
    try {
        if (voxtral_stream_reset_internal(s) == voxtral_status_internal::ok) {
            return s;
        }
    } catch (...) {
        s->ctx = nullptr;
        delete s;
        throw;
    }
    s->ctx = nullptr;
    delete s;
    return nullptr;
}

void voxtral_stream_destroy_internal(voxtral_stream * stream) {
    if (!stream) return;   // destroy(nullptr) is safe.
    // Owns only its own context (and mutable state); never frees the model. The
    // caller guarantees no operation is in flight (threading contract).
    if (stream->frontend.enc) {
        voxtral_encoder_stream_destroy(stream->frontend.enc);
    }
    if (stream->frontend.mel_fe) {
        voxtral_mel_frontend_destroy(stream->frontend.mel_fe);
    }
    if (stream->owns_context && stream->ctx) {
        g_context_free(stream->ctx);
    }
    delete stream;
}

voxtral_status_internal voxtral_stream_warmup_internal(voxtral_stream * stream) {
    if (!stream) return voxtral_status_internal::invalid_argument;
    op_guard guard(stream);
    if (!guard.engage()) return voxtral_status_internal::busy;
    if (stream->lifecycle.warmup_complete) return voxtral_status_internal::ok;
    if (stream->lifecycle.state != voxtral_stream_state_internal::created ||
        stream->lifecycle.total_samples_received != 0 || !stream->ctx || !stream->decoder.incremental) {
        set_error(stream, voxtral_status_internal::invalid_state,
                  "warmup requires a fresh production incremental stream");
        return voxtral_status_internal::invalid_state;
    }
    if (!ensure_frontend(stream) || !ensure_encoder(stream) || !stream->frontend.enc) {
        if (stream->lifecycle.last_status == voxtral_status_internal::ok) {
            set_error(stream, voxtral_status_internal::backend_error,
                      "failed to initialize production warmup state");
        }
        return stream->lifecycle.last_status;
    }
    if (!voxtral_encoder_stream_warmup(stream->frontend.enc)) {
        set_error(stream, voxtral_status_internal::backend_error,
                  "production graph warmup failed");
        return voxtral_status_internal::backend_error;
    }
    stream->frontend.enc_pushed_frames = 0;
    stream->lifecycle.warmup_complete = true;
    clear_error(stream);
    return voxtral_status_internal::ok;
}

voxtral_status_internal voxtral_stream_feed_pcm16_internal(
    voxtral_stream * stream, const int16_t * samples, size_t sample_count)
{
    return feed_common(stream, samples, sample_count, [&](std::vector<float> & dst) {
        for (size_t i = 0; i < sample_count; ++i) {
            // Deterministic S16 -> float. -32768 maps to exactly -1.0f.
            dst.push_back(static_cast<float>(samples[i]) / 32768.0f);
        }
        return true;
    });
}

voxtral_status_internal voxtral_stream_feed_f32_internal(
    voxtral_stream * stream, const float * samples, size_t sample_count)
{
    return feed_common(stream, samples, sample_count, [&](std::vector<float> & dst) {
        // Validate finiteness up front so a rejected payload never mutates the
        // buffer. The canonical range is [-1, 1]; values outside are passed
        // through unchanged (no silent clamp) to preserve batch-path parity.
        for (size_t i = 0; i < sample_count; ++i) {
            if (!std::isfinite(samples[i])) {
                set_error(stream, voxtral_status_internal::invalid_argument,
                          "float32 feed contains a non-finite sample");
                return false;
            }
        }
        dst.insert(dst.end(), samples, samples + sample_count);
        return true;
    });
}

voxtral_status_internal voxtral_stream_finish_internal(voxtral_stream * stream) {
    if (!stream) return voxtral_status_internal::invalid_argument;

    op_guard guard(stream);
    if (!guard.engage()) {
        // Reentrant finish() (only reachable from within a running finish()).
        return voxtral_status_internal::busy;
    }

    switch (stream->lifecycle.state) {
        case voxtral_stream_state_internal::completed:
            // Idempotent: never run inference a second time.
            return voxtral_status_internal::ok;
        case voxtral_stream_state_internal::cancelled:
            // Cancelled before finish: no inference, no additional events.
            return voxtral_status_internal::ok;
        case voxtral_stream_state_internal::failed:
            set_error(stream, voxtral_status_internal::invalid_state, "finish not allowed after failure");
            return voxtral_status_internal::invalid_state;
        case voxtral_stream_state_internal::finishing:
            // Unreachable while externally serialized (the guard would have
            // rejected a reentrant call with `busy` already); defensive.
            return voxtral_status_internal::busy;
        case voxtral_stream_state_internal::created:
        case voxtral_stream_state_internal::running:
            break;
    }

    // A large accepted feed may have stopped decoder delivery at the ordinary
    // queue bound while the bounded encoder continued consuming audio. Drain
    // that pre-existing work under normal backpressure before reserving the
    // small terminal headroom. This prevents finish() from turning arbitrary
    // utterance backlog into hidden queue growth.
    if (stream->decoder.incremental &&
        stream->events.decoder_backpressured) {
        const voxtral_status_internal pending = pump_incremental(stream);
        if (pending == voxtral_status_internal::limit_exceeded) {
            return pending;
        }
        if (pending != voxtral_status_internal::ok) {
            stream->events.finalizing_flush = true;
            emit_error(stream, pending, stream->lifecycle.last_error);
            stream->lifecycle.state = voxtral_stream_state_internal::failed;
            return pending;
        }
    }

    stream->lifecycle.state = voxtral_stream_state_internal::finishing;
    // finish() delivers a bounded tail (remaining audio positions + EOS) plus the
    // terminal FINAL_TEXT/COMPLETED (or ERROR). These mandatory events must never
    // be dropped, so the terminal flush uses the fixed terminal headroom beyond
    // the ordinary streaming bound.
    stream->events.finalizing_flush = true;

    // Test seam: observe the transient `finishing` state / probe reentrancy.
    // Any reentrant stream call from here returns `busy` (the guard is engaged).
    if (stream->lifecycle.finishing_hook) {
        stream->lifecycle.finishing_hook(stream, stream->lifecycle.finishing_hook_user);
    }

    // Empty stream: documented as COMPLETED with an empty final transcript and
    // no inference.
    if (stream->lifecycle.total_samples_received == 0) {
        stream->decoder.transcript.clear();
        stream->decoder.tokens.clear();
        stream->lifecycle.total_samples_consumed = 0;
        emit_final_and_completed(stream);
        stream->lifecycle.state = voxtral_stream_state_internal::completed;
        clear_error(stream);
        return voxtral_status_internal::ok;
    }

    if (stream->ctx == nullptr || stream->frontend.mel_fe == nullptr) {
        set_error(stream, voxtral_status_internal::backend_error,
                  "no execution context / Mel frontend: cannot transcribe audio");
        emit_error(stream, voxtral_status_internal::backend_error, stream->lifecycle.last_error);
        stream->lifecycle.state = voxtral_stream_state_internal::failed;
        return voxtral_status_internal::backend_error;
    }

    // Incremental frontend + encoder path: no full PCM was buffered, and the
    // causal encoder already produced most of its output DURING feed. Push the
    // streaming right zero-padding (equivalent to the batch pad_audio_streaming
    // tail), flush the final Mel frames, drain them into the encoder, finalize it
    // (which runs at most the last one/two encoder chunks — never the whole Mel),
    // then run the shared adapter/decoder over the accumulated encoder output.
    // Chunk boundaries do not affect the Mel or the encoder output (both are
    // bit-for-bit the batch result of the same audio), so tokens are invariant.
    // The even-trimmed Mel is still assembled for the session-5 introspection.
    voxtral_result result;
    bool ok = false;
    try {
        const int64_t finish_front_start_ns = stream_now_ns();
        ensure_left_pad(stream);   // no-op if already injected during feed
        const int64_t n_raw     = (int64_t) stream->lifecycle.total_samples_received;
        const int64_t mult      = VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK;
        const int64_t align_pad = (mult - (n_raw % mult)) % mult;
        const int64_t right_pad = align_pad + (int64_t) VOXTRAL_N_RIGHT_PAD_TOKENS * mult;
        voxtral_mel_frontend_feed_silence(stream->frontend.mel_fe, (size_t) right_pad);
        voxtral_mel_frontend_finish(stream->frontend.mel_fe);
        voxtral_mel_frontend_assemble_even(stream->frontend.mel_fe, stream->frontend.mel_even, stream->frontend.mel_even_frames);
        stream->frontend.full_pcm_buffered_at_finish = false;
        stream->telemetry.finish_frontend_ms = (double) (stream_now_ns() - finish_front_start_ns) / 1e6;
        voxtral_context_profile_record_internal(stream->ctx, voxtral_profile_stage::mel_compute,
                                                stream->telemetry.finish_frontend_ms);

        if (!ensure_encoder(stream)) {
            throw std::runtime_error(stream->lifecycle.last_error);
        }
        if (stream->frontend.enc) {
            // Feed the finish-flushed Mel frames, then finalize the encoder.
            voxtral_encoder_stream_note_audio(stream->frontend.enc,
                                              (int64_t) stream->lifecycle.total_samples_received,
                                              stream_now_ns(),
                                              voxtral_mel_frontend_frame_count(stream->frontend.mel_fe),
                                              stream_now_ns(),
                                              stream->telemetry.timeline_start_ns,
                                              kStreamLeftPad);
            const int64_t finish_encoder_start_ns = stream_now_ns();
            if (!drain_mel_to_encoder(stream, /*final=*/true)) {
                throw std::runtime_error(stream->lifecycle.last_error);
            }
            if (!voxtral_encoder_stream_finish(stream->frontend.enc)) {
                throw std::runtime_error("incremental encoder finish failed");
            }
            stream->telemetry.finish_encoder_ms = (double) (stream_now_ns() - finish_encoder_start_ns) / 1e6;
            const int64_t finish_decoder_start_ns = stream_now_ns();
            if (stream->decoder.incremental) {
                // Device-resident path: the adapter and decoder already ran during
                // feed(). Only the flushed tail remains — commit its groups and run
                // its decoder steps. No whole-utterance adapter or decoder replay.
                stream->decoder.tokens_before_finish = stream->decoder.decoder_tokens_emitted;
                const voxtral_status_internal ps = pump_incremental(stream);
                ok = (ps == voxtral_status_internal::ok);
                if (!ok) throw std::runtime_error(stream->lifecycle.last_error);
            } else {
                const float * enc_out    = voxtral_encoder_stream_output(stream->frontend.enc);
                const int32_t enc_frames = voxtral_encoder_stream_output_frames(stream->frontend.enc);
                ok = voxtral_transcribe_encoder_output_internal(
                    *stream->ctx, enc_out, enc_frames, stream->params.max_tokens, result);
            }
            stream->telemetry.finish_decoder_ms = (double) (stream_now_ns() - finish_decoder_start_ns) / 1e6;
        } else {
            // Defensive fallback (should not happen for inference streams): run the
            // shared Mel -> text path over the accumulated incremental Mel.
            const int64_t finish_decoder_start_ns = stream_now_ns();
            ok = voxtral_transcribe_mel_internal(*stream->ctx, stream->frontend.mel_even.data(),
                                                 stream->frontend.mel_even_frames, stream->params.max_tokens, result);
            stream->telemetry.finish_decoder_ms = (double) (stream_now_ns() - finish_decoder_start_ns) / 1e6;
        }
        stream->lifecycle.inference_runs++;   // one execution per finish, success or failure
    } catch (const std::exception & e) {
        set_error(stream, voxtral_status_internal::backend_error,
                  std::string("inference threw: ") + e.what());
        emit_error(stream, voxtral_status_internal::backend_error, stream->lifecycle.last_error);
        stream->lifecycle.state = voxtral_stream_state_internal::failed;
        return voxtral_status_internal::backend_error;
    }

    if (!ok) {
        set_error(stream, voxtral_status_internal::backend_error, "incremental Mel inference reported failure");
        emit_error(stream, voxtral_status_internal::backend_error, stream->lifecycle.last_error);
        stream->lifecycle.state = voxtral_stream_state_internal::failed;
        return voxtral_status_internal::backend_error;
    }

    if (stream->decoder.incremental) {
        // Built incrementally during feed/finish; byte-identical to decode_tokens
        // over stream->decoder.tokens (which already excludes the terminal EOS).
        stream->decoder.transcript = stream->decoder.partial_text;
    } else {
        stream->decoder.transcript = std::move(result.text);
        stream->decoder.tokens     = std::move(result.tokens);
    }
    stream->lifecycle.total_samples_consumed = stream->lifecycle.total_samples_received;
    emit_final_and_completed(stream);
    stream->lifecycle.state = voxtral_stream_state_internal::completed;
    clear_error(stream);
    return voxtral_status_internal::ok;
}

voxtral_status_internal voxtral_stream_reset_internal(voxtral_stream * stream) {
    if (!stream) return voxtral_status_internal::invalid_argument;

    op_guard guard(stream);
    if (!guard.engage()) {
        // reset() during an in-flight finish() is rejected without mutating state.
        return voxtral_status_internal::busy;
    }

    // Clears all per-stream mutable runtime state; the owned context, params,
    // and test knobs (max_events / finishing hook) are kept for cheap reuse.
    // Deliberately no shrink_to_fit(): keep buffer capacities so reuse does not
    // re-allocate.
    stream->frontend.pcm.clear();
    stream->frontend.feed_scratch.clear();
    if (stream->frontend.mel_fe) {
        voxtral_mel_frontend_reset(stream->frontend.mel_fe);   // clears rolling PCM, frames, counters
    }
    if (stream->frontend.enc) {
        voxtral_encoder_stream_reset(stream->frontend.enc);    // clears Mel window, accumulated output, counters
    }
    stream->frontend.enc_pushed_frames            = 0;
    stream->telemetry.encoder_backlog.reset();
    stream->telemetry.adapter_backlog.reset();
    stream->telemetry.decoder_backlog.reset();
    stream->frontend.left_pad_injected            = false;
    stream->frontend.full_pcm_buffered_at_finish  = false;
    // Incremental adapter/decoder state. The device rings are append-only
    // and only ever read at written positions, so resetting the counters is enough;
    // detach the shared decoder from this stream's audio ring and clear its KV.
    if (stream->ctx) {
        voxtral_ctx_decoder_reset_incremental(stream->ctx);
        voxtral_context_profile_reset_internal(stream->ctx);
    }
    stream->decoder.adapter_groups_committed = 0;
    stream->decoder.adapter_commit_calls     = 0;
    stream->decoder.decoder_prefill_done     = false;
    stream->decoder.decoder_prev_token       = 0;
    stream->decoder.decoder_position         = 0;
    stream->decoder.decoder_steps            = 0;
    stream->decoder.decoder_tokens_emitted   = 0;
    stream->decoder.tokens_before_finish     = 0;
    stream->decoder.decoder_eos              = false;
    stream->decoder.token_sequence           = 0;
    stream->decoder.partial_revision         = 0;
    stream->decoder.partial_text.clear();
    stream->decoder.partial_stable_bytes     = 0;
    stream->telemetry.first_adapter_commit_ms  = 0.0;
    stream->telemetry.first_decoder_step_ms    = 0.0;
    stream->telemetry.first_token_ms           = 0.0;
    stream->telemetry.first_visible_text_ms    = 0.0;
    std::fill(stream->telemetry.eligibility_absolute_group.begin(),
              stream->telemetry.eligibility_absolute_group.end(), -1);
    std::fill(stream->telemetry.eligibility_arrival_ms.begin(),
              stream->telemetry.eligibility_arrival_ms.end(), 0.0);
    stream->telemetry.current_audio_availability_ms       = 0.0;
    stream->telemetry.first_decoder_step_eligibility_ms   = -1.0;
    stream->telemetry.first_decoder_step_overhead_ms      = -1.0;
    stream->telemetry.first_token_eligibility_ms          = -1.0;
    stream->telemetry.first_token_overhead_ms             = -1.0;
    stream->telemetry.first_partial_eligibility_ms        = -1.0;
    stream->telemetry.first_partial_overhead_ms           = -1.0;
    stream->telemetry.token_id_d2h_bytes       = 0;
    stream->decoder.pending                  = {};      // no token in flight
    stream->events.decoder_backpressured    = false;
    stream->events.finalizing_flush         = false;
    stream->events.aggressive_partial_coalescing = false;
    stream->diagnostics.pcm_sha.reset();
    stream->diagnostics.encoder_output_sha.reset();
    stream->diagnostics.adapter_output_sha.reset();
    stream->diagnostics.output_sha_scratch.clear();
    stream->diagnostics.encoder_sha_rows = 0;
    stream->diagnostics.adapter_sha_rows = 0;
    stream->diagnostics.output_sha_d2h_bytes = 0;
    stream->frontend.mel_even.clear();
    stream->frontend.mel_even_frames        = 0;
    stream->lifecycle.total_samples_received = 0;
    stream->lifecycle.total_samples_consumed = 0;
    stream->lifecycle.feed_calls             = 0;
    stream->lifecycle.inference_runs         = 0;
    stream->telemetry.timeline_start_ns      = 0;
    stream->telemetry.finish_frontend_ms    = 0.0;
    stream->telemetry.finish_encoder_ms     = 0.0;
    stream->telemetry.finish_decoder_ms     = 0.0;
    stream->telemetry.first_mel_absolute_ms = 0.0;
    stream->decoder.tokens.clear();
    stream->decoder.transcript.clear();
    stream->events.queue.clear();
    stream->events.events_overflowed = false;
    stream->events.events_emitted             = 0;
    stream->events.token_events               = 0;
    stream->events.partial_events             = 0;
    stream->events.partial_events_coalesced   = 0;
    stream->events.event_queue_high_watermark = 0;
    stream->events.event_queue_overflow_attempts = 0;
    stream->events.events_dropped             = 0;
    stream->events.public_poll_sequence       = 0;
    stream->lifecycle.cancel_requested  = false;
    stream->lifecycle.cancelled_emitted = false;
    stream->lifecycle.state = voxtral_stream_state_internal::created;
    clear_error(stream);
    return voxtral_status_internal::ok;
}

voxtral_status_internal voxtral_stream_cancel_internal(voxtral_stream * stream) {
    if (!stream) return voxtral_status_internal::invalid_argument;

    op_guard guard(stream);
    if (!guard.engage()) {
        // cancel() cannot interrupt an in-flight finish(): it returns `busy` and
        // does NOT set `cancelled`. This is what prevents a cancelled -> completed
        // transition (the finish proceeds and completes normally).
        return voxtral_status_internal::busy;
    }

    switch (stream->lifecycle.state) {
        case voxtral_stream_state_internal::completed:
            set_error(stream, voxtral_status_internal::invalid_state, "cannot cancel a completed stream");
            return voxtral_status_internal::invalid_state;
        case voxtral_stream_state_internal::cancelled:
        case voxtral_stream_state_internal::failed:
            // Idempotent no-op; do not emit a second CANCELLED event.
            return voxtral_status_internal::ok;
        case voxtral_stream_state_internal::finishing:
            // Unreachable while externally serialized (guard rejects reentrancy
            // with `busy`); defensive — never convert an in-flight finish.
            set_error(stream, voxtral_status_internal::invalid_state,
                      "cannot cancel during finish");
            return voxtral_status_internal::invalid_state;
        case voxtral_stream_state_internal::created:
        case voxtral_stream_state_internal::running:
            break;
    }

    stream->lifecycle.cancel_requested = true;
    if (!stream->lifecycle.cancelled_emitted) {
        // Cancellation is terminal. Preserve its single observable event even
        // when ordinary streaming output has filled the configured queue.
        stream->events.finalizing_flush = true;
        voxtral_stream_event_internal ev;
        ev.type       = voxtral_stream_event_type_internal::cancelled;
        ev.t_audio_ms = samples_to_ms(stream->lifecycle.total_samples_received, stream->params.sample_rate);
        push_event(stream, std::move(ev));
        stream->lifecycle.cancelled_emitted = true;
    }
    stream->lifecycle.state = voxtral_stream_state_internal::cancelled;
    clear_error(stream);
    return voxtral_status_internal::ok;
}

// ============================================================================
// Introspection
// ============================================================================
voxtral_stream_state_internal voxtral_stream_get_state_internal(const voxtral_stream * s) {
    return s ? s->lifecycle.state : voxtral_stream_state_internal::failed;
}
voxtral_status_internal voxtral_stream_last_status(const voxtral_stream * s) {
    return s ? s->lifecycle.last_status : voxtral_status_internal::invalid_argument;
}
const std::string & voxtral_stream_last_error(const voxtral_stream * s) {
    static const std::string empty;
    return s ? s->lifecycle.last_error : empty;
}
uint64_t voxtral_stream_samples_received(const voxtral_stream * s) {
    return s ? s->lifecycle.total_samples_received : 0;
}
uint64_t voxtral_stream_samples_consumed(const voxtral_stream * s) {
    return s ? s->lifecycle.total_samples_consumed : 0;
}
uint64_t voxtral_stream_feed_calls(const voxtral_stream * s) {
    return s ? s->lifecycle.feed_calls : 0;
}
uint64_t voxtral_stream_inference_runs(const voxtral_stream * s) {
    return s ? s->lifecycle.inference_runs : 0;
}
double voxtral_stream_audio_ms(const voxtral_stream * s) {
    return s ? samples_to_ms(s->lifecycle.total_samples_received, s->params.sample_rate) : 0.0;
}
const std::vector<int32_t> & voxtral_stream_tokens(const voxtral_stream * s) {
    static const std::vector<int32_t> empty;
    return s ? s->decoder.tokens : empty;
}
const std::string & voxtral_stream_transcript(const voxtral_stream * s) {
    static const std::string empty;
    return s ? s->decoder.transcript : empty;
}
bool voxtral_stream_owns_context(const voxtral_stream * s) {
    return s ? s->owns_context : false;
}
const void * voxtral_stream_context_ptr(const voxtral_stream * s) {
    return s ? static_cast<const void *>(s->ctx) : nullptr;
}
const float * voxtral_stream_pcm_data(const voxtral_stream * s) {
    return (s && !s->frontend.pcm.empty()) ? s->frontend.pcm.data() : nullptr;
}
size_t voxtral_stream_pcm_size(const voxtral_stream * s) {
    return s ? s->frontend.pcm.size() : 0;
}






// Explicit backpressure state (maps the most recent operation's status onto the
// documented feed contract: queue_full = drain events and retry).
voxtral_stream_feed_status voxtral_stream_last_feed_status(const voxtral_stream * s) {
    if (!s) return voxtral_stream_feed_status::failed;
    switch (s->lifecycle.last_status) {
        case voxtral_status_internal::ok:             return voxtral_stream_feed_status::ok;
        case voxtral_status_internal::limit_exceeded:
            return s->events.decoder_backpressured
                ? voxtral_stream_feed_status::queue_full
                : voxtral_stream_feed_status::failed;
        case voxtral_status_internal::busy:           return voxtral_stream_feed_status::would_block;
        case voxtral_status_internal::cancelled:      return voxtral_stream_feed_status::cancelled;
        default:                             return voxtral_stream_feed_status::failed;
    }
}

// ============================================================================
// Events
// ============================================================================
bool voxtral_stream_poll_event_internal(voxtral_stream * s, voxtral_stream_event_internal & out) {
    if (!s || s->events.queue.empty()) return false;
    out = std::move(s->events.queue.front());
    s->events.queue.pop_front();
    return true;
}
size_t voxtral_stream_pending_events(const voxtral_stream * s) {
    return s ? s->events.queue.size() : 0;
}

bool voxtral_stream_set_event_capacity_internal(
    voxtral_stream * stream, size_t max_events)
{
    if (!stream || max_events == 0 || max_events > kMaxEvents ||
        stream->lifecycle.state != voxtral_stream_state_internal::created ||
        !stream->events.queue.empty()) {
        return false;
    }
    stream->events.max_events = max_events;
    return true;
}

// ============================================================================
// Name helpers (diagnostics / machine-readable output)
// ============================================================================
const char * voxtral_stream_state_name(voxtral_stream_state_internal state) {
    switch (state) {
        case voxtral_stream_state_internal::created:   return "created";
        case voxtral_stream_state_internal::running:   return "running";
        case voxtral_stream_state_internal::finishing: return "finishing";
        case voxtral_stream_state_internal::completed: return "completed";
        case voxtral_stream_state_internal::cancelled: return "cancelled";
        case voxtral_stream_state_internal::failed:    return "failed";
    }
    return "unknown";
}

const char * voxtral_stream_status_name(voxtral_status_internal status) {
    switch (status) {
        case voxtral_status_internal::ok:                       return "ok";
        case voxtral_status_internal::invalid_argument:         return "invalid_argument";
        case voxtral_status_internal::invalid_state:            return "invalid_state";
        case voxtral_status_internal::unsupported_audio_format: return "unsupported_audio_format";
        case voxtral_status_internal::cancelled:                return "cancelled";
        case voxtral_status_internal::backend_error:            return "backend_error";
        case voxtral_status_internal::out_of_memory:            return "out_of_memory";
        case voxtral_status_internal::limit_exceeded:           return "limit_exceeded";
        case voxtral_status_internal::busy:                     return "busy";
        case voxtral_status_internal::internal_error:           return "internal_error";
    }
    return "unknown";
}

const char * voxtral_stream_event_name(voxtral_stream_event_type_internal type) {
    switch (type) {
        case voxtral_stream_event_type_internal::token:        return "token";
        case voxtral_stream_event_type_internal::partial_text: return "partial_text";
        case voxtral_stream_event_type_internal::final_text:   return "final_text";
        case voxtral_stream_event_type_internal::error:        return "error";
        case voxtral_stream_event_type_internal::completed:    return "completed";
        case voxtral_stream_event_type_internal::cancelled:    return "cancelled";
    }
    return "unknown";
}

// ============================================================================
// Test seam (internal, test-only — see header). No effect unless a test calls
// these; production code always uses the defaults.
// ============================================================================
void voxtral_stream_test_set_context_factory(voxtral_stream_context_factory_fn factory) {
    g_context_factory = factory ? factory : &voxtral_init_from_model;
}
void voxtral_stream_test_set_context_free(voxtral_stream_context_free_fn free_fn) {
    g_context_free = free_fn ? free_fn : &voxtral_free;
}
void voxtral_stream_test_set_finishing_hook(
    voxtral_stream * stream, voxtral_stream_finishing_hook_fn hook, void * user)
{
    if (!stream) return;
    stream->lifecycle.finishing_hook      = hook;
    stream->lifecycle.finishing_hook_user = user;
}
void voxtral_stream_test_set_max_events(voxtral_stream * stream, size_t max_events) {
    (void) voxtral_stream_set_event_capacity_internal(stream, max_events);
}
bool voxtral_stream_test_events_overflowed(const voxtral_stream * stream) {
    return stream ? stream->events.events_overflowed : false;
}
