// ============================================================================
// Streaming audio frontend — implementation.
//
// Drives the incremental PCM -> log-Mel frontend and the causal per-layer-KV
// encoder during feed(): lazy frontend/encoder bringup, the one-shot streaming
// left zero-pad injection, and the fixed-cadence drain of newly-stable Mel
// frames into the encoder (which mirrors output into the device ring the
// adapter reads). No decoder or event logic lives here. See
// voxtral-stream-internal.h.
// ============================================================================

#include "voxtral-stream-internal.h"

#include <algorithm>

// Lazily create the incremental Mel frontend for an inference stream on first
// feed. Deferred (not done in create) so the model-free ownership tests, which
// substitute an opaque sentinel context and never feed, never dereference it.
// Returns false only on a genuine allocation failure. A lifecycle-only stream
// (no context) or a context without frontend tables leaves mel_fe null and keeps
// the full-PCM path. A valid context may be owned internally or borrowed by the
// stable public C adapter.
bool ensure_frontend(voxtral_stream * s) {
    if (s->frontend.mel_fe) return true;
    if (!s->ctx) return true;
    const float * hann    = voxtral_ctx_hann_window(s->ctx);
    const float * filters = voxtral_ctx_mel_filters(s->ctx);
    if (!hann || !filters) return true;
    s->frontend.mel_fe = voxtral_mel_frontend_create(hann, filters);
    if (!s->frontend.mel_fe) {
        set_error(s, voxtral_status_internal::out_of_memory, "failed to create incremental Mel frontend");
        return false;
    }
    voxtral_mel_frontend_set_retain_history(s->frontend.mel_fe, s->params.retain_mel_history);
    return true;
}
// Inject the streaming left zero-padding into the Mel frontend exactly once,
// before the first real sample. Mirrors the batch left_pad prefix. Bounded: the
// injected silence is emitted to stable Mel frames and the PCM tail compacted
// before this returns, so it does not create a lasting 40960-sample buffer.
void ensure_left_pad(voxtral_stream * s) {
    if (s->frontend.mel_fe && !s->frontend.left_pad_injected) {
        s->frontend.left_pad_injected = true;
        context_profile_scope profile(s->ctx, voxtral_profile_stage::mel_compute);
        voxtral_mel_frontend_feed_silence(s->frontend.mel_fe, (size_t) kStreamLeftPad);
    }
}
voxtral_mel_metrics stream_mel_metrics(const voxtral_stream * s) {
    return (s && s->frontend.mel_fe) ? voxtral_mel_frontend_metrics(s->frontend.mel_fe) : voxtral_mel_metrics{};
}
voxtral_encoder_metrics stream_encoder_metrics(const voxtral_stream * s) {
    return (s && s->frontend.enc) ? voxtral_encoder_stream_metrics(s->frontend.enc) : voxtral_encoder_metrics{};
}
// Lazily create the incremental causal encoder once the Mel frontend exists (which
// implies a real execution context with frontend tables). Model-free ownership
// tests use a sentinel context with no frontend, so mel_fe stays null and no
// encoder is created. Returns false only on a genuine allocation failure.
bool ensure_encoder(voxtral_stream * s) {
    if (s->frontend.enc) return true;
    if (!s->frontend.mel_fe || !s->ctx) return true;
    s->frontend.enc = voxtral_encoder_stream_create(s->ctx);
    if (!s->frontend.enc) {
        set_error(s, voxtral_status_internal::out_of_memory, "failed to create incremental encoder");
        return false;
    }
    // Incremental path: mirror encoder output into the device ring from the first
    // batch on, so the adapter can read complete groups on-device. The finish-only
    // reference leaves the ring copy off (no extra per-batch work).
    //
    // The incremental adapter/decoder reads encoder output from that device ring,
    // which ONLY the per-layer KV encoder fills. If the reference (bounded-window
    // recompute) encoder is selected (VOXTRAL_ENCODER_STRATEGY=reference), couple
    // the decoder to the coherent reference finish-only path rather than feed the
    // adapter an empty ring. This is a deterministic configuration coupling, not a
    // runtime error fallback.
    if (s->decoder.incremental && !voxtral_encoder_stream_uses_kv(s->frontend.enc)) {
        s->decoder.incremental = false;
    }
    if (s->decoder.incremental) voxtral_ctx_set_enc_out_ring_active(s->ctx, true);
    return true;
}
// Hand every newly-stable Mel frame from the frontend to the incremental encoder,
// which runs any encoder chunks that became available. Called after each feed and
// once more (with the finish-flushed frames) at finish().
bool drain_mel_to_encoder(voxtral_stream * s, bool final) {
    if (!s->frontend.enc || !s->frontend.mel_fe) return true;
    const int64_t total = voxtral_mel_frontend_frame_count(s->frontend.mel_fe);
    if (total <= s->frontend.enc_pushed_frames) return true;
    int64_t base = s->frontend.enc_pushed_frames;
    if (base < voxtral_mel_frontend_frames_base(s->frontend.mel_fe) || total < base) {
        set_error(s, voxtral_status_internal::backend_error, "incremental Mel history base advanced past encoder cursor");
        return false;
    }
    // The terminal frontend flush is already a bounded suffix. Preserve it as
    // one push so the encoder can batch right-padding work; ordinary feed keeps
    // the fixed absolute cadence below for feed-plan invariance.
    if (final) {
        const int64_t remaining = total - base;
        const float * frames = voxtral_mel_frontend_frame_data(s->frontend.mel_fe, base);
        if (!frames || remaining > INT32_MAX ||
            !voxtral_encoder_stream_push_mel_final(
                s->frontend.enc, frames, base, (int32_t) remaining)) {
            set_error(s, voxtral_status_internal::backend_error,
                      "incremental encoder rejected terminal Mel frames");
            return false;
        }
        s->frontend.enc_pushed_frames = total;
        voxtral_mel_frontend_discard_before(s->frontend.mel_fe, total);
        return capture_new_encoder_output_sha(s);
    }

    // Feed the encoder on a fixed absolute Mel cadence: eight Mel frames are
    // exactly four encoder frames, i.e. one future adapter group. A one-shot
    // feed and an 80/160 ms feed therefore expose identical graph-ready
    // boundaries to the KV scheduler instead of letting caller chunk size alter
    // how much future Mel happens to be resident when a graph starts.
    constexpr int32_t kMelDrainSlice = 8;
    while (base < total) {
        const int32_t n = (int32_t) std::min<int64_t>(kMelDrainSlice, total - base);
        const float * frames = voxtral_mel_frontend_frame_data(s->frontend.mel_fe, base);
        if (!frames) {
            set_error(s, voxtral_status_internal::backend_error, "incremental Mel frame is outside retained window");
            return false;
        }
        if (!voxtral_encoder_stream_push_mel(s->frontend.enc, frames, base, n)) {
            set_error(s, voxtral_status_internal::backend_error, "incremental encoder rejected Mel frames");
            return false;
        }
        if (!capture_new_encoder_output_sha(s)) return false;
        base += n;
        s->frontend.enc_pushed_frames = base;
        voxtral_mel_frontend_discard_before(s->frontend.mel_fe, base);
    }
    return true;
}

// ---- introspection accessors ----
// --- Incremental Mel frontend introspection --------------------------------
bool voxtral_stream_uses_incremental_mel(const voxtral_stream * s) {
    return s ? (s->frontend.mel_fe != nullptr) : false;
}
int64_t voxtral_stream_mel_frames(const voxtral_stream * s) {
    return stream_mel_metrics(s).frames_total;
}
int64_t voxtral_stream_mel_frames_before_finish(const voxtral_stream * s) {
    return stream_mel_metrics(s).frames_during_feed;
}
int64_t voxtral_stream_mel_frames_flushed_at_finish(const voxtral_stream * s) {
    return stream_mel_metrics(s).frames_at_finish;
}
int64_t voxtral_stream_dft_frames_computed(const voxtral_stream * s) {
    return stream_mel_metrics(s).dft_frames_computed;
}
int64_t voxtral_stream_pcm_retained_samples(const voxtral_stream * s) {
    return stream_mel_metrics(s).pcm_retained;
}
int64_t voxtral_stream_pcm_peak_retained_samples(const voxtral_stream * s) {
    return stream_mel_metrics(s).pcm_peak_retained;
}
int64_t voxtral_stream_pcm_base_sample(const voxtral_stream * s) {
    return stream_mel_metrics(s).pcm_base;
}
bool voxtral_stream_full_pcm_buffered_at_finish(const voxtral_stream * s) {
    return s ? s->frontend.full_pcm_buffered_at_finish : false;
}
bool voxtral_stream_mel_history_retained(const voxtral_stream * s) {
    return s && s->frontend.mel_fe ? (s->params.retain_mel_history && voxtral_mel_frontend_frames_base(s->frontend.mel_fe) == 0) : false;
}
const float * voxtral_stream_mel_data(const voxtral_stream * s) {
    return (s && !s->frontend.mel_even.empty()) ? s->frontend.mel_even.data() : nullptr;
}
int32_t voxtral_stream_mel_data_frames(const voxtral_stream * s) {
    return s ? s->frontend.mel_even_frames : 0;
}
// --- Incremental causal encoder introspection ------------------------------
bool voxtral_stream_uses_incremental_encoder(const voxtral_stream * s) {
    return s ? (s->frontend.enc != nullptr) : false;
}
int64_t voxtral_stream_encoder_frames(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderOutputFrames;
}
int64_t voxtral_stream_encoder_frames_before_finish(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderFramesBeforeFinish;
}
int64_t voxtral_stream_encoder_frames_flushed_at_finish(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderFramesFlushedAtFinish;
}
int64_t voxtral_stream_encoder_executions(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderExecutions;
}
int64_t voxtral_stream_encoder_input_frames_processed(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderInputFramesProcessed;
}
int64_t voxtral_stream_encoder_frames_recomputed(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderFramesRecomputed;
}
int64_t voxtral_stream_encoder_max_window_frames(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderMaxWindowFrames;
}
int64_t voxtral_stream_encoder_peak_context_frames(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderPeakContextFrames;
}
int64_t voxtral_stream_encoder_context_frames_retained(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderContextFramesRetained;
}
int64_t voxtral_stream_encoder_state_bytes(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderStateBytes;
}
int64_t voxtral_stream_encoder_output_accumulated_bytes(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderOutputAccumulatedBytes;
}
voxtral_encoder_metrics voxtral_stream_encoder_metrics_full(const voxtral_stream * s) {
    return stream_encoder_metrics(s);
}
const float * voxtral_stream_encoder_output_data(const voxtral_stream * s) {
    return (s && s->frontend.enc) ? voxtral_encoder_stream_output(s->frontend.enc) : nullptr;
}
int32_t voxtral_stream_encoder_output_frames_count(const voxtral_stream * s) {
    return (s && s->frontend.enc) ? voxtral_encoder_stream_output_frames(s->frontend.enc) : 0;
}
// Actual encoder-output device->host bytes performed by this stream. Hard gate:
// 0 in the incremental production path (the adapter reads the device ring).
int64_t voxtral_stream_encoder_output_d2h_bytes(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderOutputD2hBytes;
}
