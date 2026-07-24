// ============================================================================
// Streaming latency / backlog telemetry — implementation.
//
// Fixed-memory realtime instrumentation: the monotonic stream timeline, the
// per-group eligibility ring (when each committed audio group became available)
// and the bounded per-stage backlog reservoirs. None of this influences the
// inference result; it is sampled once per feed and never allocates per event.
// See voxtral-stream-internal.h. (The backlog percentile export and the
// per-subsystem accessors live with the introspection getters.)
// ============================================================================

#include "voxtral-stream-internal.h"

#include <algorithm>
#include <vector>

double stream_elapsed_ms(const voxtral_stream * s) {
    return s->telemetry.timeline_start_ns > 0
        ? (double) (stream_now_ns() - s->telemetry.timeline_start_ns) / 1e6 : 0.0;
}
// Record when each committed audio group became available.  Adapter and decoder
// use the same absolute group index; modulo storage is safe because the decoder
// cannot consume an entry after the bounded audio-embedding ring overwrote it.
void note_group_eligibility(voxtral_stream * s, int64_t start, int32_t count,
                            int32_t capacity) {
    if (!s || capacity <= 0 || count <= 0) return;
    if ((int32_t) s->telemetry.eligibility_absolute_group.size() != capacity) {
        s->telemetry.eligibility_absolute_group.assign((size_t) capacity, -1);
        s->telemetry.eligibility_arrival_ms.assign((size_t) capacity, 0.0);
    }
    for (int32_t i = 0; i < count; ++i) {
        const int64_t absolute = start + i;
        const size_t slot = (size_t) (absolute % capacity);
        s->telemetry.eligibility_absolute_group[slot] = absolute;
        s->telemetry.eligibility_arrival_ms[slot] = s->telemetry.current_audio_availability_ms;
    }
}
double group_eligibility_ms(const voxtral_stream * s, int64_t absolute) {
    if (!s || absolute < 0 || s->telemetry.eligibility_absolute_group.empty()) return -1.0;
    const size_t slot = (size_t) (absolute %
        (int64_t) s->telemetry.eligibility_absolute_group.size());
    return s->telemetry.eligibility_absolute_group[slot] == absolute
        ? s->telemetry.eligibility_arrival_ms[slot] : -1.0;
}
void sample_stage_backlogs(voxtral_stream * s) {
    if (!s || !s->decoder.incremental || !s->frontend.enc || !s->frontend.mel_fe ||
        s->lifecycle.total_samples_received == 0) return;
    constexpr double group_ms =
        (double) VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK * 1000.0 / VOXTRAL_SAMPLE_RATE;
    const double audio_s =
        (double) s->lifecycle.total_samples_received / (double) VOXTRAL_SAMPLE_RATE;

    // A group is backlog only after all Mel needed for its four encoder frames
    // exists. Fixed causal/model latency is residence, not queued work.
    const int64_t stable_mel = voxtral_mel_frontend_frame_count(s->frontend.mel_fe);
    const int64_t realizable_frames = stable_mel < 2 ? 0 : (stable_mel - 2) / 2 + 1;
    const int64_t encoder_ready_groups = realizable_frames / VOXTRAL_DOWNSAMPLE_FACTOR;
    const int64_t encoder_done_groups =
        voxtral_encoder_stream_output_frames(s->frontend.enc) / VOXTRAL_DOWNSAMPLE_FACTOR;
    const int64_t encoder_queued =
        std::max<int64_t>(0, encoder_ready_groups - encoder_done_groups);
    const int64_t adapter_queued =
        std::max<int64_t>(0, encoder_done_groups - s->decoder.adapter_groups_committed);

    const int64_t prompt_prefix = (int64_t) s->decoder.prompt_ids.size() - 1;
    const int64_t decoder_ready = s->decoder.adapter_groups_committed > prompt_prefix
        ? s->decoder.adapter_groups_committed - prompt_prefix : 0;
    const int64_t decoder_done = s->decoder.decoder_prefill_done
        ? std::max<int64_t>(0, s->decoder.decoder_position - prompt_prefix) : 0;
    const int64_t decoder_queued = std::max<int64_t>(0, decoder_ready - decoder_done);

    s->telemetry.encoder_backlog.add(audio_s, (double) encoder_queued * group_ms);
    s->telemetry.adapter_backlog.add(audio_s, (double) adapter_queued * group_ms);
    s->telemetry.decoder_backlog.add(audio_s, (double) decoder_queued * group_ms);
}

// ---- introspection accessors ----
double voxtral_stream_finish_frontend_ms(const voxtral_stream * s) {
    return s ? s->telemetry.finish_frontend_ms : 0.0;
}
double voxtral_stream_finish_encoder_ms(const voxtral_stream * s) {
    return s ? s->telemetry.finish_encoder_ms : 0.0;
}
double voxtral_stream_finish_decoder_ms(const voxtral_stream * s) {
    return s ? s->telemetry.finish_decoder_ms : 0.0;
}
double voxtral_stream_first_mel_absolute_ms(const voxtral_stream * s) {
    return s ? s->telemetry.first_mel_absolute_ms : 0.0;
}
double voxtral_stream_first_adapter_commit_ms(const voxtral_stream * s) {
    return s ? s->telemetry.first_adapter_commit_ms : 0.0;
}
double voxtral_stream_first_decoder_step_ms(const voxtral_stream * s) {
    return s ? s->telemetry.first_decoder_step_ms : 0.0;
}
double voxtral_stream_first_token_ms(const voxtral_stream * s) {
    return s ? s->telemetry.first_token_ms : 0.0;
}
double voxtral_stream_first_visible_text_ms(const voxtral_stream * s) {
    return s ? s->telemetry.first_visible_text_ms : 0.0;
}
double voxtral_stream_first_decoder_step_eligibility_ms(const voxtral_stream * s) {
    return s ? s->telemetry.first_decoder_step_eligibility_ms : -1.0;
}
double voxtral_stream_first_decoder_step_overhead_ms(const voxtral_stream * s) {
    return s ? s->telemetry.first_decoder_step_overhead_ms : -1.0;
}
double voxtral_stream_first_token_eligibility_ms(const voxtral_stream * s) {
    return s ? s->telemetry.first_token_eligibility_ms : -1.0;
}
double voxtral_stream_first_token_overhead_ms(const voxtral_stream * s) {
    return s ? s->telemetry.first_token_overhead_ms : -1.0;
}
double voxtral_stream_first_partial_eligibility_ms(const voxtral_stream * s) {
    return s ? s->telemetry.first_partial_eligibility_ms : -1.0;
}
double voxtral_stream_first_partial_overhead_ms(const voxtral_stream * s) {
    return s ? s->telemetry.first_partial_overhead_ms : -1.0;
}
// ---- Event-queue telemetry (events_dropped is a hard gate == 0) ------------
uint64_t voxtral_stream_events_emitted(const voxtral_stream * s) {
    return s ? s->events.events_emitted : 0;
}
uint64_t voxtral_stream_token_events(const voxtral_stream * s) {
    return s ? s->events.token_events : 0;
}
uint64_t voxtral_stream_partial_events(const voxtral_stream * s) {
    return s ? s->events.partial_events : 0;
}
uint64_t voxtral_stream_partial_events_coalesced(const voxtral_stream * s) {
    return s ? s->events.partial_events_coalesced : 0;
}
uint64_t voxtral_stream_event_queue_high_watermark(const voxtral_stream * s) {
    return s ? s->events.event_queue_high_watermark : 0;
}
uint64_t voxtral_stream_event_queue_overflow_attempts(const voxtral_stream * s) {
    return s ? s->events.event_queue_overflow_attempts : 0;
}
uint64_t voxtral_stream_events_dropped(const voxtral_stream * s) {
    return s ? s->events.events_dropped : 0;
}
static voxtral_backlog_metrics backlog_metrics_from(
    const stage_backlog_series & series) {
    voxtral_backlog_metrics out;
    out.count = series.seen;
    out.finalMs = series.final_ms;
    out.deadlineMisses = series.deadline_misses;
    out.deadlineMissRate = series.seen
        ? (double) series.deadline_misses / (double) series.seen : 0.0;
    if (series.stored == 0) return out;

    std::vector<double> sorted(
        series.values.begin(), series.values.begin() + (std::ptrdiff_t) series.stored);
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) {
        const double x = p * (double) (sorted.size() - 1);
        const size_t lo = (size_t) x;
        const size_t hi = std::min(lo + 1, sorted.size() - 1);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (x - (double) lo);
    };
    out.p50Ms = pct(0.50);
    out.p95Ms = pct(0.95);
    out.p99Ms = pct(0.99);
    out.maxMs = sorted.back();

    if (series.stored > 1) {
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
        for (size_t i = 0; i < series.stored; ++i) {
            const double x = series.audio_seconds[i];
            const double y = series.values[i];
            sx += x; sy += y; sxx += x * x; sxy += x * y;
        }
        const double n = (double) series.stored;
        const double denom = n * sxx - sx * sx;
        if (denom > 0.0) out.slopeMsPerSec = (n * sxy - sx * sy) / denom;
    }
    return out;
}
voxtral_backlog_metrics voxtral_stream_encoder_backlog(const voxtral_stream * s) {
    return s ? backlog_metrics_from(s->telemetry.encoder_backlog) : voxtral_backlog_metrics{};
}
voxtral_backlog_metrics voxtral_stream_adapter_backlog(const voxtral_stream * s) {
    return s ? backlog_metrics_from(s->telemetry.adapter_backlog) : voxtral_backlog_metrics{};
}
voxtral_backlog_metrics voxtral_stream_decoder_backlog(const voxtral_stream * s) {
    return s ? backlog_metrics_from(s->telemetry.decoder_backlog) : voxtral_backlog_metrics{};
}
