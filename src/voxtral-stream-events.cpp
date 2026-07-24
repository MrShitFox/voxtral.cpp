// ============================================================================
// Streaming event queue — implementation.
//
// Owns the bounded event deque and its ordering/backpressure guarantees:
//   * mandatory events (token / final_text / error / completed) are never
//     silently dropped — a full queue is surfaced as explicit feed
//     backpressure (queue_full) so the caller drains and retries;
//   * replaceable PARTIAL_TEXT snapshots coalesce to the newest revision;
//   * the finish() terminal flush (finalizing_flush) bypasses the bound so the
//     bounded final tail is always delivered.
// See voxtral-stream-internal.h.
// ============================================================================

#include "voxtral-stream-internal.h"

#include <string>
#include <utility>

// Bookkeeping after a successful enqueue: aggregate + per-type counters and the
// running queue-depth high-watermark.
static void note_enqueued(voxtral_stream * s, voxtral_stream_event_type type) {
    s->events.events_emitted++;
    if (type == voxtral_stream_event_type::token) s->events.token_events++;
    else if (type == voxtral_stream_event_type::partial_text) s->events.partial_events++;
    if (s->events.queue.size() > s->events.event_queue_high_watermark)
        s->events.event_queue_high_watermark = s->events.queue.size();
}
// Bounded push for mandatory events. Returns false WITHOUT enqueuing when the
// queue is at its bound — the caller turns that into explicit backpressure
// (feed → queue_full) and never drops the event. During finish()'s terminal
// flush (finalizing_flush) the small bounded finish tail is always delivered, so
// FINAL_TEXT / COMPLETED / ERROR can never be lost.
bool push_event(voxtral_stream * s, voxtral_stream_event ev) {
    if (!s->events.finalizing_flush && s->events.queue.size() >= s->events.max_events) {
        s->events.events_overflowed = true;
        s->events.event_queue_overflow_attempts++;
        set_error(s, voxtral_status::limit_exceeded,
                  std::string("event queue full (bound ") + std::to_string(s->events.max_events) +
                  "): backpressure on " + voxtral_stream_event_name(ev.type) + " event");
        return false;
    }
    const auto type = ev.type;
    s->events.queue.push_back(std::move(ev));
    note_enqueued(s, type);
    return true;
}
void emit_final_and_completed(voxtral_stream * s) {
    context_profile_scope profile(s ? s->ctx : nullptr, voxtral_profile_stage::event_processing);
    const double t_ms = samples_to_ms(s->lifecycle.total_samples_received, s->params.sample_rate);
    {
        voxtral_stream_event ev;
        ev.type       = voxtral_stream_event_type::final_text;
        ev.text       = s->decoder.transcript;   // owned copy
        ev.t_audio_ms = t_ms;
        push_event(s, std::move(ev));
    }
    {
        voxtral_stream_event ev;
        ev.type       = voxtral_stream_event_type::completed;
        ev.t_audio_ms = t_ms;
        push_event(s, std::move(ev));
    }
}
void emit_error(voxtral_stream * s, voxtral_status status, const std::string & msg) {
    voxtral_stream_event ev;
    ev.type       = voxtral_stream_event_type::error;
    ev.text       = msg;
    ev.error_code = static_cast<int32_t>(status);
    ev.t_audio_ms = samples_to_ms(s->lifecycle.total_samples_received, s->params.sample_rate);
    push_event(s, std::move(ev));
}
// Real-audio end sample for an audio position, derived from the 80 ms cadence and
// the injected left pad. Clamped to 0 for positions inside the left-pad region.
static int64_t audio_end_sample_for(int64_t position) {
    const int64_t s = (position + 1) * VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK - kStreamLeftPad;
    return s > 0 ? s : 0;
}
// Push an event, coalescing PARTIAL_TEXT: a full queue keeps only the newest
// revision (partials are a replaceable snapshot, so coalescing/dropping one is
// allowed and is NOT a mandatory-event drop). Mandatory events (token / final /
// error / completed) never coalesce: a full queue returns false so the caller
// can raise explicit backpressure instead of losing the event. The finish()
// terminal flush (finalizing_flush) bypasses the bound entirely.
static bool push_event_coalesced(voxtral_stream * s, voxtral_stream_event ev) {
    if (ev.type == voxtral_stream_event_type::partial_text &&
        s->events.aggressive_partial_coalescing) {
        // A multi-minute one-shot feed cannot let replaceable partial snapshots
        // consume half of the mandatory TOKEN queue. Keep exactly the newest
        // snapshot, moving it behind the current token so observable ordering
        // remains coherent. The queue stays fixed-size; token events are never
        // coalesced or dropped.
        for (auto it = s->events.queue.begin(); it != s->events.queue.end(); ++it) {
            if (it->type == voxtral_stream_event_type::partial_text) {
                s->events.queue.erase(it);
                s->events.queue.push_back(std::move(ev));
                s->events.partial_events_coalesced++;
                return true;
            }
        }
    }
    if (s->events.finalizing_flush || s->events.queue.size() < s->events.max_events) {
        const auto type = ev.type;
        s->events.queue.push_back(std::move(ev));
        note_enqueued(s, type);
        return true;
    }
    if (ev.type == voxtral_stream_event_type::partial_text) {
        for (auto it = s->events.queue.rbegin(); it != s->events.queue.rend(); ++it) {
            if (it->type == voxtral_stream_event_type::partial_text) {
                *it = std::move(ev);                 // keep only the newest revision
                s->events.partial_events_coalesced++;
                return true;
            }
        }
        s->events.partial_events_coalesced++;
        return false;   // no prior partial to replace; drop newest (partials are lossy)
    }
    return push_event(s, std::move(ev));   // mandatory: caller raises backpressure
}
// Returns false when the mandatory TOKEN event did not fit the bounded queue; the
// caller must NOT advance the decoder past it (the token is stashed and retried).
// The sequence id is committed only on a successful enqueue, so no gaps appear.
bool emit_token_event(voxtral_stream * s, int32_t token, int64_t position, bool special) {
    voxtral_stream_event ev;
    ev.type                    = voxtral_stream_event_type::token;
    ev.token                   = token;
    ev.text                    = voxtral_token_piece_internal(s->model, token);
    ev.special                 = special;
    ev.sequence                = s->decoder.token_sequence + 1;   // tentative
    ev.decoder_position        = position;
    ev.audio_end_sample        = audio_end_sample_for(position);
    ev.emitted_at_monotonic_ns = stream_now_ns();
    ev.t_audio_ms              = (double) ev.audio_end_sample * 1000.0 / (double) s->params.sample_rate;
    if (!push_event_coalesced(s, std::move(ev))) return false;   // queue full → backpressure
    ++s->decoder.token_sequence;                                          // commit only on success
    return true;
}
void emit_partial_text_event(voxtral_stream * s, int64_t position) {
    voxtral_stream_event ev;
    ev.type               = voxtral_stream_event_type::partial_text;
    ev.text               = s->decoder.partial_text;   // full snapshot
    ev.revision           = ++s->decoder.partial_revision;
    ev.stable_prefix_bytes= s->decoder.partial_stable_bytes;
    ev.audio_end_sample   = audio_end_sample_for(position);
    ev.t_audio_ms         = (double) ev.audio_end_sample * 1000.0 / (double) s->params.sample_rate;
    push_event_coalesced(s, std::move(ev));
}
