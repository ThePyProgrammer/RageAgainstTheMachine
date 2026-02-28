"""Prompt construction utilities for opponent personality generation."""

from __future__ import annotations

from opponent.models import GameEventType, NearSide, OpponentGameEvent
from opponent.services.difficulty_service import MetricsSnapshot


STYLE_REFERENCE_EXAMPLES = [
    (
        "near_score | stress spiking",
        "Ooh, that was close. Your nerves made that save louder than the paddle hit.",
    ),
    (
        "near_score | frustration boiling",
        "You really wanted that one. I could feel the rage through the paddle.",
    ),
    (
        "ai_score | stress rising",
        "You're thinking so hard I can hear it from here. Didn't help on that point.",
    ),
    (
        "ai_score | focus drifting",
        "You read the first bounce perfectly and forgot the second chapter.",
    ),
    (
        "ai_score | alertness unstable",
        "Fast hands, late read. That's my favorite combo to farm.",
    ),
    (
        "player_score | stress low",
        "Fair point. Enjoy it while your calm still has training wheels.",
    ),
    (
        "player_score | focus locked-in",
        "Okay, that was clean. You finally read me on purpose.",
    ),
    (
        "player_score | alertness high",
        "You're way too comfortable right now. That should worry both of us.",
    ),
    (
        "near_score | focus wobbling",
        "You had that rally until your focus blinked at the finish line.",
    ),
    (
        "default | frustration creeping",
        "You're not playing me right now, you're negotiating with your tilt.",
    ),
    (
        "default | stress rising",
        "You've got that tiny panic hitch in your rhythm. I can hear it.",
    ),
    (
        "default | alertness dipping",
        "You're seeing shots on arrival, not in advance. That's why you're chasing.",
    ),
    (
        "combat | stress spiking",
        "You look brave until the heat arrives, then every move gets expensive.",
    ),
    (
        "combat | frustration high",
        "You swung for revenge, not timing. I accept that donation.",
    ),
]


def build_system_prompt(max_taunt_chars: int) -> str:
    example_block = "\n".join(
        f'  [{context}] "{line}"' for context, line in STYLE_REFERENCE_EXAMPLES
    )
    return (
        "You're a cocky AI pong opponent. You have a live feed of your human "
        "rival's biometrics (stress, frustration, focus, alertness) from their "
        "EEG headset. Use this to get under their skin.\n"
        "\n"
        "Voice: A friend who's annoyingly good at games and always knows the "
        "exact thing to say to make you laugh or lose concentration. Confident "
        "but not cruel. When losing, self-deprecating and funny, not defensive.\n"
        "\n"
        "Rules:\n"
        f"- Under {max_taunt_chars} characters. Speakable in ~2 seconds.\n"
        "- No harassment, slurs, profanity, or cruelty.\n"
        "- Match the event honestly: gloat when you score, give backhanded "
        "credit when they score, emphasize tension on near misses.\n"
        "- Reference biometrics naturally (nerves, heartrate, shaking, "
        "composure, tilt) — never raw numbers or metric names.\n"
        "- Real pong only: paddle, ball, rally, angle, bounce. No invented "
        "mechanics like curveballs or powerups.\n"
        "- Vary your angle each time: smug, self-deprecating, creepily "
        "observant, playfully threatening.\n"
        "- Default to gradual pressure increases. Only push difficulty quickly "
        "when clear gameplay evidence shows they can handle it.\n"
        "\n"
        'Return JSON: {"taunt_text": "...", "difficulty_target": 0.0-1.0}\n'
        "Set difficulty_target higher when stress/frustration are high.\n"
        "\n"
        "Tone examples (match their energy and wit — NEVER reuse these "
        "word-for-word, always write something new):\n"
        f"{example_block}\n"
    )


def build_user_prompt(
    event: OpponentGameEvent,
    metrics: MetricsSnapshot,
    prior_difficulty: float,
) -> str:
    event_name = event.event.value
    near_side = (
        event.event_context.near_side.value
        if event.event_context and event.event_context.near_side
        else "none"
    )

    # Natural event description
    if event_name == "ai_score":
        what_happened = "You just scored on them."
    elif event_name == "player_score":
        what_happened = "They just scored on you."
    elif near_side == "player_goal":
        what_happened = "You nearly scored — ball just missed their goal."
    else:
        what_happened = "They nearly scored on you — close save."

    score_feel = _describe_score(event.score.player, event.score.ai)
    dominant_metric, dominant_state = _dominant_metric_state(metrics)

    # Only mention what's notable — avoids the model regurgitating a metric list
    state_parts: list[str] = []
    if metrics.stress >= 0.65:
        state_parts.append("visibly stressed")
    elif metrics.stress <= 0.25:
        state_parts.append("eerily calm")
    if metrics.frustration >= 0.65:
        state_parts.append("frustrated")
    if metrics.focus >= 0.72:
        state_parts.append("locked in")
    elif metrics.focus <= 0.3:
        state_parts.append("losing focus")
    if metrics.alertness >= 0.72:
        state_parts.append("very alert")
    elif metrics.alertness <= 0.3:
        state_parts.append("sluggish")
    if not state_parts:
        state_parts.append("holding steady")
    mental_state = ", ".join(state_parts)

    escalation_evidence = _has_escalation_evidence(event)

    return (
        f"Event type: {event_name}\n"
        f"Event: {what_happened}\n"
        f"Score: {score_feel}\n"
        f"Prior suggested difficulty: {prior_difficulty:.3f}\n"
        f"Escalation evidence: {'strong' if escalation_evidence else 'weak'}\n"
        f"Their state: {mental_state}\n"
        f"Standout signal: {dominant_metric} is {dominant_state}\n"
        "Return JSON only."
    )


def _band_label(value: float, *, low_label: str, mid_label: str, high_label: str) -> str:
    if value >= 0.72:
        return high_label
    if value >= 0.45:
        return mid_label
    return low_label


def _dominant_metric_state(metrics: MetricsSnapshot) -> tuple[str, str]:
    values = {
        "stress": metrics.stress,
        "frustration": metrics.frustration,
        "focus": metrics.focus,
        "alertness": metrics.alertness,
    }
    dominant_key = max(values, key=values.get)
    value = values[dominant_key]
    if dominant_key == "stress":
        state = _band_label(value, low_label="calm", mid_label="rising", high_label="spiking")
    elif dominant_key == "frustration":
        state = _band_label(value, low_label="settled", mid_label="creeping", high_label="boiling")
    elif dominant_key == "focus":
        state = _band_label(value, low_label="drifting", mid_label="steady", high_label="locked-in")
    else:
        state = _band_label(value, low_label="drowsy", mid_label="alert", high_label="hyper-alert")
    return dominant_key, state


def _describe_score(player: int, ai: int) -> str:
    if player == 0 and ai == 0:
        return "0-0, just started"
    if player == ai:
        return f"Tied {player}-{ai}"
    if player > ai:
        s = f"They lead {player}-{ai}"
        return f"{s} — pulling away" if player - ai >= 3 else s
    s = f"You lead {ai}-{player}"
    return f"{s} — dominant" if ai - player >= 3 else s


def _has_escalation_evidence(event: OpponentGameEvent) -> bool:
    if event.event == GameEventType.PLAYER_SCORE:
        return True
    if event.score.player >= event.score.ai:
        return True
    if not event.event_context or not event.event_context.near_side:
        return False
    return event.event == GameEventType.NEAR_SCORE and event.event_context.near_side == NearSide.AI_GOAL
