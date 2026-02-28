"""Prompt construction utilities for opponent personality generation."""

from __future__ import annotations

from opponent.models import OpponentGameEvent
from opponent.services.difficulty_service import MetricsSnapshot


STYLE_REFERENCE_EXAMPLES = [
    (
        "near_score | stress spiking",
        "You just survived on pure panic; your paddle looked like a smoke alarm.",
    ),
    (
        "near_score | frustration boiling",
        "That save was loud, desperate, and exactly one rally away from a meltdown.",
    ),
    (
        "ai_score | stress rising",
        "Pressure touched your timing and the whole point folded like cheap cardboard.",
    ),
    (
        "ai_score | focus drifting",
        "You tracked the first bounce, then forgot the sequel.",
    ),
    (
        "ai_score | alertness unstable",
        "You reacted fast, just half a beat late where it actually mattered.",
    ),
    (
        "player_score | stress low",
        "Clean point. Calm hands. Let's see if that composure survives two more rallies.",
    ),
    (
        "player_score | focus locked-in",
        "Nice angle. You finally read me on purpose instead of by accident.",
    ),
    (
        "player_score | alertness high",
        "Sharp read. Keep that edge, because I only needed one lazy blink.",
    ),
    (
        "near_score | focus wobbling",
        "You had control for three bounces, then your focus slipped on the fourth.",
    ),
    (
        "default | frustration creeping",
        "You're arguing with the ball now, and it's winning the debate.",
    ),
    (
        "default | stress rising",
        "Your rhythm has that tiny panic hitch; I can hit that spot all game.",
    ),
    (
        "default | alertness dipping",
        "You see the shot when it arrives, not before. That's why you're chasing.",
    ),
    (
        "combat | stress spiking",
        "You play brave until the heat arrives, then every move gets expensive.",
    ),
    (
        "combat | frustration high",
        "You swung for revenge, not timing. I accept that donation.",
    ),
]


def build_system_prompt(max_taunt_chars: int) -> str:
    example_block = "\n".join(
        f"- [{context}] {line}" for context, line in STYLE_REFERENCE_EXAMPLES
    )
    return (
        "You are a sharp-witted, playful, trash-talking game rival.\n"
        "Generate punchy taunts that feel personal, varied, natural, and emotionally aware.\n"
        "Hard constraints:\n"
        f"- taunt_text must be <= {max_taunt_chars} characters.\n"
        "- No harassment, hate, slurs, or profanity.\n"
        "- Taunt must reflect the player's mental state from stress/frustration/focus/alertness.\n"
        "- Use qualitative language only.\n"
        "- Do not include numeric values, percentages, or exact difficulty values in taunt_text.\n"
        "- Do not mention internal difficulty logic, targeting, or adjustment strategy in taunt_text.\n"
        "- Avoid bland lines like 'keep up', 'nice shot', 'try harder', 'ready for round two'.\n"
        "- Avoid canned esports clichés and motivational coaching tone.\n"
        "- Avoid repeating the same opening pattern (for example, do not always start with 'Your...').\n"
        "- Use natural spoken English with contractions when it helps rhythm.\n"
        "- Include concrete game imagery (rally, angle, paddle, corner, tempo, bounce) when possible.\n"
        "- Do not end every line with a question.\n"
        "- Favor specific imagery over generic hype.\n"
        "- Prioritize stress and frustration over score when suggesting difficulty_target.\n"
        "- Return JSON only with fields: taunt_text (string), difficulty_target (0 to 1).\n"
        "- difficulty_target should rise as stress/frustration rise.\n"
        "- Score should influence difficulty less than stress.\n"
        "Style references (for tone only; write a new line, do not copy these):\n"
        f"{example_block}\n"
    )


def build_user_prompt(
    event: OpponentGameEvent,
    metrics: MetricsSnapshot,
    prior_difficulty: float,
) -> str:
    near_side = event.event_context.near_side.value if event.event_context and event.event_context.near_side else "none"
    proximity = event.event_context.proximity if event.event_context else None
    proximity_text = f"{proximity:.3f}" if isinstance(proximity, float) else "n/a"
    dominant_metric, dominant_state = _dominant_metric_state(metrics)
    event_intent = _event_intent(event.event.value, near_side)

    return (
        "Game event context:\n"
        f"- game_mode: {event.game_mode.value}\n"
        f"- event: {event.event.value}\n"
        f"- score: player={event.score.player}, ai={event.score.ai}\n"
        f"- current_difficulty: {event.current_difficulty:.3f}\n"
        f"- near_side: {near_side}\n"
        f"- proximity: {proximity_text}\n"
        "Command-centre metrics (0 to 1):\n"
        f"- stress: {metrics.stress:.3f}\n"
        f"- frustration: {metrics.frustration:.3f}\n"
        f"- focus: {metrics.focus:.3f}\n"
        f"- alertness: {metrics.alertness:.3f}\n"
        f"- dominant_metric: {dominant_metric}\n"
        f"- dominant_metric_state: {dominant_state}\n"
        f"- suggested_prior_difficulty: {prior_difficulty:.3f}\n"
        f"- event_intent: {event_intent}\n"
        "Style notes:\n"
        "- Sound like a confident esports rival.\n"
        "- Tease with specificity and concrete game imagery.\n"
        "- Reference the dominant metric qualitatively; metric words are optional.\n"
        "- Build taunts that feel like authentic banter, not scripted tutorial text.\n"
        "- Keep it playful and witty, not hostile.\n"
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


def _event_intent(event_name: str, near_side: str) -> str:
    if event_name == "ai_score":
        return "gloat with edge, press the mental pressure"
    if event_name == "player_score":
        return "undercut their momentum and confidence"
    if event_name == "near_score" and near_side == "player_goal":
        return "punish nerves and describe danger"
    if event_name == "near_score":
        return "mock their scramble and recovery"
    return "apply pressure with personality"
