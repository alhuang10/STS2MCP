#!/usr/bin/env python3
"""Small OpenAI-compatible model harness for STS2_MCP.

The loop is intentionally boring:
  1. Fetch current STS2 state from the mod's REST API.
  2. Ask an OpenAI-compatible model for exactly one JSON action.
  3. Validate and execute that action.
  4. Repeat.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


Json = dict[str, Any]
BodyBuilder = Callable[[Json], Json]

DEFAULT_LOCAL_LLM_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LOCAL_LLM_MODEL = "qwen3.5-9b"
DEFAULT_RUNPOD_LLM_MODEL = "Qwen3.6-27B"
DEFAULT_RUNPOD_PORT = 8000
DEFAULT_RUNPOD_DOMAIN = "proxy.runpod.net"
DEFAULT_HTTP_USER_AGENT = "STS2MCP-Harness/0.1"


@dataclass(frozen=True)
class ActionSpec:
    description: str
    build_body: BodyBuilder | None = None


def optional_target(body: Json, args: Json) -> Json:
    target = coerce_target_value(args.get("target", args.get("target_entity_id")))
    if target not in (None, ""):
        body["target"] = target
    return body


def optional_seed(body: Json, args: Json) -> Json:
    seed = args.get("seed")
    if seed not in (None, ""):
        body["seed"] = str(seed)
    return body


def int_arg(args: Json, *names: str) -> int:
    for name in names:
        if name in args:
            return int(args[name])
    raise ValueError(f"missing integer argument; expected one of {', '.join(names)}")


def str_arg(args: Json, *names: str) -> str:
    for name in names:
        value = args.get(name)
        if value is not None:
            return str(value)
    raise ValueError(f"missing string argument; expected one of {', '.join(names)}")


ACTION_FIELD_NAMES = {
    "action",
    "action_name",
    "tool",
    "tool_name",
    "name",
    "next_action",
    "rationale",
}


TOP_LEVEL_ARG_FIELDS = {
    "option",
    "option_id",
    "seed",
    "slot",
    "target",
    "target_entity_id",
    "target_name",
    "card_index",
    "index",
    "reward_index",
    "node_index",
    "option_index",
    "item_index",
    "bundle_index",
    "relic_index",
    "tool",
    "x",
    "y",
}


def coerce_target_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        for key in ("entity_id", "id", "target", "name"):
            nested = value.get(key)
            if nested not in (None, ""):
                return str(nested)
        return None
    return str(value)


def normalize_arg_values(args: Json) -> Json:
    normalized_args = dict(args)
    if "target" not in normalized_args and "target_entity_id" in normalized_args:
        normalized_args["target"] = normalized_args["target_entity_id"]
    if "target" in normalized_args:
        target = coerce_target_value(normalized_args["target"])
        if target is None:
            normalized_args.pop("target", None)
        else:
            normalized_args["target"] = target
    return normalized_args


def collect_model_args(raw: Json, nested_args: Json | None) -> Json:
    args = raw.get("args") or raw.get("arguments") or nested_args or {}
    if not isinstance(args, dict):
        raise ValueError("model response field 'args' must be an object")
    merged = dict(args)
    for key in TOP_LEVEL_ARG_FIELDS:
        if key in raw and key not in ACTION_FIELD_NAMES and key not in merged:
            merged[key] = raw[key]
    return normalize_arg_values(merged)


ACTION_SPECS: dict[str, ActionSpec] = {
    "wait": ActionSpec("Do nothing briefly, then poll game state again."),
    "stop": ActionSpec("Stop the harness cleanly."),
    "menu_select": ActionSpec(
        "Select a visible menu or game-over option by option id.",
        lambda a: optional_seed(
            {"action": "menu_select", "option": str_arg(a, "option", "option_id")},
            a,
        ),
    ),
    "use_potion": ActionSpec(
        "Use a potion slot, optionally with target.",
        lambda a: optional_target({"action": "use_potion", "slot": int_arg(a, "slot")}, a),
    ),
    "discard_potion": ActionSpec(
        "Discard a potion slot.",
        lambda a: {"action": "discard_potion", "slot": int_arg(a, "slot")},
    ),
    "proceed_to_map": ActionSpec(
        "Proceed from rewards/rest/shop/treasure to the map.",
        lambda a: {"action": "proceed"},
    ),
    "combat_play_card": ActionSpec(
        "Play a card from hand by 0-based card_index, optionally with target entity id.",
        lambda a: optional_target(
            {"action": "play_card", "card_index": int_arg(a, "card_index", "index")},
            a,
        ),
    ),
    "combat_end_turn": ActionSpec(
        "End the current player turn.",
        lambda a: {"action": "end_turn"},
    ),
    "combat_select_card": ActionSpec(
        "Select a card during in-combat selection prompts.",
        lambda a: {"action": "combat_select_card", "card_index": int_arg(a, "card_index", "index")},
    ),
    "combat_confirm_selection": ActionSpec(
        "Confirm an in-combat card selection.",
        lambda a: {"action": "combat_confirm_selection"},
    ),
    "rewards_claim": ActionSpec(
        "Claim a post-combat reward by reward_index.",
        lambda a: {"action": "claim_reward", "index": int_arg(a, "reward_index", "index")},
    ),
    "rewards_pick_card": ActionSpec(
        "Pick a card from a card reward screen by card_index.",
        lambda a: {"action": "select_card_reward", "card_index": int_arg(a, "card_index", "index")},
    ),
    "rewards_skip_card": ActionSpec(
        "Skip the current card reward.",
        lambda a: {"action": "skip_card_reward"},
    ),
    "map_choose_node": ActionSpec(
        "Choose a map node by node_index.",
        lambda a: {"action": "choose_map_node", "index": int_arg(a, "node_index", "index")},
    ),
    "rest_choose_option": ActionSpec(
        "Choose a rest site option by option_index.",
        lambda a: {"action": "choose_rest_option", "index": int_arg(a, "option_index", "index")},
    ),
    "shop_purchase": ActionSpec(
        "Purchase a shop item by item_index.",
        lambda a: {"action": "shop_purchase", "index": int_arg(a, "item_index", "index")},
    ),
    "event_choose_option": ActionSpec(
        "Choose an event option by option_index.",
        lambda a: {"action": "choose_event_option", "index": int_arg(a, "option_index", "index")},
    ),
    "event_advance_dialogue": ActionSpec(
        "Advance event dialogue.",
        lambda a: {"action": "advance_dialogue"},
    ),
    "deck_select_card": ActionSpec(
        "Select or toggle a card in an out-of-combat deck-selection screen.",
        lambda a: {"action": "select_card", "index": int_arg(a, "card_index", "index")},
    ),
    "deck_confirm_selection": ActionSpec(
        "Confirm an out-of-combat deck-selection screen.",
        lambda a: {"action": "confirm_selection"},
    ),
    "deck_cancel_selection": ActionSpec(
        "Cancel or skip an out-of-combat deck-selection screen.",
        lambda a: {"action": "cancel_selection"},
    ),
    "bundle_select": ActionSpec(
        "Open a bundle preview by bundle_index.",
        lambda a: {"action": "select_bundle", "index": int_arg(a, "bundle_index", "index")},
    ),
    "bundle_confirm_selection": ActionSpec(
        "Confirm the current bundle preview.",
        lambda a: {"action": "confirm_bundle_selection"},
    ),
    "bundle_cancel_selection": ActionSpec(
        "Cancel the current bundle preview.",
        lambda a: {"action": "cancel_bundle_selection"},
    ),
    "relic_select": ActionSpec(
        "Choose a relic from a relic-selection screen by relic_index.",
        lambda a: {"action": "select_relic", "index": int_arg(a, "relic_index", "index")},
    ),
    "relic_skip": ActionSpec(
        "Skip the current relic selection.",
        lambda a: {"action": "skip_relic_selection"},
    ),
    "treasure_claim_relic": ActionSpec(
        "Claim a treasure relic by relic_index.",
        lambda a: {"action": "claim_treasure_relic", "index": int_arg(a, "relic_index", "index")},
    ),
    "crystal_sphere_set_tool": ActionSpec(
        "Switch Crystal Sphere tool to 'big' or 'small'.",
        lambda a: {"action": "crystal_sphere_set_tool", "tool": str_arg(a, "tool")},
    ),
    "crystal_sphere_click_cell": ActionSpec(
        "Click a Crystal Sphere cell.",
        lambda a: {
            "action": "crystal_sphere_click_cell",
            "x": int_arg(a, "x"),
            "y": int_arg(a, "y"),
        },
    ),
    "crystal_sphere_proceed": ActionSpec(
        "Continue after Crystal Sphere finishes.",
        lambda a: {"action": "crystal_sphere_proceed"},
    ),
}


ALIASES = {
    "play_card": "combat_play_card",
    "end_turn": "combat_end_turn",
    "choose_map_node": "map_choose_node",
    "choose_event_option": "event_choose_option",
    "choose_rest_option": "rest_choose_option",
    "select_card_reward": "rewards_pick_card",
    "skip_card_reward": "rewards_skip_card",
    "claim_reward": "rewards_claim",
    "proceed": "proceed_to_map",
}


GAME_CONTEXT = """Game context:
- Slay the Spire is a roguelike deckbuilder. The goal is to climb the map, survive combats/events, improve the deck, beat elites/bosses, and avoid dying.
- Each turn you draw a hand and spend energy to play cards. Only cards with can_play=true are legal. At 0 energy, normally end the turn.
- HP is a resource, but dying ends the run. Block only matters for the current turn, so block mainly when enemies intend to attack.
- Deckbuilding goal: make a small, reliable deck with enough front-loaded damage, block, scaling, and synergy. Do not add mediocre cards just because they are offered.
- Early act priorities: take good damage cards, value strong relics, use potions instead of dying, fight elites only when HP/deck quality can support it.
- Card rewards: pick cards that solve a problem or improve the deck plan; skip weak/off-plan cards when skipping is available.
- Map pathing: prefer paths with useful rewards and a rest before boss; elites are valuable but risky.
"""


SYSTEM_PROMPT = f"""You are playing Slay the Spire 2 through a localhost action API.

{GAME_CONTEXT}

Return exactly one JSON object and no prose:
{{"action":"ACTION_NAME","args":{{}},"rationale":"short public reason"}}

Rules:
- Keep rationale to 20 words or fewer.
- Choose only one action per response. The harness will fetch fresh state after every action.
- Use the current state exactly. Use visible menu option ids and listed indices.
- State types monster, elite, and boss are combat screens. Combat details are under battle.
- Only play cards whose current hand entry has can_play=true. Never play cards with unplayable_reason.
- In combat, use target entity ids exactly as shown. If a single-target card needs a target, provide one.
- If there are multiple enemies and you are attacking, choose a target explicitly; prefer low-HP attackers.
- Playing cards changes hand indices. Prefer one card at a time; if planning multiple cards, play from higher indices first.
- In combat, wait does not end the turn. If you have 0 energy or no useful playable cards, choose combat_end_turn.
- Choose wait only for transient states where the game is already resolving and no player action is accepted.
- On a rewards screen with rewards.items, use rewards_claim(reward_index) to take a reward, or proceed_to_map to skip remaining rewards when available.
- Use rewards_pick_card only after card choices are visible.
- If enemies are not attacking, prefer useful offense/setup. If lethal is available, kill rather than block.
- If no useful action is available, choose wait. If the run is over or blocked, choose stop.
- Do not request get_game_state. State is already provided every step.
"""


def compact_json(value: Any, max_chars: int | None = None) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "\n...TRUNCATED..."
    return text


def action_menu() -> str:
    lines = []
    for name, spec in ACTION_SPECS.items():
        lines.append(f"- {name}: {spec.description}")
    return "\n".join(lines)


def state_for_prompt(state: Json) -> Json:
    state_type = state.get("state_type")
    if state_type == "map":
        game_map = state.get("map", {})
        return {
            "state_type": state_type,
            "run": state.get("run"),
            "player": state.get("player"),
            "map": {
                "current_position": game_map.get("current_position"),
                "visited": game_map.get("visited", []),
                "next_options": game_map.get("next_options", []),
                "nodes": game_map.get("nodes", []),
                "boss": game_map.get("boss"),
                "bosses": game_map.get("bosses", []),
            },
        }
    if state_type == "menu" and state.get("menu_screen") == "character_select":
        characters = state.get("characters", [])
        return {
            "state_type": state_type,
            "menu_screen": state.get("menu_screen"),
            "message": state.get("message"),
            "run": state.get("run"),
            "options": state.get("options"),
            "characters": [
                {
                    "id": character.get("id"),
                    "name": character.get("name"),
                    "locked": character.get("locked"),
                    "hp": character.get("hp"),
                    "starting_deck": character.get("starting_deck"),
                    "starting_relics": character.get("starting_relics"),
                }
                for character in characters
            ],
        }
    return state


def build_user_prompt(state: Json, step: int, max_state_chars: int, history: list[str]) -> str:
    recent_history = "\n".join(history[-6:]) if history else "No prior actions in this harness run."
    return f"""Step: {step}

Available actions:
{action_menu()}

Recent action history:
{recent_history}

Current game state JSON:
{compact_json(state_for_prompt(state), max_state_chars)}

Pick the next single action now."""


def request_json(method: str, url: str, body: Json | None = None, timeout: float = 60.0) -> Any:
    data = None
    headers = {
        "Accept": "application/json",
        "User-Agent": os.environ.get("STS2_HARNESS_USER_AGENT", DEFAULT_HTTP_USER_AGENT),
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {raw_error[:2000]}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def build_url(base: str, path: str, params: Json | None = None) -> str:
    url = base.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return url


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return int(raw)


def ensure_openai_v1_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def runpod_openai_url(runpod_id: str, port: int = DEFAULT_RUNPOD_PORT, domain: str = DEFAULT_RUNPOD_DOMAIN) -> str:
    """Build the OpenAI-compatible /v1 base URL for a Runpod proxy endpoint."""
    value = runpod_id.strip().rstrip("/")
    if not value:
        raise ValueError("runpod id cannot be empty")

    if value.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(value)
        path = parsed.path.rstrip("/")
        if path and path != "/v1":
            raise ValueError("Runpod URL should point at the proxy root or /v1 API root; use --llm-url for custom paths.")
        return ensure_openai_v1_url(value)

    if "/" in value:
        raise ValueError("runpod id should be an id or host, not a path")

    clean_domain = domain.strip().strip(".")
    if not clean_domain:
        raise ValueError("runpod domain cannot be empty")

    if "." in value:
        host = value
    elif value.endswith(f"-{port}"):
        host = f"{value}.{clean_domain}"
    else:
        host = f"{value}-{port}.{clean_domain}"
    return ensure_openai_v1_url(f"https://{host}")


def add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-url",
        help=f"OpenAI-compatible API base URL. Defaults to {DEFAULT_LOCAL_LLM_URL}; overrides Runpod URL generation.",
    )
    parser.add_argument(
        "--runpod-id",
        help=(
            "Runpod proxy id or host, e.g. i6qkeo2utydx8c-64411136. "
            "Builds https://<id>-<port>.proxy.runpod.net/v1."
        ),
    )
    parser.add_argument(
        "--runpod-port",
        type=int,
        default=env_int("RUNPOD_PORT", DEFAULT_RUNPOD_PORT),
        help="Exposed Runpod proxy port used with --runpod-id.",
    )
    parser.add_argument(
        "--runpod-domain",
        default=os.environ.get("RUNPOD_DOMAIN", DEFAULT_RUNPOD_DOMAIN),
        help="Runpod proxy domain used with --runpod-id.",
    )
    parser.add_argument("--model", help="Model name to send to the OpenAI-compatible chat endpoint.")


def resolve_llm_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve local/custom/Runpod LLM configuration onto an argparse namespace."""
    cli_llm_url = (args.llm_url or "").strip()
    cli_runpod_id = (args.runpod_id or "").strip()
    env_llm_url = os.environ.get("LLM_URL", "").strip()
    env_runpod_id = os.environ.get("RUNPOD_ID", "").strip()
    source = "local"

    if cli_llm_url:
        args.llm_url = cli_llm_url.rstrip("/")
        source = "custom"
    elif cli_runpod_id:
        args.runpod_id = cli_runpod_id
        args.llm_url = runpod_openai_url(cli_runpod_id, args.runpod_port, args.runpod_domain)
        source = "runpod"
    elif env_llm_url:
        args.llm_url = env_llm_url.rstrip("/")
        source = "custom-env"
    elif env_runpod_id:
        args.runpod_id = env_runpod_id
        args.llm_url = runpod_openai_url(env_runpod_id, args.runpod_port, args.runpod_domain)
        source = "runpod-env"
    else:
        args.llm_url = DEFAULT_LOCAL_LLM_URL

    if not args.model:
        if source.startswith("runpod"):
            args.model = os.environ.get("RUNPOD_MODEL") or os.environ.get("LLM_MODEL") or DEFAULT_RUNPOD_LLM_MODEL
        else:
            args.model = os.environ.get("LLM_MODEL") or DEFAULT_LOCAL_LLM_MODEL

    args.llm_source = source
    return args


def get_game_state(sts2_url: str) -> Json:
    result = request_json(
        "GET",
        build_url(sts2_url, "/api/v1/singleplayer", {"format": "json"}),
        timeout=10,
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected state response: {result!r}")
    return result


def post_game_action(sts2_url: str, action: str, args: Json) -> Any:
    spec = ACTION_SPECS[action]
    if spec.build_body is None:
        raise ValueError(f"action {action} is not a POST action")
    body = spec.build_body(args)
    return request_json("POST", build_url(sts2_url, "/api/v1/singleplayer"), body=body, timeout=15)


def living_enemies(state: Json) -> list[Json]:
    enemies = state.get("battle", {}).get("enemies", [])
    return [
        enemy
        for enemy in enemies
        if isinstance(enemy, dict) and enemy.get("entity_id") and int(enemy.get("hp", 0)) > 0
    ]


def hand_card(state: Json, card_index: int) -> Json | None:
    for card in state.get("player", {}).get("hand", []):
        if isinstance(card, dict) and int(card.get("index", -1)) == card_index:
            return card
    return None


def playable_hand_cards(state: Json) -> list[Json]:
    return [
        card
        for card in state.get("player", {}).get("hand", [])
        if isinstance(card, dict) and card.get("can_play") is True
    ]


def first_int(value: Any) -> int:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else 0


def incoming_attack_damage(state: Json) -> int:
    total = 0
    for enemy in living_enemies(state):
        for intent in enemy.get("intents", []):
            if isinstance(intent, dict) and str(intent.get("type", "")).lower() == "attack":
                total += first_int(intent.get("label") or intent.get("description"))
    return total


def card_block_value(card: Json) -> int:
    return first_int(card.get("description")) if "block" in str(card.get("description", "")).lower() else 0


def card_damage_value(card: Json) -> int:
    return first_int(card.get("description")) if "damage" in str(card.get("description", "")).lower() else 0


def choose_default_playable_card_index(state: Json) -> int | None:
    playable = playable_hand_cards(state)
    if not playable:
        return None

    player = state.get("player", {})
    block_needed = max(0, incoming_attack_damage(state) - int(player.get("block", 0)))
    if block_needed > 0:
        block_cards = [card for card in playable if card_block_value(card) > 0]
        if block_cards:
            return int(max(block_cards, key=card_block_value).get("index", 0))

    attacks = [card for card in playable if "Enemy" in str(card.get("target_type", ""))]
    if attacks:
        return int(max(attacks, key=card_damage_value).get("index", 0))

    return int(playable[0].get("index", 0))


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def enemy_is_attacking(enemy: Json) -> bool:
    for intent in enemy.get("intents", []):
        if isinstance(intent, dict) and str(intent.get("type", "")).lower() == "attack":
            return True
    return False


def choose_fallback_enemy(enemies: list[Json]) -> Json | None:
    if not enemies:
        return None
    attacking = [enemy for enemy in enemies if enemy_is_attacking(enemy)]
    pool = attacking or enemies
    return min(pool, key=lambda enemy: int(enemy.get("hp", 0)))


def match_enemy_from_text(enemies: list[Json], text: str) -> Json | None:
    haystack = normalized(text)
    if not haystack:
        return None
    for enemy in enemies:
        entity_id = str(enemy.get("entity_id", ""))
        name = str(enemy.get("name", ""))
        candidates = {
            normalized(entity_id),
            normalized(entity_id.replace("_", " ")),
            normalized(name),
            normalized(re.sub(r"\([^)]*\)", "", name)),
        }
        if any(candidate and candidate in haystack for candidate in candidates):
            return enemy
    return None


def enrich_action_args(state: Json, action: str, action_args: Json, rationale: str) -> Json:
    enriched = dict(action_args)
    if action not in {"combat_play_card", "use_potion"} or enriched.get("target"):
        return enriched

    enemies = living_enemies(state)
    if action == "combat_play_card":
        try:
            card_index = int_arg(enriched, "card_index", "index")
        except ValueError:
            return enriched
        card = hand_card(state, card_index)
        target_type = str((card or {}).get("target_type", ""))
        if "Enemy" not in target_type:
            return enriched

    enemy = enemies[0] if len(enemies) == 1 else None
    if enemy is None:
        target_text = " ".join(
            str(enriched.get(name, ""))
            for name in ("target_entity_id", "target_name", "target")
        )
        enemy = match_enemy_from_text(enemies, f"{target_text} {rationale}")
    if enemy is None:
        enemy = choose_fallback_enemy(enemies)
    if enemy is None:
        return enriched

    enriched["target"] = enemy["entity_id"]
    enriched["_auto_targeted"] = True
    return enriched


def is_combat_state(state: Json) -> bool:
    return state.get("state_type") in {"monster", "elite", "boss"}


def maybe_replace_wait(state: Json, action: str, action_args: Json) -> tuple[str, Json]:
    if action != "wait" or not is_combat_state(state):
        return action, action_args
    battle = state.get("battle", {})
    if battle.get("turn") != "player" or battle.get("is_play_phase") is not True:
        return action, action_args

    player = state.get("player", {})
    energy = int(player.get("energy", 0))
    playable_cards = [
        card
        for card in player.get("hand", [])
        if isinstance(card, dict) and card.get("can_play") is True
    ]
    if energy == 0 or not playable_cards:
        return "combat_end_turn", {"_auto_from_wait": True}
    return action, action_args


def maybe_replace_unplayable_card(state: Json, action: str, action_args: Json) -> tuple[str, Json]:
    if action != "combat_play_card" or not is_combat_state(state):
        return action, action_args
    battle = state.get("battle", {})
    if battle.get("turn") != "player" or battle.get("is_play_phase") is not True:
        return action, action_args

    player = state.get("player", {})
    try:
        card_index = int_arg(action_args, "card_index", "index")
    except ValueError:
        return action, action_args
    card = hand_card(state, card_index)
    if not isinstance(card, dict) or card.get("can_play") is not False:
        return action, action_args

    playable_cards = playable_hand_cards(state)
    if not playable_cards:
        return "combat_end_turn", {
            "_auto_from_unplayable_card": True,
            "_blocked_card_index": card_index,
            "_blocked_reason": card.get("unplayable_reason"),
        }
    return action, action_args


def maybe_fill_missing_required_args(state: Json, action: str, action_args: Json) -> tuple[str, Json]:
    filled = dict(action_args)

    if action == "rewards_claim" and "reward_index" not in filled and "index" not in filled:
        reward_items = state.get("rewards", {}).get("items")
        if isinstance(reward_items, list) and reward_items:
            filled["reward_index"] = int(reward_items[0].get("index", 0))
            filled["_auto_filled_arg"] = "reward_index"
        return action, filled

    if action == "rewards_pick_card" and "card_index" not in filled and "index" not in filled:
        cards = state.get("card_reward", {}).get("cards")
        if isinstance(cards, list) and cards:
            filled["card_index"] = int(cards[0].get("index", 0))
            filled["_auto_filled_arg"] = "card_index"
        return action, filled

    if action == "map_choose_node" and "node_index" not in filled and "index" not in filled:
        next_options = state.get("map", {}).get("next_options")
        if isinstance(next_options, list) and next_options:
            filled["node_index"] = int(next_options[0].get("index", 0))
            filled["_auto_filled_arg"] = "node_index"
        return action, filled

    if action == "combat_play_card" and "card_index" not in filled and "index" not in filled:
        card_index = choose_default_playable_card_index(state)
        if card_index is None and is_combat_state(state):
            return "combat_end_turn", {"_auto_from_missing_card_index": True}
        if card_index is not None:
            filled["card_index"] = card_index
            filled["_auto_filled_arg"] = "card_index"
        return action, filled

    return action, filled


def maybe_replace_reward_pick(state: Json, action: str, action_args: Json) -> tuple[str, Json]:
    if action != "rewards_pick_card" or state.get("state_type") != "rewards":
        return action, action_args
    reward_items = state.get("rewards", {}).get("items")
    if not isinstance(reward_items, list):
        return action, action_args
    for item in reward_items:
        if isinstance(item, dict) and item.get("type") == "card":
            return "rewards_claim", {
                "reward_index": int(item.get("index", action_args.get("card_index", 0))),
                "_auto_from_card_pick": True,
            }
    return action, action_args


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    return text.replace("```", "").strip()


def has_action_key(value: Json) -> bool:
    return any(
        key in value
        for key in ("action", "action_name", "tool", "tool_name", "name", "next_action")
    )


def try_repair_json_object(text: str, start: int) -> Json | None:
    candidate = text[start:]
    depth = 0
    in_string = False
    escape = False
    for ch in candidate:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    if in_string or depth <= 0:
        return None
    repaired = candidate + ("}" * depth)
    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_json_object(text: str) -> Json:
    cleaned = strip_thinking(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start_positions = [i for i, ch in enumerate(cleaned) if ch == "{"]
    first_valid: Json | None = None
    for start in start_positions:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(cleaned)):
            ch = cleaned[index]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        if first_valid is None:
                            first_valid = parsed
                        if has_action_key(parsed):
                            return parsed
                    break
    for start in start_positions:
        repaired = try_repair_json_object(cleaned, start)
        if repaired is None:
            continue
        if first_valid is None:
            first_valid = repaired
        if has_action_key(repaired):
            return repaired
    if first_valid is not None:
        return first_valid
    raise ValueError(f"could not parse JSON object from model response: {text!r}")


def call_llm(
    llm_url: str,
    model: str,
    messages: list[Json],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> tuple[str, Json]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = request_json(
        "POST",
        build_url(llm_url, "/chat/completions"),
        body=payload,
        timeout=timeout,
    )
    if not isinstance(response, dict):
        raise RuntimeError(f"unexpected LLM response: {response!r}")
    message = response["choices"][0]["message"]
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content), response


def normalize_action(raw: Json) -> tuple[str, Json, str]:
    candidate = (
        raw.get("action")
        or raw.get("action_name")
        or raw.get("tool")
        or raw.get("tool_name")
        or raw.get("name")
        or raw.get("next_action")
    )
    nested_args: Json | None = None
    if isinstance(candidate, dict):
        nested_args_value = candidate.get("args", candidate.get("arguments"))
        nested_args = nested_args_value if isinstance(nested_args_value, dict) else None
        candidate = candidate.get("name") or candidate.get("action") or candidate.get("tool")
    action = candidate
    if not isinstance(action, str):
        raise ValueError("model response must include string field 'action'")
    action = ALIASES.get(action, action)
    if action not in ACTION_SPECS:
        raise ValueError(f"unknown action: {action}")
    args = collect_model_args(raw, nested_args)
    rationale = raw.get("rationale", "")
    return action, args, str(rationale)


def log_event(log_path: Path, event: Json) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def print_step(step: int, state: Json, action: str, args: Json, rationale: str) -> None:
    state_type = state.get("state_type", "unknown")
    screen = state.get("menu_screen") or state.get("screen") or state.get("room_type") or ""
    suffix = f" ({screen})" if screen else ""
    arg_text = compact_json(args)
    print(f"[{step}] {state_type}{suffix} -> {action} {arg_text}", flush=True)
    if rationale:
        print(f"    {rationale}", flush=True)


def history_line(step: int, state: Json, action: str, action_args: Json, result: Any) -> str:
    state_type = state.get("state_type", "unknown")
    result_status = result.get("status") if isinstance(result, dict) else type(result).__name__
    compact_args = json.dumps(action_args, ensure_ascii=False, sort_keys=True)
    return f"{step}. {state_type}: {action} {compact_args} -> {result_status}"


def run(args: argparse.Namespace) -> int:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"sts2-harness-{stamp}.jsonl"

    action_history: list[str] = []
    consecutive_errors = 0

    print(f"Using LLM ({getattr(args, 'llm_source', 'custom')}): {args.model} at {args.llm_url}", flush=True)

    for step in range(1, args.steps + 1):
        raw_text: str | None = None
        parsed: Json | None = None
        state: Json | None = None
        try:
            state = get_game_state(args.sts2_url)
            user_prompt = build_user_prompt(state, step, args.max_state_chars, action_history)
            turn_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            raw_text, raw_response = call_llm(
                args.llm_url,
                args.model,
                turn_messages,
                args.temperature,
                args.max_tokens,
                args.llm_timeout,
            )
            parsed = extract_json_object(raw_text)
            action, action_args, rationale = normalize_action(parsed)
            action, action_args = maybe_replace_wait(state, action, action_args)
            action, action_args = maybe_replace_unplayable_card(state, action, action_args)
            action, action_args = maybe_replace_reward_pick(state, action, action_args)
            action_args = enrich_action_args(state, action, action_args, rationale)
            print_step(step, state, action, action_args, rationale)

            result: Any
            if action == "stop":
                result = {"status": "stopped"}
                log_event(
                    log_path,
                    {
                        "step": step,
                        "state": state,
                        "model_text": raw_text,
                        "parsed": parsed,
                        "result": result,
                    },
                )
                break
            if action == "wait" or args.dry_run:
                result = {"status": "dry_run" if args.dry_run else "wait"}
            else:
                result = post_game_action(args.sts2_url, action, action_args)

            log_event(
                log_path,
                {
                    "step": step,
                    "state": state,
                    "model_text": raw_text,
                    "parsed": parsed,
                    "result": result,
                    "llm_usage": raw_response.get("usage") if isinstance(raw_response, dict) else None,
                },
            )

            action_history.append(history_line(step, state, action, action_args, result))
            if len(action_history) > args.history_turns:
                action_history = action_history[-args.history_turns :]

            if args.show_results:
                print(compact_json(result, args.max_result_chars), flush=True)
            consecutive_errors = 0
            time.sleep(args.sleep)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 130
        except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
            consecutive_errors += 1
            print(f"[{step}] error: {exc}", file=sys.stderr, flush=True)
            log_event(
                log_path,
                {
                    "step": step,
                    "error": repr(exc),
                    "state_for_prompt": state_for_prompt(state) if isinstance(state, dict) else None,
                    "model_text": raw_text,
                    "parsed": parsed,
                },
            )
            if consecutive_errors >= args.max_consecutive_errors:
                print(f"Too many consecutive errors. Log: {log_path}", file=sys.stderr, flush=True)
                return 1
            time.sleep(args.sleep)

    print(f"Log: {log_path}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Let an OpenAI-compatible model play STS2 through STS2_MCP.")
    parser.add_argument("--steps", type=int, default=12, help="Maximum number of model actions to take.")
    parser.add_argument("--sleep", type=float, default=1.25, help="Seconds to wait after each step.")
    parser.add_argument("--sts2-url", default=os.environ.get("STS2_URL", "http://localhost:15526"))
    add_llm_arguments(parser)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--max-state-chars", type=int, default=22000)
    parser.add_argument("--max-result-chars", type=int, default=6000)
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--dry-run", action="store_true", help="Do not send actions to the game.")
    parser.add_argument("--show-results", action="store_true", help="Print raw action results.")
    return resolve_llm_args(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
