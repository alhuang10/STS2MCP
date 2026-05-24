#!/usr/bin/env python3
"""MCP-backed OpenAI-compatible model harness for STS2_MCP.

This keeps the small explicit game loop from sts2_harness.py, but delegates
tool discovery and execution to mcp/server.py instead of duplicating REST
request bodies in the harness.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from prompt_templates import (
    PromptBundle,
    load_sts2_mcp_action_prompts,
    render_system_prompt,
    render_user_prompt,
    sha256_text,
)
from sts2_harness import (
    ALIASES,
    add_llm_arguments,
    call_llm,
    compact_json,
    extract_json_object,
    GAME_CONTEXT,
    history_line,
    log_event,
    normalize_action,
    print_step,
    resolve_llm_args,
    state_for_prompt,
)


Json = dict[str, Any]


EXCLUDED_TOOL_PREFIXES = ("mp_",)
EXCLUDED_TOOL_NAMES = {
    "get_game_state",
    "get_profile",
    "get_compendium",
    "search_wiki",
    "list_profiles",
    "switch_profile",
    "delete_profile",
    "eval_start_run",
}

LOCAL_ACTIONS = {
    "wait": "Do nothing briefly, then poll game state again.",
    "stop": "Stop the harness cleanly.",
}

COMBAT_STATE_TYPES = {"monster", "elite", "boss"}
TRANSIENT_STATE_TYPES = {"unknown"}
TRANSITION_SETTLE_ACTIONS = {
    "map_choose_node",
    "proceed_to_map",
    "crystal_sphere_proceed",
}
DEFAULT_HARNESS_STEPS = 300
DEFAULT_SEEDED_EVAL_STEPS = 10000
SEEDED_EVAL_CHARACTER = "IRONCLAD"
SEEDED_EVAL_SEED = "CW967RN0QC"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Json


@dataclass(frozen=True)
class LegalAction:
    id: str
    tool: str
    args: Json
    summary: str
    category: str = ""

    def for_prompt(self) -> Json:
        payload: Json = {
            "id": self.id,
            "tool": self.tool,
            "args": self.args,
            "summary": self.summary,
        }
        if self.category:
            payload["category"] = self.category
        return payload


class ModelActionError(ValueError):
    def __init__(self, message: str, trace: Json):
        super().__init__(message)
        self.trace = trace


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def short_description(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.strip().splitlines()[0].split())


def is_action_tool(name: str) -> bool:
    if name in EXCLUDED_TOOL_NAMES:
        return False
    return not name.startswith(EXCLUDED_TOOL_PREFIXES)


def coerce_tool_schema(schema: Any) -> Json:
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_dump"):
        return schema.model_dump()
    return {}


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def safe_dict(value: Any) -> Json:
    return value if isinstance(value, dict) else {}


def indexed_values(values: Any, *, enabled_only: bool = False) -> list[int]:
    indices: list[int] = []
    for item in safe_list(values):
        if not isinstance(item, dict) or "index" not in item:
            continue
        if enabled_only:
            if item.get("enabled") is False or item.get("is_enabled") is False or item.get("is_locked") is True:
                continue
            if item.get("is_stocked") is False or item.get("can_afford") is False:
                continue
        try:
            indices.append(int(item["index"]))
        except (TypeError, ValueError):
            pass
    return indices


def menu_option_names(state: Json) -> list[str]:
    options = safe_list(state.get("options"))
    names: list[str] = []
    for item in options:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("enabled") is not False:
            name = item.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def potion_slots(state: Json) -> list[int]:
    slots: list[int] = []
    for index, potion in enumerate(safe_list(safe_dict(state.get("player")).get("potions"))):
        if not isinstance(potion, dict):
            continue
        raw_slot = potion.get("slot", potion.get("index", index))
        try:
            slots.append(int(raw_slot))
        except (TypeError, ValueError):
            pass
    return slots


def shop_items(state: Json) -> list[Any]:
    if state.get("state_type") == "fake_merchant":
        return safe_list(safe_dict(safe_dict(state.get("fake_merchant")).get("shop")).get("items"))
    return safe_list(safe_dict(state.get("shop")).get("items"))


def shop_can_proceed(state: Json) -> bool:
    if state.get("state_type") == "fake_merchant":
        return safe_dict(safe_dict(state.get("fake_merchant")).get("shop")).get("can_proceed") is True
    return safe_dict(state.get("shop")).get("can_proceed") is True


def add_available(existing: set[str], tools: dict[str, ToolSpec], *names: str) -> None:
    for name in names:
        if name in tools:
            existing.add(name)


def available_tool_names(state: Json, tools: dict[str, ToolSpec]) -> set[str]:
    """Return MCP tools that are plausible for the current visible screen."""
    state_type = state.get("state_type")
    names: set[str] = set()

    if state_type in {"menu", "game_over"}:
        add_available(names, tools, "menu_select")
    elif state_type in COMBAT_STATE_TYPES:
        battle = safe_dict(state.get("battle"))
        if battle.get("turn") in {None, "player"} and battle.get("is_play_phase") is not False:
            add_available(names, tools, "combat_play_card", "combat_end_turn")
    elif state_type == "hand_select":
        hand_select = safe_dict(state.get("hand_select"))
        if indexed_values(hand_select.get("cards")):
            add_available(names, tools, "combat_select_card")
        if hand_select.get("can_confirm") is True:
            add_available(names, tools, "combat_confirm_selection")
    elif state_type == "rewards":
        rewards = safe_dict(state.get("rewards"))
        if indexed_values(rewards.get("items")):
            add_available(names, tools, "rewards_claim")
        if rewards.get("can_proceed") is True:
            add_available(names, tools, "proceed_to_map")
    elif state_type == "card_reward":
        card_reward = safe_dict(state.get("card_reward"))
        if indexed_values(card_reward.get("cards")):
            add_available(names, tools, "rewards_pick_card")
        if card_reward.get("can_skip") is True:
            add_available(names, tools, "rewards_skip_card")
    elif state_type == "map":
        if indexed_values(safe_dict(state.get("map")).get("next_options")):
            add_available(names, tools, "map_choose_node")
    elif state_type == "event":
        event = safe_dict(state.get("event"))
        if event.get("in_dialogue") is True:
            add_available(names, tools, "event_advance_dialogue")
        elif indexed_values(event.get("options"), enabled_only=True):
            add_available(names, tools, "event_choose_option")
    elif state_type == "rest_site":
        rest = safe_dict(state.get("rest_site"))
        if indexed_values(rest.get("options"), enabled_only=True):
            add_available(names, tools, "rest_choose_option")
        if rest.get("can_proceed") is True:
            add_available(names, tools, "proceed_to_map")
    elif state_type in {"shop", "fake_merchant"}:
        if indexed_values(shop_items(state), enabled_only=True):
            add_available(names, tools, "shop_purchase")
        if shop_can_proceed(state):
            add_available(names, tools, "proceed_to_map")
    elif state_type == "treasure":
        treasure = safe_dict(state.get("treasure"))
        if indexed_values(treasure.get("relics")):
            add_available(names, tools, "treasure_claim_relic")
        if treasure.get("can_proceed") is True:
            add_available(names, tools, "proceed_to_map")
    elif state_type == "card_select":
        selection = safe_dict(state.get("card_select"))
        if indexed_values(selection.get("cards")):
            add_available(names, tools, "deck_select_card")
        if selection.get("can_confirm") is True:
            add_available(names, tools, "deck_confirm_selection")
        if selection.get("can_cancel") is True or selection.get("can_skip") is True:
            add_available(names, tools, "deck_cancel_selection")
    elif state_type == "bundle_select":
        selection = safe_dict(state.get("bundle_select"))
        if indexed_values(selection.get("bundles")):
            add_available(names, tools, "bundle_select")
        if selection.get("can_confirm") is True:
            add_available(names, tools, "bundle_confirm_selection")
        if selection.get("can_cancel") is True:
            add_available(names, tools, "bundle_cancel_selection")
    elif state_type == "relic_select":
        selection = safe_dict(state.get("relic_select"))
        if indexed_values(selection.get("relics")):
            add_available(names, tools, "relic_select")
        if selection.get("can_skip") is True:
            add_available(names, tools, "relic_skip")
    elif state_type == "crystal_sphere":
        sphere = safe_dict(state.get("crystal_sphere"))
        if sphere.get("can_use_big_tool") is True or sphere.get("can_use_small_tool") is True:
            add_available(names, tools, "crystal_sphere_set_tool")
        if safe_list(sphere.get("clickable_cells")):
            add_available(names, tools, "crystal_sphere_click_cell")
        if sphere.get("can_proceed") is True:
            add_available(names, tools, "crystal_sphere_proceed")

    if state_type not in {"menu", "game_over", "unknown", "overlay", None} and potion_slots(state):
        add_available(names, tools, "use_potion", "discard_potion")

    return names


def tools_for_state(state: Json, tools: dict[str, ToolSpec]) -> dict[str, ToolSpec]:
    names = available_tool_names(state, tools)
    return {name: tools[name] for name in sorted(names) if name in tools}


def slug(value: Any, max_len: int = 42) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return (text or "x")[:max_len]


def first_number(value: Any) -> int:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else 0


def card_damage_value(card: Json) -> int:
    description = str(card.get("description", ""))
    return first_number(description) if "damage" in description.lower() else 0


def card_block_value(card: Json) -> int:
    description = str(card.get("description", ""))
    return first_number(description) if "block" in description.lower() else 0


def enemy_hp_with_block(enemy: Json) -> int:
    return int(enemy.get("hp", 0) or 0) + int(enemy.get("block", 0) or 0)


def living_enemies(state: Json) -> list[Json]:
    enemies = []
    for enemy in safe_list(safe_dict(state.get("battle")).get("enemies")):
        if not isinstance(enemy, dict) or not enemy.get("entity_id"):
            continue
        try:
            if int(enemy.get("hp", 0)) <= 0:
                continue
        except (TypeError, ValueError):
            pass
        enemies.append(enemy)
    return enemies


def incoming_attack_damage(state: Json) -> int:
    total = 0
    for enemy in living_enemies(state):
        for intent in safe_list(enemy.get("intents")):
            if not isinstance(intent, dict):
                continue
            if str(intent.get("type", "")).lower() == "attack":
                total += first_number(intent.get("label") or intent.get("description"))
    return total


def target_type_needs_enemy(target_type: Any) -> bool:
    text = str(target_type or "")
    return "Enemy" in text and "All" not in text


def target_label(enemy: Json) -> str:
    name = enemy.get("name") or enemy.get("entity_id") or "enemy"
    hp = enemy.get("hp", "?")
    block = enemy.get("block", 0)
    block_text = f", {block} block" if block else ""
    return f"{name} ({hp} HP{block_text})"


def card_cost_value(card: Json) -> int | None:
    try:
        return int(str(card.get("cost")))
    except (TypeError, ValueError):
        return None


def add_legal_action(
    actions: list[LegalAction],
    tools: dict[str, ToolSpec],
    action_id: str,
    tool: str,
    args: Json | None,
    summary: str,
    category: str = "",
) -> None:
    if tool not in LOCAL_ACTIONS and tool not in tools:
        return
    existing = {action.id for action in actions}
    base = slug(action_id, 72)
    unique_id = base
    suffix = 2
    while unique_id in existing:
        unique_id = f"{base}_{suffix}"
        suffix += 1
    actions.append(LegalAction(unique_id, tool, args or {}, summary, category))


def indexed_label(item: Json, *keys: str) -> str:
    index = item.get("index", "?")
    for key in keys:
        value = item.get(key)
        if value:
            return f"{index}: {value}"
    return str(index)


def combat_card_summary(state: Json, card: Json, enemy: Json | None) -> str:
    name = card.get("name") or card.get("id") or "card"
    cost = card.get("cost", "?")
    damage = card_damage_value(card)
    block = card_block_value(card)
    pieces = [f"Play {name} (hand index {card.get('index')}, cost {cost})"]
    cost_value = card_cost_value(card)
    energy = safe_dict(state.get("player")).get("energy")
    if isinstance(cost_value, int) and isinstance(energy, int):
        pieces.append(f"energy after: {energy - cost_value}")
    if enemy is not None:
        pieces.append(f"target: {target_label(enemy)}")
        if damage:
            enemy_total = enemy_hp_with_block(enemy)
            lethal = "; listed damage is lethal" if damage >= enemy_total else ""
            pieces.append(f"listed damage: {damage} vs HP+block {enemy_total}{lethal}")
    elif damage:
        pieces.append(f"listed damage: {damage}")
    if block:
        incoming = incoming_attack_damage(state)
        pieces.append(f"listed block: {block}; incoming attack: {incoming}")
    description = card.get("description")
    if description:
        pieces.append(str(description))
    return ". ".join(pieces)


def potion_summary(potion: Json, enemy: Json | None = None) -> str:
    name = potion.get("name") or potion.get("id") or "Potion"
    slot = potion.get("slot", "?")
    pieces = [f"Use {name} from potion slot {slot}"]
    if enemy is not None:
        pieces.append(f"target: {target_label(enemy)}")
    description = potion.get("description")
    if description:
        pieces.append(str(description))
    return ". ".join(pieces)


def add_potion_actions(state: Json, tools: dict[str, ToolSpec], actions: list[LegalAction]) -> None:
    state_type = state.get("state_type")
    in_combat = state_type in COMBAT_STATE_TYPES
    battle = safe_dict(state.get("battle"))
    if in_combat and (battle.get("turn") not in {None, "player"} or battle.get("is_play_phase") is False):
        return

    potions = [
        potion
        for potion in safe_list(safe_dict(state.get("player")).get("potions"))
        if isinstance(potion, dict) and potion.get("slot") is not None
    ]
    enemies = living_enemies(state)
    for potion in potions:
        target_type = potion.get("target_type")
        slot = int(potion.get("slot"))
        if target_type_needs_enemy(target_type):
            if not in_combat:
                continue
            for enemy in enemies:
                add_legal_action(
                    actions,
                    tools,
                    f"potion_{slot}_{potion.get('id') or potion.get('name')}_{enemy.get('entity_id')}",
                    "use_potion",
                    {"slot": slot, "target": enemy["entity_id"]},
                    potion_summary(potion, enemy),
                    "potion",
                )
        else:
            if in_combat and potion.get("can_use_in_combat") is False:
                continue
            add_legal_action(
                actions,
                tools,
                f"potion_{slot}_{potion.get('id') or potion.get('name')}",
                "use_potion",
                {"slot": slot},
                potion_summary(potion),
                "potion",
            )

    max_slots = safe_dict(state.get("player")).get("max_potion_slots")
    if isinstance(max_slots, int) and len(potions) >= max_slots:
        for potion in potions:
            slot = int(potion.get("slot"))
            name = potion.get("name") or potion.get("id") or "potion"
            add_legal_action(
                actions,
                tools,
                f"discard_potion_{slot}_{name}",
                "discard_potion",
                {"slot": slot},
                f"Discard {name} from slot {slot} to free a potion slot.",
                "potion",
            )


def build_legal_actions(state: Json, tools: dict[str, ToolSpec]) -> list[LegalAction]:
    state_type = state.get("state_type")
    actions: list[LegalAction] = []

    if state_type in {"menu", "game_over"}:
        for option in menu_option_names(state):
            add_legal_action(
                actions,
                tools,
                f"menu_{option}",
                "menu_select",
                {"option": option},
                f"Select visible menu option '{option}'.",
                "menu",
            )
    elif state_type in COMBAT_STATE_TYPES:
        battle = safe_dict(state.get("battle"))
        if battle.get("turn") in {None, "player"} and battle.get("is_play_phase") is not False:
            enemies = living_enemies(state)
            for card in safe_list(safe_dict(state.get("player")).get("hand")):
                if not isinstance(card, dict) or card.get("can_play") is not True:
                    continue
                card_index = int(card.get("index"))
                if target_type_needs_enemy(card.get("target_type")):
                    for enemy in enemies:
                        add_legal_action(
                            actions,
                            tools,
                            f"card_{card_index}_{card.get('id') or card.get('name')}_{enemy.get('entity_id')}",
                            "combat_play_card",
                            {"card_index": card_index, "target": enemy["entity_id"]},
                            combat_card_summary(state, card, enemy),
                            "combat",
                        )
                else:
                    add_legal_action(
                        actions,
                        tools,
                        f"card_{card_index}_{card.get('id') or card.get('name')}",
                        "combat_play_card",
                        {"card_index": card_index},
                        combat_card_summary(state, card, None),
                        "combat",
                    )
            add_legal_action(
                actions,
                tools,
                "combat_end_turn",
                "combat_end_turn",
                {},
                f"End the current combat turn. Incoming attack: {incoming_attack_damage(state)}.",
                "combat",
            )
    elif state_type == "hand_select":
        hand_select = safe_dict(state.get("hand_select"))
        for card in safe_list(hand_select.get("cards")):
            if not isinstance(card, dict):
                continue
            index = int(card.get("index"))
            add_legal_action(
                actions,
                tools,
                f"hand_select_{index}_{card.get('id') or card.get('name')}",
                "combat_select_card",
                {"card_index": index},
                f"Select hand card {indexed_label(card, 'name', 'id')} for: {hand_select.get('prompt', 'selection')}.",
                "selection",
            )
        if hand_select.get("can_confirm") is True:
            add_legal_action(actions, tools, "hand_select_confirm", "combat_confirm_selection", {}, "Confirm the in-combat card selection.", "selection")
    elif state_type == "rewards":
        rewards = safe_dict(state.get("rewards"))
        items = safe_list(rewards.get("items"))
        for item in items:
            if not isinstance(item, dict):
                continue
            index = int(item.get("index"))
            reward_type = item.get("type", "reward")
            add_legal_action(
                actions,
                tools,
                f"reward_{index}_{reward_type}",
                "rewards_claim",
                {"reward_index": index},
                f"Claim reward {index}: {reward_type}. {item.get('description', '')}".strip(),
                "rewards",
            )
        if rewards.get("can_proceed") is True:
            summary = "Leave rewards and proceed to the map." if not items else "Skip remaining rewards and proceed to the map."
            add_legal_action(actions, tools, "proceed_to_map", "proceed_to_map", {}, summary, "navigation")
    elif state_type == "card_reward":
        card_reward = safe_dict(state.get("card_reward"))
        for card in safe_list(card_reward.get("cards")):
            if not isinstance(card, dict):
                continue
            index = int(card.get("index"))
            summary = f"Pick card reward {indexed_label(card, 'name', 'id')}. {card.get('type', '')} {card.get('rarity', '')}. {card.get('description', '')}".strip()
            add_legal_action(actions, tools, f"card_reward_{index}_{card.get('id') or card.get('name')}", "rewards_pick_card", {"card_index": index}, summary, "rewards")
        if card_reward.get("can_skip") is True:
            add_legal_action(actions, tools, "card_reward_skip", "rewards_skip_card", {}, "Skip this card reward.", "rewards")
    elif state_type == "map":
        for node in safe_list(safe_dict(state.get("map")).get("next_options")):
            if not isinstance(node, dict):
                continue
            index = int(node.get("index"))
            leads = ", ".join(str(child.get("type")) for child in safe_list(node.get("leads_to")) if isinstance(child, dict) and child.get("type"))
            leads_text = f"; next choices after it: {leads}" if leads else ""
            add_legal_action(actions, tools, f"map_{index}_{node.get('type')}", "map_choose_node", {"node_index": index}, f"Choose map node {index}: {node.get('type')}{leads_text}.", "map")
    elif state_type == "event":
        event = safe_dict(state.get("event"))
        if event.get("in_dialogue") is True:
            add_legal_action(actions, tools, "event_advance_dialogue", "event_advance_dialogue", {}, f"Advance dialogue for {event.get('event_name', 'event')}.", "event")
        else:
            for option in safe_list(event.get("options")):
                if not isinstance(option, dict) or option.get("is_locked") is True:
                    continue
                index = int(option.get("index"))
                title = option.get("title") or option.get("description") or "option"
                add_legal_action(actions, tools, f"event_{index}_{title}", "event_choose_option", {"option_index": index}, f"Choose event option {index}: {title}. {option.get('description', '')}".strip(), "event")
    elif state_type == "rest_site":
        rest = safe_dict(state.get("rest_site"))
        for option in safe_list(rest.get("options")):
            if not isinstance(option, dict) or option.get("is_enabled") is False:
                continue
            index = int(option.get("index"))
            add_legal_action(actions, tools, f"rest_{index}_{option.get('id') or option.get('name')}", "rest_choose_option", {"option_index": index}, f"Choose rest option {indexed_label(option, 'name', 'id')}. {option.get('description', '')}".strip(), "rest")
        if rest.get("can_proceed") is True:
            add_legal_action(actions, tools, "proceed_to_map", "proceed_to_map", {}, "Leave the rest site and proceed to the map.", "navigation")
    elif state_type in {"shop", "fake_merchant"}:
        for item in shop_items(state):
            if not isinstance(item, dict):
                continue
            if item.get("is_stocked") is False or item.get("can_afford") is False:
                continue
            index = int(item.get("index"))
            label = item.get("card_name") or item.get("relic_name") or item.get("potion_name") or item.get("category") or "shop item"
            price = item.get("price", item.get("cost", "?"))
            add_legal_action(actions, tools, f"shop_{index}_{label}", "shop_purchase", {"item_index": index}, f"Buy shop item {index}: {label} for {price} gold.", "shop")
        if shop_can_proceed(state):
            add_legal_action(actions, tools, "proceed_to_map", "proceed_to_map", {}, "Leave the shop and proceed to the map.", "navigation")
    elif state_type == "treasure":
        treasure = safe_dict(state.get("treasure"))
        for relic in safe_list(treasure.get("relics")):
            if not isinstance(relic, dict):
                continue
            index = int(relic.get("index"))
            add_legal_action(actions, tools, f"treasure_{index}_{relic.get('id') or relic.get('name')}", "treasure_claim_relic", {"relic_index": index}, f"Claim treasure relic {indexed_label(relic, 'name', 'id')}. {relic.get('description', '')}".strip(), "treasure")
        if treasure.get("can_proceed") is True:
            add_legal_action(actions, tools, "proceed_to_map", "proceed_to_map", {}, "Leave the treasure room and proceed to the map.", "navigation")
    elif state_type == "card_select":
        selection = safe_dict(state.get("card_select"))
        for card in safe_list(selection.get("cards")):
            if not isinstance(card, dict):
                continue
            index = int(card.get("index"))
            add_legal_action(actions, tools, f"deck_select_{index}_{card.get('id') or card.get('name')}", "deck_select_card", {"card_index": index}, f"Select deck card {indexed_label(card, 'name', 'id')} for: {selection.get('prompt', 'selection')}. {card.get('description', '')}".strip(), "selection")
        if selection.get("can_confirm") is True:
            add_legal_action(actions, tools, "deck_confirm_selection", "deck_confirm_selection", {}, "Confirm the selected deck cards.", "selection")
        if selection.get("can_cancel") is True or selection.get("can_skip") is True:
            add_legal_action(actions, tools, "deck_cancel_selection", "deck_cancel_selection", {}, "Cancel or skip the card selection screen.", "selection")
    elif state_type == "bundle_select":
        selection = safe_dict(state.get("bundle_select"))
        for bundle in safe_list(selection.get("bundles")):
            if not isinstance(bundle, dict):
                continue
            index = int(bundle.get("index"))
            cards = ", ".join(str(card.get("name") or card.get("id")) for card in safe_list(bundle.get("cards")) if isinstance(card, dict))
            add_legal_action(actions, tools, f"bundle_{index}", "bundle_select", {"bundle_index": index}, f"Open bundle {index}: {cards}.", "selection")
        if selection.get("can_confirm") is True:
            add_legal_action(actions, tools, "bundle_confirm_selection", "bundle_confirm_selection", {}, "Confirm the current bundle preview.", "selection")
        if selection.get("can_cancel") is True:
            add_legal_action(actions, tools, "bundle_cancel_selection", "bundle_cancel_selection", {}, "Cancel the bundle preview.", "selection")
    elif state_type == "relic_select":
        selection = safe_dict(state.get("relic_select"))
        for relic in safe_list(selection.get("relics")):
            if not isinstance(relic, dict):
                continue
            index = int(relic.get("index"))
            add_legal_action(actions, tools, f"relic_{index}_{relic.get('id') or relic.get('name')}", "relic_select", {"relic_index": index}, f"Choose relic {indexed_label(relic, 'name', 'id')}. {relic.get('description', '')}".strip(), "relic")
        if selection.get("can_skip") is True:
            add_legal_action(actions, tools, "relic_skip", "relic_skip", {}, "Skip the relic choice.", "relic")
    elif state_type == "crystal_sphere":
        sphere = safe_dict(state.get("crystal_sphere"))
        if sphere.get("can_use_big_tool") is True:
            add_legal_action(actions, tools, "crystal_tool_big", "crystal_sphere_set_tool", {"tool": "big"}, "Switch to the big Crystal Sphere tool.", "crystal_sphere")
        if sphere.get("can_use_small_tool") is True:
            add_legal_action(actions, tools, "crystal_tool_small", "crystal_sphere_set_tool", {"tool": "small"}, "Switch to the small Crystal Sphere tool.", "crystal_sphere")
        for cell in safe_list(sphere.get("clickable_cells")):
            if not isinstance(cell, dict) or cell.get("x") is None or cell.get("y") is None:
                continue
            x = int(cell.get("x"))
            y = int(cell.get("y"))
            add_legal_action(actions, tools, f"crystal_cell_{x}_{y}", "crystal_sphere_click_cell", {"x": x, "y": y}, f"Reveal Crystal Sphere cell ({x}, {y}).", "crystal_sphere")
        if sphere.get("can_proceed") is True:
            add_legal_action(actions, tools, "crystal_sphere_proceed", "crystal_sphere_proceed", {}, "Finish the Crystal Sphere minigame.", "crystal_sphere")

    if state_type not in {"menu", "game_over", "unknown", "overlay", None}:
        add_potion_actions(state, tools, actions)

    if not actions:
        add_legal_action(actions, tools, "wait", "wait", {}, "No accepted player action is visible; wait briefly and poll state again.", "local")
    if state_type in {"game_over", "unknown", "overlay"}:
        add_legal_action(actions, tools, "stop", "stop", {}, "Stop the harness cleanly.", "local")

    return actions


def legal_action_map(actions: list[LegalAction]) -> dict[str, LegalAction]:
    return {action.id: action for action in actions}


def legal_tools(actions: list[LegalAction], tools: dict[str, ToolSpec]) -> dict[str, ToolSpec]:
    return {action.tool: tools[action.tool] for action in actions if action.tool in tools}


def valid_actions_json(actions: list[LegalAction]) -> str:
    payload = [action.for_prompt() for action in actions]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_prompt_metadata(bundle: PromptBundle, system_prompt: str) -> Json:
    metadata = bundle.metadata()
    metadata["game_context_sha256"] = sha256_text(GAME_CONTEXT)
    metadata["rendered_system_sha256"] = sha256_text(system_prompt)
    return metadata


def names_for(items: list[Any], *, name_keys: tuple[str, ...] = ("name", "title", "type", "category")) -> list[str]:
    labels: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        for key in name_keys:
            value = item.get(key)
            if isinstance(value, str) and value:
                labels.append(f"{index}:{value}")
                break
        else:
            labels.append(str(index))
    return labels


def limited(values: list[str], limit: int = 10) -> str:
    if not values:
        return "none"
    shown = values[:limit]
    suffix = f", ... +{len(values) - limit}" if len(values) > limit else ""
    return ", ".join(shown) + suffix


def action_state_hint(state: Json, action: str) -> str:
    if action == "menu_select":
        return f"valid option values: {limited(menu_option_names(state))}"
    if action == "combat_play_card":
        hand = [
            card
            for card in safe_list(safe_dict(state.get("player")).get("hand"))
            if isinstance(card, dict) and card.get("can_play") is True
        ]
        cards = names_for(hand, name_keys=("name", "id"))
        targets = [
            str(enemy.get("entity_id"))
            for enemy in safe_list(safe_dict(state.get("battle")).get("enemies"))
            if isinstance(enemy, dict) and enemy.get("entity_id")
        ]
        return f"playable card_index values: {limited(cards)}; target entity_ids: {limited(targets)}"
    if action == "combat_end_turn":
        return "ends the player turn; use when no useful playable cards remain"
    if action == "combat_select_card":
        cards = names_for(safe_list(safe_dict(state.get("hand_select")).get("cards")), name_keys=("name", "id"))
        return f"valid card_index values: {limited(cards)}"
    if action == "rewards_claim":
        rewards = names_for(safe_list(safe_dict(state.get("rewards")).get("items")), name_keys=("type", "description"))
        return f"valid reward_index values: {limited(rewards)}"
    if action == "rewards_pick_card":
        cards = names_for(safe_list(safe_dict(state.get("card_reward")).get("cards")), name_keys=("name", "id"))
        return f"valid card_index values: {limited(cards)}"
    if action == "map_choose_node":
        nodes = names_for(safe_list(safe_dict(state.get("map")).get("next_options")), name_keys=("type",))
        return f"valid node_index values: {limited(nodes)}"
    if action == "event_choose_option":
        options = [
            item
            for item in safe_list(safe_dict(state.get("event")).get("options"))
            if isinstance(item, dict) and item.get("is_locked") is not True
        ]
        return f"valid option_index values: {limited(names_for(options, name_keys=('title', 'description')))}"
    if action == "rest_choose_option":
        options = [
            item
            for item in safe_list(safe_dict(state.get("rest_site")).get("options"))
            if isinstance(item, dict) and item.get("is_enabled") is not False
        ]
        return f"valid option_index values: {limited(names_for(options, name_keys=('name', 'id')))}"
    if action == "shop_purchase":
        items = [
            item
            for item in shop_items(state)
            if isinstance(item, dict) and item.get("is_stocked") is not False and item.get("can_afford") is not False
        ]
        return f"affordable item_index values: {limited(names_for(items, name_keys=('card_name', 'relic_name', 'potion_name', 'category')))}"
    if action == "deck_select_card":
        cards = names_for(safe_list(safe_dict(state.get("card_select")).get("cards")), name_keys=("name", "id"))
        return f"valid card_index values: {limited(cards)}"
    if action == "bundle_select":
        bundles = names_for(safe_list(safe_dict(state.get("bundle_select")).get("bundles")), name_keys=("card_count",))
        return f"valid bundle_index values: {limited(bundles)}"
    if action == "relic_select":
        relics = names_for(safe_list(safe_dict(state.get("relic_select")).get("relics")), name_keys=("name", "id"))
        return f"valid relic_index values: {limited(relics)}"
    if action == "treasure_claim_relic":
        relics = names_for(safe_list(safe_dict(state.get("treasure")).get("relics")), name_keys=("name", "id"))
        return f"valid relic_index values: {limited(relics)}"
    if action == "use_potion":
        return f"valid slot values: {limited([str(slot) for slot in potion_slots(state)])}"
    if action == "discard_potion":
        return f"valid slot values: {limited([str(slot) for slot in potion_slots(state)])}"
    if action == "crystal_sphere_set_tool":
        sphere = safe_dict(state.get("crystal_sphere"))
        tools = []
        if sphere.get("can_use_big_tool") is True:
            tools.append("big")
        if sphere.get("can_use_small_tool") is True:
            tools.append("small")
        return f"valid tool values: {limited(tools)}"
    if action == "crystal_sphere_click_cell":
        cells = [
            f"{cell.get('x')},{cell.get('y')}"
            for cell in safe_list(safe_dict(state.get("crystal_sphere")).get("clickable_cells"))
            if isinstance(cell, dict) and cell.get("x") is not None and cell.get("y") is not None
        ]
        return f"valid x,y cells: {limited(cells)}"
    if action == "proceed_to_map":
        return "leave this completed screen and return to the map"
    return ""


def tool_menu(tools: dict[str, ToolSpec], state: Json | None = None) -> str:
    lines = []
    for name in LOCAL_ACTIONS:
        lines.append(f"- {name}: {LOCAL_ACTIONS[name]}")
    for name in sorted(tools):
        spec = tools[name]
        schema = spec.input_schema or {}
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        arg_bits = []
        for arg_name, arg_schema in properties.items():
            marker = "" if arg_name in required else "?"
            arg_type = "value"
            if isinstance(arg_schema, dict):
                arg_type = str(arg_schema.get("type") or arg_schema.get("title") or "value")
            arg_bits.append(f"{arg_name}{marker}:{arg_type}")
        args_text = f"({', '.join(arg_bits)})" if arg_bits else "()"
        hint = action_state_hint(state, name) if state is not None else ""
        hint_text = f" State hint: {hint}." if hint else ""
        lines.append(f"- {name}{args_text}: {spec.description}{hint_text}")
    return "\n".join(lines)


def build_user_prompt(
    state: Json,
    step: int,
    max_state_chars: int,
    history: list[str],
    legal_actions: list[LegalAction],
    prompt_bundle: PromptBundle,
) -> str:
    recent_history = "\n".join(history[-6:]) if history else "No prior actions in this harness run."
    return render_user_prompt(
        prompt_bundle,
        step=step,
        valid_actions_json=valid_actions_json(legal_actions),
        recent_history=recent_history,
        state_json=compact_json(state_for_prompt(state), max_state_chars),
    )


def extract_text_content(result: Any) -> str:
    pieces = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            pieces.append(str(text))
    if pieces:
        return "\n".join(pieces)
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)
    return str(result)


def parse_tool_json(result: Any) -> Any:
    text = extract_text_content(result)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def mcp_get_state(session: ClientSession) -> Json:
    result = await session.call_tool("get_game_state", {"format": "json"})
    parsed = parse_tool_json(result)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"unexpected get_game_state result: {parsed!r}")
    return parsed


async def mcp_call_action(session: ClientSession, tools: dict[str, ToolSpec], action: str, args: Json) -> Any:
    cleaned_args = {k: v for k, v in args.items() if not k.startswith("_")}
    schema = tools.get(action).input_schema if action in tools else {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if properties:
        cleaned_args = {k: v for k, v in cleaned_args.items() if k in properties}
    result = await session.call_tool(action, cleaned_args)
    return parse_tool_json(result)


async def mcp_call_setup_tool(session: ClientSession, tool_name: str, args: Json) -> Any:
    result = parse_tool_json(await session.call_tool(tool_name, args))
    if isinstance(result, dict) and result.get("status") == "error":
        raise RuntimeError(f"{tool_name} failed: {result.get('error')}")
    return result


def compact_state_summary(state: Json | None) -> Json | None:
    if not isinstance(state, dict):
        return None
    player = safe_dict(state.get("player"))
    run = safe_dict(state.get("run"))
    summary: Json = {
        "state_type": state.get("state_type"),
        "menu_screen": state.get("menu_screen"),
        "run": run or None,
    }
    if player:
        summary["player"] = {
            "character": player.get("character"),
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "gold": player.get("gold"),
            "relic_count": len(safe_list(player.get("relics"))),
            "potion_count": len(safe_list(player.get("potions"))),
            "deck_count": len(safe_list(player.get("deck"))),
        }
    if state.get("state_type") in COMBAT_STATE_TYPES:
        battle = safe_dict(state.get("battle"))
        summary["battle"] = {
            "room_type": state.get("state_type"),
            "turn": battle.get("turn"),
            "enemies": [
                {
                    "name": enemy.get("name"),
                    "entity_id": enemy.get("entity_id"),
                    "hp": enemy.get("hp"),
                    "max_hp": enemy.get("max_hp"),
                }
                for enemy in safe_list(battle.get("enemies"))
                if isinstance(enemy, dict)
            ],
        }
    if state.get("state_type") == "event":
        event = safe_dict(state.get("event"))
        summary["event"] = {
            "event_id": event.get("event_id"),
            "event_name": event.get("event_name"),
            "in_dialogue": event.get("in_dialogue"),
        }
    if state.get("state_type") == "map":
        game_map = safe_dict(state.get("map"))
        summary["map"] = {
            "current_position": game_map.get("current_position"),
            "next_option_count": len(safe_list(game_map.get("next_options"))),
            "boss": game_map.get("boss"),
            "bosses": game_map.get("bosses"),
        }
    if state.get("state_type") == "game_over":
        summary["game_over"] = safe_dict(state.get("game_over"))
    return summary


def eval_outcome(stop_reason: str, final_state: Json | None) -> str:
    if isinstance(final_state, dict) and final_state.get("state_type") == "game_over":
        return "game_over"
    if stop_reason == "step_limit":
        return "step_limit"
    return stop_reason


def result_is_ok(result: Any) -> bool:
    return isinstance(result, dict) and result.get("status") == "ok"


def state_progress_signature(state: Json | None) -> str:
    return json.dumps(compact_state_summary(state) or {}, ensure_ascii=False, sort_keys=True)


def write_eval_summary(
    summary_path: Path,
    *,
    args: argparse.Namespace,
    log_path: Path,
    started_at: dt.datetime,
    ended_at: dt.datetime,
    steps_taken: int,
    stop_reason: str,
    setup: Json | None,
    final_state: Json | None,
    error: str | None = None,
) -> None:
    summary: Json = {
        "kind": "sts2_seeded_eval",
        "character": SEEDED_EVAL_CHARACTER,
        "seed": SEEDED_EVAL_SEED,
        "max_steps": args.steps,
        "steps_taken": steps_taken,
        "outcome": eval_outcome(stop_reason, final_state),
        "stop_reason": stop_reason,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round((ended_at - started_at).total_seconds(), 3),
        "log_path": str(log_path),
        "model": args.model,
        "llm_url": args.llm_url,
        "llm_source": getattr(args, "llm_source", "custom"),
        "setup": setup,
        "final_state": compact_state_summary(final_state),
    }
    if error:
        summary["error"] = error
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def wait_for_non_menu_state(session: ClientSession, timeout: float, sleep_seconds: float) -> Json:
    deadline = time.monotonic() + timeout
    last_state: Json = {}
    while time.monotonic() < deadline:
        last_state = await mcp_get_state(session)
        if last_state.get("state_type") != "menu" and last_state.get("state_type") not in TRANSIENT_STATE_TYPES:
            return last_state
        time.sleep(sleep_seconds)
    raise RuntimeError(
        "timed out waiting for eval run to leave menu/setup transition after eval_start_run; "
        f"last state: {compact_state_summary(last_state)}"
    )


async def settle_after_transition_action(
    session: ClientSession,
    previous_state: Json,
    timeout: float,
    sleep_seconds: float,
) -> Json | None:
    previous_signature = state_progress_signature(previous_state)
    deadline = time.monotonic() + timeout
    last_state: Json | None = None
    while time.monotonic() < deadline:
        time.sleep(sleep_seconds)
        last_state = await mcp_get_state(session)
        state_type = last_state.get("state_type")
        if state_type in TRANSIENT_STATE_TYPES:
            continue
        if state_progress_signature(last_state) != previous_signature:
            return last_state
    return last_state


async def start_seeded_eval_run(session: ClientSession, args: argparse.Namespace, log_path: Path) -> Json:
    setup: Json = {
        "character": SEEDED_EVAL_CHARACTER,
        "seed": SEEDED_EVAL_SEED,
        "events": [],
    }

    def record(event: str, state: Json | None = None, result: Any = None) -> None:
        entry: Json = {"event": event}
        if state is not None:
            entry["state"] = compact_state_summary(state)
        if result is not None:
            entry["result"] = result
        setup["events"].append(entry)
        log_event(
            log_path,
            {
                "phase": "eval_setup",
                "event": event,
                "state": state,
                "result": result,
                "character": SEEDED_EVAL_CHARACTER,
                "seed": SEEDED_EVAL_SEED,
            },
        )

    state = await mcp_get_state(session)
    record("initial_state", state=state)
    if state.get("state_type") != "menu":
        raise RuntimeError(
            "seeded eval setup requires the game to be at the main menu, singleplayer menu, "
            f"or standard character-select screen; current state is {state.get('state_type')!r}."
        )

    menu_screen = state.get("menu_screen")
    if menu_screen == "main":
        result = await mcp_call_setup_tool(session, "menu_select", {"option": "singleplayer"})
        record("menu_select_singleplayer", result=result)
        time.sleep(args.eval_setup_sleep)
        state = await mcp_get_state(session)
        record("after_singleplayer", state=state)
        menu_screen = state.get("menu_screen")

    if menu_screen == "singleplayer":
        result = await mcp_call_setup_tool(session, "menu_select", {"option": "standard"})
        record("menu_select_standard", result=result)
        time.sleep(args.eval_setup_sleep)
        state = await mcp_get_state(session)
        record("after_standard", state=state)
        menu_screen = state.get("menu_screen")

    if menu_screen != "character_select":
        raise RuntimeError(f"seeded eval setup expected character_select, got menu_screen={menu_screen!r}.")

    result = await mcp_call_setup_tool(
        session,
        "eval_start_run",
        {"character": SEEDED_EVAL_CHARACTER, "seed": SEEDED_EVAL_SEED},
    )
    record("eval_start_run", result=result)
    final_state = await wait_for_non_menu_state(session, args.eval_setup_timeout, args.eval_setup_sleep)
    record("run_started", state=final_state)
    setup["started_state"] = compact_state_summary(final_state)
    return setup


def normalize_mcp_action(raw: Json, tools: dict[str, ToolSpec]) -> tuple[str, Json, str]:
    action, args, rationale = normalize_action_against_tools(raw, tools)
    action = ALIASES.get(action, action)
    if action not in LOCAL_ACTIONS and action not in tools:
        raise ValueError(f"unknown MCP action: {action}")
    return action, args, rationale


def normalize_action_against_tools(raw: Json, tools: dict[str, ToolSpec]) -> tuple[str, Json, str]:
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
    if not isinstance(candidate, str):
        raise ValueError("model response must include string field 'action'")
    action = candidate
    args = raw.get("args") or raw.get("arguments") or nested_args or {}
    if not isinstance(args, dict):
        raise ValueError("model response field 'args' must be an object")
    if action not in LOCAL_ACTIONS and action not in tools and action not in ALIASES:
        # Let the shared direct harness parser produce a more familiar error
        # when it happens to know this alias/action.
        action, args, rationale = normalize_action(raw)
        return action, args, rationale
    return action, args, str(raw.get("rationale", ""))


def required_tool_args(tools: dict[str, ToolSpec], action: str) -> list[str]:
    spec = tools.get(action)
    if spec is None:
        return []
    required = spec.input_schema.get("required", [])
    return [str(item) for item in required] if isinstance(required, list) else []


def valid_indices(values: Any) -> set[int]:
    if not isinstance(values, list):
        return set()
    indices: set[int] = set()
    for item in values:
        if isinstance(item, dict) and "index" in item:
            try:
                indices.add(int(item["index"]))
            except (TypeError, ValueError):
                pass
    return indices


def parse_int_arg(args: Json, name: str) -> int | None:
    try:
        return int(args[name])
    except (KeyError, TypeError, ValueError):
        return None


def index_validation_error(state: Json, action: str, args: Json) -> str | None:
    checks: dict[str, tuple[str, list[int]]] = {
        "combat_select_card": ("card_index", indexed_values(safe_dict(state.get("hand_select")).get("cards"))),
        "event_choose_option": (
            "option_index",
            indexed_values(safe_dict(state.get("event")).get("options"), enabled_only=True),
        ),
        "rest_choose_option": (
            "option_index",
            indexed_values(safe_dict(state.get("rest_site")).get("options"), enabled_only=True),
        ),
        "shop_purchase": ("item_index", indexed_values(shop_items(state), enabled_only=True)),
        "deck_select_card": ("card_index", indexed_values(safe_dict(state.get("card_select")).get("cards"))),
        "bundle_select": ("bundle_index", indexed_values(safe_dict(state.get("bundle_select")).get("bundles"))),
        "relic_select": ("relic_index", indexed_values(safe_dict(state.get("relic_select")).get("relics"))),
        "treasure_claim_relic": ("relic_index", indexed_values(safe_dict(state.get("treasure")).get("relics"))),
    }
    check = checks.get(action)
    if check is None:
        return None
    arg_name, indices = check
    value = parse_int_arg(args, arg_name)
    if value is None:
        return f"{action} requires args.{arg_name} as an integer."
    if indices and value not in indices:
        return f"{arg_name} must be one of {indices}."
    return None


def validate_model_action(
    state: Json,
    tools: dict[str, ToolSpec],
    action: str,
    args: Json,
    allowed_tools: dict[str, ToolSpec],
) -> str | None:
    if action in LOCAL_ACTIONS:
        return None
    if action not in tools:
        return f"{action!r} is not an available MCP action."

    if action not in allowed_tools:
        allowed = sorted(allowed_tools)
        if allowed:
            return (
                f"{action} is not valid for current state_type {state.get('state_type')!r}. "
                f"Choose one of the current-state actions: {allowed}."
            )
        return f"{action} is not valid for current state_type {state.get('state_type')!r}. Choose wait or stop."

    missing = [name for name in required_tool_args(tools, action) if name not in args]
    if missing:
        return f"{action} requires args.{', args.'.join(missing)}. Put required fields inside the args object."

    if "target" in args and not isinstance(args["target"], str):
        return "args.target must be a string entity_id, for example \"NIBBIT_0\"."

    if action == "menu_select":
        option = args.get("option")
        options = menu_option_names(state)
        if not isinstance(option, str):
            return "menu_select requires args.option as a string."
        if options and option not in options:
            return f"option must be one of {options}."

    state_type = state.get("state_type")

    if action == "rewards_pick_card" and state_type != "card_reward":
        if state_type == "rewards":
            reward_items = state.get("rewards", {}).get("items", [])
            card_indices = [
                item.get("index")
                for item in reward_items
                if isinstance(item, dict) and item.get("type") == "card"
            ]
            return (
                "rewards_pick_card is only valid on state_type card_reward. "
                f"Current state_type is rewards. Use rewards_claim with args.reward_index first"
                f"{f' (card reward indices: {card_indices})' if card_indices else ''}."
            )
        return f"rewards_pick_card is only valid on state_type card_reward. Current state_type is {state_type!r}."

    if action == "rewards_claim" and state_type != "rewards":
        return f"rewards_claim is only valid on state_type rewards. Current state_type is {state_type!r}."

    if action == "map_choose_node" and state_type != "map":
        return f"map_choose_node is only valid on state_type map. Current state_type is {state_type!r}."

    if action == "combat_play_card":
        if not state_type in {"monster", "elite", "boss"}:
            return f"combat_play_card is only valid in combat. Current state_type is {state_type!r}."
        try:
            card_index = int(args["card_index"])
        except (KeyError, TypeError, ValueError):
            return "combat_play_card requires args.card_index as a hand index integer."
        hand = state.get("player", {}).get("hand", [])
        card = next(
            (item for item in hand if isinstance(item, dict) and int(item.get("index", -1)) == card_index),
            None,
        )
        if card is None:
            return f"card_index {card_index} is not in the current hand."
        if card.get("can_play") is not True:
            return f"card_index {card_index} cannot be played: {card.get('unplayable_reason')}."
        if "Enemy" in str(card.get("target_type", "")) and not args.get("target"):
            enemies = [
                enemy.get("entity_id")
                for enemy in state.get("battle", {}).get("enemies", [])
                if isinstance(enemy, dict) and enemy.get("entity_id")
            ]
            return f"card_index {card_index} needs args.target. Valid enemy entity_ids: {enemies}."

    if action == "rewards_claim" and state.get("state_type") == "rewards":
        indices = valid_indices(state.get("rewards", {}).get("items"))
        reward_index = parse_int_arg(args, "reward_index")
        if reward_index is None:
            return "rewards_claim requires args.reward_index as an integer."
        if indices and reward_index not in indices:
            return f"reward_index must be one of {sorted(indices)}."

    if action == "rewards_pick_card" and state.get("state_type") == "card_reward":
        indices = valid_indices(state.get("card_reward", {}).get("cards"))
        card_index = parse_int_arg(args, "card_index")
        if card_index is None:
            return "rewards_pick_card requires args.card_index as an integer."
        if indices and card_index not in indices:
            return f"card_index must be one of {sorted(indices)}."

    if action == "map_choose_node" and state.get("state_type") == "map":
        indices = valid_indices(state.get("map", {}).get("next_options"))
        node_index = parse_int_arg(args, "node_index")
        if node_index is None:
            return "map_choose_node requires args.node_index as an integer."
        if indices and node_index not in indices:
            return f"node_index must be one of {sorted(indices)}."

    if action in {"use_potion", "discard_potion"}:
        slots = potion_slots(state)
        slot = parse_int_arg(args, "slot")
        if slot is None:
            return f"{action} requires args.slot as an integer."
        if slots and slot not in slots:
            return f"slot must be one of {slots}."

    if action == "crystal_sphere_set_tool":
        tool = args.get("tool")
        sphere = safe_dict(state.get("crystal_sphere"))
        valid_tools = []
        if sphere.get("can_use_big_tool") is True:
            valid_tools.append("big")
        if sphere.get("can_use_small_tool") is True:
            valid_tools.append("small")
        if tool not in valid_tools:
            return f"tool must be one of {valid_tools}."

    if action == "crystal_sphere_click_cell":
        try:
            x = int(args["x"])
            y = int(args["y"])
        except (KeyError, TypeError, ValueError):
            return "crystal_sphere_click_cell requires integer args.x and args.y."
        cells = {
            (int(cell.get("x")), int(cell.get("y")))
            for cell in safe_list(safe_dict(state.get("crystal_sphere")).get("clickable_cells"))
            if isinstance(cell, dict) and cell.get("x") is not None and cell.get("y") is not None
        }
        if cells and (x, y) not in cells:
            return f"x,y must be one of {sorted(cells)}."

    index_error = index_validation_error(state, action, args)
    if index_error is not None:
        return index_error

    return None


def normalize_action_id_choice(raw: Json, actions: list[LegalAction], tools: dict[str, ToolSpec]) -> tuple[LegalAction, str]:
    action_by_id = legal_action_map(actions)
    action_id = raw.get("action_id") or raw.get("id")
    if isinstance(action_id, dict):
        action_id = action_id.get("id") or action_id.get("action_id")
    rationale = str(raw.get("rationale", ""))
    if isinstance(action_id, str):
        if action_id in action_by_id:
            return action_by_id[action_id], rationale
        close = ", ".join(list(action_by_id)[:12])
        raise ValueError(f"unknown action_id {action_id!r}. Choose one of the listed valid action ids. First ids: {close}")

    if any(key in raw for key in ("action", "action_name", "tool", "tool_name", "name", "next_action")):
        action, args, rationale = normalize_mcp_action(raw, tools)
        for candidate in actions:
            if candidate.tool == action and candidate.args == args:
                return candidate, rationale
        raise ValueError(
            "model returned tool/args instead of action_id, and those args do not match a listed valid action. "
            "Return only {\"action_id\":\"...\",\"rationale\":\"...\"}."
        )

    raise ValueError("model response must include string field 'action_id'")


def correction_prompt(state: Json, bad_text: str, error: str, legal_actions: list[LegalAction]) -> str:
    return f"""Your previous action JSON was invalid and no tool was called.

Validation error:
{error}

Previous response:
{bad_text}

Current state has not changed:
{compact_json(state_for_prompt(state), 12000)}

valid_actions:
{valid_actions_json(legal_actions)}

Return corrected JSON only: {{"action_id":"one listed id","rationale":"short public reason"}}."""


def clone_messages(messages: list[Json]) -> list[Json]:
    return [{"role": str(item.get("role", "")), "content": str(item.get("content", ""))} for item in messages]


def first_choice(response: Json) -> Json:
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def compact_llm_response(response: Json) -> Json:
    choice = first_choice(response)
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    return {
        "message": message,
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
    }


def model_request_trace(args: argparse.Namespace, messages: list[Json], prompt_metadata: Json) -> Json:
    return {
        "llm_url": args.llm_url,
        "llm_source": getattr(args, "llm_source", "custom"),
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "prompt": prompt_metadata,
        "messages": clone_messages(messages),
    }


def choose_model_action(
    args: argparse.Namespace,
    state: Json,
    tools: dict[str, ToolSpec],
    legal_actions: list[LegalAction],
    user_prompt: str,
    system_prompt: str,
    prompt_metadata: Json,
) -> tuple[str, Json, str, Json, str, Json, Json]:
    allowed_tools = legal_tools(legal_actions, tools)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error = ""
    raw_response: Json = {}
    raw_text = ""
    parsed: Json = {}
    trace: Json = {
        "prompt": prompt_metadata,
        "valid_actions": [action.for_prompt() for action in legal_actions],
        "attempts": [],
    }

    for attempt in range(args.repair_attempts + 1):
        parsed = {}
        request = model_request_trace(args, messages, prompt_metadata)
        raw_text, raw_response = call_llm(
            args.llm_url,
            args.model,
            messages,
            args.temperature,
            args.max_tokens,
            args.llm_timeout,
        )
        action = ""
        action_args: Json = {}
        rationale = ""
        validation_error = None
        action_id = ""
        try:
            parsed = extract_json_object(raw_text)
            selected, rationale = normalize_action_id_choice(parsed, legal_actions, tools)
            action_id = selected.id
            action = selected.tool
            action_args = dict(selected.args)
            validation_error = validate_model_action(state, tools, action, action_args, allowed_tools)
        except ValueError as exc:
            validation_error = str(exc)

        attempt_trace = {
            "attempt": attempt + 1,
            "request": request,
            "response": compact_llm_response(raw_response),
            "raw_text": raw_text,
            "parsed": parsed,
            "normalized": {
                "action_id": action_id,
                "action": action,
                "args": action_args,
                "rationale": rationale,
            },
            "validation_error": validation_error,
        }
        trace["attempts"].append(attempt_trace)

        if validation_error is None:
            trace["final_attempt"] = attempt + 1
            return action, action_args, rationale, parsed, raw_text, raw_response, trace

        last_error = validation_error
        if attempt < args.repair_attempts:
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": correction_prompt(state, raw_text, validation_error, legal_actions)})

    trace["final_error"] = last_error
    raise ModelActionError(f"model produced invalid action after {args.repair_attempts + 1} attempt(s): {last_error}", trace)


async def discover_tools(session: ClientSession) -> dict[str, ToolSpec]:
    result = await session.list_tools()
    tools: dict[str, ToolSpec] = {}
    for tool in result.tools:
        if not is_action_tool(tool.name):
            continue
        tools[tool.name] = ToolSpec(
            name=tool.name,
            description=short_description(tool.description),
            input_schema=coerce_tool_schema(tool.inputSchema),
        )
    return tools


def make_server_params(args: argparse.Namespace) -> StdioServerParameters:
    root = repo_root()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return StdioServerParameters(
        command=args.mcp_python,
        args=[
            str(root / "mcp/server.py"),
            "--host",
            args.sts2_host,
            "--port",
            str(args.sts2_port),
        ],
        cwd=str(root / "mcp"),
        env=env,
    )


async def run_async(args: argparse.Namespace) -> int:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_prefix = "mcp-sts2-seeded-eval" if args.seeded_eval else "mcp-sts2-harness"
    log_path = log_dir / f"{log_prefix}-{stamp}.jsonl"
    summary_path = (
        Path(args.eval_summary_path)
        if args.eval_summary_path
        else log_dir / f"{log_prefix}-{stamp}.summary.json"
    )
    prompt_version = args.prompt_version
    system_prompt_version = args.system_prompt_version or prompt_version
    user_prompt_version = args.user_prompt_version or prompt_version
    prompt_bundle = load_sts2_mcp_action_prompts(
        system_version=system_prompt_version,
        user_version=user_prompt_version,
    )
    system_prompt = render_system_prompt(prompt_bundle, game_context=GAME_CONTEXT)
    prompt_metadata = build_prompt_metadata(prompt_bundle, system_prompt)

    action_history: list[str] = []
    consecutive_errors = 0
    server_params = make_server_params(args)
    started_at = dt.datetime.now()
    setup_summary: Json | None = None
    final_state: Json | None = None
    final_error: str | None = None
    stop_reason = "step_limit"
    last_step = 0
    exit_code = 0

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await discover_tools(session)
            print(f"Loaded {len(tools)} MCP action tools.", flush=True)
            print(f"Using LLM ({getattr(args, 'llm_source', 'custom')}): {args.model} at {args.llm_url}", flush=True)
            print(
                "Using prompts: "
                f"system={system_prompt_version} ({prompt_bundle.system.sha256[:12]}), "
                f"user={user_prompt_version} ({prompt_bundle.user.sha256[:12]})",
                flush=True,
            )
            if args.seeded_eval:
                print(
                    "Seeded eval: "
                    f"character={SEEDED_EVAL_CHARACTER}, seed={SEEDED_EVAL_SEED}, max_steps={args.steps}",
                    flush=True,
                )
                try:
                    setup_summary = await start_seeded_eval_run(session, args, log_path)
                except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
                    stop_reason = "setup_error"
                    final_error = repr(exc)
                    print(f"Seeded eval setup failed: {exc}", file=sys.stderr, flush=True)
                    try:
                        final_state = await mcp_get_state(session)
                    except Exception:
                        final_state = None
                    write_eval_summary(
                        summary_path,
                        args=args,
                        log_path=log_path,
                        started_at=started_at,
                        ended_at=dt.datetime.now(),
                        steps_taken=last_step,
                        stop_reason=stop_reason,
                        setup=setup_summary,
                        final_state=final_state,
                        error=final_error,
                    )
                    print(f"Eval summary: {summary_path}", flush=True)
                    print(f"Log: {log_path}", flush=True)
                    return 1

            for step in range(1, args.steps + 1):
                last_step = step
                raw_text: str | None = None
                parsed: Json | None = None
                raw_response: Json = {}
                state: Json | None = None
                model_trace: Json | None = None
                try:
                    state = await mcp_get_state(session)
                    final_state = state
                    legal_actions = build_legal_actions(state, tools)
                    user_prompt = build_user_prompt(
                        state,
                        step,
                        args.max_state_chars,
                        action_history,
                        legal_actions,
                        prompt_bundle,
                    )
                    action, action_args, rationale, parsed, raw_text, raw_response, model_trace = choose_model_action(
                        args,
                        state,
                        tools,
                        legal_actions,
                        user_prompt,
                        system_prompt,
                        prompt_metadata,
                    )
                    print_step(step, state, action, action_args, rationale)

                    if action == "stop":
                        result: Any = {"status": "stopped"}
                        stop_reason = "model_stop"
                        log_event(
                            log_path,
                            {
                                "step": step,
                                "state": state,
                                "model_text": raw_text,
                                "parsed": parsed,
                                "model_trace": model_trace,
                                "result": result,
                            },
                        )
                        break
                    if action == "wait" or args.dry_run:
                        result = {"status": "dry_run" if args.dry_run else "wait"}
                    else:
                        result = await mcp_call_action(session, tools, action, action_args)

                    log_event(
                        log_path,
                        {
                            "step": step,
                            "state": state,
                            "model_text": raw_text,
                            "parsed": parsed,
                            "model_trace": model_trace,
                            "result": result,
                            "llm_usage": raw_response.get("usage") if isinstance(raw_response, dict) else None,
                        },
                    )

                    if result_is_ok(result) and action in TRANSITION_SETTLE_ACTIONS:
                        settled_state = await settle_after_transition_action(
                            session,
                            state,
                            args.transition_settle_timeout,
                            args.transition_settle_sleep,
                        )
                        if settled_state is not None:
                            final_state = settled_state
                            log_event(
                                log_path,
                                {
                                    "step": step,
                                    "phase": "post_action_settle",
                                    "action": action,
                                    "state": settled_state,
                                },
                            )

                    action_history.append(history_line(step, state, action, action_args, result))
                    if len(action_history) > args.history_turns:
                        action_history = action_history[-args.history_turns :]

                    if args.show_results:
                        print(compact_json(result, args.max_result_chars), flush=True)
                    if state.get("state_type") == "game_over":
                        stop_reason = "game_over"
                        print("Reached game_over state.", flush=True)
                        break
                    consecutive_errors = 0
                    time.sleep(args.sleep)
                except KeyboardInterrupt:
                    print("\nInterrupted.", flush=True)
                    stop_reason = "interrupted"
                    exit_code = 130
                    break
                except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
                    consecutive_errors += 1
                    final_error = repr(exc)
                    print(f"[{step}] error: {exc}", file=sys.stderr, flush=True)
                    log_event(
                        log_path,
                        {
                            "step": step,
                            "error": repr(exc),
                            "state_for_prompt": state_for_prompt(state) if isinstance(state, dict) else None,
                            "model_text": raw_text,
                            "parsed": parsed,
                            "model_trace": exc.trace if isinstance(exc, ModelActionError) else model_trace,
                        },
                    )
                    if consecutive_errors >= args.max_consecutive_errors:
                        stop_reason = "error"
                        exit_code = 1
                        print(f"Too many consecutive errors. Log: {log_path}", file=sys.stderr, flush=True)
                        break
                    time.sleep(args.sleep)

            if args.seeded_eval:
                try:
                    final_state = await mcp_get_state(session)
                except Exception as exc:
                    if final_error is None:
                        final_error = f"failed to fetch final state: {exc!r}"
                write_eval_summary(
                    summary_path,
                    args=args,
                    log_path=log_path,
                    started_at=started_at,
                    ended_at=dt.datetime.now(),
                    steps_taken=last_step,
                    stop_reason=stop_reason,
                    setup=setup_summary,
                    final_state=final_state,
                    error=final_error,
                )
                print(f"Eval summary: {summary_path}", flush=True)

    print(f"Log: {log_path}", flush=True)
    return exit_code


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Let an OpenAI-compatible model play STS2 through MCP.")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Maximum number of model actions to take. Defaults to "
            f"{DEFAULT_HARNESS_STEPS}, or {DEFAULT_SEEDED_EVAL_STEPS} with --seeded-eval."
        ),
    )
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds to wait after each step.")
    add_llm_arguments(parser)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--history-turns", type=int, default=6)
    parser.add_argument("--max-state-chars", type=int, default=22000)
    parser.add_argument("--max-result-chars", type=int, default=6000)
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    parser.add_argument("--repair-attempts", type=int, default=2, help="Ask the model to fix invalid action JSON this many times before skipping the step.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument(
        "--prompt-version",
        default=os.environ.get("STS2_PROMPT_VERSION", "v1"),
        help="Default version for the MCP action system and user prompt templates.",
    )
    parser.add_argument(
        "--system-prompt-version",
        default=os.environ.get("STS2_SYSTEM_PROMPT_VERSION"),
        help="Override the MCP action system prompt template version.",
    )
    parser.add_argument(
        "--user-prompt-version",
        default=os.environ.get("STS2_USER_PROMPT_VERSION"),
        help="Override the MCP action user prompt template version.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not send actions to the game.")
    parser.add_argument("--show-results", action="store_true", help="Print raw action results.")
    parser.add_argument(
        "--seeded-eval",
        action="store_true",
        help=(
            "Start the hardcoded seeded eval run before the model loop "
            f"({SEEDED_EVAL_CHARACTER} on {SEEDED_EVAL_SEED})."
        ),
    )
    parser.add_argument(
        "--eval-summary-path",
        help="Optional output path for the seeded eval summary JSON.",
    )
    parser.add_argument(
        "--eval-setup-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for eval_start_run to leave character select.",
    )
    parser.add_argument(
        "--eval-setup-sleep",
        type=float,
        default=0.5,
        help="Seconds to wait between eval setup navigation actions.",
    )
    parser.add_argument(
        "--transition-settle-timeout",
        type=float,
        default=3.0,
        help="Seconds to poll for a changed state after transition actions like map_choose_node.",
    )
    parser.add_argument(
        "--transition-settle-sleep",
        type=float,
        default=0.1,
        help="Seconds between post-transition state polls.",
    )
    parser.add_argument("--sts2-host", default=os.environ.get("STS2_HOST", "localhost"))
    parser.add_argument("--sts2-port", type=int, default=int(os.environ.get("STS2_PORT", "15526")))
    parser.add_argument(
        "--mcp-python",
        default=os.environ.get("MCP_PYTHON", str(root / ".local/mcp-venv/bin/python")),
        help="Python executable with mcp and httpx installed.",
    )
    args = parser.parse_args()
    if args.seeded_eval and args.dry_run:
        parser.error("--seeded-eval cannot be combined with --dry-run because eval setup starts a real game run.")
    if args.steps is None:
        args.steps = DEFAULT_SEEDED_EVAL_STEPS if args.seeded_eval else DEFAULT_HARNESS_STEPS
    return resolve_llm_args(args)


def main() -> int:
    return asyncio.run(run_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
