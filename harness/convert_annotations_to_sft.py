#!/usr/bin/env python3
"""Convert STS2 in-game annotations into chat SFT JSONL."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

from prompt_templates import (
    PromptBundle,
    load_sts2_mcp_action_prompts,
    render_system_prompt,
    render_user_prompt,
    sha256_text,
)
from sts2_harness import GAME_CONTEXT, compact_json, state_for_prompt


Json = dict[str, Any]


def iter_json_values(path: Path) -> list[Json]:
    """Read true JSONL, a JSON array, or pretty-printed concatenated JSON objects."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []

    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"{path} contains JSON but not a list")
        return [item for item in value if isinstance(item, dict)]

    decoder = json.JSONDecoder()
    index = 0
    records: list[Json] = []
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        value, next_index = decoder.raw_decode(text, index)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{index} contains a non-object JSON value")
        records.append(value)
        index = next_index
    return records


def valid_actions_json(actions: list[Json]) -> str:
    return json.dumps(actions, ensure_ascii=False, indent=2)


def prompt_metadata(bundle: PromptBundle, system_prompt: str) -> Json:
    metadata = bundle.metadata()
    metadata["game_context_sha256"] = sha256_text(GAME_CONTEXT)
    metadata["rendered_system_sha256"] = sha256_text(system_prompt)
    return metadata


def first_sentence(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
    rationale = parts[0].strip()
    if len(rationale) <= max_chars:
        return rationale
    return rationale[: max_chars - 1].rstrip() + "..."


def selected_action_for(record: Json) -> Json | None:
    selected = record.get("selected_action")
    if isinstance(selected, dict):
        return selected

    selected_id = record.get("selected_action_id")
    for action in record.get("valid_actions") or []:
        if isinstance(action, dict) and action.get("id") == selected_id:
            return action
    return None


def build_user_content(
    bundle: PromptBundle,
    record: Json,
    *,
    step: int,
    max_state_chars: int,
    recent_history: str,
) -> str:
    state = record.get("state")
    if not isinstance(state, dict):
        raise ValueError("record is missing object field 'state'")
    valid_actions = record.get("valid_actions")
    if not isinstance(valid_actions, list):
        raise ValueError("record is missing list field 'valid_actions'")

    return render_user_prompt(
        bundle,
        step=step,
        valid_actions_json=valid_actions_json([action for action in valid_actions if isinstance(action, dict)]),
        recent_history=recent_history,
        state_json=compact_json(state_for_prompt(state), max_state_chars),
    )


def convert_record(
    record: Json,
    *,
    bundle: PromptBundle,
    system_prompt: str,
    prompt_meta: Json,
    source_path: Path,
    source_index: int,
    step: int,
    max_state_chars: int,
    rationale_max_chars: int,
    strict: bool,
) -> Json | None:
    selected_id = record.get("selected_action_id")
    if not isinstance(selected_id, str) or not selected_id:
        message = "record is missing string field 'selected_action_id'"
        if strict:
            raise ValueError(message)
        print(f"skip {source_path}:{source_index}: {message}", file=sys.stderr)
        return None

    valid_actions = [action for action in record.get("valid_actions") or [] if isinstance(action, dict)]
    valid_ids = {str(action.get("id")) for action in valid_actions if action.get("id") is not None}
    if selected_id not in valid_ids:
        message = f"selected_action_id {selected_id!r} is not in valid_actions"
        if strict:
            raise ValueError(message)
        print(f"skip {source_path}:{source_index}: {message}", file=sys.stderr)
        return None

    selected_action = selected_action_for(record) or {}
    reasoning = str(record.get("reasoning_trace") or "").strip()
    rationale = first_sentence(str(selected_action.get("summary") or selected_id), rationale_max_chars)
    final_json = json.dumps(
        {
            "action_id": selected_id,
            "rationale": rationale,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assistant_content = f"<think>\n{reasoning}\n</think>\n\n{final_json}"
    recent_history = str(record.get("recent_history") or "No prior actions in this harness run.")

    state = record.get("state") if isinstance(record.get("state"), dict) else {}
    output = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_user_content(
                    bundle,
                    record,
                    step=step,
                    max_state_chars=max_state_chars,
                    recent_history=recent_history,
                ),
            },
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": record.get("source"),
            "source_file": str(source_path),
            "source_index": source_index,
            "schema_version": record.get("schema_version"),
            "timestamp_utc": record.get("timestamp_utc"),
            "converted_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "prompt": prompt_meta,
            "state_type": state.get("state_type"),
            "valid_actions_count": len(valid_actions),
            "selected_action_id": selected_id,
            "selected_action_tool": selected_action.get("tool"),
            "selected_action_category": selected_action.get("category"),
            "rationale_source": "selected_action.summary:first_sentence",
        },
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert STS2 annotations into chat SFT JSONL.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Annotation files to convert.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output true JSONL dataset path.")
    parser.add_argument("--prompt-version", default="v1", help="Default system/user prompt template version.")
    parser.add_argument("--system-prompt-version", default=None, help="Override system prompt template version.")
    parser.add_argument("--user-prompt-version", default=None, help="Override user prompt template version.")
    parser.add_argument("--max-state-chars", type=int, default=22000)
    parser.add_argument("--rationale-max-chars", type=int, default=200)
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping invalid records.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    system_version = args.system_prompt_version or args.prompt_version
    user_version = args.user_prompt_version or args.prompt_version
    bundle = load_sts2_mcp_action_prompts(system_version=system_version, user_version=user_version)
    system_prompt = render_system_prompt(bundle, game_context=GAME_CONTEXT)
    prompt_meta = prompt_metadata(bundle, system_prompt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seen = 0
    by_state: dict[str, int] = {}
    with args.output.open("w", encoding="utf-8") as handle:
        for input_path in args.inputs:
            for source_index, record in enumerate(iter_json_values(input_path), start=1):
                seen += 1
                example = convert_record(
                    record,
                    bundle=bundle,
                    system_prompt=system_prompt,
                    prompt_meta=prompt_meta,
                    source_path=input_path,
                    source_index=source_index,
                    step=source_index,
                    max_state_chars=args.max_state_chars,
                    rationale_max_chars=args.rationale_max_chars,
                    strict=args.strict,
                )
                if example is None:
                    continue
                state_type = str(example["metadata"].get("state_type") or "unknown")
                by_state[state_type] = by_state.get(state_type, 0) + 1
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1

    print(json.dumps({"seen": seen, "written": written, "by_state": by_state, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
