# Trace Viewer DPO Annotation Format

The trace viewer can turn a logged model decision into a JSONL preference row.
Start it with an annotation path:

```bash
.local/mcp-venv/bin/python harness/trace_viewer_server.py \
  --port 8765 \
  --annotation-path data/sts2_dpo_annotations.jsonl
```

Open the DPO tab, choose the corrected action from the logged `valid_actions`,
write the human reasoning trace, optionally add notes, and save. Each click
appends one JSON object to `--annotation-path`.

## Top-Level Shape

```json
{
  "kind": "sts2_dpo_annotation",
  "schema_version": 1,
  "created_at": "2026-05-25T12:00:00.000Z",
  "saved_at": "2026-05-25T19:00:00.000000+00:00",
  "source": {},
  "prompt": {},
  "game_state": {},
  "valid_actions": [],
  "rejected": {},
  "chosen": {},
  "dpo": {},
  "annotator_notes": ""
}
```

- `created_at`: browser-side timestamp when the row is assembled.
- `saved_at`: server-side timestamp added when the row is appended.
- `source`: provenance for the source trace record.
- `prompt`: exact prompt metadata and chat messages sent to the model.
- `game_state`: game state JSON for the decision point.
- `valid_actions`: legal action menu shown to the model.
- `rejected`: original model decision from the trace.
- `chosen`: human-corrected decision from the annotation UI.
- `dpo`: convenience chat message fields for DPO conversion.
- `annotator_notes`: optional private note from the annotator.

## Source

```json
{
  "log_name": "mcp-sts2-seeded-eval-20260525-153613.jsonl",
  "source_label": "logs/mcp-sts2-seeded-eval-20260525-153613.jsonl",
  "record_index": 17,
  "step": 6,
  "record_kind": "step",
  "model": "Qwen3.6-27B",
  "llm_url": "https://example/v1"
}
```

Use this to trace an annotation back to the original replay step. `record_index`
is the zero-based line position in the loaded trace viewer record array.

## Prompt

```json
{
  "metadata": {
    "prompt_id": "sts2_mcp_action",
    "system": {
      "version": "v1",
      "path": "harness/prompts/sts2_mcp_action_system.v1.md",
      "sha256": "..."
    },
    "user": {
      "version": "v1",
      "path": "harness/prompts/sts2_mcp_action_user.v1.txt",
      "sha256": "..."
    },
    "game_context_sha256": "...",
    "rendered_system_sha256": "..."
  },
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

`prompt.messages` is the actual prompt for the decision. If a source trace is
too old to include `model_trace.attempts[].request.messages`, this array may be
empty and the row should not be used for training until reconstructed.

## Valid Actions

Each entry is one legal action offered to the model:

```json
{
  "id": "card_1_bash_enemy_0",
  "tool": "combat_play_card",
  "args": {"card_index": 1, "target": "LEAF_SLIME_S_0"},
  "summary": "Play Bash ...",
  "category": "combat"
}
```

The model and annotator choose by `id`; the harness executes the associated
`tool` and `args`.

## Rejected Decision

`rejected` is extracted from the model's final attempt for that step:

```json
{
  "action_id": "combat_end_turn",
  "action": "combat_end_turn",
  "args": {},
  "rationale": "No useful playable cards remain.",
  "reasoning": "Full emitted reasoning, if present.",
  "final_text": "{\"action_id\":\"combat_end_turn\",\"rationale\":\"...\"}",
  "raw_text": "<think>...</think>\n\n{\"action_id\":\"combat_end_turn\",\"rationale\":\"...\"}",
  "validation_error": null
}
```

`reasoning` is populated from `message.reasoning_content`, `message.reasoning`,
or `<think>...</think>` blocks when present. With thinking disabled it may be
empty. `raw_text` preserves the exact model response used as the rejected
assistant message when available.

## Chosen Decision

`chosen` is the human correction:

```json
{
  "action_id": "card_1_bash_enemy_0",
  "action": "combat_play_card",
  "args": {"card_index": 1, "target": "LEAF_SLIME_S_0"},
  "rationale": "Apply Vulnerable before attacks.",
  "reasoning": "The enemy is attacking and Bash sets up a faster kill...",
  "final_text": "{\"action_id\":\"card_1_bash_enemy_0\",\"rationale\":\"Apply Vulnerable before attacks.\"}",
  "assistant_content": "<think>\nThe enemy is attacking...\n</think>\n\n{\"action_id\":\"card_1_bash_enemy_0\",\"rationale\":\"Apply Vulnerable before attacks.\"}"
}
```

The DPO tab requires `reasoning` before saving. `assistant_content` is the
preferred-form completion for training with reasoning traces: a `<think>` block
followed by the compact final JSON.

## DPO Convenience Fields

```json
{
  "prompt_messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "chosen_messages": [
    {"role": "assistant", "content": "<think>\n...\n</think>\n\n{\"action_id\":\"...\",\"rationale\":\"...\"}"}
  ],
  "rejected_messages": [
    {"role": "assistant", "content": "original model response"}
  ]
}
```

These fields are included so a converter can produce common preference-training
formats without re-parsing the rest of the annotation row. The canonical data is
still the richer top-level `prompt`, `game_state`, `valid_actions`, `chosen`,
and `rejected` objects.

## Notes

- Rows are JSONL: one complete JSON object per line.
- The file is append-only; saving the same step twice creates two rows.
- A row where `chosen.action_id` equals `rejected.action_id` may be useful for
  reasoning/rationale refinement, but it is not an action preference pair.
- The trace viewer only records human preference data. It does not label
  downstream outcome quality unless a later converter adds outcome fields.
