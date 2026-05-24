# STS2 Model Harness

This is a small single-purpose harness for letting an OpenAI-compatible chat
model choose actions against the STS2_MCP localhost API.

## Start Qwen3.5-9B

From the repo root:

```bash
./harness/start_llama_server.sh
```

Defaults:

- model: `bartowski/Qwen_Qwen3.5-9B-GGUF:Q4_K_M`
- URL: `http://127.0.0.1:8080/v1`
- context: `8192`
- Metal offload: `-ngl 99`
- reasoning: off, to keep action JSON compact

Override examples:

```bash
MODEL_REPO='bartowski/Qwen_Qwen3.5-9B-GGUF:Q5_K_M' ./harness/start_llama_server.sh
LLAMA_CTX=12288 LLAMA_REASONING=auto ./harness/start_llama_server.sh
LLAMA_REASONING=on LLAMA_REASONING_FORMAT=deepseek ./harness/start_llama_server.sh
```

## Use A Runpod Model

The harness can also build a Runpod proxy URL from a pod/proxy id. For a proxy
host like:

```text
https://i6qkeo2utydx8c-64411136-8000.proxy.runpod.net
```

run the MCP harness with the id before the port:

```bash
.local/mcp-venv/bin/python harness/mcp_sts2_harness.py \
  --runpod-id i6qkeo2utydx8c-64411136 \
  --model Qwen3.6-27B \
  --steps 300 --sleep 0.05 --show-results
```

Equivalent environment-variable form:

```bash
RUNPOD_ID=i6qkeo2utydx8c-64411136 RUNPOD_MODEL=Qwen3.6-27B \
  .local/mcp-venv/bin/python harness/mcp_sts2_harness.py --steps 300
```

Useful remote flags:

- `--runpod-port 8000`: exposed Runpod proxy port.
- `--runpod-domain proxy.runpod.net`: proxy domain, if Runpod changes it.
- `--llm-url https://.../v1`: explicit OpenAI-compatible API base URL; this overrides Runpod URL generation.

## Run a Few Actions

With Slay the Spire 2 running and the mod responding on `localhost:15526`:

```bash
python3 harness/sts2_harness.py --steps 12
```

Useful flags:

- `--dry-run`: ask the model what it would do without sending game actions.
- `--sleep 1.5`: wait between actions so the game UI can settle.
- `--llm-url http://127.0.0.1:8080/v1`: point at a different OpenAI-compatible server.
- `--runpod-id i6qkeo2utydx8c-64411136`: build `https://<id>-8000.proxy.runpod.net/v1`.
- `--sts2-url http://localhost:15526`: point at a non-default STS2_MCP mod server.

The harness logs JSONL traces under `logs/`.

## Run Through MCP

The MCP-backed harness uses `mcp/server.py` for tool discovery and tool calls,
while keeping the same small model-driven game loop:

```bash
.local/mcp-venv/bin/python harness/mcp_sts2_harness.py --steps 300
```

It creates a subprocess for `mcp/server.py`, calls `get_game_state`, asks the
configured model for one action, executes that MCP tool, then repeats. Use
`--dry-run` to inspect the chosen action without changing the game.

## Eval Run Setup

The mod exposes an explicit eval-only setup action for deterministic runs:

```json
{"action":"eval_start_run","character":"IRONCLAD","seed":"CW967RN0QC"}
```

This bypasses profile-locked seed UI by applying the game's debug seed override
before embarking. It expects the game to already be on the standard
singleplayer character-select screen, so navigate there first:

```bash
curl -s -X POST http://localhost:15526/api/v1/singleplayer \
  -H 'Content-Type: application/json' \
  -d '{"action":"menu_select","option":"singleplayer"}'
curl -s -X POST http://localhost:15526/api/v1/singleplayer \
  -H 'Content-Type: application/json' \
  -d '{"action":"menu_select","option":"standard"}'
curl -s -X POST http://localhost:15526/api/v1/singleplayer \
  -H 'Content-Type: application/json' \
  -d '{"action":"eval_start_run","character":"IRONCLAD","seed":"CW967RN0QC"}'
```

`eval_start_run` is excluded from the model-visible legal action menu and is
intended only for harness-controlled benchmark setup.

## Fixed Seed Eval Harness

For a repeatable benchmark run, start STS2 at the main menu and run:

```bash
.local/mcp-venv/bin/python harness/mcp_sts2_harness.py \
  --seeded-eval \
  --runpod-id i6qkeo2utydx8c-64411136 \
  --model Qwen3.6-27B \
  --sleep 0.05
```

`--seeded-eval` currently hardcodes `IRONCLAD` on seed `CW967RN0QC`, launches
that run through `eval_start_run`, and defaults to `--steps 10000`. It writes
the normal JSONL trace plus a sibling `.summary.json` containing setup details,
the configured seed/character, `steps_taken`, the stop reason, and a compact
final-state summary.

After transition actions such as `map_choose_node`, the harness briefly polls
for a changed state so very low `--sleep` values do not read a stale map while
the game is already loading the next room.

## Trace Viewer

The MCP harness now asks the model to choose from fully-instantiated legal
actions. Each step builds a `valid_actions` list where every item has an
`id`, MCP `tool`, concrete `args`, and a short summary. The model should return:

```json
{"action_id":"one_listed_id"}
```

The harness resolves `action_id` back to the attached MCP tool and args. This
avoids asking the model to invent indices, targets, or required tool arguments.
If the model is run with explicit reasoning enabled, reasoning should live in
the model's reasoning trace rather than a separate JSON `rationale` field.

Prompt text for the MCP harness lives under `harness/prompts/` and is loaded by
version. The default is `v1` for both the system and user prompt templates.
Override both together:

```bash
.local/mcp-venv/bin/python harness/mcp_sts2_harness.py --prompt-version v1
```

Or override them independently:

```bash
STS2_SYSTEM_PROMPT_VERSION=v1 STS2_USER_PROMPT_VERSION=v1 \
  .local/mcp-venv/bin/python harness/mcp_sts2_harness.py
```

Trace records include prompt template versions and SHA-256 hashes so training
datasets and evaluation runs can be tied back to the exact prompt source.

The harness also logs a `model_trace` block for each step, including the exact
chat messages sent to the model, the valid actions shown, raw model output,
parsed action JSON, validation result, tool result, and token usage.

Start the viewer:

```bash
.local/mcp-venv/bin/python harness/trace_viewer_server.py --port 8765
```

Open `http://127.0.0.1:8765`.

If reasoning is enabled and the model emits `<think>...</think>` blocks or a
reasoning field in the chat response, the viewer shows that under the Reasoning
tab. Older logs can still be loaded, but they will not include prompt messages.

## SFT Dataset Conversion

Convert in-game annotation captures into true chat-format JSONL:

```bash
/workspace/qwen36-venv/bin/python harness/convert_annotations_to_sft.py \
  /workspace/annotations-20260523.jsonl \
  -o data/sts2_sft_poc_20260523.jsonl \
  --prompt-version v1
```

Each output row contains rendered `messages` plus metadata for the annotation
source, selected action, prompt versions, and prompt hashes. The assistant
message stores the human reasoning trace in `<think>...</think>` and keeps the
final action JSON compact.

## QLoRA Overfit POC

Run a small W&B-tracked overfit test against the local Qwen3.6-27B model:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_PROJECT=sts2mcp-qwen-lora \
  /workspace/qwen36-venv/bin/python harness/train_qwen_lora_poc.py \
    --dataset data/sts2_sft_poc_20260523.jsonl \
    --model-dir /workspace/models/Qwen3.6-27B \
    --output-dir /workspace/qwen36-sts2-lora-poc-20260523-4k \
    --max-seq-length 4096 \
    --too-long skip \
    --max-steps 30 \
    --learning-rate 5e-4 \
    --wandb-project sts2mcp-qwen-lora \
    --wandb-run-name qwen36-27b-sts2-overfit-poc-20260523-4k
```

Use `--dry-run-tokenize --no-wandb` first to inspect sequence lengths without
loading the model.
