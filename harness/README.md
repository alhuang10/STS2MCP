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

## Trace Viewer

The MCP harness now asks the model to choose from fully-instantiated legal
actions. Each step builds a `valid_actions` list where every item has an
`id`, MCP `tool`, concrete `args`, and a short summary. The model should return:

```json
{"action_id":"one_listed_id","rationale":"short public reason"}
```

The harness resolves `action_id` back to the attached MCP tool and args. This
avoids asking the model to invent indices, targets, or required tool arguments.

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
