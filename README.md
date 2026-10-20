# Ornith-1.5-122B on one M5 Ultra

Serve [ornith-ai/Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B), a Qwen3.5-MoE 122B-A10B with a
vision tower, on one Apple M5 Ultra (256 GB): converted here to 4-bit or 8-bit MLX affine (groups of 64) with its MTP
drafter, served by [vllm-mlx](https://github.com/BobClawblaw/vllm-mlx) on mlx-vlm and mlx-lm, with continuous
batching, kept prompt states between turns, tool calls, `reasoning_content`, images, an OpenAI-compatible API, and a
launchd unit for the LAN.

- **Checkpoints** (`python3 -m mac convert`): 4-bit, 65 GB, 4.544 bits a weight (routers and shared-expert gates
  8-bit); 8-bit, 122 GB, 8.528 bits a weight. Each with its MTP layer beside it (`-mtp`, 1.3 and 2.5 GB).
- **Served and measured** on one M5 Ultra (2026-10-20, the 4-bit checkpoint, drafts off): below. [PLAN.md](PLAN.md) is
  the work log.

## Measured on this Mac

Mac Studio M5 Ultra (256 GB), vllm-mlx 0.6.0 on mlx-vlm 0.7.9 and mlx-lm 0.31.3, the 4-bit checkpoint, greedy, thinking
off. The server loads in 12 s. The recipe harness's ruler (256-token replies, a distinct prompt a phase; run
`20261020-111724-mlx-speed`, target `ornith-15-122b-m5`), in tok/s summed over the streams:

| Streams | prose | code | structured | list | first token |
|---|---|---|---|---|---|
| 1 | 66.2 | 66.6 | 66.4 | 66.6 | 0.08-0.11 s |
| 2 | 113.4 | 113.2 | 113.2 | 114.0 | 0.14-0.19 s |
| 4 | 175.9 | 178.1 | 177.6 | 180.4 | 0.25-0.35 s |

Prompts: a 20,016-token turn takes 5.8-6.3 s (about 3,300 tok/s); the next turn of that conversation answers in 0.16 s
from the prefix cache, and two such conversations at once both resume (0.27 s).

| Feature | Check through the chat API |
|---|---|
| Text | 17*23 answered with thinking off and on (`reasoning_content` present); plain greedy decoding is deterministic |
| Tools | a `get_weather` call with the right argument (`--tool-call-parser qwen3.5`) |
| Structured output | a `json_schema` reply parses |
| Images | PNG, JPEG and WebP data URLs answered by colour; text in an image read exactly; three circles counted |
| Prompt cache | a 20k-token conversation's later turns resume in 0.16 s (with the shim below) |
| Streaming, concurrency | streamed chunks; two requests at once each right |

Known limits:
- **The MTP drafter is off by default** (`mtp = false`). The checkpoint's MTP layer works through mlx-vlm's drafter
  (patch `0001`, applied by `venv`, gives the model's layer the attribute it keys on) at 71% acceptance: code 75-80
  tok/s against 64 plain, prose 60-61 against 63. mlx-vlm's drafter is exact against its own batch-invariant forward
  (the same greedy tokens with no drafts, wrong, right or mixed ones, 120 tokens on two prompts), but that forward
  and ordinary decode are two bf16 reduction orders of the model that part after 35-59 tokens, and vllm-mlx mixes
  them: a prompt served cold and the same prompt resumed from the prefix cache then decode differently, so greedy
  replies with drafts on depend on the cache's history. Plain decode is deterministic. `mtp = true` is for code-heavy
  use that accepts that; `mtp_draft_tokens` has no measurable effect (the layer drafts its native block).
- **A conversation with an image re-reads its prompt each turn** (vllm-mlx keeps no prefix cache for multimodal
  requests): a 20k-token turn with an image takes 6 s and so does the next turn. Text-only conversations resume.
- **An exact repeat of a long prompt is not served from the cache** (vllm-mlx falls back to a full pass when it
  cannot rewind an exact hit); a continuation of it is.
- **The shim.** vllm-mlx batches requests with mlx-lm's cache classes, and mlx-vlm's Qwen3.5 DeltaNet layers call
  two methods on them that only mlx-vlm's class has, so a resumed turn failed with a 500. `tools/sitecustomize.py`
  (installed by `python3 -m mac venv`) gives mlx-lm's class those two methods.
- **Quality:** not measured yet; the harness quality suite has not been run on this server.

## Quick start

```bash
git clone https://github.com/BobClawblaw/ornith-15-122b-mlx-m5-ultra && cd ornith-15-122b-mlx-m5-ultra
python3 -m mac venv                                    # uv: the pinned serving stack into .venv
hf download ornith-ai/Ornith-1.5-122B --revision 346ec829dd63e107aad8771d74e885e901cec95d   # 250 GB
cp local.example.toml local.toml                       # paths and the port for this Mac
python3 -m mac convert --bits 4                        # ~/models/Ornith-1.5-122B-MLX-4bit (+ -mtp), about a minute
python3 -m mac up && python3 -m mac chat "Chirp?"      # the server in the background, a log under ~/.cache/ornith-15-122b
python3 -m mac launchd --port 8888                     # the launchd unit (not loaded) serving the LAN
```

`python3 -m mac` also has `down`, `status`, `logs`, `command` (the serve command it would run) and `validate`.

## Settings

`recipe.toml` holds the defaults, `local.toml` (git-ignored) this Mac's overrides, an upper-case environment variable
either (`PORT=8891 python3 -m mac up`).

| Setting | Default | What it does |
|---|---|---|
| `model_dir`, `bits` | the 4-bit checkpoint | which conversion to serve |
| `max_num_seqs` | 4 | requests decoded together |
| `max_request_tokens` | 262144 | the checkpoint's window |
| `prefix_cache` | true | kept prompt states between turns |
| `mtp`, `mtp_draft_tokens` | true, 3 | the MTP layer drafts |
| `kv_cache_bits` | 0 (bf16) | 8 or 4 quantizes the KV cache |
| `reasoning_parser` | qwen3 | thinking blocks come back as `reasoning_content` |

## Requirements

A Mac with 256 GB of unified memory for the 4-bit checkpoint (about 67 GiB resident with its drafter, plus the KV
cache); the 8-bit one needs about 125 GiB. Nothing else that holds the GPU's memory should run beside it.

## Layout

| Path | What it holds |
|---|---|
| `mac/` | `python3 -m mac`: settings, the venv, the conversion, the server, the launchd unit |
| `recipe.toml`, `local.example.toml`, `requirements.txt` | the defaults, the per-Mac example, the pinned stack |
| `PLAN.md` | the work log, with how each number was measured |
