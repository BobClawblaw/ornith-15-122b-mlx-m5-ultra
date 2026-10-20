# Ornith-1.5-122B on one M5 Ultra

Serve [ornith-ai/Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B), a Qwen3.5-MoE 122B-A10B with a
vision tower, on one Apple M5 Ultra (256 GB): converted here to 4-bit or 8-bit MLX affine (groups of 64) with its MTP
drafter, served by [vllm-mlx](https://github.com/BobClawblaw/vllm-mlx) on mlx-vlm and mlx-lm, with continuous
batching, kept prompt states between turns, tool calls, `reasoning_content`, images, an OpenAI-compatible API, and a
launchd unit for the LAN.

- **Checkpoints** (`python3 -m mac convert`): 4-bit, 65 GB, 4.544 bits a weight (routers and shared-expert gates
  8-bit); 8-bit, 122 GB, 8.528 bits a weight. Each with its MTP layer beside it (`-mtp`, 1.3 and 2.5 GB).
- **Status:** being brought up (2026-10-20). Measurements and the checks table follow in this file once the server
  has been through them; [PLAN.md](PLAN.md) is the work log.

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
