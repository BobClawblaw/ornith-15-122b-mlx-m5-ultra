# Ornith-1.5-122B on one M5 Ultra

Serve Ornith-1.5-122B, a Qwen3.5-MoE 122B-A10B with a vision tower, on one Apple M5 Ultra (256 GB) with
[TensorFold](https://github.com/ashhart/TensorFold)'s native Metal engine. The weights are the publisher's own 4-bit MLX
checkpoint, [ornith-ai/Ornith-1.5-122B-MLX-4bit](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX-4bit) at
revision `06a57aba4fba`, served exactly as published: affine 4-bit in groups of 64, the router and the shared-expert
gate at 8 bits. The recipe adds the [incoai/Ornith-1.5-122B-DFlash2](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2)
drafter, images, a prompt cache kept between turns, up to four streams, tools and structured output, an
OpenAI-compatible API and a launchd unit for the LAN. Each stream holds the checkpoint's 262,144-token window; longer
windows are not supported.

The engine is TensorFold [PR #1125](https://github.com/ashhart/TensorFold/pull/1125), the 27B family taught the routed
block, at the commit `local.example.toml` names (branch `ornith-pr-v2` of
[BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold)); `python3 -m mac build` builds it.
[PLAN.md](PLAN.md) is the work log.

Faster variants (wider prompt chunks, our own conversion) live on the
[`optimizations`](https://github.com/BobClawblaw/ornith-15-122b-mlx-m5-ultra/tree/optimizations) branch. They are
not this recipe's default.

## Measured on this Mac

Mac Studio M5 Ultra (256 GB), engine aa472b6d2, the publisher's checkpoint, four streams of 262,144 tokens, a 32 GiB
prompt cache, images and the drafter, 2026-10-26. The engine loads in 9 s; the server reports 93.3 GiB in use once loaded
(92.6 GiB kept wired), and the 32 GiB prompt cache fits under 70% of the Mac's memory.

The recipe harness's ruler (256-token greedy replies, a distinct prompt a phase; target `ornith-15-122b-native-m5`,
run `20261026-084207-mlx-speed`), drafts on, in tok/s summed over the streams:

| Streams | prose | code | structured | list |
|---|---|---|---|---|
| 1 | 94.1 | 157.7 | 132.9 | 149.8 |
| 2 | 191.6 | 263.2 | 230.7 | 317.3 |
| 4 | 232.4 | 314.0 | 290.5 | 349.5 |

DFlash2 drafts one stream; the others decode plainly, and a drafted greedy reply equals the undrafted one. On this
routed model a 16-row verify window reads most experts' weights, so drafts pay on code, lists and structured output
and about break even on prose. Plain decoding was not re-measured on this checkpoint.

Cold prompts through the server (harness `prefill`, run `20261026-084338-prefill`):

| Prompt | thinking off | thinking on |
|---|---|---|
| about 2,200 tokens | 2,499 tok/s (0.89 s) | 2,158 tok/s (1.02 s) |
| about 8,800 tokens | 2,482 tok/s (3.55 s) | 2,462 tok/s (3.56 s) |
| about 35,000 tokens | 2,159 tok/s (16.2 s) | 2,132 tok/s (16.4 s) |
| about 140,000 tokens | 1,334 tok/s (104.8 s) | 1,333 tok/s (104.8 s) |

Serving checks (harness `quick`, run `20261026-083724-quick`):

- **Pass:** seeded sampling equals its repeat and the undrafted reply (9/9). Structured output obeys its schema
  (18/18, drafted equal to plain 12/12). A resumed turn equals a cold one, with 5,624 of 5,643 tokens from the cache.
  Sixteen streams together match the same requests alone (12/12), and a 60-turn soak beside a decoding stream
  matches too (60/60). Stop strings, max-tokens, streaming and multi-turn recall (8/8) pass.
- **Fail:** a named tool call with thinking on repeats its call. Image URLs are refused: the recipe serves `--vision`
  without `--vision-urls`, so send images as data URLs.

The engine's own checks for PR #1125 (kernel contract, attention precision, chunk independence, a dense 27B A/B
against TensorFold main) are in the PR.

Quality: the harness quality suites have not been run on the publisher's checkpoint yet. The figures published
earlier were measured on a different 4-bit conversion; the `optimizations` branch keeps them.

## Quick start

```bash
git clone https://github.com/BobClawblaw/ornith-15-122b-mlx-m5-ultra && cd ornith-15-122b-mlx-m5-ultra
cp local.example.toml local.toml                       # this Mac's paths, the TensorFold checkout and commit
python3 -m mac build                                   # the engine, libtfgrammar and libjpeg-turbo under ~/.cache/ornith-15-122b
python3 -m mac download                                # the publisher's checkpoint (70 GB) and the drafter, both pinned
python3 -m mac up && python3 -m mac chat "Chirp?"      # the server in the background, a log under ~/.cache/ornith-15-122b
python3 -m mac launchd --port 8888                     # the launchd unit (not loaded) serving the LAN
```

`python3 -m mac` also has `down`, `status`, `logs`, `command` (the serve command it would run) and `validate`. The
engine needs Zig 0.17.0, cmake (libjpeg-turbo) and git; `download` needs the `hf` CLI.

## Settings

`recipe.toml` holds the defaults, `local.toml` (git-ignored) this Mac's overrides, an upper-case environment variable
either (`PORT=8891 python3 -m mac up`).

| Setting | Default | What it does |
|---|---|---|
| `model_dir` | `~/models/Ornith-1.5-122B-MLX-4bit` | where `download` puts the publisher's checkpoint |
| `tensorfold_src`, `tensorfold_sha`, `zig` | local.toml | the TensorFold checkout, the pinned commit, Zig 0.17.0 |
| `context` | 262144 | a stream's window, at most the checkpoint's 262,144 |
| `parallel` | 4 | streams decoded together (1 to 16; more streams need a smaller `prompt_cache_gib`) |
| `drafts`, `drafter_bits` | true, 4 | DFlash2 drafts for one stream, the drafter's projections in 4-bit |
| `thinking` | false | the template's thinking block by default |
| `max_tokens`, `thinking_budget` | 50176, 49152 | a reply's ceiling, and the reasoning a thinking reply may take before its answer |
| `prompt_cache_gib` | 32 | kept prompt states between turns |
| `vision`, `vision_image_tokens`, `vision_max_images` | true, 4096, 4 | the image tower, an image's tokens at most, images a request |
| `temperature` | the server's (0.6, top_k 20, top_p 0.95) | default sampling |

## Requirements

A Mac with 256 GB of unified memory. With four streams of 262,144 tokens the server reports 93.3 GiB in use once
loaded and 19.8 GiB reserved outside that reading, and the prompt cache takes its 32 GiB under the 70% cap. Nothing
else that holds the GPU's memory should run beside it.

## Layout

| Path | What it holds |
|---|---|
| `mac/` | `python3 -m mac`: settings, the download, the engine build, the server, the launchd unit |
| `recipe.toml`, `local.example.toml` | the defaults, the per-Mac example |
| `PLAN.md` | the work log |
