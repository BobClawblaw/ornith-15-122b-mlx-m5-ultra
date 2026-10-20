## Re-Running Benchmarks due to a bug: Oct.20.2026 @ 3:07pm CDT

# Ornith-1.5-122B on one M5 Ultra

Serve [ornith-ai/Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B), a Qwen3.5-MoE 122B-A10B with a
vision tower, on one Apple M5 Ultra (256 GB) with [TensorFold](https://github.com/ashhart/TensorFold)'s native Metal
engine: the checkpoint converted to 4-bit MLX affine (groups of 64, public at
[BobClawblaw/Ornith-1.5-122B-MLX-4bit](https://huggingface.co/BobClawblaw/Ornith-1.5-122B-MLX-4bit)), the
[incoai/Ornith-1.5-122B-DFlash2](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2) drafter, images, a prompt
cache kept between turns, up to four streams, tools and structured output, an OpenAI-compatible API, and a launchd
unit for the LAN.

The engine is TensorFold's 27B family taught the routed block, from branch `ornith-main` of
[BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold) at the commit `local.example.toml` names;
`python3 -m mac build` builds it. [PLAN.md](PLAN.md) is the work log, with how each number was measured.

> **Correction (2026-10-20):** the numbers below are withdrawn. The engine commit they were measured on had a routing
> bug: Ornith's config omits `norm_topk_prob`, which means renormalized top-k weights, and the Metal port read it as
> unset. Replies were measurably wrong (a first quality run scored GSM8K 60/250). The "exactness" figures were measured
> against an mlx-vlm reference whose routing had been rewritten with the same omission, so they did not show it. The
> fix is TensorFold commit 30e5dcb4 (branch `ornith-main`); against stock mlx-vlm it matches to KL 0.0001 one token at a
> time. Quality and speed are being re-measured on it, and this README will be replaced with those results. Until then,
> do not rely on the tables below or on builds of earlier commits.

## Measured on this Mac (withdrawn, see the correction above)

Mac Studio M5 Ultra (256 GB), the 4-bit checkpoint, four streams of 262,144 tokens, a 32 GiB prompt cache, images,
the drafter (2026-10-20). The engine loads in 10 s and keeps 93 GiB resident (weights, drafter, tower) plus the
streams' caches as they fill. The recipe harness's ruler (256-token greedy replies, a distinct prompt a phase; target
`ornith-15-122b-native-m5`), in tok/s summed over the streams:

| Streams | prose | code | structured | list | first token |
|---|---|---|---|---|---|
| 1, drafts off (run `20261020-141648`) | 95.6 | 95.4 | 95.1 | 95.5 | 0.05-0.07 s |
| 2, drafts off | 152.9 | 159.9 | 167.7 | 156.4 | 0.07-0.10 s |
| 4, drafts off | 234.0 | 240.1 | 253.3 | 241.6 | 0.12-0.17 s |
| 1, drafts on (run `20261020-141514`) | 89.7 | 314.3 | 151.1 | 177.6 | 0.05-0.07 s |
| 2, drafts on | 169.9 | 285.0 | 246.4 | 282.2 | 0.08-0.11 s |
| 4, drafts on | 204.5 | 357.4 | 361.4 | 389.7 | 0.13-0.17 s |

DFlash2 drafts one stream (the others decode plainly); a drafted greedy reply is identical to the undrafted one. The
drafter lands 4.3 tokens a round on prose and 9.7 on code (a 300-word story 115 tok/s, a CSV parser 274, a list 201,
a JSON array 248; PLAN.md). On this routed model a 16-row verify window reads most experts' weights, so a round costs
about three plain steps: prose about breaks even at one stream, code runs 3.3x plain. A decode step is 10 ms of GPU
time: the shared expert runs as a ninth slot of the experts' kernel beside the eight routed ones.

Prompts: 2,576 tok/s at the engine's 1,024-row chunks (8,192 tokens in 3.2 s; `tf-qwen27-run`, PLAN.md); through the server a
19,235-token turn took 10.9 s, its next turn 0.10 s from the cache, an exact repeat 0.5 s.

| Feature | Check through the chat API |
|---|---|
| Text | greedy replies deterministic; thinking off by default, on with `chat_template_kwargs: {"enable_thinking": true}` (`reasoning_content`) |
| Tools, structured output | a `get_weather` call with its argument; `tool_choice: required`; a `json_schema` reply parses (libtfgrammar) |
| Images | PNG, JPEG, WebP (with alpha) by colour; text in an image read exactly; three circles counted; four images in order; a 3000x2000 JPEG at 4,080 tokens; GIF refused with a clear error |
| Prompt cache | a 19k-token conversation's later turns resume in 0.10 s; a conversation with an image resumes too |
| Long context | a code hidden at the start of 90k- and 230k-token prompts is recalled (PLAN.md) |
| Drafts | drafted and undrafted greedy replies identical on prose, code, a list and a story |
| Streaming, concurrency | streamed chunks; four requests at once each right |

Exactness of the port: against mlx-vlm 0.7.9 on the same checkpoint with HF-style routing, the prompt pass's logits
agree to KL 0.0023 at all 48 layers (the same top token at every depth) and teacher-forced decode to KL 0.0006-0.0014
with 98-100% top-1 agreement. mlx-vlm rounds its router probabilities to bf16 before top-k, which ties experts and
changes some answers (17*23 among them); the engine routes as the reference implementation does, in fp32.

Known limits:
- **Quality:** the harness quality suite has not been run on this server.
- **Drafts never propose rare tokens:** the drafter's head keeps the 27B's vocabulary spans (ids 0-98,303 and
  248,032-248,319), so CJK, Cyrillic and other rare tokens are only verified, never drafted; replies stay exact.
- **One stream drafts** at a time; the others decode without drafts.

## Quick start

```bash
git clone https://github.com/BobClawblaw/ornith-15-122b-mlx-m5-ultra && cd ornith-15-122b-mlx-m5-ultra
cp local.example.toml local.toml                       # this Mac's paths, the TensorFold checkout and commit
python3 -m mac build                                   # the engine, libtfgrammar and libjpeg-turbo under ~/.cache/ornith-15-122b
hf download ornith-ai/Ornith-1.5-122B --revision 346ec829dd63e107aad8771d74e885e901cec95d   # 250 GB
python3 -m mac venv && python3 -m mac convert          # ~/models/Ornith-1.5-122B-MLX-4bit (mlx-vlm, about a minute)
hf download incoai/Ornith-1.5-122B-DFlash2 --revision a5ed7396f5f9e5b1abf7ff50b7899a3186961432
python3 -m mac up && python3 -m mac chat "Chirp?"      # the server in the background, a log under ~/.cache/ornith-15-122b
python3 -m mac launchd --port 8888                     # the launchd unit (not loaded) serving the LAN
```

The converted checkpoint can also be downloaded: `hf download BobClawblaw/Ornith-1.5-122B-MLX-4bit --local-dir
~/models/Ornith-1.5-122B-MLX-4bit`. `python3 -m mac` also has `down`, `status`, `logs`, `command` (the serve command
it would run) and `validate`. The engine needs Zig 0.17.0, cmake (libjpeg-turbo) and git.

## Settings

`recipe.toml` holds the defaults, `local.toml` (git-ignored) this Mac's overrides, an upper-case environment variable
either (`PORT=8891 python3 -m mac up`).

| Setting | Default | What it does |
|---|---|---|
| `model_dir`, `bits` | the 4-bit checkpoint | which conversion to serve (`convert --bits 8` writes the 8-bit one) |
| `tensorfold_src`, `tensorfold_sha`, `zig` | local.toml | the TensorFold checkout, the pinned commit, Zig 0.17.0 |
| `context` | 262144 | a stream's window |
| `parallel` | 4 | streams decoded together |
| `drafts`, `drafter_bits` | true, 4 | DFlash2 drafts for one stream, the drafter's projections in 4-bit |
| `thinking` | false | the template's thinking block by default |
| `prompt_cache_gib` | 32 | kept prompt states between turns |
| `vision`, `vision_image_tokens`, `vision_max_images` | true, 4096, 4 | the image tower, an image's tokens at most, images a request |
| `temperature` | the server's (0.6, top_k 20, top_p 0.95) | default sampling |

## Requirements

A Mac with 256 GB of unified memory: the 4-bit checkpoint with its drafter and tower keeps 93 GiB resident, four
streams of 262,144 tokens take up to 20 GiB more, and the prompt cache its 32 GiB. Nothing else that holds the GPU's
memory should run beside it.

## Layout

| Path | What it holds |
|---|---|
| `mac/` | `python3 -m mac`: settings, the engine build, the conversion, the server, the launchd unit |
| `recipe.toml`, `local.example.toml`, `requirements.txt` | the defaults, the per-Mac example, the conversion stack |
| `PLAN.md` | the work log, with how each number was measured |
