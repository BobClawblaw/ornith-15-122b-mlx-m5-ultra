# Ornith-1.5-122B on one M5 Ultra

Serve Ornith-1.5-122B, a Qwen3.5-MoE 122B-A10B with a vision tower, on one Apple M5 Ultra (256 GB) with
[TensorFold](https://github.com/ashhart/TensorFold)'s native Metal engine. The weights are the publisher's own MLX
checkpoints, served exactly as published. By default that is
[ornith-ai/Ornith-1.5-122B-MLX-4bit](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX-4bit) at revision
`06a57aba4fba` (affine 4-bit in groups of 64, the router and the shared-expert gate at 8 bits, 70 GB). `weights = "8bit"`
serves [ornith-ai/Ornith-1.5-122B-MLX-8bit](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX-8bit) at revision
`5e8df2d2e504` instead (affine 8-bit in groups of 64, 122 GB): nearer the bf16 release, which does not fit one 256 GB
Mac, but about 40% slower to decode and with room for a 6 GiB prompt cache rather than 32 ([below](#the-8-bit-checkpoint)).
The recipe adds the [incoai/Ornith-1.5-122B-DFlash2](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2)
drafter, images, a prompt cache kept between turns, up to four streams, tools and structured output, an
OpenAI-compatible API and a launchd unit for the LAN. Each stream holds the checkpoint's 262,144-token window; longer
windows are not supported.

The engine is TensorFold [PR #1125](https://github.com/ashhart/TensorFold/pull/1125), the 27B family taught the routed
block, at the commit `local.example.toml` names (`a410eb56a`, branch `ornith-main` of
[BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold)); `python3 -m mac build` builds it.
[PLAN.md](PLAN.md) is the work log.

Faster variants (wider prompt chunks, our own conversion) live on the
[`optimizations`](https://github.com/BobClawblaw/ornith-15-122b-mlx-m5-ultra/tree/optimizations) branch. They are
not this recipe's default.

## Measured on this Mac

Mac Studio M5 Ultra (256 GB), engine a410eb56a, the publisher's 4-bit checkpoint, four streams of 262,144 tokens, a
32 GiB prompt cache, images and the drafter, 2026-10-29. The engine loads in 7 s; the server reports 92.9 GiB in use once
loaded (92.6 GiB kept wired), and the 32 GiB prompt cache fits under 70% of the Mac's memory.

The recipe harness's ruler (256-token greedy replies, a distinct prompt a phase; target `ornith-15-122b-native-m5`,
run `20261029-045414-mlx-speed`), drafts on, in tok/s summed over the streams:

| Streams | prose | code | structured | list |
|---|---|---|---|---|
| 1 | 93.2 | 155.9 | 132.6 | 149.2 |
| 2 | 192.5 | 265.6 | 220.5 | 298.3 |
| 4 | 230.2 | 315.0 | 280.3 | 349.3 |

DFlash2 drafts one stream; the others decode plainly, and a drafted greedy reply equals the undrafted one. On this
routed model a 16-row verify window reads most experts' weights, so drafts pay on code, lists and structured output
and about break even on prose. Plain decoding was not re-measured on this checkpoint.

Cold prompts through the server (harness `prefill`, run `20261029-045545-prefill`):

| Prompt | thinking off | thinking on |
|---|---|---|
| about 2,200 tokens | 2,433 tok/s (0.92 s) | 2,196 tok/s (1.01 s) |
| about 8,800 tokens | 2,477 tok/s (3.55 s) | 2,466 tok/s (3.55 s) |
| about 35,000 tokens | 2,152 tok/s (16.3 s) | 2,147 tok/s (16.3 s) |
| about 140,000 tokens | 1,328 tok/s (105.3 s) | 1,325 tok/s (105.5 s) |

Serving checks (harness `quick`, run `20261031-045144-quick`):

- **Pass:** seeded sampling equals its repeat and the undrafted reply (9/9). Structured output obeys its schema
  (18/18, drafted equal to plain 12/12). A resumed turn equals the same turn again and a cold one after a restart,
  with 5,624 of 5,643 tokens from the cache (first token 0.07 s against 2.6 s cold). Sixteen streams together match
  the same requests alone (12/12), and a 60-turn soak beside a decoding stream matches too (60/60). Drafts (12/12),
  stop strings, max-tokens, streaming, the thinking toggle, copying and multi-turn recall (8/8) pass. Image URLs:
  the recipe serves `--vision-urls`, so a public HTTPS image is fetched and answered, while loopback and plain-HTTP
  URLs are refused with HTTP 400 (set `vision_urls = false` to accept data URLs only).
- **Fail:** a named tool call with thinking on repeats its call (tools 4/5).

Images (harness `glm-vision`, run `20261029-050010-glm-vision`, the suite's vision checks with images sent as data
URLs): 4/4 right, covering two shapes with their colours and caption, text read
exactly, three triangles counted, and two images named in order. Sixteen image requests together give the same replies
as each alone (16/16).

Video is not supported. The checkpoint declares video input, but TensorFold's qwen27 family has no video path (only
its GLM and Flash Next families do), so the server answers `video_url` parts with HTTP 400 ("content parts must be text
or image_url"). Both of the suite's video cases fail that way.

The engine's own checks for PR #1125 (kernel contract, attention precision, chunk independence, a dense 27B A/B
against TensorFold main) are in the PR.

On the [TensorFold leaderboard](https://tensorfold.dev/benchmarks?result=5615f516-2770-4aa5-b71a-e29f25f70318) (the
board's community-v1 suite, the publisher's checkpoint at 06a57aba with the drafter; engine aa472b6d2 plus the suite
commit, 2026-10-26), in stream delivery tok/s: code 208.6 and chat 121.7 at temperature 1, code 202.2 and chat 96.2
greedy.

Quality, the recipe harness's suites on the same checkpoint and settings with engine aa472b6d2, the PR's previous head
(the same family code on an older TensorFold main; greedy tokens on the PR's chunk check are identical), greedy, one
run each:

| Check | thinking off (`20261026-103314-dsv41-quality-off`) |
|---|---|
| tools | 55/60 |
| json (schema asked in the prompt) | 30/30 |
| IFEval | 120/150 |
| GSM8K | 245/250 |
| MGSM | 220/240 |
| MMLU | 298/342 |
| MATH-500 | 438/500 |

Drafts (12/12), streaming, multi-turn, stop strings, the thinking toggle, copying and max-tokens all pass. With thinking
on and a 49,152-token reasoning budget (run `20261026-110423-thinking`), GSM8K scores 244/250, and the harness's ten
word problems are all right, each with its reasoning in `reasoning_content`, none cut at 50,176 tokens.

These quality runs predate the 8-bit option and the current engine pin; they were not repeated on a410eb56a.

The `optimizations` branch's own 4-bit conversion scored within a few items of these (tools 56, IFEval 114, GSM8K 241,
MGSM 212, MMLU 300, MATH-500 432; GSM8K 245 thinking on), on its own engine. At these sizes one standard error is about
five items on IFEval and six on MGSM, so the two are about even, with the publisher's weights slightly ahead. The PR
head 0f033773 with that conversion ran the same speed as these weights to within 1% at one stream (1% to 4% at four).

## The 8-bit checkpoint

`weights = "8bit"` (in `local.toml`, or `WEIGHTS=8bit`) serves the publisher's
[8-bit checkpoint](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX-8bit) at revision
`5e8df2d2e50478f221602f0997b8cbbec51f9c8a`: affine 8-bit in groups of 64 in every projection, the vision tower as
published. Its 29 large files match the sha256 sums Hugging Face lists at that revision. It switches the checkpoint's
repository, revision, folder and prompt cache together; `validate` refuses a folder whose `config.json` holds other
bits than the choice names.

Measured on the same Mac and engine (a410eb56a), four streams of 262,144 tokens, images and the drafter, 2026-10-29,
target `ornith-15-122b-8bit-native-m5`. The server reports 150.2 GiB in use once loaded (149.8 GiB kept wired), which
leaves 7.3 GiB under the engine's 70% cap: the recipe keeps a **6 GiB prompt cache** with this checkpoint (the server
refuses 32 GiB at start; fewer streams, or `prompt_cache_over_cap`, leave more).

Ruler (run `20261029-050046-mlx-speed`), drafts on, tok/s summed over the streams:

| Streams | prose | code | structured | list |
|---|---|---|---|---|
| 1 | 57.3 | 112.1 | 91.0 | 83.7 |
| 2 | 116.0 | 171.5 | 146.5 | 202.3 |
| 4 | 153.8 | 239.6 | 188.4 | 247.4 |

One stream decodes at about 60% of the 4-bit's speed on prose and 56% on lists. The suite's drafts check gives 86 tok/s
drafted against 57 undrafted on the same requests (1.5x; the 4-bit's, run `20261029-044908-quick`: 136 against 84, 1.6x).

Cold prompts (run `20261029-020057-prefill`):

| Prompt | thinking off | thinking on |
|---|---|---|
| about 2,200 tokens | 1,642 tok/s (1.35 s) | 1,640 tok/s (1.35 s) |
| about 8,800 tokens | 1,728 tok/s (5.09 s) | 1,726 tok/s (5.08 s) |
| about 35,000 tokens | 1,573 tok/s (22.2 s) | 1,560 tok/s (22.5 s) |
| about 140,000 tokens | 1,078 tok/s (129.7 s) | 1,051 tok/s (132.9 s) |

Serving checks (run `20261029-015221-quick`) all pass but image URLs, as with the 4-bit: tools 5/5 (the named call
with thinking on included), structured output 18/18 (drafted equal to plain 12/12), seeded sampling 9/9, a resumed turn
equal to a cold one after a restart (first token 0.09 s against 3.9 s cold), sixteen streams together equal to alone
(12/12), the 60-turn soak (60/60), drafts, streaming, stop strings, thinking, copying, max-tokens and multi-turn recall.
Images (run `20261029-020027-glm-vision`): 4/4, and sixteen requests together 16/16; video is refused as above.

Fidelity: the engine teacher-forced on MLX's greedy replies (mlx-lm 0.31.3, MLX 0.32.2, the same 8-bit files; six chat
prompts of 256 tokens each, the check PR #1125 uses for the 4-bit) picks MLX's top token at 1,495 of 1,536 positions
(97.33%). MLX's own prompt pass against its one-token steps agrees at 1,493 (97.20%). MLX's margin at the engine's 41
misses has median 0.25 and maximum 2.0 logits.

Quality: the harness's quality suites were **not run** on the 8-bit, so whether its answers beat the 4-bit's is not
measured here. It is the nearer of the two to the release by construction (8 bits in every projection where the 4-bit has 4, all but
the router and the shared-expert gate), at the speed and cache cost above.

## Quick start

```bash
git clone https://github.com/BobClawblaw/ornith-15-122b-mlx-m5-ultra && cd ornith-15-122b-mlx-m5-ultra
cp local.example.toml local.toml                       # this Mac's paths, the TensorFold checkout and commit
python3 -m mac build                                   # the engine, libtfgrammar and libjpeg-turbo under ~/.cache/ornith-15-122b
python3 -m mac download                                # the publisher's checkpoint (70 GB; 122 GB with weights = "8bit") and the drafter
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
| `weights` | `4bit` | the publisher's checkpoint to serve: `4bit` or `8bit` (each `[weights.*]` table pins its repository and revision) |
| `model_dir` | `~/models/Ornith-1.5-122B-MLX-4bit` (`-8bit`) | where `download` puts that checkpoint; set it under `[weights.4bit]` or `[weights.8bit]` in `local.toml` |
| `tensorfold_src`, `tensorfold_sha`, `zig` | local.toml | the TensorFold checkout, the pinned commit, Zig 0.17.0 |
| `context` | 262144 | a stream's window, at most the checkpoint's 262,144 |
| `parallel` | 4 | streams decoded together (1 to 16; more streams need a smaller `prompt_cache_gib`) |
| `drafts`, `drafter_bits` | true, 4 | DFlash2 drafts for one stream, the drafter's projections in 4-bit |
| `thinking` | false | the template's thinking block by default |
| `max_tokens`, `thinking_budget` | 50176, 49152 | a reply's ceiling, and the reasoning a thinking reply may take before its answer |
| `prompt_cache_gib` | 32 (8-bit: 6) | kept prompt states between turns |
| `vision`, `vision_image_tokens`, `vision_max_images` | true, 4096, 4 | the image tower, an image's tokens at most, images a request |
| `temperature` | the server's (0.6, top_k 20, top_p 0.95) | default sampling |

## Requirements

A Mac with 256 GB of unified memory. With four streams of 262,144 tokens the 4-bit server reports 92.9 GiB in use once
loaded and 19.8 GiB reserved outside that reading, and the prompt cache takes its 32 GiB under the 70% cap; the 8-bit
reports 150.2 GiB and keeps 6 GiB of prompt cache. Nothing
else that holds the GPU's memory should run beside it.

## Layout

| Path | What it holds |
|---|---|
| `mac/` | `python3 -m mac`: settings, the download, the engine build, the server, the launchd unit |
| `recipe.toml`, `local.example.toml` | the defaults, the per-Mac example |
| `PLAN.md` | the work log |
