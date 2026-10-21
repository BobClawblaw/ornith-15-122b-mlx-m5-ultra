# Ornith-1.5-122B on one M5 Ultra

Serve [ornith-ai/Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B), a Qwen3.5-MoE 122B-A10B with a
vision tower, on one Apple M5 Ultra (256 GB) with [TensorFold](https://github.com/ashhart/TensorFold)'s native Metal
engine: the checkpoint converted to 4-bit MLX affine (groups of 64, public at
[BobClawblaw/Ornith-1.5-122B-MLX-4bit](https://huggingface.co/BobClawblaw/Ornith-1.5-122B-MLX-4bit)), the
[incoai/Ornith-1.5-122B-DFlash2](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2) drafter, images, a prompt cache
kept between turns, up to four streams, tools and structured output, an OpenAI-compatible API, and a launchd unit for
the LAN. A 1M-token window (YaRN, one stream) is a setting away; the default stays the checkpoint's 262,144.

The engine is TensorFold's 27B family taught the routed block, from branch `ornith-main` of
[BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold) at the commit `local.example.toml` names;
`python3 -m mac build` builds it. [PLAN.md](PLAN.md) is the work log, with how each number was measured.

## Measured on this Mac

Mac Studio M5 Ultra (256 GB), TensorFold `ornith-main` at 30e5dcb4 (the pinned 9328aff8 adds YaRN, below; at 262,144 it
gives the same greedy replies on six prompts up to 20k tokens, and its ruler, run `20261021-022004`, matches), the 4-bit
checkpoint, four streams of 262,144 tokens, a 32 GiB prompt cache, images, the drafter (2026-10-20). The engine loads in
10 s and keeps 93 GiB resident (weights, drafter, tower) plus the streams' caches as they fill. The recipe harness's
ruler (256-token greedy replies, a distinct prompt a phase; target `ornith-15-122b-native-m5`), in tok/s summed over the
streams:

| Streams | prose | code | structured | list | first token |
|---|---|---|---|---|---|
| 1, drafts off (run `20261020-152513`) | 95.7 | 95.6 | 95.7 | 95.6 | 0.05-0.07 s |
| 2, drafts off | 152.7 | 154.7 | 154.9 | 156.5 | 0.08-0.10 s |
| 4, drafts off | 232.9 | 233.4 | 234.1 | 240.9 | 0.12-0.17 s |
| 1, drafts on (run `20261020-152333`) | 93.7 | 157.0 | 132.9 | 149.7 | 0.05-0.07 s |
| 2, drafts on | 191.5 | 274.2 | 224.2 | 315.2 | 0.08-0.11 s |
| 4, drafts on | 223.2 | 319.2 | 282.6 | 349.3 | 0.13-0.18 s |

DFlash2 drafts one stream (the others decode plainly); a drafted greedy reply is identical to the undrafted one. On
this routed model a 16-row verify window reads most experts' weights, so a drafted round costs about three plain
steps: drafts pay on code, lists and structured output and about break even on prose at one stream. A plain decode
step is 10 ms of GPU time (the shared expert runs as a ninth slot of the experts' kernel).

More streams (harness stream plan, 262,144-token streams and a 4 GiB prompt cache; drafted run `20261020-163412`,
plain run `20261020-163959`), tok/s summed over the streams:

| Streams | plain: prose / code / structured / list | drafted: prose / code / structured / list | first token |
|---|---|---|---|
| 1 | 95.3 / 95.5 / 95.4 / 95.5 | 93.8 / 157.1 / 132.6 / 149.5 | 0.05-0.07 s |
| 4 | 227.1 / 232.7 / 233.5 / 240.2 | 239.4 / 326.0 / 282.6 / 391.0 | 0.13-0.18 s |
| 8 | 315.0 / 322.5 / 327.2 / 338.6 | 231.8 / 357.8 / 302.7 / 385.1 | 0.22-0.31 s |
| 16 | 404.5 / 415.0 / 414.9 / 437.5 | 355.6 / 410.5 / 373.1 / 422.1 | 0.42-0.59 s |

Sixteen streams together give each stream the reply it gets alone (the suite's concurrent check, 12/12). Past four
streams drafting stops paying: the drafted stream's 16-row windows make every shared round heavier, so for many users
serve plain (`drafts = false`). Each 262,144-token stream takes 5.9 GiB, and the engine keeps the weights (77 GiB),
the streams and the prompt cache inside 70% of the Mac's memory: 11 streams fit beside the default 32 GiB prompt
cache, 15 beside 8 GiB, 16 beside 4 GiB (`PARALLEL=16 PROMPT_CACHE_GIB=4 DRAFTS=false python3 -m mac up`).

Prompts: a 19,235-token turn takes 8.2 s through the server (2,350 tok/s), its next turn 0.15 s from the cache; a
96,042-token prompt 59 s, a 245,370-token one 262 s.

Quality, thinking off (the recipe harness's suite, run `20261020-145307-dsv41-quality-off`, greedy):

| GSM8K | MATH-500 | MGSM | MMLU | IFEval (loose / strict) | tools | JSON |
|---|---|---|---|---|---|---|
| 245/250 (98.0%) | 438/500 (87.6%) | 220/240 (91.7%) | 294/342 (86.0%) | 120/150 / 116/150 | 55/60 | 30/30 |

The suite's serving checks pass too: drafted replies identical to undrafted (12/12), streaming identical, multi-turn
recall 8/8, stop strings, thinking (`reasoning_content`, no leaked tags), max-tokens.

Quality, thinking on (the harness's `thinking` plan, run `20261020-161950-thinking`: a 48k reasoning budget, 49,152
tokens of reasoning then 1,024 for the answer): GSM8K 244/250 (97.6%), none of the 250 replies cut at the 50,176-token
cap; the harness's reasoning check 10/10 right with no leaked tags. MATH-500 and GPQA with thinking on are not
measured.

| Feature | Check through the chat API |
|---|---|
| Text | greedy replies deterministic; thinking off by default, on with `chat_template_kwargs: {"enable_thinking": true}` (`reasoning_content`) |
| Tools, structured output | a `get_weather` call with its argument; `tool_choice: required`; a `json_schema` reply parses (libtfgrammar) |
| Images | PNG, JPEG, WebP (with alpha) by colour; text in an image read exactly; three circles counted; four images in order; a 3000x2000 JPEG at 4,080 tokens; GIF refused with a clear error |
| Prompt cache | a 19k-token conversation's later turns resume in 0.15 s; a conversation with an image resumes too |
| Long context | a code hidden at the start of 96k- and 245k-token prompts is recalled; at 1M (YaRN), five codes in a 1,027,846-token prompt |
| Drafts | drafted and undrafted greedy replies identical |
| Streaming, concurrency | streamed replies complete; four requests at once each right |

Exactness of the port: against stock mlx-vlm 0.7.9 on the same checkpoint, fed one token at a time, the engine's
logits agree to KL 0.0001 with the same top token (and a constant offset across the vocabulary, which softmax and
sampling ignore); its tensor-unit prompt pass is within KL 0.01-0.04 on a short prompt, where mlx-vlm's own one-pass
and one-token-at-a-time logits differ by KL 1.0-1.5.

**A correction (2026-10-20):** the first version of this recipe published numbers from an engine commit with a
routing bug. Ornith's config omits `norm_topk_prob`, which means renormalized top-k weights, and the port read it as
unset; a first quality run scored GSM8K 60/250. That version's exactness figures had been measured against an
mlx-vlm reference whose routing was rewritten with the same omission. The bug is fixed in 30e5dcb4; every number
above is measured on it. PLAN.md has the details.

## A 1M-token window (YaRN)

Ornith's model card validates static YaRN x4 over its 262,144-token window. Set `context` past the window
(`CONTEXT=1048576 PARALLEL=1 python3 -m mac up`) and the engine stretches the rotation itself: the factor is the window
multiple rounded up (x2 to 524,288, x4 to 1,048,576; only x4 is measured), as vLLM's YaRN builds it (attention factor
0.1 ln(factor) + 1, precise cos/sin on that path). The checkpoint is unchanged; a config with a `rope_scaling: yarn`
block works too. The default stays at 262,144 because YaRN changes every position, short prompts too, and costs a little
quality there.

Measured on 5ec491c8 (2026-10-20; 9328aff8 differs from it only in comments), one stream of 1,048,576 tokens, drafts,
images, a 32 GiB prompt cache, target `ornith-15-122b-yarn1m-native-m5`:
- **Memory:** one stream only; the engine holds 94 GiB once loaded, and two 1M streams pass the 70% cap.
- **Recall:** five codes at depths 2%, 25%, 50%, 75% and 97% of a 1,027,846-token prompt (Python source), asked one a
  turn, all five right. The first turn's prompt took 4,651 s (77.5 min, about 220 tok/s); each later turn resumed from
  the prompt cache in 1.9-2.1 s. Replies at that depth come at about 6.5 tok/s.
- **Speed at short prompts** (ruler, one stream): drafts off 95.3/95.3/95.1/95.2 tok/s (run `20261020-213756`), drafted
  92.4/193.7/123.8/130.0 (run `20261020-213454`), the same as without YaRN.
- **Exactness:** against stock mlx-vlm with its rotary set to the same YaRN table (fed one token at a time), the
  engine's YaRN logits agree as closely as its plain-rope logits do with YaRN off.

Quality with YaRN on, against the default (thinking off: run `20261020-214200-dsv41-quality-off`; thinking on: run
`20261020-222403-thinking`):

| | GSM8K | MATH-500 | MGSM | MMLU | IFEval loose | tools | JSON | GSM8K, thinking |
|---|---|---|---|---|---|---|---|---|
| 262,144, plain rope | 245 | 438 | 220 | 294 | 120 | 55 | 30 | 244 |
| 1,048,576, YaRN x4 | 245 | 428 | 215 | 294 | 116 | 56 | 30 | 247 |

The serving checks pass with YaRN on too (drafts 12/12, streaming, multi-turn, stop, thinking, max-tokens), and the
image checks give the same results (GIF refused). Turn it on for long documents; leave it off otherwise.

Known limits:
- **Thinking-mode quality** is measured on GSM8K only.
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
| `context` | 262144 | a stream's window; past 262,144 YaRN, up to 1,048,576 with `parallel = 1` (above) |
| `parallel` | 4 | streams decoded together (1 to 16; more streams need a smaller `prompt_cache_gib`, above) |
| `drafts`, `drafter_bits` | true, 4 | DFlash2 drafts for one stream, the drafter's projections in 4-bit |
| `thinking` | false | the template's thinking block by default |
| `max_tokens`, `thinking_budget` | 50176, 49152 | a reply's ceiling, and the reasoning a thinking reply may take before its answer |
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
