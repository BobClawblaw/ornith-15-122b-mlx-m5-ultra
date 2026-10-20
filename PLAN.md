# Work log

Each entry says what was done, how it was measured, and what it found. Newest last.

- Scope (2026-10-20): serve [ornith-ai/Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B) on one Mac
  Studio M5 Ultra (256 GB), converted to 4-bit and 8-bit, and retire the DeepSeek-V4.1 server on this Mac once it
  serves. Ornith is a Qwen3.5-MoE 122B-A10B: 48 layers (DeltaNet linear attention, full attention every fourth), 256
  routed experts with 8 active and a gated shared expert, hidden 3,072, 32 query heads of 256 over 2 KV heads, a
  27-layer vision tower (1,152 wide, 16-pixel patches merged 2x2 into 3,072), a 248,320-token vocabulary, a 262,144
  window, one MTP layer. bf16 release: 250 GB in 62 shards, revision `346ec829`.
- Engine choice: TensorFold's native Metal `qwen3_5` family is Qwen3.5-2B dense (its shapes are constants, no routed
  experts, a tied head); the MoE variant exists only on its CUDA side. A native port means the MoE block (router,
  experts on the core indexed-expert kernels, the shared expert), the larger shapes, the untied head, the vision
  tower and the MTP drafter: days of engine work. mlx-vlm 0.7.9 has `qwen3_5_moe` with the tower and mlx-lm 0.31.3
  the text side, and vllm-mlx 0.6.0 serves both with continuous batching, a prefix cache, an MTP drafter, tool-call
  and reasoning parsers. The recipe serves with vllm-mlx first; the native family is the follow-on.
- Conversion (2026-10-20, `python3 -m mac convert`): mlx-vlm's converter, affine groups of 64, bf16 activations,
  `--mtp` writing the MTP layer beside the checkpoint. Quantizing the 250 GB release took 53 s (4-bit) and 29 s (8-bit)
  with the shards in the page cache, peak memory 37 GB. Results: 4-bit 65 GB at 4.544 bits a weight (the routers and
  shared-expert gates stay 8-bit: mlx-vlm's rule), drafter 1.3 GB; 8-bit 122 GB at 8.528 bits a weight, drafter 2.5 GB.
  The vision tower is quantized with the rest (mlx-vlm's default; a predicate could keep it bf16).
  - Pitfall: mlx-vlm copies the source's json files into the output, then `save_pretrained` rewrites them. From a
    Hugging Face cache snapshot the copies inherit the blobs' read-only mode and the rewrite fails with
    PermissionError after the weights are written, leaving a checkpoint without its `quantization` block that does not
    load. `convert` therefore stages the snapshot: shards symlinked, the small files copied writable.
- Memory: the 4-bit checkpoint and its drafter need about 67 GiB resident plus the KV cache; DeepSeek-V4.1 holds ~200
  GiB of the 256, so the two cannot serve together. Ornith is brought up with DeepSeek down, and DeepSeek's LaunchAgent
  is retired once Ornith serves.
- Serving (2026-10-20, 4-bit, vllm-mlx 0.6.0): the server loads in 12 s. Through the chat API (greedy, thinking off
  unless said): 17*23 right with thinking off and with thinking on (`reasoning_content` present); a `get_weather`
  tool call with the right argument; a `json_schema` reply that parses; PNG, JPEG and WebP data URLs answered by
  colour; text in an image read exactly; three circles counted; a 20,016-token turn in 6.3 s and the next turn in
  0.5 s from the prefix cache; streaming; two requests at once; 256-token replies at 63 tok/s end to end (prose and
  code alike). Plain decoding is deterministic (two identical greedy runs).
  - The MTP drafter through vllm-mlx's multimodal path fails on every round: `'Qwen3_5MoeDecoderLayer' object has no
    attribute 'layer_type'` (mlx_vlm/speculative/mtp.py keys shared KV states by it; mlx-vlm's Qwen3.5-MoE layer has
    `is_linear` only). With a two-line patch giving the layer that attribute (tools/patches/0001) the drafter runs at
    69-71% acceptance: code 72.5-79.8 tok/s, prose 55.5-58.2 (slower than plain). But its greedy replies differ from
    plain ones (the essay diverges at character 286 of 1,307, the code inside its docstring), so it is not an exact
    drafter as wired, and the recipe keeps `mtp = false`. The patch stays for the experiment, not applied by `venv`.
- Prefix cache (2026-10-20): a text conversation's second turn failed with a 500: `'ArraysCache' object has no attribute
  'update_window'`, then `update_recurrent`. vllm-mlx batches requests with mlx-lm's cache classes and hands each
  request's recurrent layers back as mlx-lm's ArraysCache (extracted from the batch cache, stored in the prefix cache,
  cloned on a hit), while mlx-vlm's DeltaNet layers call those two methods, which only mlx-vlm's ArraysCache has.
  Converting the class at the clone, the reconstruction or the extraction did not reach every path (three tries), so
  tools/sitecustomize.py gives mlx-lm's class the two methods (mlx-vlm's window and recurrent updates without the
  speculative history), installed by `python3 -m mac venv`. After it: a 20,016-token turn 5.8-6.3 s, the next turns
  0.16 s, two such conversations at once both resume in 0.27 s; all 14 API checks pass on the launchd server on 8888.
  An exact repeat of the 20k prompt is not served from the cache (5.8 s); a conversation with an image re-reads its
  prompt each turn (6 s a turn).
- Ruler (run 20261020-111724-mlx-speed, target ornith-15-122b-m5, before the shim; decode is unaffected by it): one
  stream 66.2/66.6/66.4/66.6 tok/s (prose/code/structured/list), two 113.4/113.2/113.2/114.0, four
  175.9/178.1/177.6/180.4; first token 0.08-0.11 s.
- Cut-over (2026-10-20 11:20): Ornith serves :8888 under launchd (`com.ornith.ornith-15-122b`); the DeepSeek
  LaunchAgent's plist was moved to ~/.cache/tensorfold-deepseek-v41/retired/. Hermes' config still names the DeepSeek
  model (an edit the session could not make); its base_url already points at :8888.
- The drafter, looked at again (2026-10-20): a block verify is exact against one-token decode on this model (47/47
  argmax, plain forward and mlx-vlm's batch-invariant verifier), rounds with no drafts through mlx-vlm's transaction
  are exact (0.00 logit difference), and the invariant path gives identical greedy tokens for no, wrong, right and
  mixed drafts (120 tokens, two prompts). But the invariant forward's logits sit about one logit from ordinary
  decode's at every position (two reduction orders), parting at token 59 (essay) and 35 (code). Served through
  vllm-mlx with MTP on, block sizes 1, 2 and 3 behave the same (acceptance 71%; prose 60.5-61.5 tok/s, code
  75-80; plain 63.5 and 63.7); a prompt's reply differs between its cold run, its first cached run and its second
  (the same pattern at every block size), because prefill and resumed states come from the ordinary forward and the
  verify from the invariant one. Kept off by default; patch 0001 is applied by `venv` so `mtp = true` works.

