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

- Native port (2026-10-20, afternoon): the vllm-mlx server was a stopgap; the recipe now serves TensorFold's native
  Metal engine. The 27B family (`zig/src/families/qwen27`) learned `qwen3_5_moe`: the config reads the experts (the
  shared expert's width is the layer's `intermediate`, `norm_topk_prob` false for Ornith), `widths.zig` admits the
  router, shared gate and stacked experts' widths, the loader borrows the stacked `switch_mlp` tensors where the
  shards pack them and copies router + shared gate into one 264-row matrix, a routed block (`moe_gpu.zig`,
  `kernels/metal/qwen27/moe.metal`) routes by fp32 softmax over bf16 logits with ties to the lower id (HF's semantics),
  runs decode rows through the row kernels by expert id and prompt chunks sorted by expert on the tensor units
  (core.affine_mm), and combines with the reference's bf16 roundings; the attention kernels take the KV group as a
  compile-time define (24 over 4 for the 27B, 32 over 2 here); the DFlash2 runtime takes eight taps and a block of 16.
- Checks of the port, prompt pass (`tf-qwen27-run`, `TF_LAYERS=N` truncates the stack, `TF_LOGITS_DUMP` writes logits)
  against mlx-vlm 0.7.9 on the same 4-bit checkpoint with its routing done HF's way (fp32 softmax and top-k; mlx-vlm
  itself rounds the probabilities to bf16 before top-k, which ties experts and changes answers: 17*23 is 391 under
  mlx-vlm's ties and 47 under HF's routing on this 4-bit checkpoint, the native engine agreeing with HF): KL 0.0000 at
  1 layer, 0.0001 at 4, 0.0002 at 16, 0.0006 at 32, 0.0023 at 48 (sky prompt), 0.0035 (arithmetic prompt), the same
  top token at every depth. Decode (one-token steps, teacher-forced 64 and 96 tokens): KL mean 0.0014 and 0.0006,
  top-1 agreement 98.4% and 100%. mlx-vlm's own logits move by KL 1.0-1.5 at 48 layers between its accumulation
  orders (one pass, two chunks, a token a step), so the checkpoint itself is that sensitive at high-entropy spots.
- Served (`tensorfold-native serve`, 4 streams, 32 GiB prompt cache, drafter, images): greedy deterministic;
  streaming; tools; json_schema and required tools (libtfgrammar beside the binary); a 19,235-token prompt in 10.9 s
  then its next turn in 0.10 s from the cache (19,274 cached) and an exact repeat in 0.5 s; four requests at once;
  images 16/17 (PNG, JPEG, WebP, alpha, text read, counting, four in order, a 3000x2000 JPEG at 4,080 tokens, a
  second turn after an image from the cache, thinking on, streaming, two at once, an image after 20k tokens; GIF
  refused by design). DFlash2 (incoai/Ornith-1.5-122B-DFlash2, `--drafter-bits 4`): drafted greedy replies identical
  to undrafted on prose, code, a list and a story (400 tokens each); 2.8-4.4 tokens a round.
- Speed (harness target `ornith-15-122b-native-m5`, plan mlx-speed): undrafted run 20261020-124945, one stream
  85.0/85.1/85.3/85.1 tok/s (prose/code/structured/list), two 143.4/144.7/143.3/142.5, four 219.4/221.4/237.1/225.0;
  drafted run 20261020-125604, one 40.4/79.3/71.4/71.8 (the first phase after the load; the manual checks right
  after gave prose 101.6, code 112.7, list 157.0), two 152.1/281.8/164.6/250.7, four 197.3/408.0/322.4/331.1.
  Prompt (`tf-qwen27-run --max-tokens 0`, 8,192 tokens): 1,773 tok/s with 512-row chunks before the router moved to
  the tensor units, then 2,230 / 2,576 / 2,762 tok/s at 512 / 1,024 / 2,048 rows; the family's default for routed
  checkpoints is 1,024 (a chunk reads every expert's weights once). A decode step is 12.2 ms of GPU time: dense
  projections 5.7, experts 3.0, route 1.0, norms 0.6, attention 0.5, DeltaNet 1.0.
- Recipe: `mac/` builds the pinned engine (`python3 -m mac build`: a worktree at tensorfold_sha, libtfgrammar from
  xgrammar v0.2.8, libjpeg-turbo 3.1.4.1), serves it (`up`, launchd), converts with mlx-vlm (`venv`, `convert`,
  without the MTP layer: the drafter is DFlash2); the vllm-mlx shim and patch are gone. The 4-bit checkpoint is public
  at BobClawblaw/Ornith-1.5-122B-MLX-4bit.
- Final measurements on the pinned engine (32dad79b; the drafting policy now lets plain rounds compete for routed
  models, and the router matrix is padded to 64-row tiles): drafted run 20261020-133658, one stream
  42.6/84.1/70.5/71.4 tok/s (prose/code/structured/list), two 155.1/252.2/165.3/246.5, four 194.5/363.7/329.6/322.7;
  plain run 20261020-133912, one 85.0/84.8/84.6/84.9, two 139.2/143.6/146.2/141.7, four 218.7/223.6/238.1/227.3.
  Why drafted prose loses at one stream: a 16-row window on this MoE reads most experts' weights (about 100 of 256),
  so the host times it at 28.7 ms against 12.5 ms for one row, a drafted round takes 34-38 ms, and the ruler's prose
  prompt lands 1.3-1.8 drafts a round (its code prompt 5.7). Other prose prompts did better (66 tok/s, 2.3 a round).
  `zig build test`, test-golden, fuzz-cortheon, test-kernel-contract and the lean check (zero new findings) pass on
  that commit; the full-depth logits check is unchanged (KL 0.0017 and 0.0035).
