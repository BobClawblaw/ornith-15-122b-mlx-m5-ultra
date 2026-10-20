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
  tower and the MTP drafter: days of engine work. mlx-vlm 0.7.9 has `qwen3_5_moe` with the tower and mlx-lm 0.32.2
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
