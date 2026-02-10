# Developer Notes

## What This Project Does

Tests how much Cambrian-8B VLM relies on images vs language priors. Runs 6 benchmarks under 4 image conditions (normal, no image, wrong image, noise) and compares accuracy. If accuracy barely drops without images, the model is using language shortcuts.

## Architecture Decisions

**Why custom eval scripts (not lmms-eval)?** Cambrian has a non-standard 4-encoder vision tower. Image interception is trivial in our own code (`condition.apply()` before `process_images()`), but would require writing custom model wrappers + per-task `doc_to_visual` functions in lmms-eval.

**Why no quantization?** FP16 matches published eval conditions. INT8 failed on Pascal (compute 6.1 — `named symbol not found` in bitsandbytes). INT4 only gives ~15-25% speedup for short-answer generation.

**Why single GPU only?** Multi-GPU tensor parallel is fundamentally incompatible with Cambrian. The SVA re-aggregation at decoder layers 0,3,6,...,27 does cross-device tensor operations that fail with accelerate's device_map splitting.

## Cambrian Gotchas

- `process_images()` returns a **list of 4 tensors** (one per encoder: SigLIP, CLIP, DINOv2, ConvNeXt), not a single tensor
- `model.generate()` returns **only new tokens** (not input+generated) — don't slice off input_ids length
- Builder.py line 33: `if device != "cuda"` overrides device_map to single device — always pass `device="cuda"` for multi-GPU (moot now, but FYI)
- MMMU questions contain `<image 1>` placeholders that must be stripped before adding Cambrian's `<im_start><image><im_end>` tokens
- Vision encoders wrap forward pass in `torch.set_grad_enabled(self.unfreeze_mm_vision_tower)` — set flag to True to enable gradient flow for optimization

## Dataset Quirks

- **GQA**: images and questions are in separate HF configs (`testdev_balanced_images` and `testdev_balanced_instructions`), joined by `imageId`
- **MMMU**: images stored as `image_1`, `image_2`, ..., `image_7` (not `image`). Validation split spans many subject configs that must be concatenated.
- **MMMU**: options field is sometimes a string repr of a list — needs `eval()`

## Key Files

```
src/model/loader.py          # load_cambrian() — model loading
src/model/inference.py        # run_inference() — single-sample generation
src/benchmarks/base.py        # BenchmarkSample dataclass
src/image_conditions/base.py  # ImageCondition ABC
src/evaluation/runner.py      # Main eval loop with checkpoint/resume
src/optimization/             # Noise embedding optimization (experimental)
scripts/run_machine.py        # Runs all 24 benchmark x condition jobs
scripts/start.sh              # Background launcher (tmux/screen/nohup)
```

## Noise Optimization (Experimental)

Optimizes 4 encoder-output tensors to maximize P(correct answer) via teacher forcing. Monkey-patches `model.encode_images()` to inject learnable tensors, uses `model.forward()` with labels for differentiable loss. If this works well, next step is pixel-space optimization (requires differentiable reimplementation of `process_images()` and enabling vision encoder gradients).
