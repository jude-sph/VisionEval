# Noise Optimization Ideas

Ideas for crafting optimized inputs that maximize VLM benchmark accuracy without real images.

## Currently Implemented

### Per-Question Embedding Optimization
- Optimize 4 encoder-output tensors (one per vision encoder) to minimize teacher-forcing loss
- Fresh random initialization per question, 50 steps of Adam
- Bypasses vision encoders entirely via monkey-patching `encode_images()`
- Answers: "given unlimited freedom in embedding space, can we make the model answer correctly?"

### Hardware Setup
- **Noise optimization**: 4x Titan X Pascal (12GB each, 48GB total) on Machine A
  - `python scripts/optimize_noise.py --gpu_ids 0,1,2,3 --benchmark mmmu`
  - Flash Attention auto-disabled (Pascal compute 6.1 < 8.0 required)
  - Model split across 4 GPUs via `device_map="auto"` (~4.25GB weights per GPU)
  - 48GB total = plenty of headroom for backprop activations, no memory workarounds needed
- **Benchmark evals**: 1x RTX 3090 (24GB) on Machine B — inference only, no backprop
- These run on separate machines and can execute in parallel

### Memory Notes (for future single-GPU attempts)
- On 3090 (24GB) the optimization OOMs by ~112MB during backward pass
- Three workarounds are available if needed:
  1. **Detach ConvNeXt features**: ConvNeXt outputs 9216 tokens (16x more than other encoders at 576). Its SVA attention creates massive backward intermediates. Freezing its feature tensor (no gradients) saves ~1-2GB. The other 3 encoders still get optimized. Code change: in `_init_features`, call `.detach()` on the ConvNeXt tensor (index 3).
  2. **Vision encoder CPU offload**: Already implemented (conditional on single-GPU). Saves ~3.8GB by moving unused encoders to CPU.
  3. **Gradient checkpointing**: Already implemented. Saves ~10GB on LLM decoder activations.

## Future Ideas

### Universal Embedding Optimization
- Find ONE set of 4 embedding tensors that works across ALL questions in a benchmark
- Train on a subset (e.g., 200 questions), evaluate on held-out questions
- Accumulate gradients across all training questions before each optimizer step
- Much harder than per-question, but a stronger result if it works
- The finding: "this single abstract input achieves X% on a college exam"

### Pixel-Space Optimization
- Optimize actual pixel values `[3, 384, 384]` instead of abstract embeddings
- Produces a real image you can display — the visual demo
- Requires:
  - Differentiable reimplementation of `process_images()` in pure PyTorch (`F.interpolate` for resize, manual normalization)
  - Enabling gradient flow through frozen vision encoders (flip `unfreeze_mm_vision_tower=True` flag)
  - Gradient checkpointing on LLM to fit in 24GB
- Result: a noise image that looks like TV static but makes the model answer correctly

### Universal Pixel Noise (The Demo)
- One noise image for an entire benchmark
- "Show two noise images and a real photo. Which does the model do best with?"
- Train on subset, eval on held-out — if it generalizes, very strong finding
- The optimized noise will likely show some structure (texture, color patterns) that reveals what the vision encoders respond to

### Single-Encoder Optimization
- Instead of backpropagating through all 4 encoders, optimize through just CLIP
- Much cheaper (one encoder instead of four)
- Tests whether CLIP alone carries the signal, or if all 4 encoders contribute
- Could compare: optimize through CLIP only vs SigLIP only vs DINOv2 only → which encoder matters most?

### Embedding Inversion (Two-Stage)
1. First find optimal embeddings (cheap, guaranteed to work)
2. Then find pixel images that produce those embeddings when run through an encoder
- CLIP inversion is well-studied (used in DALL-E, Stable Diffusion, etc.)
- Produces interpretable images that reveal what features the model uses
- Could use existing CLIP inversion libraries

### Diffusion Model + RL
- Train a small diffusion model to generate images
- Use benchmark accuracy as reward signal via REINFORCE
- The diffusion model learns to generate "helpful" images for any question
- Most complex approach, but produces a generative model rather than static noise
- Could condition on the question text → question-specific generated images

### Adversarial Patch
- Instead of full-image noise, find a small patch (e.g., 64x64) that can be pasted onto ANY image to flip the answer
- Like adversarial patches in image classification, but for VLMs
- Tests robustness: "does adding this sticker change the model's answer?"

### Gradient Visualization
- Even without full optimization, visualize the gradient of the loss w.r.t. input pixels
- Shows which regions of the image the model is actually attending to
- Saliency maps for each question — compare normal vs wrong image gradients
- Cheap (one backward pass per question)

### Cross-Benchmark Transfer
- Optimize noise on one benchmark (e.g., MMMU), test on another (e.g., ScienceQA)
- If universal noise transfers between benchmarks, it suggests the model has a generic "image-present" mode rather than actually processing visual content
- Very strong finding if it works

### Phase 2 Image Conditions (from original plan)
These aren't optimization-based but test related hypotheses:
- **Heavy blur** (Gaussian radius=50): preserves color distribution, destroys detail
- **Shuffled patches** (8x8 grid randomly rearranged): destroys spatial layout, preserves texture statistics
- Together with noise + wrong image, creates a spectrum from "no information" to "partial information" to "full information"
