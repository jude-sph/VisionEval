# Two-Bit Bottleneck: Transmitting Answers Through a Narrow Vision Channel

## 1. Motivation

The existing noise optimization experiments show that optimized encoder-output embeddings can steer Cambrian-8B toward correct MCQ answers without any real image. But those embeddings are enormous: 4 encoders produce ~11,000 tokens with 1024–1152 dimensions each — millions of floats for what is fundamentally a 2-bit signal (one of four choices: A/B/C/D).

**Core question:** How narrow can the vision-to-language channel be and still transmit the answer?

## 2. The Experiment

We replace the entire vision pipeline (4 encoders + SVA projector + thousands of image tokens) with a small number of **learnable token embeddings** injected directly into the LLM's input embedding space. For each benchmark question, we optimize these tokens via gradient descent to minimize teacher-forcing loss on the correct answer.

The tokens bypass the vision encoders and multimodal projector entirely — they are inserted directly where image tokens would normally appear in the LLM's input sequence.

## 3. Design Decisions and Open Problems

### 3.1 Where do we optimize?

We optimize in the **LLM's input embedding space** (R^4096 for LLaMA-8B). The learnable vectors are continuous — they are not constrained to correspond to any real vocabulary token. They sit in the same vector space as text token embeddings but can take any value the optimizer finds useful.

This is distinct from the existing embedding optimizer, which works in the vision encoder output space (pre-projector). Our approach is simpler: no monkey-patching of `encode_images`, no interaction with the multimodal projector at all. We build `inputs_embeds` directly and call `model.forward(inputs_embeds=...)`.

### 3.2 Sequence length mismatch (the positional encoding problem)

**Problem:** Cambrian was trained with ~576+ image tokens occupying a fixed region of the input sequence. If we insert only 2 tokens, all subsequent text (the question, answer choices) shifts to much earlier positions than the model ever saw during training. This creates two confounds:

1. **RoPE positional encoding shift** — the question text lands at positions ~20 instead of ~600. The model's learned attention patterns assume the question follows a large image block.
2. **Attention pattern mismatch** — the model's middle layers learned to aggregate over a large image region. With 2 tokens, those attention heads have almost nothing to attend to.

**Consequence:** If the bottleneck fails, we cannot distinguish "2 tokens aren't enough information" from "the model is confused by the wrong sequence format." If it succeeds, it might be exploiting the positional shift rather than genuinely decoding the tokens.

### 3.3 Proposed fix: outer product parameterization

To maintain the expected sequence length while keeping the information bottleneck narrow, we considered filling 576 positions using an outer product (rank-2 decomposition):

```
basis = learnable [2, 4096]       — the 2 vectors we optimize
alpha = fixed     [576, 2]        — position-dependent mixing weights
image_tokens = alpha @ basis      — [576, 4096], what the LLM sees
```

Each position gets a unique blend of the 2 basis vectors (e.g., a linear ramp from pure-A to pure-B). This gives the model the correct sequence length and positional encoding alignment, with only 2 vectors worth of learnable information.

**Problem with this fix:** After the matrix product, the model sees 576 real tokens and processes them through nonlinear layers (LayerNorm, GeLU activations, multi-head attention). Because of these nonlinearities, attending to position 100 (mostly vector A) vs position 400 (mostly vector B) can produce genuinely different downstream effects. The effective information capacity is somewhere between 2 tokens and 576 tokens — it's a soft bottleneck, not a hard one.

**Current status:** We implement the simple 2-token version first (hard bottleneck, known positional confound) and document the positional shift as a limitation. The outer product version can be added as a comparison condition. Having both versions is actually more informative than either alone:
- If both succeed → the information content is truly small, and the format doesn't matter much
- If only the outer product succeeds → the model needs the right format but not the full bandwidth
- If neither succeeds → the vision channel carries more than 2 tokens of real information (for these benchmarks)

### 3.4 What we expect to find: codebook emergence

For MCQ benchmarks (MMMU, ScienceQA), there are only 4 possible answers. If we optimize across hundreds of questions and cluster the resulting token pairs by ground truth answer (A/B/C/D), we expect ~4 distinct clusters. This would mean the optimizer discovered a consistent "code" — a mapping from token patterns to answers.

**Analysis plan:**
- PCA and t-SNE scatter plots of optimized tokens, colored by answer
- Within-class vs between-class cosine similarity
- Confusion matrix (what gets confused with what)
- LM head decoding: project each token through LayerNorm + lm_head to see what vocabulary word it most resembles. Does it literally learn the token "A"? Or something more abstract?

## 4. Mechanistic Hypothesis: Attention Head Selection

Beyond asking "can it work?", we have a specific hypothesis about *how* it works at the circuit level.

### 4.1 The hypothesis

The LLM has many attention heads across 32 layers. Some heads have OV (output-value) circuits that naturally "write" the direction for token A (or B, C, D) into the residual stream when they attend to something. Our hypothesis:

**When optimizing a bottleneck token, the optimizer does not learn an abstract code. Instead, it finds the input direction that maximally activates the attention head whose OV circuit already points toward the correct answer.**

In other words: the optimized token is essentially the top singular vector of one head's QK circuit, selected because that head's OV circuit produces the right answer.

### 4.2 Testable prediction

The attention head that attends most strongly to the bottleneck token (after optimization) should be one of the top-N heads that was already predisposed to output the correct answer — identifiable from the weight matrices alone, before any optimization.

### 4.3 How to test it

1. **Before optimization:** For each attention head, characterize its "answer preference" — how strongly does its OV circuit write in the direction of A, B, C, D? This can be approximated by looking at the head's W_OV matrix projected through the unembedding layer.

2. **After optimization:** Hook every attention head and record which heads attend to the bottleneck positions. Check whether a single head dominates or whether activation is distributed.

3. **Compare:** Is the dominant head one of the top-ranked heads from step 1? If yes, the optimizer is just finding the right key for a lock that already exists.

### 4.4 Implications

If confirmed, this would mean:
- The optimizer isn't teaching the model anything new — it's finding the input that triggers an existing circuit
- The "codebook" isn't arbitrary — it's determined by the model's weight geometry
- The vision channel in normal operation might function similarly: real image features happen to activate specific heads whose OV circuits produce relevant outputs
- This connects the bottleneck experiment to mechanistic interpretability of how VLMs use image tokens in general

## 5. Data Collection

The implementation collects rich per-step data to support graphing and analysis:

**Every optimization step (cheap — tensor operations on 2 vectors):**
- Loss value
- L2 norm of each bottleneck token
- Gradient L2 norm of each token
- Cosine similarity between the 2 tokens

**At snapshot intervals (every 5 steps — requires forward pass):**
- Accuracy check: does the model answer correctly at this point?
- LM head decoding: what vocabulary words do the tokens most resemble?

**Codebook analysis (after all questions are optimized):**
- PCA and t-SNE projections with initial→final trajectory arrows
- Confusion matrix
- Pairwise cosine similarity heatmap (sorted by answer class)
- Aggregated loss curves by answer class (mean ± std)
- Step-accuracy evolution curve
- Token norm and gradient norm evolution by answer class
- Inter-token cosine similarity evolution by answer class
- LM head decoded words per answer class centroid
- Per-answer final loss distributions (for box/violin plots)

**Attention head instrumentation (planned, not yet implemented):**
- Per-layer, per-head attention weights on bottleneck positions
- Activation differences between optimized vs random tokens
- OV circuit characterization per head

## 6. Relationship to the Broader Project

This experiment sits alongside the existing conditions (normal image, no image, wrong image, Gaussian noise, optimized embeddings, optimized pixels) as another way to probe what the vision channel actually contributes.

| Experiment | Question |
|-----------|----------|
| No image | Does the model need image tokens at all? |
| Wrong image | Does the model need the *right* image? |
| Gaussian noise | Does structured (natural image) input matter? |
| Optimized embeddings (full) | Can arbitrary encoder outputs steer the answer? |
| Optimized pixels | Can a visible noise image steer the answer? |
| **2-token bottleneck** | **How little information suffices to steer the answer?** |
| **Attention head analysis** | **What circuit does the model use to read the signal?** |

If 2 tokens suffice, it suggests the vision-language interface in Cambrian-8B functions more like a narrow control signal than a rich perceptual channel — at least for MCQ benchmarks where the answer is determined by language priors and the image just nudges a decision.

## 7. Running the Experiment

```bash
# Smoke test (1 question, verify it runs)
python scripts/optimize_bottleneck.py --max_samples 1 --num_steps 5 --gpu_id 0

# Full run on MMMU (50 questions)
python scripts/optimize_bottleneck.py --benchmark mmmu --max_samples 50 --gpu_id 0

# Larger run for more statistical power
python scripts/optimize_bottleneck.py --benchmark mmmu --max_samples 200 --num_steps 50 --gpu_id 0
```

Results are saved to `results/two_bit/`, separate from the main optimization results in `results/optimization/`.

Hardware: Single GPU is sufficient (3070 8GB, 3090 24GB, etc.). Vision encoders are offloaded to CPU since they are not used. Gradient checkpointing is enabled. Memory footprint is lighter than the full embedding optimizer since image tokens do not expand the sequence length.
