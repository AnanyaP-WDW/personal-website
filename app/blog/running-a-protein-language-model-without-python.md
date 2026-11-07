---

title: ESMC.cpp - Running a Protein Language Model Without Python
date: 2026-11-7
description: The aim of this project is to port ESM Cambrian 300M to a single C++ binary on ggml/llama.cpp — same embeddings as PyTorch, no Python at inference time, model fits in ~230 MB.
tags: ["protein language models", "c++", "ggml", "quantization", "inference"]
---

# Running a Protein Language Model Without Python

## Table of Contents

1. [The idea](#the-idea)
2. [What is a protein language model?](#what-is-a-protein-language-model)
3. [Why port it to C++?](#why-port-it-to-c)
4. [What made this hard](#what-made-this-hard)
5. [Validation of the port](#validation-of-the-port)
6. [The numbers](#the-numbers)
7. [Quantization: smaller models, honest quality](#quantization-smaller-models-honest-quality)
8. [Does it still work on real biology?](#does-it-still-work-on-real-biology)
9. [Speed: batching, and where it helps](#speed-batching-and-where-it-helps)
10. [What we learned](#what-we-learned)
11. [Try it yourself](#try-it-yourself)
12. [References](#references)

---

## The idea
[ESM Cambrian (ESM-C)](https://www.evolutionaryscale.ai/blog/esm-cambrian) is a priotein model ie. when I input a protein sequence it outputs back a numerical "embedding"/ representtion of the sequence. This protein repsresetation backbone can in turn be finetuned to predict mutations, functions or toxicity etc. 

The official way to run ESM-C is PyTorch. That works great on a GPU server. It works poorly on a laptop, an offline batch machine, an embedded device, or anywhere you would rather ship one binary than a Python environment, PyTorch, and a pile of dependencies.

The hypothesis: the llama.cpp project proved that large *language* models can run as a single self-contained C++ binary. Can the same engineering discipline work for a *protein* model?

The answer — a standalone C/C++ runtime for ESM-C 300M built on top of [ggml](https://github.com/ggml-org/llama.cpp) (the tensor library behind llama.cpp), with Apple Metal GPU acceleration, a GGUF model format, and a quantizer. No Python at inference time. The full model, 4-bit quantized, is **228 MB on disk** and uses **~520 MB of RAM**. It reproduces the original PyTorch embeddings to a mean per-residue cosine similarity of **0.9999996** — the remaining gap is pure floating-point rounding, not a mistake.

This post is about how I got there, the three major silent bugs that nearly shipped wrong science, and the numbers.

## What is a protein language model?

The ESMC models are trained on hundreds of millions of protein sequences the way a text LLM is trained on books — except the "words" are amino acids, and the goal is to understand each position *in the context of the whole chain*.

Unlike GPT-style models, ESM-C is **encoder-only**: one forward pass looks at the entire sequence in both directions (no left-to-right generation), and produces one vector per amino acid. There is no autoregressive loop, no growing key/value cache — just a single clean input-output mapping. That makes it a natural candidate for a lightweight runtime.

ESM-C comes in 300M, 600M, and 6B parameter sizes. I ported the 300M — small enough to experiment with, big enough to be useful.

## Why port it to C++?

- **Cold start.** Importing PyTorch and loading a checkpoint can take upto 10 seconds. A C++ binary loads in a fraction of that.
- **Dependencies.** Python, PyTorch, the right CUDA/Metal versions... vs. one executable and one model file.
- **Memory.** The PyTorch stack has overhead. The worst measured configuration (full F32, longest sequences) peaks at 2.6 GB — and the recommended deployment config (Metal, 4-bit) sits at ~520 MB.
- **Embeddability.** A C library can be called from anything. A Python process cannot.

The idea was that llama.cpp's backbone — ggml's CPU and Metal backends, the GGUF single-file format, the block-quantization kernels — would transfer to a *bidirectional encoder*. It mostly did. The interesting parts were the places it didn't, which is what most of this post is about.

## What made this hard 

Porting a transformer is not the hard part. The hard part is that **numerical ports fail silently**: the code runs, the output looks like plausible numbers, and it is subtly, biologically wrong. Nothing crashes. Nothing warns you.

ESM-C is not ESM2 with different weights. It has a gated SwiGLU feed-forward (three weight matrices instead of two), no biases anywhere except one LayerNorm, rotary position embeddings in the NeoX style, a query-scaling step that must happen *before* rotation, and a residual scaling factor. Get any of these wrong and you don't get an error — you get confident nonsense.

Three major bugs from the port, each of which produced wrong embeddings without raising a single exception:

**1. The two amino acids mistake.** ESM-C's vocabulary is only 33 tokens (20 amino acids plus special tokens), so it's a simple letter-number lookup table. I got the order of two of them wrong: glutamine (Q) and asparagine (N) were swapped. Test sequences that didn't contain Q or N passed validation perfectly. Longer real-world sequences silently dropped to ~0.95 cosine similarity — looks fine on a plot, wrong for any downstream science. 

**2. Scale in the wrong order.** The query matrix must be scaled by 1/√d *before* rotary position embedding is applied. I scaled it after. Layer-0 query norms were off by **700%**. After moving one line, the error dropped to 0.0025%.

**3. The forgotten bias.** ESM-C's feed-forward LayerNorm has a bias term. The original runtime dropped it. It slipped through my implementation. After the fix, every cached reference embedding in the project had to be regenerated.

## Validation of the port

1. **Tokenizer**: exact token-ID match against the official tokenizer.
2. **Layer-0 attention probe**: after the first layer, compare query/key vector norms against a hand-written PyTorch reference of the native checkpoint (≤5% relative error). This catches scaling-order and rotation-style bugs *early*, before they compound through 30 layers.
3. **Full forward**: per-residue cosine similarity > 0.999 (mean) and > 0.99 (worst residue) against the reference, plus a mean-pooled error check, across 100 real Swiss-Prot human sequences bucketed by length.

The reference itself is a small hand-written PyTorch reimplementation that reads the original checkpoint directly — deliberately independent of the HuggingFace transformers path, so a bug in one ecosystem doesn't hide a bug in the other.

Every optimization we shipped later — flash attention, GPU scheduling, batching, graph caching — had to keep passing these gates. Correctness was never a phase; it was a wall each change hit.

## The numbers

All on an Apple M4 Max (36 GB unified memory). The full tables are in the [paper](https://github.com/AnanyaP-WDW/esmc.cpp/blob/main/paper_esmc_cpp.pdf) and [repo](https://github.com/AnanyaP-WDW/esmc.cpp); the highlights:

**Accuracy vs. PyTorch** (100 Swiss-Prot sequences, per-residue cosine):

| Precision | Mean cosine | File size |
|---|---|---|
| F16 | 0.9999996 | 634 MB |
| Q8_0 | 0.99987 | 337 MB |
| Q4_K_M | 0.9953 | 238 MB |
| Q4_K_S | 0.9942 | 228 MB |

F16's residual error is pure rounding — a separate F32 build matches the reference to 1.000000, confirming the architecture is right and the GPU backend (Metal) adds no extra error beyond what CPU gives.

**Speed** (Metal, single sequence):

| Length | Throughput |
|---|---|
| short (47 tokens) | 95.7 seq/s |
| medium (235 tokens) | 38.3 seq/s |
| long (850 tokens) | 9.5 seq/s |

F16 on Metal is the fastest configuration — Apple Silicon has real F16 tensor cores, and for a 300M model the dequantization cost of smaller formats outweighs their bandwidth savings. **Quantization's value here is file size and RAM, not speed.**

**Memory**: 36/36 tested configurations fit in the 36 GB budget; the most efficient (Metal, 4-bit) peaks at ~512 MB — 1.4% of the machine.

## Quantization: smaller models, honest quality

The interesting engineering constraint: ggml's modern k-quant blocks need row dimensions that are multiples of 256. ESM-C 300M's model dimension is **960** — not a multiple. So only one of the seven weight matrices per layer (the feed-forward down-projection, 2560 wide) gets true k-quant blocks; the rest fall back to older 32-element block formats. Our first attempt (cheap 4-bit fallbacks everywhere) cratered cosine similarity to ~0.98; raising the fallbacks to 5/8-bit restored ~0.995.

The honest summary, measured on a real downstream task below: **8-bit is a safe default** (half the size, cosine 0.99987, biology preserved). **4-bit works for many uses but is not free** — treat it as an approximation and measure on *your* task.

## Does it still work on real biology?

Cosine similarity is a proxy. We also ran a genuine downstream task: **ProteinGym** variant-effect prediction over 10 deep-mutational-scanning assays (1000 variants each). Each mutation is scored by how much its embedding differs from the wild-type protein's embedding; we then measure whether that ranking matches experimental reality, and whether quantization preserves the *rank correlations* the PyTorch model achieves.

- **F16**: passes on all 10 assays (Spearman within 0.01 of PyTorch, mean gap 0.0007)
- **Q8_0**: also 10/10, mean gap 0.0027
- **4-bit**: 4–5 of 10 assays pass the same gate — usable for exploration, not for claiming parity

So: shrink the model by half with 8-bit and your biology stays intact. Shrink it to a third with 4-bit and you should re-validate on your own assays.

## Speed: batching, and where it helps

Two throughput stories, told honestly:

**Batching helps short sequences — not long ones.** Feeding many sequences through the GPU at once reaches 3.6× single-sequence throughput at batch 16 for short sequences. For medium and long sequences the GPU is already saturated by one sequence, so batching is roughly neutral once padding is included. Optimal batch size is length-dependent (8 for short, 4 for medium, 1–2 for long) — our `--fasta` mode uses a length-aware schedule by default.

**Load once, embed a whole corpus.** A 1000-sequence Swiss-Prot sample embeds ~10.6× faster than a shell loop that reloads the model per sequence, with only 2% padding overhead.

We also recorded the negative results: a length-bucketed graph cache and an F16-activation variant both failed to help on measurement, and we left them out. The lab notebook has the corpses.

## What we learned

1. **Silent numerical failure is the real antagonist.** Every bug in this project ran without errors and produced plausible numbers. Staged, alphabet-covering validation caught them; intuition did not.
2. **Validate your yardstick too.** The missing-bias bug hid because runtime and reference shared the same wrong assumption. Independent reference implementations are worth the effort.
3. **Consumer hardware is enough.** A 300M protein model at 38 seq/s, half a gigabyte of RAM, on a laptop that also runs your browser. The edge/batch deployment case for protein embeddings is real, and it doesn't need a datacenter GPU.
4. **Quantization is a science, not a slider.** Block-size constraints, fallback formats, and downstream rank-correlation gates all matter. Publish the honest numbers, not the flattering ones.
5. **Negative results belong in the notebook.** Two optimizations we planned didn't work. That's two fewer lies in the paper.

## Try it yourself

Everything is open source: **[github.com/AnanyaP-WDW/esmc.cpp](https://github.com/AnanyaP-WDW/esmc.cpp)** — runtime, converter, quantizer, benchmarks, and the full lab notebook.

Pre-converted GGUF models (F32/F16/Q8_0/Q4_K_M/Q4_K_S) with a model card: **[huggingface.co/AnanyaPathak/esmc-300m-gguf](https://huggingface.co/AnanyaPathak/esmc-300m-gguf)**.

```bash
# build (no Python needed at inference time)
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# embed a protein sequence, mean-pooled, 8-bit model
./build/esmc-embed -m models/esmc-300m-Q8_0.gguf \
    -s "MKVLAIGTGCGASHLG..." --pool mean -o embedding.npy

# or a whole FASTA corpus in length-sorted batches
./build/esmc-embed -m models/esmc-300m-f16.gguf --fasta proteins.fasta
```

The full reproduction recipe (including regenerating the PyTorch reference and every benchmark in this post) is in the paper's appendix.

## References

- EvolutionaryScale Team (2024). [ESM Cambrian: a frontier protein language model](https://www.evolutionaryscale.ai/blog/esm-cambrian).
- Pathak, A., Pathak, A. & Sharma, S. *esmc.cpp: A Zero-Dependency, Metal-Accelerated C/C++ Runtime for ESM Cambrian Protein Embeddings.* CAISc 2026 ([paper](https://github.com/AnanyaP-WDW/esmc.cpp/blob/main/paper_esmc_cpp.pdf), [repo](https://github.com/AnanyaP-WDW/esmc.cpp)).
- Gerganov, G. et al. [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) — the substrate this runtime is built on.
- Notin, P. et al. (2023). [ProteinGym](https://proteingym.org/) — the downstream variant-effect benchmark.
