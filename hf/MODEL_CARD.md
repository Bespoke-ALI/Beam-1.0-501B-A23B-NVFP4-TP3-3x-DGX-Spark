---
license: apache-2.0
base_model: reflection/Beam-1.0-501B-A23B
base_model_relation: quantized
library_name: vllm
pipeline_tag: text-generation
tags:
  - nvfp4
  - tensor-parallel
  - dgx-spark
  - gb10
  - beam
  - reflection
  - multi-node
language:
  - en
---
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark (unofficial)

Reflection's [Beam-1.0-501B-A23B-NVFP4](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4), zero-padded so vLLM can split it evenly across **three** GPUs (tensor parallelism 3). It was built for three NVIDIA DGX Spark (GB10) nodes, where it runs on a stock vLLM image with one constant changed.

**This is a modified version of Reflection's model, not an official release.** It is not affiliated with or endorsed by Reflection AI or NVIDIA. Full instructions, the padder, the equivalence proof and the launch scripts: **https://github.com/Bespoke-ALI/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark**

## At a glance

| | |
|---|---|
| Hardware | 3 × NVIDIA DGX Spark (GB10, 128 GB unified memory each), linked in a switchless ring of three QSFP cables |
| Serving | vLLM, tensor parallelism 3, one rank per node; pinned image plus a one-line patch |
| Context | 131,072 tokens; KV-cache pool about 178,000–180,000 tokens (about 1.4 full-length requests at once) |
| Speed | about 18 tokens/s decode for one request at any context length; about 32 tokens/s in total with 4 short requests at once |
| Start-up | 17 minutes on freshly rebooted nodes (both Oct 21 launches); an estimated 65–70 minutes on our first-ever launch |
| Size | about 309 GB (53 shards); each node needs the full copy and about 103 GB of it in memory |
| Weights | Reflection's NVFP4 values, unchanged; only zeros added |

## Quick start

The GitHub repo has the full, tested walkthrough with a pass/fail check after every step: **https://github.com/Bespoke-ALI/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark**. In outline:

1. **Network:** cable and address the three-node ring and check every link (GitHub Step 1).
2. **Download this checkpoint** to one node and verify it:

```bash
pip install -U "huggingface_hub>=1.0"
hf download Bespoke-ALI/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark --local-dir /srv/models/tp3/Beam-1.0-501B-A23B-NVFP4-TP3
cd /srv/models/tp3/Beam-1.0-501B-A23B-NVFP4-TP3 && sha256sum SHA256SUMS && sha256sum --quiet -c SHA256SUMS && echo SUMS_MATCH
```

   `sha256sum SHA256SUMS` must print `9f57e87b492a27c3fa3cea11b0e45e1f4784069a35ee05ea2b3e0f1c100cd5dd`, the value published on GitHub, so a changed checksum list cannot pass unnoticed. Then `SUMS_MATCH`.
3. **Copy it to the other two nodes**, and **build the serving image** from the repo's `docker/Dockerfile` (GitHub Steps 4–5).
4. **Launch** with `scripts/launch_node_v0_1_4.sh` on all three nodes, node0 first (GitHub Step 6), then run `scripts/smoke_test_v0_1_1.sh` on node0 (Step 7).

## What was changed

Zeros were appended so every split divides by 3. No original value was altered.

| | Original | Padded |
|---|---|---|
| Query heads | 48 | 54 |
| Key-value heads | 8 | 9 |
| Dense feed-forward width (layers 0–2) | 20,480 | 20,736 |
| Expert feed-forward width (256 experts × 49 layers) | 2,560 | 2,688 |
| Vocabulary | 151,936 | unchanged; vLLM's padding multiple is set to 192 at serve time |

The attention output gate and its per-head bias, and the NVFP4 expert weights with their FP8 block scales, were padded to match. `config.json` changes in four fields only. `PADDING_MANIFEST.json` records the padder version, its arguments, every shard's SHA-256 and the config changes. See `NOTICE` for the Apache-2.0 modification notice.

## How it was checked

- **Source:** revision `3bd1dccf13c8f7aa3d6732c5aaa48ba107a2c086` of the NVFP4 repo; all 63 files matched the Hub's hashes before padding.
- **Byte-level:** all 151,262 tensors compared with the source. Original regions identical, padded regions zero, untouched tensors identical: 0 failures.
- **Numeric:** float64 forward passes on sampled layers. Exact unsharded, and within 5.3×10⁻¹⁶ relative error when split three ways.
- **Served on 3 × DGX Spark:** about 18 tokens/s single-stream decode; a correct tool call (one probe); exact retrieval of a planted fact from a 41,522-token prompt. In Reflection's Beam Verifier, 131 of 133 checks passed, including every structured-output and tool-calling check; the 2 failures are one request-validation check (a message with an invalid role is accepted instead of refused), which happens before the model runs.
- **Public benchmarks (one pass each, temperature 0.7, top_p 0.99, reasoning effort left at Beam's default, the maximum):**

| Benchmark | This build | Reflection's NVFP4 figure |
|---|---|---|
| GPQA Diamond | 89.9% at a 32,768-token cap; 90.9% with the 2 cut-off answers re-asked at 65,536 | 89.3% |
| IFBench, strict / lenient | 75.0% / 77.7% at 16,384; 76.7% / 79.3% with the 13 cut-off answers re-asked at 32,768 (unchanged after re-asking the last 3 at 65,536) | 79.2% |
| GSM8K (first 200) / HumanEval | 97.5% / 98.2% | — (a community 3-bit build reports 97.5% / 97.6%) |

All are consistent with the reference figures by a rule fixed before scoring. Method, intervals and caveats are in the GitHub repo.
- **Not yet done:** a full-model comparison of output probabilities against an unpadded reference.

For context, diffbot's [EXL3 build](https://huggingface.co/diffbot/Beam-1.0-501B-A23B-EXL3-3bpw-2x-RTX-PRO-6000) reports a KL divergence from the full-precision model of 0.0116 for NVFP4 as served on Blackwell. Padding adds only exact zeros, so this build is expected to match served NVFP4, not to improve on it. That is expected, not yet measured.

## Performance (one request unless stated; `vllm bench serve`, eager mode)

| Prompt length | Time to first token (1 request) | Decode speed (1 request) | Total output with 4 requests at once |
|---|---|---|---|
| 1,024 tokens | 0.7 s | 18.2 tokens/s | 31.9 tokens/s |
| 8,192 tokens | 3.5 s | 18.2 tokens/s | 26.5 tokens/s |
| 32,768 tokens | 12.5 s | 18.1 tokens/s | 16.4 tokens/s |
| 130,000 tokens | 71.7 s | 18.0 tokens/s | not run (the KV pool holds about 1.4 full-length requests) |

Medians from one run on Oct 20 with random-token prompts, 256-token answers and small samples (2–16 requests per cell). The last column counts the whole run time, prompt reading included, so it falls as prompts get longer. Full tables, 99th percentiles and raw files are on GitHub.

## Use

Serve at `--tensor-parallel-size 3` with vLLM's `DEFAULT_VOCAB_PADDING_SIZE` set to 192 (the repo's Dockerfile does this, on top of `vllm/vllm-openai@sha256:c1a6be6ad08016f8cebcb449146b3d992791fe1868f9a26f3e48a51040517b77`), with `--moe-backend b12x`, `--kv-cache-dtype fp8` and `--gpu-memory-utilization 0.885` (0.88 left too little KV-cache memory for the full 131,072-token context on two of three launches; see the GitHub README's "Memory margin"). Each node holds about 103 GB of weights and needs at least 109 GiB of available memory at launch. This checkpoint does not load at TP=2, 4 or 8, because 9 key-value heads do not divide by those; for those, use Reflection's original checkpoint.

Verify your download with `sha256sum -c SHA256SUMS`. It lists the 64 files the padding step produced, with Reflection's original model card under its new name, `README_reflection.md`; this card (`README.md`) and `NOTICE` are not listed.

## Known limitations

- **Memory margin is thin.** At `--gpu-memory-utilization 0.88`, vLLM found too little KV-cache memory for the full context on two of three launches; 0.885 started on both launches we ran. If you hit `ValueError: To serve at least one request with the model's max seq len…`, raise it by 0.005 or lower `--max-model-len`.
- **Reboot before relaunching.** On our nodes, stopping the containers left about 105 GiB of memory held by the driver until reboot.
- **Eager mode only;** CUDA graphs, chunked-prefill tuning and speculative decoding are untried.
- **Equivalence is shown analytically and on sampled blocks,** not yet by a full-model comparison against an unpadded reference.
- **One cluster, two launches at the published settings.** Long-run stability at 0.885 has not yet been soak-tested (a 24-hour soak passed at 0.88).

## Other Beam builds

- [diffbot EXL3 3-bit](https://huggingface.co/diffbot/Beam-1.0-501B-A23B-EXL3-3bpw-2x-RTX-PRO-6000) for two RTX PRO 6000.
- [Baekpica ~1.5-bit VQ](https://huggingface.co/Baekpica/Beam-1.0-501B-A23B-VQ-GGUF) for a single DGX Spark, on its own engine.
- GGUF quantizations for llama.cpp: [unsloth](https://huggingface.co/unsloth/Beam-1.0-501B-A23B-GGUF), [AtomicChat](https://huggingface.co/AtomicChat/Beam-1.0-501B-A23B-GGUF).

## Support

Questions and problems: open an issue on the GitHub repo. Help is best effort, with no support promised.

## Licence and citation

Apache-2.0, as the original. Please cite Reflection:

```bibtex
@misc{reflection2026introducingbeam,
  author = {{Reflection}},
  title  = {Introducing {Beam}: {Reflection}'s 501B open-weight model},
  year   = {2026},
  month  = oct,
  url    = {https://reflection.ai/blog/introducing-beam}
}
```

To cite this build, use the "Cite this repository" button on the GitHub repo.

Padded and validated by Josh Pearlman on his own three-node DGX Spark cluster. Code and documentation drafted with Claude (Anthropic).
