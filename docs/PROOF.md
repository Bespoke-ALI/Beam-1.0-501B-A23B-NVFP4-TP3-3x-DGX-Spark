<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Why the padded checkpoint computes the same function

Zeros are appended at the **end** of every padded axis, so every original weight keeps its index. This note shows that each padded component contributes exactly zero, and that splitting the padded model three ways keeps every query head with its original key-value head.

## Notation

Beam has 48 query heads, 8 key-value (KV) heads and head dimension 128, so each KV head serves 6 query heads (grouped-query attention, GQA). Query head *h* reads KV head κ(h) = ⌊h/6⌋.

Padding changes the counts to 54 query heads and 9 KV heads. The ratio stays 6:1. Heads 48–53 are new, and all six read the new KV head 8.

## Attention

Write head *h*'s contribution to the layer output as:

$$
y_{\mathrm{attn}} = \sum_{h} W_o^{(h)} \Big[ g\big(G_h x + b_h\big) \odot \sum_{j} \operatorname{softmax}_j\big(\tilde q_h \cdot \tilde k_{\kappa(h),j}\big)\, v_{\kappa(h),j} \Big]
$$

- *g* is the output gate. It is a sigmoid in the reference model; the served image adds an optional floor and temperature. Nothing below depends on *g*'s form.
- *G_h* is the head's rows of the gate projection, and *b_h* its gate bias.
- *W_o^(h)* is the head's slice of the output projection.
- *q̃* and *k̃* are the per-head RMSNorm outputs, after RoPE on sliding-window layers or SSMax scaling on full-attention layers.

Four facts complete the argument.

1. **Padded heads (48–53) contribute zero, for two independent reasons.**
   - Their output-projection columns are zero, so *W_o^(h)*(·) = 0 whatever the bracket holds.
   - They read only KV head 8, whose value vectors are zero, so the bracket is zero too.
2. **Nothing becomes NaN on the way.** A zero query gives RMSNorm(0) = 0/√ε = 0. RoPE and SSMax keep zero at zero. A softmax over all-zero scores is uniform and finite.
3. **Original heads (0–47) are unchanged.** Their projection rows, gate rows and bias entries are byte-identical, and κ(h) < 8 still points at an unchanged KV head.
4. **Sharding keeps GQA groups whole.** At TP=3, rank *r* holds query heads 18r to 18r+17 and KV heads 3r to 3r+2. A rank-local query head *i* reads local KV head ⌊i/6⌋, which is global head 3r + ⌊i/6⌋ = ⌊(18r+i)/6⌋: its original KV head. The padder checks this mapping for every head before it writes anything (`TP_GQA_MAPPING OK`).

The row-parallel all-reduce sums each rank's partial output-projection product. That equals the unsharded sum up to floating-point reassociation, as at any TP degree.

## Feed-forward layers (dense and experts)

For each SwiGLU block, with φ = SiLU:

$$
y_{\mathrm{ffn}} = W_d\big(\varphi(W_g x) \odot W_u x\big), \qquad \varphi(0)\cdot 0 = 0
$$

Padded rows of *W_g* and *W_u* are zero, so every padded intermediate unit is φ(0)·0 = 0. The matching columns of *W_d* are also zero, so the output is unchanged. The TP argument is the same as for the output projection.

## NVFP4 experts

A dequantized weight is the E2M1 value × its FP8-E4M3 block scale × the per-tensor `weight_scale_2`.

- **Padded weights are exactly zero.** Padded bytes are 0x00, so each 4-bit value decodes to +0, and padded scale bytes decode to +0.
- **No 16-value quantization block mixes real and padded values.** The original width 2,560 is a multiple of 16, and so are the rank boundaries at 896 and 1,792. Dynamic block scales computed for real activations are therefore unchanged, and padded activation blocks are all zero. This step assumes each kernel maps an all-zero block (block maximum 0) to zero rather than to 0/0. Outputs were finite and correct, but we did not audit every kernel.
- **Routing is untouched.** The expert count (256), the router weights and the router inputs do not change.

## Exactness in floating point

Adding an exact zero leaves an IEEE sum unchanged. Differences can only arise where a kernel tiles a longer reduction differently. Per rank, the output projection's reduction length is 2,304 (6,144 unsharded in the original), and the expert down-projection's is 896. Such differences are reassociation: the same class as changing the TP degree.

## What was measured

- **Byte-level verify:** every tensor of the padded checkpoint compared with the source. Original regions identical, every padded region zero, every unpadded tensor identical: 151,262 tensors, 0 failures.
- **Numeric test:** float64 forward passes on one random 5-token input (seed 0) for attention layer 0 (computed without a causal mask, which does not affect the comparison because the same inputs go through all three versions), dense MLP layer 0, and expert 0 of MoE layer 3. Each was computed on the original weights, on the padded weights unsharded, and on the padded weights split exactly as vLLM splits them at TP=3.

| Comparison with the original | Max absolute difference | Relative difference |
|---|---|---|
| Attention layer 0, padded, unsharded | 0 | 0 |
| Dense MLP layer 0, padded, unsharded | 0 | 0 |
| Expert 0, layer 3, padded, unsharded | 0 | 0 |
| Attention layer 0, padded, TP=3 split | 8.9×10⁻¹⁶ | 4.2×10⁻¹⁶ |
| Dense MLP layer 0, padded, TP=3 split | 2.3×10⁻¹² | 5.3×10⁻¹⁶ |
| Expert 0, layer 3, padded, TP=3 split | 1.1×10⁻¹³ | 3.6×10⁻¹⁶ |

Relative difference is the maximum absolute difference divided by the largest absolute reference value. The attention test models the gate as a plain sigmoid with bias, and omits the q/k norms, RoPE and SSMax, which act per head dimension and are not altered by padding.

**Not yet measured:** a full-model comparison of output probabilities against an unpadded reference (planned: the unpadded checkpoint served at PP=3 on the same image, compared token by token at temperature 0).
