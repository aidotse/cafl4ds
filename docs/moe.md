# MoE ViT architecture

Select `encoder=moe_tiny` in configurations that already contain the encoder group. The defaults are four experts, top-1
routing, and MoE in the last two encoder blocks. `encoder.moe_blocks` uses zero-based indices; `null` selects the last
two for the configured depth. Attention, patch embedding, MAE masking, pooling, and the MAE decoder retain their
existing structure.

For example, an architecture smoke run on synthetic images can use:

```bash
uv run python scripts/run_loop.py encoder=moe_tiny ssl=mae data=synthetic img_size=16
```

Set `encoder.routing_level=image` for image routing or `encoder.top_k=2` for top-2. Image routing averages every token
entering the block's feedforward layer, including CLS. During masked MAE training only the visible patches and CLS reach
that layer.

## Explicit training outputs

The original `forward`, `forward_encoder`, and `embed` interfaces remain available. Use the auxiliary methods when
writing a training objective:

```python
result = encoder.forward_with_aux(images, mask_ratio=0.75)
tokens, mask, ids_restore = result.tokens, result.mask, result.ids_restore
loss = reconstruction_loss + aux_weight * result.aux_loss

features, aux_loss = encoder.embed_with_aux(images)
```

`aux_loss` is an unweighted scalar averaged over MoE layers from that same forward pass. It is not stored as mutable
last-forward state. Dense encoders return zero. MAE, SimSiam, and Barlow Twins add it through `ssl.aux_weight` (default
`0.01`); two-view methods average the two views' losses. Loss-gate per-sample scores exclude balancing.

## Routing and balancing

Dispatch gathers assigned tokens and runs only their selected experts, with no capacity limit or dropped tokens. Expert
weights are computed by softmax over **all experts before top-k**, then gathered without renormalizing. Top-1 scales the
chosen expert's output by its full-softmax probability. Reconstruction therefore trains the router through this weight
as well as through the auxiliary loss. Expert choice itself remains discrete. This changes the computation of legacy
checkpoints trained with selected-only normalization; historical WP-M0 results retain their original semantics.

Batch balancing adds importance and load coefficients of variation squared, following
[V-MoE's reference implementation](https://github.com/google-research/vmoe/blob/main/vmoe/nn/routing.py). Importance
uses full clean router probabilities; load uses a differentiable Gaussian expected-selection estimate. Statistics are
pooled over the batch's tokens or images, according to the routing level. The two terms have separate
`encoder.importance_weight` and `encoder.load_weight` settings (both default `1.0`).

Training logits receive Gaussian noise with standard deviation `encoder.router_noise_std / num_experts`; the configured
multiplier defaults to `1.0`. Evaluation is deterministic. Setting the multiplier to zero disables noise and makes the
load term a hard-count statistic with no gradient; importance still supplies gradients. Setting `ssl.aux_weight=0`
removes balancing from the training objective.

## Read-only routing

```python
probabilities = encoder.routing(images)
# dict[zero_based_block_index, Tensor[B, 1 + N, E]]
```

This returns full probabilities before top-k selection for each MoE block, using actual sequential block inputs. Image
routing broadcasts the image distribution over its tokens. It reads full, unmasked images without gradients or router
noise, preserves all module modes, and does not consume random numbers. `attention_maps(images)` follows the same shared
block execution.

This implementation covers architecture and SSL loss integration. EMA/bias balancing, cross-view consistency, upcycling,
corpus diagnostics, and acceptance experiments are deferred.
