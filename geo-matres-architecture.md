# Geo-MATRES: Model Architecture and Loss Design

This document describes the current architecture used in `geo-matres.ipynb`, with emphasis on the model flow, the three inference strategies, the logit definitions, and the training loss.

## 1. Task Setup

MATRES is a temporal relation classification task between two events in a sentence or context window. The label set contains four classes:

- `BEFORE`
- `AFTER`
- `EQUAL`
- `VAGUE`

The current model uses:

- a semantic branch that predicts all 4 labels directly,
- a geometry branch that models temporal relations through time-like coordinates,
- a fusion mechanism that combines both branches into the final prediction.

The final target of the model is the fused output `p_fuse`, but the two branches are kept as separate experts for supervision, interpretability, and ablation evaluation.

---

## 2. Overall Architecture

The model has four main stages:

1. Shared encoder
2. Event representation
3. Semantic branch
4. Geometry branch + fusion

### 2.1 Shared Encoder

The model takes:

- `input_ids`
- `attention_mask`
- `token_type_ids`
- event positions `e1_pos`, `e2_pos`

The encoder is a pretrained Transformer. In the current notebook, the instantiated backbone is:

- `bert-large-uncased`

The architecture itself is backbone-agnostic, but the notebook currently uses the large BERT variant via `MODEL_NAME = "bert-large-uncased"`.

The encoder produces hidden states:

\[
H \in \mathbb{R}^{B \times L \times d}
\]

where:

- `B` = batch size
- `L` = sequence length
- `d` = hidden size

### 2.2 Event Representation

For each event, the model takes the hidden state at the event position and enriches it with local context using attention.

Let the hidden state at the event position be `h`. The model uses `h` as the query in a `MultiheadAttention` layer, with the full sequence as key/value.

Then it:

- extracts attention weights,
- selects the top-`k` most relevant tokens,
- aggregates them into `h_context`,
- concatenates `h` and `h_context`,
- applies a projection layer to obtain a richer event vector `h_rich`.

For each event, the output is the enriched event representation:

\[
h_1, h_2
\]

The model does not produce temporal coordinates from each event independently at this stage.

### 2.3 Pair Representation and Semantic Branch

The model first builds a pair representation:

\[
h_{pair} = f_{pair}([h_1; h_2; |h_1-h_2|; h_1 \odot h_2])
\]

The semantic classifier then receives both event vectors and the pair representation:

\[
z_{sem} = [h_1; h_2; h_{pair}]
\]

Then the semantic classifier produces:

\[
\text{logits}_{sem} = f_{sem}(z_{sem})
\]

The output dimension is 4, corresponding to:

- `BEFORE`
- `AFTER`
- `EQUAL`
- `VAGUE`

Written explicitly:

\[
\text{logits}_{sem} =
\begin{bmatrix}
g^{sem}_{before} \\
g^{sem}_{after} \\
g^{sem}_{equal} \\
g^{sem}_{vague}
\end{bmatrix}
\]

where:

\[
g^{sem} = W_{sem} \, \phi(z_{sem}) + b_{sem}
\]

and `\phi(\cdot)` denotes the linear + normalization + ReLU + dropout stack inside `cls_head`.

### 2.4 Geometry Branch

The geometry branch predicts time-like coordinates for each event conditioned on the event pair:

- `s1, e1` for event 1
- `s2, e2` for event 2

\[
(s_1,e_1) = T([h_1; h_{pair}])
\]

\[
(s_2,e_2) = T([h_2; h_{pair}])
\]

This makes latent time pair-conditioned. The temporal coordinate of an event is not treated as an isolated global property; it is produced from the event representation together with the contextual pair representation.

From these coordinates, the model computes ordered temporal logits:

\[
g_{before} = s_2 - e_1
\]

\[
g_{after} = s_1 - e_2
\]

\[
g_{equal} = 2 \cdot \tau_{sim} - (|s_2 - s_1| + |e_1 - e_2|)
\]

where:

- `\tau_{sim}` is a learnable tolerance parameter for `EQUAL`
- `g_{before}`, `g_{after}`, `g_{equal}` are the three ordered temporal logits

These logits are then scaled by a learnable temperature:

\[
\tilde{g}_{ord} = \frac{g_{ord}}{\text{softplus}(\tau_{temp}) + 0.1}
\]

In vector form:

\[
\tilde{\mathbf{g}}_{ord} =
\begin{bmatrix}
\tilde{g}_{before} \\
\tilde{g}_{after} \\
\tilde{g}_{equal}
\end{bmatrix}
\]

### 2.5 VAGUE Logit in the Geometry Branch

Instead of creating a separate independent VAGUE head, the current design derives `VAGUE` from the combined strength of the three ordered classes:

\[
g_{vague} = \tau_{vague} - \log \sum_{k \in \{before, after, equal\}} e^{\tilde{g}_k}
\]

where:

- `\tau_{vague}` is a learnable parameter
- `logsumexp` measures the total evidence for the ordered classes

Intuition:

- if all ordered classes are weak, `g_{vague}` becomes larger,
- if one ordered class is strong, `g_{vague}` becomes smaller.

Finally:

\[
\text{logits}_{geo} = [\tilde{g}_{before}, \tilde{g}_{after}, \tilde{g}_{equal}, g_{vague}]
\]

or equivalently:

\[
\text{logits}_{geo} =
\begin{bmatrix}
g^{geo}_{before} \\
g^{geo}_{after} \\
g^{geo}_{equal} \\
g^{geo}_{vague}
\end{bmatrix}
\]

with:

\[
g^{geo}_{vague} = \tau_{vague} - \operatorname{LSE}(\tilde{g}_{before}, \tilde{g}_{after}, \tilde{g}_{equal})
\]

where `LSE` stands for `logsumexp`.

### 2.6 Fusion

The model converts both heads into probabilities:

\[
p_{sem} = \text{softmax}(\text{logits}_{sem})
\]

\[
p_{geo} = \text{softmax}(\text{logits}_{geo})
\]

Fusion is computed with a product-of-experts rule:

\[
p_{fuse,c}^{raw} = p_{sem,c} \cdot p_{geo,c}
\]

Then the fused distribution is normalized:

\[
p_{fuse,c} = \frac{p_{fuse,c}^{raw}}{\sum_j p_{fuse,j}^{raw}}
\]

This design gives high probability only to labels supported by both heads. If the semantic and geometry heads are confident on the same label, that label is amplified. If they are confident on different labels, their disagreement reduces the fused confidence.

`p_fuse` is the final output used by the main inference strategy.

---

## 3. Forward Pass

The current `GeoTREModel.forward(...)` returns:

- `s1, e1, s2, e2`
- `logits_sem`
- `logits_geo`
- `p_fuse`

Pseudocode:

```python
H = encoder(input_ids, attention_mask, token_type_ids).last_hidden_state

h1 = event_repr(H, e1_pos)
h2 = event_repr(H, e2_pos)
h_pair = pair_repr(h1, h2)

s1, e1 = time_head(concat(h1, h_pair))
s2, e2 = time_head(concat(h2, h_pair))

logits_sem = cls_head(concat(h1, h2, h_pair))
logits_geo = geo_logits(s1, e1, s2, e2)
p_fuse = fusion(logits_sem, logits_geo)

return s1, e1, s2, e2, logits_sem, logits_geo, p_fuse
```

---

## 4. Inference Strategies

The notebook supports three inference strategies:

### 4.1 Semantic Only

Prediction directly from the semantic head:

\[
\hat{y}_{sem} = \arg\max(\text{logits}_{sem})
\]

### 4.2 Geometry Only

Prediction directly from the geometry head:

\[
\hat{y}_{geo} = \arg\max(\text{logits}_{geo})
\]

### 4.3 Full GeoTRE (Fuse)

Prediction from the fused output:

\[
\hat{y}_{fuse} = \arg\max(p_{fuse})
\]

This is the main model output.

---

## 5. Loss Function

The redesigned objective avoids duplicated supervised classification losses. The model no longer uses separate `CE_sem` and `CE_geo` terms. Instead, only the final fused output is directly supervised by labels.

The auxiliary losses are used only to impose structure and agreement:

- `L_fuse`: supervised classification loss on the final fused prediction,
- `L_temporal`: structural loss on latent temporal coordinates,
- `L_agree`: Jensen-Shannon agreement loss between contextual and latent-time predictions.

### 5.1 Total Loss

\[
L = L_{fuse} + \lambda_{temp}L_{temporal} + \lambda_{agree}L_{agree}
\]

where:

- `L_fuse` is the only label-supervised classification loss,
- `L_temporal` enforces temporal logic directly on latent coordinates,
- `L_agree` encourages semantic and geometric distributions to agree.

This avoids sending the same label supervision through `p_fuse`, `logits_sem`, and `logits_geo` simultaneously.

### 5.2 Fusion Loss `L_fuse`

Because `p_fuse` is the final probability distribution, the main loss is:

\[
L_{fuse} = -\log p_{fuse}(y)
\]

In code:

```python
fuse_loss = F.nll_loss(torch.log(p_fuse.clamp_min(1e-8)), labels, weight=class_weights)
```

This is the only cross-entropy style objective that directly uses the gold label as a class target.

### 5.3 Temporal Structural Loss `L_temporal`

The temporal loss does not classify with another softmax. It directly constrains the latent coordinates according to the gold temporal relation.

For `BEFORE`, event 1 should end before event 2 starts:

\[
s_2 - e_1 \ge m
\]

The loss is:

\[
L_{before} = \max(0, m - (s_2 - e_1))
\]

For `AFTER`, event 2 should end before event 1 starts:

\[
s_1 - e_2 \ge m
\]

The loss is:

\[
L_{after} = \max(0, m - (s_1 - e_2))
\]

For `EQUAL`, both intervals should be close:

\[
|s_2 - s_1| + |e_1 - e_2| \le \epsilon
\]

The loss is:

\[
L_{equal} = \max(0, |s_2 - s_1| + |e_1 - e_2| - \epsilon)
\]

For `VAGUE`, no coordinate ordering constraint is applied. This is intentional: `VAGUE` should not be forced into a specific temporal geometry.

The batch-level temporal loss averages the active constraints:

\[
L_{temporal} = \operatorname{mean}(L_{before}, L_{after}, L_{equal})
\]

This term enforces temporal logic without adding a second classifier loss.

### 5.4 Agreement Loss `L_agree`

The agreement loss keeps the direct contextual prediction and the latent-time prediction consistent. It is implemented as Jensen-Shannon divergence:

\[
m = \frac{1}{2}(p_{sem}+p_{geo})
\]

\[
L_{agree} = \frac{1}{2}\Big(KL(p_{sem}\|m) + KL(p_{geo}\|m)\Big)
\]

with:

\[
p_{sem} = \text{softmax}(\text{logits}_{sem})
\]

\[
p_{geo} = \text{softmax}(\text{logits}_{geo})
\]

This term does not introduce another label target. It only encourages the contextual view and the latent-time view to remain compatible.

---

## 6. Loss Weights

The notebook now uses only:

- `lambda_struct` for the temporal structural loss,
- `lambda_agree` for semantic-geometry agreement.

The fused classification loss has implicit weight `1.0`:

\[
L = L_{fuse} + \lambda_{struct}L_{temporal} + \lambda_{agree}L_{agree}
\]

A practical default is:

\[
L = L_{fuse} + 0.3L_{temporal} + 0.1L_{agree}
\]

This keeps the final prediction objective dominant while still constraining the latent temporal space.

---

## 7. Why Remove `L_sem` and `L_geo`?

The previous formulation used:

\[
L_{fuse} + \lambda_{sem}L_{sem} + \lambda_{geo}L_{geo}
\]

This sends the same class label through three supervised objectives:

- one for the fused output,
- one for the semantic head,
- one for the geometry head.

That can make optimization redundant and can encourage the semantic and geometry branches to behave like independent classifiers.

The redesigned objective keeps only `L_fuse` as the supervised classification loss. The other losses now have different roles:

- `L_temporal` constrains the latent coordinates,
- `L_agree` keeps context and latent time consistent.

This better matches the intended interpretation: latent time is a structured projection of context, not a separate classifier competing with the semantic branch.

---

## 8. Short Summary

Current architecture:

- the shared encoder produces contextual hidden states,
- event representation extracts enriched vectors for both events,
- pair representation conditions latent time on both events,
- the geometry head derives temporal logits from pair-conditioned latent intervals,
- the semantic head predicts directly from contextual pair representation,
- the fusion module combines semantic and geometry outputs with product-of-experts fusion.

Loss:

\[
L = L_{fuse} + \lambda_{struct}L_{temporal} + \lambda_{agree}L_{agree}
\]

where:

- `L_fuse` is the only supervised classification objective,
- `L_temporal` imposes temporal logic on latent coordinates,
- `L_agree` encourages agreement between context and latent time.

---

## 9. Conclusion

This design is suitable for MATRES because it:

- does not rely only on direct classification,
- learns temporal structure through geometry,
- uses adaptive fusion to combine two complementary views,
- and supports independent evaluation of three inference strategies.

If needed, this document can be extended into:

- a compact thesis-ready version,
- an ASCII architecture diagram,
- or an ablation-study section for reporting results.

---

## 10. Unified Design for Datasets With and Without `VAGUE`

The same model family can be used on datasets that either include `VAGUE` or do not include it. The core architecture stays the same:

- shared encoder,
- semantic branch,
- geometry branch,
- product-of-experts fusion.

From a mathematical point of view, the fusion is always written in the same general form; the only difference is whether the heads produce 4 labels or 3 labels.

### 10.1 Dataset With `VAGUE`

If the dataset contains `VAGUE`, both branches operate over 4 labels:

\[
\text{logits}_{sem} \in \mathbb{R}^4, \quad \text{logits}_{geo} \in \mathbb{R}^4
\]

The geometry branch still derives `VAGUE` from the ordered logits:

\[
g^{geo}_{vague} = \tau_{vague} - \operatorname{LSE}(\tilde{g}_{before}, \tilde{g}_{after}, \tilde{g}_{equal})
\]

The fused probability is still:

\[
p_{fuse,c}^{raw} = p_{sem,c} \cdot p_{geo,c}
\]

\[
p_{fuse,c} = \frac{p_{fuse,c}^{raw}}{\sum_j p_{fuse,j}^{raw}}
\]

The only difference compared with a no-`VAGUE` dataset is that the semantic and geometry heads now operate in a 4-way label space.

This is the current design used in the notebook.

### 10.2 Dataset Without `VAGUE`

If the dataset does not contain `VAGUE`, the same fusion idea is kept, but the label space becomes 3-way:

\[
\text{logits}_{sem} \in \mathbb{R}^3, \quad \text{logits}_{geo} \in \mathbb{R}^3
\]

The ordered labels are:

- `BEFORE`
- `AFTER`
- `EQUAL`

The branch probabilities are:

\[
p_{sem} = \text{softmax}(\text{logits}_{sem})
\]

\[
p_{geo} = \text{softmax}(\text{logits}_{geo})
\]

The product-of-experts fusion is unchanged:

\[
p_{fuse,c}^{raw} = p_{sem,c} \cdot p_{geo,c}
\]

\[
p_{fuse,c} = \frac{p_{fuse,c}^{raw}}{\sum_j p_{fuse,j}^{raw}}
\]

In other words, the model becomes:

- the same encoder,
- the same semantic and geometry branches,
- the same product-of-experts fusion,
- but with a 3-way label space instead of 4-way.

### 10.3 Unified View

\[
L =
\begin{cases}
L_{fuse} + \lambda_{struct}L_{temporal} + \lambda_{agree}L_{agree}, & \text{if } VAGUE \text{ exists} \\
L_{fuse} + \lambda_{struct}L_{temporal} + \lambda_{agree}L_{agree}, & \text{if } VAGUE \text{ does not exist}
\end{cases}
\]

The structure of the loss stays the same. The practical difference is only:

- whether the output space is 4-way or 3-way,
- and whether the heads include the `VAGUE` class.

This makes the model easy to reuse across datasets with different label sets while keeping the formulation as similar as possible.
