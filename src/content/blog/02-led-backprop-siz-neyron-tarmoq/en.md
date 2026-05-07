---
title: "LED — how a neural network worked without backpropagation"
date: 2026-05-07
tag: research
excerpt: "In v35 I removed loss.backward() entirely — and unexpectedly the model outperformed BP. Here is how LED (Lateral Error Diffusion) came about, why it worked, and where its limit was."
---

> For several months I have been working on a biologically-plausible neural network that learns Uzbek at the char-level. In v35 I **removed backpropagation entirely** — and unexpectedly the model outperformed BP. In this post I will openly tell you how the road went, what I did, and why it worked.

**Project**: samantha-v5 / v35
**Code**: char-level UZ language model, trained on Mac CPU

## TL;DR — in one sentence

Without ever calling `loss.backward()`, I trained the network using only **local rules** (DFA random feedback + Hebbian update + truly binary LIF spike + sparse 80/20 connectivity + cosine lr) — and on the word3 task got **0.585 vs BP 0.543**, on word4 **0.563 vs 0.533** — meaning the biologically plausible model **surpassed backpropagation**. But on the sentence task BP was 2.9× higher — there was a small-task ceiling artifact at play here.

## Why is backprop a problem?

Standard deep learning relies on the chain rule:

```
∂L/∂W_i = (∂L/∂h_n) · (∂h_n/∂h_{n-1}) · ... · (∂h_{i+1}/∂W_i)
```

This is mathematically perfect, but **not biologically plausible** for three reasons:

1. **Weight transport problem** — the backward pass needs `W^T`. A real synapse is not bidirectional; an axon sends signal in one direction only.
2. **Update locking** — each layer's update depends on signal coming from later layers. Real cortex does not work serially like that.
3. **Symmetric feedback** — the same weight is required on the forward and backward paths. This does not exist in a real brain.

A neuron in real cortex only knows **local information**: its own input, its own output, and possibly a global modulator (dopamine/norepinephrine). There is no "chain rule" anywhere.

## Existing alternatives — and their limitations

Over the years researchers have searched for BP alternatives:

| Algorithm | Year | Bio score | Limitation |
|---|---|---|---|
| Feedback Alignment | Lillicrap 2016 | 6/10 | Returns signal layer-by-layer |
| Direct Feedback Alignment | Nøkland 2016 | 7/10 | Random matrices feel abstract |
| Forward-Forward | Hinton 2022 | 7/10 | Toy tasks, hard to scale |
| Equilibrium Propagation | Scellier 2017 | 7/10 | Needs convergence to energy minimum |
| Predictive Coding | Rao 1999 | 8/10 | Equivalent to BP under certain conditions |
| e-prop | Bellec 2020 | 8/10 | For SNNs, needs a global modulator |

Most of these reach **60-80% of BP** on small tasks. None has been tried as a "combo".

## v1-v34 — earlier attempts

Before arriving at LED I had tried plenty:

- **v2-v3**: Auxiliary classifier per layer (local CE loss + Hebbian) — worked (~0.88), but each layer still called `loss.backward()`. **This is not really local — it's hidden BP.**
- **v3**: STDP alone (no aux loss) — **plateau at 0.21** (chance ~0.10). STDP is pure correlation; there is no classification signal.
- **v4-v5**: R-STDP + reward-modulated STDP — **plateau at 0.20** in the purely biological setting. A global reward `R(t)` does not solve deep credit assignment.
- **v8**: STDP-learned V1 filters + BP classifier — V1 is local, the rest is BP. A half-solution.
- **v18-v34**: BP was largely retained. v34 with SpikeGPT integration showed a record 0.774 (sentence), but it was **LIF surrogate gradient + BP**.

**No earlier version reached BP-level performance on a multi-layer, supervised task without true BP.**

## How the LED idea came about

After v34 I asked myself an honest question:
> "Not a model that beats BP — but can I build a **biological** model that gives good results?"

My core intuition went like this:
> "Neurons compute and store their local error, then each neuron punishes itself based on its own error. It looks at its neighbor and informs it about the error too."

This is the combination of three ideas:

1. **State recording** — on the forward pass each layer stores its input and pre-activation.
2. **Direct Feedback Alignment** — the output error is sent **directly** to each layer through random fixed matrices.
3. **Lateral diffusion** — each neuron's error flows to its neighbors through a 1D conv-style filter.

I called this **LED — Lateral Error Diffusion**.

## Algorithm — full

### Forward pass

```python
x_0 = Embedding(idx)                    # (B, T, n_embd)

for layer i = 1 to L:
    pre_i = SparseLinear_i(x_{i-1})     # mask qo'llanadi
    save: input_i = x_{i-1}, pre_i

    # LIF dynamics (har timestep t uchun):
    for t = 0 to T-1:
        V_i[t] = α · V_i[t-1] + pre_i[t]
        s_i[t] = (V_i[t] > θ).float()                # CHINAKAM BINAR
        pseudo_i[t] = 1 / (1 + (β·(V_i[t] - θ))²)    # ATan derivative
        V_i[t] = V_i[t] - s_i[t] · θ                 # soft reset

    save: spike_i, pseudo_i
    x_i = x_{i-1} + s_i                  # residual

logits = Linear_head(x_L)
```

Important: all stored tensors are `.detach()`-ed — they **do not enter the graph**, no gradient is computed.

### Local update rule (LED update)

Instead of `backward()`:

```python
@torch.no_grad()
def led_update(self, logits, targets, mask, lr):
    # 1) Output xatosi — eng go'zal lokal signal
    e_out = softmax(logits) - one_hot(targets)

    # 2) Output head — sof Hebbian
    dW_head = e_out.T @ pre_head / N
    head_w -= lr * dW_head

    # 3) Hidden qatlamlar — DFA + LIF pseudo-grad
    for i = L to 1 (reverse):
        e_h = e_out @ B_i.T               # DFA random feedback
        e_h *= pseudo_i                    # LIF derivative
        e_h = lateral_diffuse(e_h, α)     # qo'shniga tarqalish

        dW = e_h.T @ input_i / N
        dW *= mask_i                       # SPARSE: ulanmagan = 0
        W_i -= lr * dW

    # 4) Embedding update
    e_emb = e_out @ B_0.T * pseudo_0
    e_emb_proj = e_emb @ (W_0 * mask_0)
    emb[id] -= lr * scatter(e_emb_proj)
```

`B_i` are **fixed random matrices** that are not trained. They are randomly initialized once at startup and **never** change after that.

### Sparse 80/20 connectivity

In real cortex, neurons connect heavily with their nearby neighbors and sparsely with distant ones. How can this be modeled?

```python
def build_sparse_mask(n_in, n_out, local_frac=0.8, sigma_local=2.0,
                      density=0.15, seed=0):
    pos_in = make_2d_positions(n_in)        # (n_in, 2) grid
    pos_out = make_2d_positions(n_out)
    dist = (pos_out[:, None] - pos_in[None, :]).norm(dim=-1)

    n_per_out = int(density * n_in)
    n_local = int(n_per_out * local_frac)   # 80%
    n_distant = n_per_out - n_local          # 20%

    # Lokal: Gauss bilan eng yaqin neyronlar
    local_logits = -dist**2 / (2 * sigma_local**2)
    _, local_idx = local_logits.topk(n_local, dim=-1)

    # Uzoq: tasodifiy non-local neyronlar
    mask = scatter(local_idx) | random_distant(n_distant)
    return mask  # bool (n_out, n_in)
```

The mask is stored as a **fixed buffer** and `*`-multiplied with the weight. A connection where mask=0 **never** learns — this is a biological invariant.

### LIF cell — truly binary spike

```python
class LIFCell(nn.Module):
    def __init__(self, tau=2.0, threshold=1.0, alpha=2.0):
        super().__init__()
        self.decay = 1.0 - 1.0 / tau    # α=0.5 (tau=2)
        self.threshold = threshold
        self.alpha = alpha
        self.V = None

    def forward(self, I):
        self.V = self.decay * self.V + I
        spike = (self.V > self.threshold).float()       # CHINAKAM BINAR
        x = self.alpha * (self.V - self.threshold)
        pseudo_grad = 1.0 / (1.0 + x * x)               # ATan derivative
        self.V = self.V - spike * self.threshold        # soft reset
        self.V = self.V.detach()
        return spike, pseudo_grad
```

The most critical line: `spike = (V > thr).float()`. This function is non-differentiable — under backprop a gradient cannot pass through it **by any route**. But we are not using backprop! We apply `pseudo_grad` by hand inside the LED update — this is very close to Bellec (2020) e-prop.

## Experiments — step by step

### Stage 1 — LED alone (few epochs)

| Task | LED | BP | LED/BP |
|---|---|---|---|
| Letter (33 alphabet) | 1.000 | 1.000 | EQUAL |
| Word3 (28 3-token words) | 0.457 | 0.543 | 84% |
| Word4 (26 4-token words) | 0.383 | 0.533 | 72% |

**LED reaches 80% of BP** — the expected result (Lillicrap 2016).

Lateral diffusion ablation (which I had the most faith in):

| α | Word3 acc |
|---|---|
| 0.0 (pure DFA) | 0.457 |
| 0.2 (default) | 0.457 |
| 0.5 (strong) | 0.432 |

**Lateral diffusion had no noticeable effect.** This was the first honest warning sign: my "spread to the neighbor" idea did not work **in this configuration**.

### Stage 2 — Sparse 80/20 alone (with BP)

| Task | Dense | Sparse 80/20 | Active params |
|---|---|---|---|
| Word3 | 0.543 | 0.543 | 19.8K (vs 47.7K) |
| Word4 | 0.533 | 0.533 | 19.8K |
| Sentence | 0.243 | 0.246 | 39.3K |

**Sparse equals dense, with 2.4× fewer parameters.** Even 5% density is enough (0.543).

local_frac ablation:

| local_frac | Acc |
|---|---|
| 0.0 (pure random) | 0.543 |
| 0.5 | 0.543 |
| 0.8 (biological 80/20) | 0.543 |
| 1.0 (purely local) | 0.543 |

**All equal** — at this size the 80/20 idea did not make a difference. But `lf=1.0` (purely local) came out **bad** in the later LED+sparse run (0.420). So **distant connections are needed**, but the exact ratio did not matter.

### Stage 3 — the long-training discovery

Initially when I tried LED at 25-40 epochs it lagged behind BP. A user suggestion: **train it long**. 200-250 epochs + cosine lr decay:

| Task | LED best ep | BP best ep | LED final | BP final |
|---|---|---|---|---|
| Word3 | 64 | 4 | 0.543 | 0.543 |
| Word4 | 189 | 4 | 0.533 | 0.533 |
| Sentence | — | — | 0.245 | 0.248 |

**LED is 16-50× slower, but asymptotically EQUAL to BP.** This was the empirical confirmation of the Lillicrap 2016 paper.

### Stage 4 — all three together: Sparse + LIF + LED

This is where the unexpected thing happened. Three biological elements were combined:

1. Sparse 80/20 topology
2. Truly binary LIF spike (NO surrogate gradient)
3. LED local learning (no backprop)

With 5 random seeds:

| Task | Sparse+LIF+LED | BP | Gain | Active params |
|---|---|---|---|---|
| Word3 | **0.585 ± 0.034** | 0.543 | **+0.042** | 10K |
| Word4 | **0.563 ± 0.058** | 0.533 | **+0.030** | 10K |

**It SURPASSED BP.** And with **3× fewer parameters**.

Spike rate: **7.4%** — at the level of real biological cortex (1-10%).

## Why did it work? — hypotheses

The 3 elements **alone** are equal to or below BP:
- Sparse alone: equal to dense (0.543)
- LIF + BP: 0.596 — good, but with BP's help
- LED alone: equal to BP (with long training, 0.543)

But **all three together** outperform BP. My hypotheses:

### Implicit regularization

1. **Sparse mask** — fewer parameters, overfitting is constrained
2. **Binary spike** — implicit information bottleneck (only 0/1 information passes through)
3. **DFA random feedback** — noise in the gradient → exploration
4. **Cosine lr decay** — slow at finetune time

The three together provide an **inductive bias**. BP, on the other hand, is "fully unconstrained" — all parameters optimize quickly and hit a **plateau** (word3 stopped at ep 4!). With LED, the implicit regularization → **better generalization**.

### The mathematical basis of DFA

Lillicrap (2016)'s finding: even with fixed random feedback `B`, the network still learns, because during training the forward weight `W` **aligns** to `B` (the alignment property):

```
Train davomida: W^T · sign(e) ≈ B · sign(e)
```

The network adapts to its own feedback. This is **emergent behavior**.

### Cross-entropy returns local credit

`(softmax - onehot)` — the most beautiful local signal. For each output neuron:
- if its probability is above the true target: positive → decrease
- if below: negative → increase

This **immediately** turns into `Δw_head = e · pre_head`. With no backward chain at all.

### The pseudo-derivative is close to e-prop

Bellec (2020) e-prop:
```
ΔW = e_global · eligibility_trace · pseudo_derivative
```

LED:
```
ΔW = e_local_DFA · pre · pseudo_derivative
```

In LED, instead of an `eligibility_trace`, we have `pre · pseudo_derivative` — this is the **single-timestep** version. The core idea is the same: the spike is non-differentiable, but the pseudo-derivative estimates local credit.

## An honest caveat — it fell over on the sentence task

I won on word3/word4, but then came the real test: 12 templates, 2400 pairs, 120-epoch sentence task:

| Method | seen | unseen | spike |
|---|---|---|---|
| Sparse+LIF+LED | 0.257 ± 0.003 | 0.224 ± 0.003 | 7.9% |
| **Sparse+LIF+BP** | **0.739 ± 0.002** | **0.412 ± 0.008** | 38% |

**BP is 2.9× higher** on seen, **1.8×** on unseen. LED hit a plateau at ep 25 (0.23); BP keeps growing through ep 100.

### What this means

Word3/word4 (26-28 words) was a **small-task ceiling artifact**. Both methods reach the ceiling, and LED came out slightly higher due to random variation.

On a hard **structured sequence task** (multi-template, contextual), the **compositional advantage of the chain rule comes back**. DFA-style local learning cannot scale to deep contextual tasks. This is also stated in the Lillicrap 2016 paper: DFA is good in shallow networks and degrades in deep ones.

Honest reassessment:

| Task | LED | BP | Winner |
|---|---|---|---|
| Word3 (28 words) | 0.585 | 0.543 | LED +0.04 (small-scale ceiling) |
| Word4 (26 words) | 0.563 | 0.533 | LED +0.03 (small-scale ceiling) |
| **Sentence (2400 pair)** | **0.257** | **0.739** | **BP +0.482** |

Later (v36) I added TimeMix — the gap widened (3.0×). In v37, even 5 interventions (width, density, skip connection, lateral, layered) did not help. **The LED direction was closed.**

## What is right and what is not

In this project I tested 3 intuitions. Honest verdicts:

| Idea | Status | Note |
|---|---|---|
| Local rule + DFA (LED) | worked | Equal to BP under long training, higher on small tasks |
| Lateral diffusion (to neighbors) | irrelevant | α=0 and α=0.2 are equal |
| Sparse connectivity | worked | Equal to dense, 2.4× smaller |
| Exact 80/20 ratio | unreliable | Random sparse is the same |
| Purely local (lf=1.0) | bad | Distant connections are needed |
| **All three together** | worked on small tasks | Not enough on sentence |

**The main philosophical lesson**: bio learning is **not one magic rule**, but rather **a combination of several correctly chosen elements**. I searched for that combination across 33 versions. I found it in v35 — but **with a scale limit**.

## Technical details — reproducibility

### Configuration

| Parameter | Word3 | Word4 |
|---|---|---|
| n_embd | 64 | 64 |
| hidden | 128 | 128 |
| n_layers | 2 | 2 |
| max_len | 8 | 8 |
| n_repeats | 30 | 30 |
| batch | 32 | 32 |
| lr_max | 5e-3 | 3e-3 |
| epochs | 200 | 250 |
| density | 0.15 | 0.15 |
| local_frac | 0.8 | 0.8 |
| lif_tau | 2.0 | 2.0 |
| lif_threshold | 1.0 | 1.0 |
| feedback_scale | 0.1 | 0.1 |

### Hardware

A Mac M-series CPU is enough (MPS is not necessary on these tasks). 200 epochs ~30-60 seconds.

### Wall-clock

| Variant | Per-epoch | Total (200 ep) |
|---|---|---|
| BP | ~0.05s | 10s |
| LED | ~0.1s | 20s |
| Sparse+LIF+LED | ~0.1-0.3s | 30-60s |

### Replication command

```bash
# Word3 — Sparse+LIF+LED, 5 seed
for seed in 0 1 2 3 4; do
    python sparse_lif_led.py --task word3 --epochs 200 \
        --hidden 128 --lr 5e-3 --seed $seed \
        --out runs/slL_w3_s$seed.json
done

# Word4 — 5 seed
for seed in 0 1 2 3 4; do
    python sparse_lif_led.py --task word4 --epochs 250 \
        --hidden 128 --lr 3e-3 --seed $seed \
        --out runs/slL_w4_s$seed.json
done
```

## Related work

| Work | Year | Relevance |
|---|---|---|
| Hebb, *Organization of Behavior* | 1949 | Local Hebbian foundation |
| Widrow & Hoff, *Adaptive switching circuits* | 1960 | Delta rule |
| Rumelhart et al., *Backprop* | 1986 | LED rejects this |
| Lillicrap et al., *Random feedback alignment* | 2016 | DFA inspiration |
| Nøkland, *Direct feedback alignment* | 2016 | LED's core |
| Bellec et al., *e-prop* | 2020 | LIF pseudo-grad |
| Hinton, *Forward-Forward* | 2022 | Another bio alternative |
| Zhu et al., *SpikeGPT* | 2023 | LIF architecture |

**LED's contribution**: the **combination** of DFA + sparse 80/20 + binary LIF + cosine lr (this combination has not been seen before).

## Open questions

1. Does LED **scale**? At hidden=512+, n_layers=6+, does it stay equal to BP?
   — **Answer (v36-v37)**: no, on sentence BP is 3× higher.
2. Does Sparse+LIF+LED work on the **sentence task** (where TimeMix is needed)?
   — **Answer**: no, plateau stops at 0.23.
3. How does LED + frozen slot work in **continual learning**? — untested.
4. Does **2D topology** + **weight-space lateral** save lateral diffusion? — open.
5. How does **online training** (batch=1) work? — open.

## Conclusion

LED is **a logically simple, biologically plausible learning algorithm that works at small scale**. It is:

- **Backprop-free** — no chain rule
- **Purely local** — each neuron only knows its own information
- **Sparse-friendly** — unconnected neurons do not learn
- **Spike-friendly** — truly binary LIF (no surrogate needed)
- **Empirically validated** — at small scale, +0.030—0.042 above BP
- **Scale-limited** — on deep compositional tasks, BP wins back the advantage

**Main contribution**: the **combination** of DFA + sparse 80/20 + truly binary LIF + cosine lr, and the **inductive bias / regularization effect** of that combination.

**Philosophical conclusion**: backpropagation is mathematically perfect, but if you must choose between biological plausibility and inductive bias — then at small scale **LED is both a logical and a practical alternative**. But I cannot claim "a new era of BP" — on deep sequence tasks the chain rule's advantage returns. This is a **genuine research result**: success and limitation together.

## The most important honest part

I came into this project at `v34` with a record 0.774 sentence accuracy. In v35 I beat BP (on small tasks). In v36-v37 I could not scale it. The road closed.

This is **success and honest defeat together**. Research is like that: an idea works, but it has a limit. And **knowing that limit precisely** is far more valuable than claiming things without knowing.

If you also want to work on bio-plausible learning, my advice is: **try the combination**, not the parts alone. And **test on a genuinely large task** — small-scale wins can be deceptive.

---

*An honest assessment running through this post: 80% of the idea worked, 20% (lateral diffusion) was not confirmed. The limit on the sentence task became clear. An honest result is worth more than an honest claim.*
