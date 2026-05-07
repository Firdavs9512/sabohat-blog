---
title: "From Samantha to Sabohat — the start of the first project"
date: 2026-04-03
tag: log
excerpt: "A film, a name, and the idea of building an AI that works like a brain. This is the story of how the Sabohat project began — why GNN, why biological mechanisms, and why the name changed."
---

This blog is something I've been meaning to write for a long time. It's the first post about the **Sabohat** project. The project is still at a very early stage, but the road starts here.

## The beginning — Samantha

It all started with the film *Her* (2013). Or more precisely, with its AI character, **Samantha**. She wasn't your typical chatbot — she learned, she felt things, she changed over time. She "lived" alongside the person she talked to. I wanted to build something like that — not in a technical sense, but in spirit.

I gave the project a simple name: `samantha`. Made a folder, wrote a `pyproject.toml`, installed `torch` and `numpy`. That was it.

## Why GNN, why "brain-like"

The classic Transformer architecture is brilliant, but it has one problem: it's **static**. Once you've trained it, you're done. New data shows up? You have to retrain. And worst of all — if you fine-tune it for a new task, it **forgets** the old one. That's the phenomenon known as *catastrophic forgetting*.

The human brain doesn't work that way. A child goes through kindergarten, school, university — picking up new things at every stage, but holding on to what came before. So something is genuinely different in the brain. Whether you can find that "something different" and reproduce it on a small scale is the central question of this project.

That's why I built the architecture as a **Graph Neural Network** (GNN):

- **L1 — Character Graph** — character-level nodes
- **L2 — Morpheme Graph** — affixes and roots
- **L3 — Word Graph** — the word level
- **L4 — Semantic Graph** — the meaning level

Every node talks to the others through **dynamic connections** — much like neurons. When new information arrives, new nodes can be added (DEN — *Dynamic Expandable Networks*), unnecessary ones can be pruned, and the important ones are protected (TWP — *Topology-aware Weight Preserving*).

## A "living" model — Life Cycle

But there was one more thing I wanted to try: what if the model were "alive"? Meaning, not stuck in a single mode, but cycling through different states:

```
ACTIVE  →  IDLE  →  SLEEP  →  DREAM  →  ACTIVE
```

- **ACTIVE** — learns, responds
- **IDLE** — prunes connections it no longer needs
- **SLEEP** — consolidates what it has learned (like NREM sleep)
- **DREAM** — explores imaginary states looking for new patterns (like REM sleep)

This isn't sci-fi — these phenomena really do exist in neurobiology. **SWR** (Sharp-Wave Ripples) replay important memories during sleep. The **Hopfield Network** memory mechanism is a mathematical analog of the hippocampus. **MESU** (*Metaplasticity from Synaptic Uncertainty*) gives you Bayesian-style synapses, where each weight stores not just a value but also an **uncertainty**.

The goal is to bring all of this together into one system: small, but self-learning, and always running.

## And then — the name

The project moved along. The first prototypes got written (`brainnet_v2.py`, `brainnet_v3.py`, `pcn.py`, `dopamine_v3.py`). The dataset came together — kun.uz, daryo.uz, gazeta.uz, the Uzbek Wikipedia. My first training runs started up.

One day, mid-conversation with Claude, I just said: *"it would be nice to find an Uzbek name for the project — samantha is a good name, but it feels a little distant."* Claude suggested a few options. One of them was **Sabohat**.

Sabohat comes from Arabic, from *sabah* ("morning"). In Uzbek it's commonly used as a woman's name. It also carries the meaning of "beauty". But for me, the main meaning was something else — **a new dawn, a new beginning**.

When I really thought about this project, that's exactly what it is. A new approach, a new architecture, a new question: can AI learn on its own? Outside the classic *backprop + gradient descent* loop, *taking biological mechanisms into account*, can you build a small but living system?

Since that day I've been using both names:
- **Sabohat** — in Uzbek, the primary name
- **Samantha** — in English, for scientific writing

They mean the same thing. The primary name is Sabohat.

## Where things stand now

What's been done so far:

- Several versions of the `BrainNet` architecture (v2, v3) — the GNN stage
- `PCN` (Predictive Coding Network) — a local update rule
- `dopamine_v3` — modulating the learning rate with a "dopamine signal"
- `train_v5`, `train_v6` — various training pipelines
- A sizeable Uzbek-language dataset (kun.uz, daryo.uz, Wikipedia, books)

Over 200 experiments so far. Most of them failed, but each one taught me something.

**An important note:** the GNN-based architecture I described above is the *initial* stage of the project. After several experiments, I moved over to the **SNN** (*Spiking Neural Network*) approach — because it's even closer to the biological brain, more energy-efficient, and naturally lends itself to continual learning through spike timing. I'll write about that transition, and what changed afterwards, in a separate post.

It's a long road. The ROADMAP says 28 weeks; I know from experience it'll take roughly twice that. But the important thing is — it has started.

## What's next

I'll be writing this blog as I go — a journal of what worked, what didn't, and what I learned at each step. There'll be deep technical notes, and there'll be ordinary thoughts. All of it lives here.

Sabohat — a new dawn. It has begun.
