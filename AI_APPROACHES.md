# Teaching AI to Play Jamaican Dominoes

## What is Jamaican Style Dominoes?

- 4 players, 28 domino tiles (double-six set)
- Two modes: **Cutthroat** (free-for-all) and **Partner** (teams of 2, seated across)
- Each player gets 7 tiles; the rest of the game is hidden
- Players take turns placing tiles matching the open suits on either end of the board
- If you can't play, you **pass** — and everyone sees that
- First player (or team) to empty their hand wins the round
- If nobody can play, it's a **block** — lowest pip count wins

---

## The Core AI Challenge: Partial Observability

In poker, you can't see other players' cards. Dominoes has the same problem.

You know:
- Your own hand (7 tiles)
- Every tile that's been played on the board
- Who passed and when (huge information signal)
- How many tiles each player has left

You **don't** know:
- What specific tiles the other 3 players hold
- What tiles they *could* play but chose not to

This makes dominoes a **Partially Observable Markov Decision Process (POMDP)** — the true game state is hidden, and you have to make decisions based on incomplete information.

---

## Three AI Approaches

We built three progressively more sophisticated approaches to tackle this problem:

| | **MLP (Snapshot)** | **Attention+MLP (Hybrid)** | **Sequence Transformer** |
|---|---|---|---|
| **Input** | 126-dim vector | 126-dim → 28 tokens | Variable-length move sequence |
| **Sees history?** | No (single frame) | No (single frame) | Yes (full game replay) |
| **Architecture** | Feed-forward MLP | Self-attention → MLP | Full transformer decoder |
| **Parameters** | ~16K | ~22K | ~73K |
| **Analogy** | Looking at a photo | Looking at a photo with a magnifying glass | Watching the whole movie |

---

## Approach 1: Pure MLP — The Snapshot Player

### How it works

The MLP sees a single **snapshot** of the current game state, encoded as a flat 126-dimensional vector:

```
[  Player Hand (28)  |  Board State (28)  |  Suit State (14)  |  Pass Memory (28)  |  Cards Remaining (28)  ]
        ↓                     ↓                    ↓                    ↓                      ↓
   Which tiles          Which tiles         What suits are      Who passed on         How many tiles
   am I holding?        are on board?       open L/R?           which suit?           does each player have?
```

This vector gets fed through a standard feed-forward neural network:

```
126 inputs → [256 hidden, ReLU] → [128 hidden, ReLU] → 56 outputs
```

The 56 outputs represent: **28 tiles x 2 sides (left/right)** = every possible action.

### Why it's a POMDP

A true **Markov Decision Process (MDP)** assumes you can see the full state. But we can't see other players' hands, so this is a **POMDP**. The 126-dim vector is our **observation** — a lossy compression of the true game state.

The pass memory is our best attempt at state recovery: if Player 2 passed when 3s were open, they probably don't have any 3s. But this is a heuristic — the network has to learn what it *really* means.

### The POMDP problem illustrated

Imagine two different game states that produce the **exact same 126-dim input**:

- **State A**: Opponent has [6-6, 5-5, 4-4] — very strong hand, you should play defensively
- **State B**: Opponent has [0-1, 0-2, 0-3] — weak hand, you can play aggressively

The MLP literally cannot distinguish these situations. It sees the same photo and must make the same decision for both.

### Training: Reinforcement Learning

The MLP learns by playing thousands of games and receiving **dense rewards** after each move:

- **+7.0**: Round win
- **-7.0**: Round loss
- **+1.0**: Opponent passes after your move (you blocked them)
- **-1.0**: You have to pass next turn
- **+3.0 bonus**: Playing doubles strategically
- **+0.3 per card**: Board control bonus (how many of your remaining tiles are still playable)
- **x1.5 multiplier**: Winning by a block (all players stuck)

These rewards encode **human-like domino strategy**: block opponents, maintain board control, play doubles wisely. The reward is normalized to [0, 1] and used to update the network weights via backpropagation.

### Strengths and weaknesses

**Strengths:**
- Fast inference (~microseconds per decision)
- Small model, trains quickly on CPU
- Learns basic strategy: play doubles early, block opponent suits, maintain playable options

**Weaknesses:**
- No temporal reasoning — every decision is independent
- Can't model opponent behavior over time
- Can't learn partner signaling patterns
- Same input = same output, regardless of how we got here

---

## Approach 2: Attention + MLP — The Magnifying Glass

### The insight

Instead of treating the 126-dim input as a flat blob, what if we recognized that it contains **28 domino tiles**, each with their own features? This is a structured input — let's let the network learn which tiles to focus on.

### How it works

The 126-dim vector gets **tokenized** into 28 tokens, one per domino tile:

```
Domino tile i → [in_hand?, on_board?, opponent_passed_suit?, cards_remaining?, suit_relevant?]
```

Each token is a 5-dimensional feature vector. Then self-attention lets each token "look at" every other token:

```
28 tokens × 5 features
       ↓
   Self-Attention (Q, K, V projections)
       ↓
  "Which other tiles should I pay attention to when deciding about this tile?"
       ↓
   28 tokens × dModel features
       ↓
      Flatten
       ↓
   MLP → 56 outputs
```

### What the attention learns

The attention mechanism learns **relationships between tiles**. For example:

- "I have the 6-5 and the board shows 6 on left — pay attention to all other 6-suit tiles"
- "The 3-3 double is in my hand and multiple 3s have been played — this might be a blocking opportunity"
- "All high doubles are already on the board — I can play more aggressively"

This is like looking at the same photo, but with a magnifying glass that knows where to focus.

### Same POMDP limitation

Critically, this approach **still sees only a single snapshot**. The attention helps the network process the snapshot better, but it still can't reason about:
- "Player 1 passed twice on 3s, then suddenly played a 3 — they must have been saving it"
- "My partner played a 5 early — they might be signaling they have more 5s"

---

## Approach 3: Sequence Transformer — Watching the Movie

### The fundamental shift

Instead of compressing the game into a single vector, we treat the entire game as a **sequence of events** — like a sentence in natural language.

Each game move becomes a **token**:

```
{Player: 2, Card: 6-4, Side: Left}  →  Token with 3 learned embeddings summed together
{Player: 3, Card: PASS}             →  Token (passes are informative!)
{Player: 0, Card: 5-5, Side: Posed} →  Token (first move of the round)
```

The full input sequence looks like:

```
[my hand]  [SEP]  [move history...]  [QUERY]
   ↓         ↓          ↓               ↓
 "Here's   "Now,    "Here's what     "Given all
  what I    let me    happened so      this, what
  hold"     think"    far..."          should I play?"
```

### Architecture

This is a **decoder-only transformer** (same family as GPT):

```
Token Sequence (up to 40 tokens)
        ↓
   Token Embedding = PlayerEmbed + CardEmbed + SideEmbed + ModeEmbed + PositionalEncoding
        ↓
   ┌─────────────────────────────────┐
   │  Transformer Layer 1            │
   │  ├─ Multi-Head Attention (2 heads, causal mask)
   │  ├─ Layer Normalization         │
   │  ├─ Feed-Forward Network        │
   │  └─ Layer Normalization         │
   └─────────────────────────────────┘
        ↓
   ┌─────────────────────────────────┐
   │  Transformer Layer 2            │
   │  (same structure)               │
   └─────────────────────────────────┘
        ↓
   Last Token (QUERY) Representation → Linear Head → 56-dim output
```

**Key design choices:**

| Component | Choice | Why |
|-----------|--------|-----|
| Attention heads | 2 | Balance between expressiveness and CPU cost |
| dModel | 64 | Embedding dimension — enough to capture domino relationships |
| dFF | 128 | Feed-forward expansion ratio of 2x |
| Layers | 2 | Deeper = better temporal reasoning, but more compute |
| Max sequence | 40 | 7 hand + 1 SEP + up to 31 moves + 1 query |
| Causal mask | Yes | Can't attend to future moves (prevents cheating) |
| Positional encoding | Sinusoidal | Standard fixed encoding for position awareness |

### The causal mask: no peeking at the future

The causal mask is crucial. When the transformer processes the sequence, each position can only attend to **itself and earlier positions**:

```
Token:    [Hand1] [Hand2] [SEP] [Move1] [Move2] [Move3] [QUERY]
Can see:   self    ≤2      ≤3    ≤4      ≤5      ≤6      all
```

This means the QUERY token can see everything — the full hand and complete move history — and must synthesize it into a decision.

### What the transformer can learn that the MLP cannot

1. **Pass timing patterns**: "Player 1 has passed on 3s three times in a row — they definitely don't have 3s, so playing a 3 is safe"

2. **Opponent modeling**: "Player 2 played aggressively early but is slowing down — they might be running out of options"

3. **Partner signaling** (in partner mode): "My partner played a 5 when they had other options — they might be telling me to play 5s"

4. **Game phase awareness**: "We're in the endgame (few tiles left) — I should switch from board control to trying to win by emptying my hand"

5. **Sequential dependencies**: "The reason the board shows 6-3 isn't just 'those suits are open' — it's because Player 1 played 6-4 left, then Player 3 played 3-4 right, and both of those moves carry information"

### The POMDP advantage

The sequence transformer doesn't fully solve the POMDP — it still can't see opponents' hands. But it dramatically **reduces the partial observability** by maintaining a memory of the entire game. Every pass, every play, and the order they happened in are all visible to the model.

Think of it this way:
- **MLP**: "The board shows 6 and 3. I should probably play a 6."
- **Transformer**: "The board shows 6 and 3. Player 1 played the 6-4 left in move 2, then passed on 3s in move 5. Player 3 played 3-4 right in move 3. My partner hasn't played a 6 yet but played a 3 in move 4. Given this sequence, I should play my 6-2 on the left to block Player 1."

---

## Training All Three

All three models use the same **dense reward function** — the same signal that encodes human domino strategy. The difference is in what information the model has access to when making decisions.

```
                    ┌──────────────┐
                    │  Game Engine │
                    │  (Go)        │
                    └──────┬───────┘
                           │ Game Events
                    ┌──────▼───────┐
                    │  Reward      │
                    │  Function    │  ← Dense rewards encoding human strategy:
                    │              │     opponent blocks, board control,
                    └──────┬───────┘     doubles, win/loss
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌─────────┐ ┌──────────┐ ┌───────────┐
         │  MLP    │ │ Attn+MLP │ │Transformer│
         │ (126-d) │ │ (28 tok) │ │ (sequence)│
         └─────────┘ └──────────┘ └───────────┘
```

Training loop for all models:
1. Play a full round of dominoes (4 players)
2. Collect all game events
3. For each move the AI made, compute reward from what happened next
4. Backpropagate the error signal through the network
5. Repeat thousands of times

The models train against a baseline **random computer AI** — one that follows the rules but picks randomly among legal moves. This gives a clear signal: if the model is learning, its win rate should climb well above 50%.

---

## Comparison Summary

| Feature | MLP | Attention+MLP | Transformer |
|---------|-----|---------------|-------------|
| Sees game history | No | No | Yes |
| Processes structure | No (flat vector) | Yes (28 tokens) | Yes (move sequence) |
| Temporal reasoning | None | None | Full sequence |
| Opponent modeling | Static (pass memory) | Static (pass memory) | Dynamic (from sequence) |
| Partner signaling | Cannot learn | Cannot learn | Can potentially learn |
| Parameters | ~16K | ~22K | ~73K |
| Inference speed | Fastest | Fast | Moderate |
| POMDP handling | Observation only | Better observation | Observation + memory |

---

## Following the Footsteps of Computer Vision

Our progression from MLP to attention+MLP to transformer isn't something we invented — it mirrors a well-documented evolution in the broader AI research community, particularly in computer vision.

### The same three-act story

**Act 1 — Pure feed-forward networks (CNNs / MLPs)**

In computer vision, Convolutional Neural Networks dominated from 2012 (AlexNet) through 2017. They processed images through stacked layers of local filters — powerful, but each filter could only see a small patch of the image. Long-range relationships ("the left eye should look like the right eye") were hard to learn.

Our MLP has the same limitation. It sees the 126-dim game state as a flat vector — no structure, no relationships between tiles, no notion that the 6-5 in my hand relates to the 6 on the left side of the board.

**Act 2 — Bolting attention onto the existing architecture (SENet, CBAM / our Attention+MLP)**

In 2017-2018, researchers started adding attention modules to CNNs:
- **Squeeze-and-Excitation Networks** (Hu et al., 2017) added channel attention — the network learns "which feature channels matter most for this image?" This won ImageNet 2017.
- **CBAM** (Woo et al., 2018) added both channel and spatial attention — "which channels matter?" + "where in the image matters?"

These were **incremental improvements**. The CNN was still doing the heavy lifting. Attention was a preprocessing step that helped the CNN focus on what mattered.

Our attention+MLP follows this exact pattern. We tokenize the 126-dim input into 28 domino tiles, run one round of self-attention so the network can learn tile relationships, then flatten and feed into the same MLP. The MLP still makes the decision — attention just gives it a better-organized input. Same snapshot, better magnifying glass.

**Act 3 — Replacing the backbone entirely with a transformer (ViT / our Sequence Transformer)**

In 2020, the Vision Transformer (ViT) from Google showed that you could **throw away the CNN entirely**. Cut the image into patches, treat each patch as a token, and run a standard transformer. It matched or beat CNNs on ImageNet, and the field hasn't looked back.

This is what our sequence transformer does. We don't bolt attention onto the MLP — we replace the entire architecture. The game becomes a sequence of tokens. Attention isn't a feature extractor; it **is** the model. Multi-head attention, layer normalization, residual connections, positional encoding, causal masking — the full transformer stack.

### Why this progression matters

Each step wasn't just "add more parameters." Each step changed **what the model could fundamentally represent**:

```
CNN / MLP                    →  Learns local patterns from static input
CNN + Attention / Attn+MLP   →  Learns which parts of static input relate to each other
Transformer / Seq. Transformer →  Learns patterns across sequences with full context
```

In computer vision, this progression unlocked image generation (DALL-E, Stable Diffusion). In NLP, it unlocked language understanding (BERT, GPT). In our domino AI, it unlocks **temporal reasoning** — the ability to understand not just where the game is, but how it got there and what that means for what comes next.

### The lesson

We didn't start with a transformer because we didn't need to. The MLP proved that a neural network *could* learn domino strategy. The attention+MLP proved that structured input processing *helped*. Each step validated the hypothesis that more context = better decisions, which justified the complexity of the next step.

This is the same reason the AI research community didn't jump straight from AlexNet to GPT-4. Each generation proved something that made the next generation possible — and each generation is still useful for the problems it's well-suited to.

---

## Key Takeaways

1. **Dominoes is a POMDP** — you never see the full state, so AI has to make decisions under uncertainty

2. **The MLP approach** treats each decision as independent — like looking at a photograph of the board and choosing. Simple, fast, but memoryless.

3. **The Attention+MLP approach** adds structure-awareness within a single snapshot — the network can learn which tiles relate to each other. Still no memory.

4. **The Sequence Transformer** is the first approach that can reason about **how the game unfolded** — not just where it is now. This is how humans actually play: by remembering who passed on what, when, and adjusting strategy accordingly.

5. **All three use the same reward signal** — dense rewards encoding human-like domino strategy. The architecture determines what information the model can extract from each game state.

6. **More parameters aren't always better** — the transformer has 4.5x more parameters than the MLP, but its advantage comes from seeing a fundamentally different (and richer) input representation, not just being bigger.

---

## What's Next

- Monitor transformer win% against baseline as training progresses
- Compare trained transformer vs trained MLP head-to-head
- Experiment with curriculum learning: start simple, increase game complexity
- Explore whether the transformer learns emergent partner communication strategies
