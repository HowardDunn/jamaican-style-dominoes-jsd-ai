Context

 The current AI models (JSDNN with/without attention) see each game state as a static snapshot — a 126-dim vector encoding the current board, hand, and pass memory. While the attention-enhanced models outperform
 regular MLPs (especially in partner mode), they can't reason about temporal patterns: pass timing, opponent modeling, partner signaling, or how the game unfolded. Human players excel at this sequential reasoning.

 This plan adds a full sequence transformer as a completely separate model path. Each game move becomes a token in a sentence, and the model learns to reason about the sequence of play — just like the architecture
  described in myunderstanding.md.

 Architecture Summary

 - Input: Variable-length sequence of game move tokens (hand context + move history)
 - Token: Each move encoded as [PlayerID, CardID, SideID] → summed embeddings
 - Game mode: Partner/cutthroat encoded as a global embedding added to all tokens
 - Model: 2 transformer layers, 2 attention heads, dModel=64, dFF=128
 - Output: 56-dim action logits (28 cards x 2 sides) from the last token
 - Training: Manual forward + backward pass (consistent with existing attention.go patterns)
 - ~73K parameters — small enough for fast CPU training

 New Files

 All under jsd-3.0/jsd-ai/nn/:
 ┌───────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────┐
 │         File          │                                          Purpose                                          │
 ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
 │ transformer.go        │ SequenceTransformer struct, constructor, Player interface, Save/Load, Clone, forward pass │
 ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
 │ transformer_layers.go │ Multi-head attention, layer norm, FFN, causal mask — reusable building blocks             │
 ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
 │ transformer_embed.go  │ Embedding tables, positional encoding, tokenization from GameEvents                       │
 ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
 │ transformer_train.go  │ TrainReinforced, TrainReinforcedPartner, full backprop through transformer                │
 ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
 │ transformer_test.go   │ Unit tests for components + integration tests                                             │
 └───────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────┘
 Plus additions to main.go for the training loop.

 Token Design (transformer_embed.go)

 Each game event becomes a moveToken{playerID, cardID, sideID}:
 ┌──────────┬──────────────────────────────────┬───────────────────────────┐
 │ Feature  │            Vocabulary            │         Embedding         │
 ├──────────┼──────────────────────────────────┼───────────────────────────┤
 │ PlayerID │ 0-3 players, 4=padding           │ [5 x 64]                  │
 ├──────────┼──────────────────────────────────┼───────────────────────────┤
 │ CardID   │ 0-27 dominos, 28=PASS, 29=SEP    │ [30 x 64]                 │
 ├──────────┼──────────────────────────────────┼───────────────────────────┤
 │ SideID   │ 0=Left, 1=Right, 2=Posed, 3=Pass │ [4 x 64]                  │
 ├──────────┼──────────────────────────────────┼───────────────────────────┤
 │ GameMode │ 0=cutthroat, 1=partner           │ [2 x 64] (added globally) │
 └──────────┴──────────────────────────────────┴───────────────────────────┘
 Token embedding = playerEmbed[pid] + cardEmbed[cid] + sideEmbed[sid] + modeEmbed[mode] + posEncode[position]

 Positional encoding: sinusoidal (standard transformer formula), precomputed, not trained.

 Input Sequence Construction

 At each decision point, the input sequence is built as:

 [hand_card_1] [hand_card_2] ... [hand_card_N] [SEP] [move_1] [move_2] ... [move_T] [QUERY]

 1. Hand tokens (up to 7): Each card in hand → {player:0, card:idx, side:Posed}
 2. SEP token: {player:0, card:29, side:Pass} — boundary marker
 3. History tokens: Each PlayedCard/PosedCard/Passed event from the round
 4. Query token: {player:0, card:28, side:Pass} — "it's my turn, what do I play?"

 Max sequence length: 40 (7 hand + 1 SEP + up to 31 moves + 1 query).

 Output is taken from the query token's final representation → linear head → 56-dim logits.

 Transformer Layer Architecture (transformer_layers.go)

 Each layer follows the original "Attention is All You Need" post-norm pattern:

 x = input + MultiHeadAttention(input, causalMask)
 x = LayerNorm1(x)
 x = x + FFN(x)
 x = LayerNorm2(x)

 Multi-head attention (2 heads, dHead=32):
 - Q, K, V projections: [dModel x dModel] weights + bias
 - Split into heads, scaled dot-product attention with causal mask
 - Concatenate heads, output projection

 Causal mask: -inf for positions j > i, preventing attention to future moves.

 FFN: Two linear layers with ReLU: dModel → dFF → dModel (64 → 128 → 64).

 Layer norm: Standard gamma * (x - mean) / sqrt(var + eps) + beta.

 Reuse existing functions from attention.go: matMul2D, matMulTransB, matMulTransA, rowSoftmax, softmaxBackward, addBias, colSum.

 Struct Definition (transformer.go)

 type SequenceTransformer struct {
     dModel, nHeads, nLayers, dFF, maxSeqLen, outputDim int

     // Embeddings
     playerEmbed, cardEmbed, sideEmbed, modeEmbed *tensor.Dense
     posEncode *tensor.Dense  // sinusoidal, non-trainable

     // Transformer layers
     layers []*transformerLayer

     // Output head
     wOut, bOut *tensor.Dense  // [outputDim x dModel], [outputDim]

     // Player interface
     *dominos.ComputerPlayer
     gameHistory   []moveToken
     gameMode      string
     passMemory    [28]float64

     // Training config
     Epsilon          float64
     OutputActivation string
     TotalWins, TotalWins2 int
 }

 Player Interface & History Tracking (transformer.go)

 The transformer must accumulate game events to build its input sequence:

 - ObserveEvent(gameEvent): Called after each PlayedCard/Passed/PosedCard, appends to gameHistory
 - ResetHistory(): Called at round boundaries, clears history
 - PlayCard(gameEvent, doms): Builds sequence from hand + history, runs forward pass, returns best valid action

 To feed events to the transformer during gameplay, add an EventObserver interface check in playGame/playGamePartner:

 type EventObserver interface {
     ObserveEvent(gameEvent *dominos.GameEvent)
     ResetHistory()
 }

 After each game event in playGame, check if the player implements EventObserver and call it. This is backward-compatible — existing JSDNN players don't implement it and are unaffected.

 Training (transformer_train.go)

 Manual forward + backward pass (consistent with existing attention.go backprop patterns).

 The backward pass propagates gradients through:
 1. Output head (linear, same as existing final layer backprop)
 2. Each transformer layer in reverse: LayerNorm2 → FFN → LayerNorm1 → Multi-head Attention
 3. Embedding gradients (update learned embedding tables)

 Each component's backward pass is a well-documented formula:
 - LayerNorm backward: Existing attention.go doesn't have it but it's a standard formula
 - Multi-head attention backward: Extension of existing attentionBackward in attention.go to handle multiple heads
 - FFN backward: Same as MLP layer backward (existing pattern in player_nn.go)

 Reward computation: Reuse the same reinforced learning reward logic from ConvertCardChoiceToTensorReinforced (line 340) and the partner variant (line 450) in player_nn.go.

 Training function signature:
 func (t *SequenceTransformer) TrainReinforced(history []moveToken, gameEvent *GameEvent, learnRate float64, nextEvents [16]*GameEvent) (float64, error)
 func (t *SequenceTransformer) TrainReinforcedPartner(history []moveToken, gameEvent *GameEvent, learnRate float64, nextEvents [16]*GameEvent) (float64, error)

 Training Loop Integration (main.go)

 New function: trainReinforcedTransformer() — follows the same 3-phase pattern as trainReinforcedAttention() (line 979):

 - Phase 1: 4 transformer models in fixed positions, play games via playGamePartner, apply training
 - Phase 2: Shuffled positions for position-invariance
 - Phase 3: Benchmark each model vs 3 random players

 New helper: applyTrainingTransformer() — like applyTrainingPartner() but passes accumulated history with each training event.

 Model names: jasai_transformer1 through jasai_transformer4 — fully separate from existing models.

 Save separate CSV: elo_history_transformer.csv.

 Save/Load (transformer.go)

 GOB encoding (same pattern as JSDNN.Save/JSDNN.Load):

 1. Architecture params: dModel, nHeads, nLayers, dFF, maxSeqLen
 2. Embedding tables: playerEmbed, cardEmbed, sideEmbed, modeEmbed
 3. Per layer: wQ, wK, wV, wO, bQ, bK, bV, bO, ln1Gamma, ln1Beta, ff1W, ff1B, ff2W, ff2B, ln2Gamma, ln2Beta
 4. Output head: wOut, bOut

 Implementation Order

 1. transformer_layers.go — LayerNorm, multi-head attention, FFN, causal mask (building blocks + unit tests)
 2. transformer_embed.go — Token construction, embedding computation, sinusoidal positional encoding
 3. transformer.go — Struct, constructor, forward pass, Player interface, Save/Load, Clone
 4. transformer_train.go — Backprop through full transformer, TrainReinforced/TrainReinforcedPartner
 5. transformer_test.go — Component tests + integration test (forward pass shape, loss decreases over training)
 6. main.go — Add EventObserver to playGame/playGamePartner, add trainReinforcedTransformer()

 Key Reference Files

 - nn/player_nn.go — Player interface impl, reward computation, Save/Load, utility functions (fillRandom, clipGrad, relu, weightDecay)
 - nn/attention.go — Matrix ops to reuse (matMul2D, rowSoftmax, etc.), attention forward/backward pattern
 - nn/autograd.go — Gorgonia graph pattern (future extension)
 - main.go:979 — trainReinforcedAttention() as template for training loop
 - main.go:466 — playGamePartner() for partner game simulation
 - main.go:527 — applyTrainingPartner() for training event distribution

 Verification

 1. Unit tests: each transformer component (layer norm, multi-head attention, FFN) produces correct output shapes
 2. Forward pass test: construct transformer, feed synthetic sequence, verify 56-dim output
 3. Training test: verify loss decreases over repeated training on same sample
 4. Integration test: run trainReinforcedTransformer() for a few iterations, verify models save/load correctly and Elo tracking works
 5. Run with ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 env var (gorgonia dep)