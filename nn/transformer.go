package nn

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"

	"github.com/HowardDunn/go-dominos/dominos"
	jsdonline "github.com/HowardDunn/jsd-online-game/game"
	log "github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/stat/distuv"
)

// EventObserver is implemented by players that want to observe game events
// for building temporal context (e.g., the sequence transformer).
type EventObserver interface {
	ObserveEvent(gameEvent *dominos.GameEvent)
	ResetHistory()
}

// SequenceTransformer is a full sequence transformer model for domino play.
// It processes variable-length sequences of game moves as tokens and outputs
// 56-dim action logits (28 cards x 2 sides).
type SequenceTransformer struct {
	dModel, nHeads, nLayers, dFF, maxSeqLen, outputDim int

	// Embeddings (learned)
	playerEmbed *embeddingTable
	cardEmbed   *embeddingTable
	sideEmbed   *embeddingTable
	modeEmbed   *embeddingTable

	// Positional encoding (fixed sinusoidal)
	posEncode []float64 // [maxSeqLen x dModel]

	// Transformer layers
	layers []*transformerLayer

	// Output head
	wOut []float64 // [outputDim x dModel]
	bOut []float64 // [outputDim]

	// Player interface
	*dominos.ComputerPlayer
	gameHistory []moveToken
	gameMode    string
	passMemory  [28]float64

	// Training config
	Epsilon          float64
	OutputActivation string  // "linear" (default)
	WeightDecay      float64 // per-step multiplicative decay (default 1.0 = no decay)
	TotalWins        int
	TotalWins2       int

	// PlayerRotation is the original player index used to rotate history
	// player IDs so they are relative to the current player's perspective.
	// Set this before calling TrainSupervised or Predict in supervised training.
	// Default 0 means no rotation (correct for reinforced/online play).
	PlayerRotation int
}

// NewSequenceTransformer creates a new transformer with the given architecture.
func NewSequenceTransformer(dModel, nHeads, nLayers, dFF, maxSeqLen, outputDim int) *SequenceTransformer {
	if dModel%nHeads != 0 {
		panic("dModel must be divisible by nHeads")
	}
	dHead := dModel / nHeads

	t := &SequenceTransformer{
		dModel:    dModel,
		nHeads:    nHeads,
		nLayers:   nLayers,
		dFF:       dFF,
		maxSeqLen: maxSeqLen,
		outputDim: outputDim,

		playerEmbed: newEmbeddingTable(playerVocab, dModel),
		cardEmbed:   newEmbeddingTable(cardVocab, dModel),
		sideEmbed:   newEmbeddingTable(sideVocab, dModel),
		modeEmbed:   newEmbeddingTable(modeVocab, dModel),

		posEncode: buildSinusoidalPositions(maxSeqLen, dModel),

		ComputerPlayer:   &dominos.ComputerPlayer{},
		gameMode:         "partner",
		OutputActivation: "linear",
		WeightDecay:      1.0,
	}

	// Initialize transformer layers
	t.layers = make([]*transformerLayer, nLayers)
	for l := 0; l < nLayers; l++ {
		t.layers[l] = newTransformerLayer(dModel, nHeads, dHead, dFF)
	}

	// Output head: [outputDim x dModel]
	t.wOut = make([]float64, outputDim*dModel)
	fillRandom(t.wOut, float64(dModel))
	t.bOut = make([]float64, outputDim)

	return t
}

// newTransformerLayer initializes a single transformer layer with Xavier init.
func newTransformerLayer(dModel, nHeads, dHead, dFF int) *transformerLayer {
	initMatrix := func(rows, cols int) *denseMatrix {
		m := newDenseMatrix(rows, cols)
		dist := distuv.Uniform{
			Min: -math.Sqrt(6.0 / float64(cols)),
			Max: math.Sqrt(6.0 / float64(cols)),
		}
		for i := range m.data {
			m.data[i] = dist.Rand()
		}
		return m
	}

	layer := &transformerLayer{
		nHeads: nHeads,
		dModel: dModel,
		dHead:  dHead,
		dFF:    dFF,

		wQ: initMatrix(dModel, dModel),
		wK: initMatrix(dModel, dModel),
		wV: initMatrix(dModel, dModel),
		wO: initMatrix(dModel, dModel),
		bQ: make([]float64, dModel),
		bK: make([]float64, dModel),
		bV: make([]float64, dModel),
		bO: make([]float64, dModel),

		ln1Gamma: make([]float64, dModel),
		ln1Beta:  make([]float64, dModel),
		ln2Gamma: make([]float64, dModel),
		ln2Beta:  make([]float64, dModel),

		ff1W: make([]float64, dFF*dModel),
		ff1B: make([]float64, dFF),
		ff2W: make([]float64, dModel*dFF),
		ff2B: make([]float64, dModel),
	}

	// Initialize layer norm gamma to 1
	for i := range layer.ln1Gamma {
		layer.ln1Gamma[i] = 1.0
	}
	for i := range layer.ln2Gamma {
		layer.ln2Gamma[i] = 1.0
	}

	// Initialize FFN weights
	fillRandom(layer.ff1W, float64(dModel))
	fillRandom(layer.ff2W, float64(dFF))

	return layer
}

// forwardCache holds all intermediate results needed for backward pass.
type forwardCache struct {
	tokens    []moveToken
	embedded  []float64 // [seqLen x dModel]
	layerCaches []*transformerLayerCache
	mask      []float64 // [seqLen x seqLen]
	seqLen    int
	lastTokenOut []float64 // [dModel] the last token's representation
}

// forward runs the full transformer forward pass.
// Returns 56-dim output logits and cache for backward.
func (t *SequenceTransformer) forward(tokens []moveToken) ([]float64, *forwardCache) {
	seqLen := len(tokens)
	dModel := t.dModel

	// Embed tokens
	embedded := t.embedTokens(tokens)

	// Build causal mask
	mask := buildCausalMask(seqLen)

	// Run through transformer layers
	x := embedded
	layerCaches := make([]*transformerLayerCache, t.nLayers)
	for l := 0; l < t.nLayers; l++ {
		var lc *transformerLayerCache
		x, lc = transformerLayerForward(x, t.layers[l], mask, seqLen)
		layerCaches[l] = lc
	}

	// Extract last token representation
	lastBase := (seqLen - 1) * dModel
	lastTokenOut := make([]float64, dModel)
	copy(lastTokenOut, x[lastBase:lastBase+dModel])

	// Output head: lastTokenOut @ wOut^T + bOut → [outputDim]
	// wOut is [outputDim x dModel]
	output := make([]float64, t.outputDim)
	for i := 0; i < t.outputDim; i++ {
		sum := t.bOut[i]
		for j := 0; j < dModel; j++ {
			sum += lastTokenOut[j] * t.wOut[i*dModel+j]
		}
		output[i] = sum
	}

	cache := &forwardCache{
		tokens:      tokens,
		embedded:    embedded,
		layerCaches: layerCaches,
		mask:        mask,
		seqLen:      seqLen,
		lastTokenOut: lastTokenOut,
	}

	return output, cache
}

func (*SequenceTransformer) GetPlayerType() string {
	return "transformerPlayer"
}

func (t *SequenceTransformer) GetNumHidden() int {
	return t.nLayers
}

// ObserveEvent implements EventObserver — called after each game event.
func (t *SequenceTransformer) ObserveEvent(gameEvent *dominos.GameEvent) {
	t.observeGameEvent(gameEvent)
	// Also update pass memory for compatibility
	if gameEvent.EventType == dominos.Passed {
		t.UpdatePassMemory(gameEvent)
	}
}

// ResetHistory implements EventObserver — called at round boundaries.
func (t *SequenceTransformer) ResetHistory() {
	t.gameHistory = nil
	t.passMemory = [28]float64{}
}

// UpdatePassMemory updates the pass memory tracking (same as JSDNN).
func (t *SequenceTransformer) UpdatePassMemory(gameEvent *dominos.GameEvent) {
	gameEvent = jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
	card := gameEvent.Card
	player := gameEvent.Player
	doms := dominos.GetCLIDominos()
	dom := doms[card]
	suit1, suit2 := dom.GetSuits()
	suit1Index := uint(player*7) + suit1
	suit2Index := uint(player*7) + suit2
	t.passMemory[suit1Index] = 1.0
	t.passMemory[suit2Index] = 1.0
}

// PlayCard implements the Player interface for game play.
func (t *SequenceTransformer) PlayCard(gameEvent *dominos.GameEvent, doms []dominos.Domino) (uint, dominos.BoardSide) {
	t.PlayerRotation = int(gameEvent.Player)
	gameEvent = jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)

	// Epsilon-greedy exploration
	if t.Epsilon > 0 && rand.Float64() < t.Epsilon {
		compatibleCards := []dominos.CardChoice{}
		for _, card := range gameEvent.PlayerHands[gameEvent.Player] {
			if card >= 28 {
				continue
			}
			suit1, suit2 := doms[card].GetSuits()
			if gameEvent.BoardState.CardPosed {
				if suit1 == gameEvent.BoardState.LeftSuit || suit2 == gameEvent.BoardState.LeftSuit {
					compatibleCards = append(compatibleCards, dominos.CardChoice{Card: card, Side: dominos.Left})
				}
				if suit1 == gameEvent.BoardState.RightSuit || suit2 == gameEvent.BoardState.RightSuit {
					compatibleCards = append(compatibleCards, dominos.CardChoice{Card: card, Side: dominos.Right})
				}
			}
		}
		if len(compatibleCards) > 0 {
			pick := compatibleCards[rand.Intn(len(compatibleCards))]
			return pick.Card, pick.Side
		}
	}

	if !gameEvent.BoardState.CardPosed {
		return t.ComputerPlayer.PlayCard(gameEvent, doms)
	}

	// Build sequence and run forward pass
	tokens := t.buildSequenceFromGameEvent(gameEvent)
	output, _ := t.forward(tokens)

	// Find best valid action
	maxConfidence := math.Inf(-1)
	bestCard := uint(0)
	bestSide := dominos.Left
	found := false

	contained := func(card uint) bool {
		hand := gameEvent.PlayerHands[gameEvent.Player]
		for _, h := range hand {
			if h == card {
				return true
			}
		}
		return false
	}

	compatible := func(card uint, side dominos.BoardSide) bool {
		suit1, suit2 := doms[card].GetSuits()
		if side == dominos.Left {
			return suit1 == gameEvent.BoardState.LeftSuit || suit2 == gameEvent.BoardState.LeftSuit
		}
		return suit1 == gameEvent.BoardState.RightSuit || suit2 == gameEvent.BoardState.RightSuit
	}

	for i, conf := range output {
		side := dominos.Left
		card := i
		if i >= 28 {
			side = dominos.Right
			card -= 28
		}
		if card >= 28 {
			continue
		}
		if conf > maxConfidence && contained(uint(card)) && compatible(uint(card), side) {
			maxConfidence = conf
			bestCard = uint(card)
			bestSide = side
			found = true
		}
	}

	if !found {
		log.Warn("Transformer: no valid card found, falling back to random")
		return t.ComputerPlayer.PlayCard(gameEvent, doms)
	}

	return bestCard, bestSide
}

// PoseCard chooses which card to pose (first play of the round).
func (t *SequenceTransformer) PoseCard(doms []dominos.Domino) uint {
	// Reuse JSDNN's pose strategy: prefer double in strongest suit
	hand := t.ComputerPlayer.GetHand()
	suitCount := [7]int{}
	for _, c := range hand {
		if int(c) >= len(doms) {
			continue
		}
		dom := doms[c]
		suit1, _ := dom.GetSuits()
		if dom.IsDouble() {
			suitCount[suit1]++
		}
	}
	for _, c := range hand {
		if int(c) >= len(doms) {
			continue
		}
		dom := doms[c]
		suit1, suit2 := dom.GetSuits()
		suitSum := suitCount[suit1] + suitCount[suit2]
		if !dom.IsDouble() && suitSum > 0 {
			suitCount[suit1]++
			suitCount[suit2]++
		}
	}
	maxSuit := 0
	maxCount := 0
	for i, c := range suitCount {
		if c >= maxCount {
			maxCount = c
			maxSuit = i
		}
	}
	for _, c := range hand {
		dom := doms[c]
		suit1, _ := dom.GetSuits()
		if dom.IsDouble() && suit1 == uint(maxSuit) {
			return c
		}
	}
	return t.ComputerPlayer.PoseCard(doms)
}

// Clone creates a deep copy of the transformer.
func (t *SequenceTransformer) Clone() *SequenceTransformer {
	clone := &SequenceTransformer{
		dModel:    t.dModel,
		nHeads:    t.nHeads,
		nLayers:   t.nLayers,
		dFF:       t.dFF,
		maxSeqLen: t.maxSeqLen,
		outputDim: t.outputDim,

		posEncode:        t.posEncode, // shared, immutable
		ComputerPlayer:   &dominos.ComputerPlayer{},
		gameMode:         t.gameMode,
		Epsilon:          t.Epsilon,
		OutputActivation: t.OutputActivation,
	}

	// Clone embeddings
	clone.playerEmbed = cloneEmbeddingTable(t.playerEmbed)
	clone.cardEmbed = cloneEmbeddingTable(t.cardEmbed)
	clone.sideEmbed = cloneEmbeddingTable(t.sideEmbed)
	clone.modeEmbed = cloneEmbeddingTable(t.modeEmbed)

	// Clone layers
	clone.layers = make([]*transformerLayer, t.nLayers)
	for l := range t.layers {
		clone.layers[l] = cloneTransformerLayer(t.layers[l])
	}

	// Clone output head
	clone.wOut = make([]float64, len(t.wOut))
	copy(clone.wOut, t.wOut)
	clone.bOut = make([]float64, len(t.bOut))
	copy(clone.bOut, t.bOut)

	return clone
}

func cloneEmbeddingTable(e *embeddingTable) *embeddingTable {
	data := make([]float64, len(e.data))
	copy(data, e.data)
	return &embeddingTable{data: data, vocabSize: e.vocabSize, dModel: e.dModel}
}

func cloneTransformerLayer(src *transformerLayer) *transformerLayer {
	dst := &transformerLayer{
		nHeads: src.nHeads,
		dModel: src.dModel,
		dHead:  src.dHead,
		dFF:    src.dFF,
	}

	cloneMatrix := func(m *denseMatrix) *denseMatrix {
		d := make([]float64, len(m.data))
		copy(d, m.data)
		return &denseMatrix{data: d, rows: m.rows, cols: m.cols}
	}
	cloneSlice := func(s []float64) []float64 {
		d := make([]float64, len(s))
		copy(d, s)
		return d
	}

	dst.wQ = cloneMatrix(src.wQ)
	dst.wK = cloneMatrix(src.wK)
	dst.wV = cloneMatrix(src.wV)
	dst.wO = cloneMatrix(src.wO)
	dst.bQ = cloneSlice(src.bQ)
	dst.bK = cloneSlice(src.bK)
	dst.bV = cloneSlice(src.bV)
	dst.bO = cloneSlice(src.bO)
	dst.ln1Gamma = cloneSlice(src.ln1Gamma)
	dst.ln1Beta = cloneSlice(src.ln1Beta)
	dst.ff1W = cloneSlice(src.ff1W)
	dst.ff1B = cloneSlice(src.ff1B)
	dst.ff2W = cloneSlice(src.ff2W)
	dst.ff2B = cloneSlice(src.ff2B)
	dst.ln2Gamma = cloneSlice(src.ln2Gamma)
	dst.ln2Beta = cloneSlice(src.ln2Beta)

	return dst
}

// Save writes the transformer to a file using GOB encoding.
func (t *SequenceTransformer) Save(fileName string) error {
	f, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)

	// Architecture
	if err := enc.Encode(t.dModel); err != nil {
		return err
	}
	if err := enc.Encode(t.nHeads); err != nil {
		return err
	}
	if err := enc.Encode(t.nLayers); err != nil {
		return err
	}
	if err := enc.Encode(t.dFF); err != nil {
		return err
	}
	if err := enc.Encode(t.maxSeqLen); err != nil {
		return err
	}
	if err := enc.Encode(t.outputDim); err != nil {
		return err
	}

	// Embeddings
	if err := enc.Encode(t.playerEmbed.data); err != nil {
		return err
	}
	if err := enc.Encode(t.cardEmbed.data); err != nil {
		return err
	}
	if err := enc.Encode(t.sideEmbed.data); err != nil {
		return err
	}
	if err := enc.Encode(t.modeEmbed.data); err != nil {
		return err
	}

	// Layers
	for _, layer := range t.layers {
		for _, data := range [][]float64{
			layer.wQ.data, layer.wK.data, layer.wV.data, layer.wO.data,
			layer.bQ, layer.bK, layer.bV, layer.bO,
			layer.ln1Gamma, layer.ln1Beta,
			layer.ff1W, layer.ff1B, layer.ff2W, layer.ff2B,
			layer.ln2Gamma, layer.ln2Beta,
		} {
			if err := enc.Encode(data); err != nil {
				return err
			}
		}
	}

	// Output head
	if err := enc.Encode(t.wOut); err != nil {
		return err
	}
	if err := enc.Encode(t.bOut); err != nil {
		return err
	}

	return nil
}

// Load reads the transformer from a file using GOB decoding.
func (t *SequenceTransformer) Load(fileName string) error {
	f, err := os.Open(fileName)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := gob.NewDecoder(f)

	// Architecture
	if err := dec.Decode(&t.dModel); err != nil {
		return err
	}
	if err := dec.Decode(&t.nHeads); err != nil {
		return err
	}
	if err := dec.Decode(&t.nLayers); err != nil {
		return err
	}
	if err := dec.Decode(&t.dFF); err != nil {
		return err
	}
	if err := dec.Decode(&t.maxSeqLen); err != nil {
		return err
	}
	if err := dec.Decode(&t.outputDim); err != nil {
		return err
	}

	dHead := t.dModel / t.nHeads

	// Embeddings
	if err := dec.Decode(&t.playerEmbed.data); err != nil {
		return err
	}
	if err := dec.Decode(&t.cardEmbed.data); err != nil {
		return err
	}
	if err := dec.Decode(&t.sideEmbed.data); err != nil {
		return err
	}
	if err := dec.Decode(&t.modeEmbed.data); err != nil {
		return err
	}

	// Rebuild positional encoding
	t.posEncode = buildSinusoidalPositions(t.maxSeqLen, t.dModel)

	// Layers
	t.layers = make([]*transformerLayer, t.nLayers)
	for l := 0; l < t.nLayers; l++ {
		layer := &transformerLayer{
			nHeads: t.nHeads,
			dModel: t.dModel,
			dHead:  dHead,
			dFF:    t.dFF,
		}
		// Decode weight matrices
		layer.wQ = newDenseMatrix(t.dModel, t.dModel)
		layer.wK = newDenseMatrix(t.dModel, t.dModel)
		layer.wV = newDenseMatrix(t.dModel, t.dModel)
		layer.wO = newDenseMatrix(t.dModel, t.dModel)

		for _, ptr := range []*[]float64{
			&layer.wQ.data, &layer.wK.data, &layer.wV.data, &layer.wO.data,
			&layer.bQ, &layer.bK, &layer.bV, &layer.bO,
			&layer.ln1Gamma, &layer.ln1Beta,
			&layer.ff1W, &layer.ff1B, &layer.ff2W, &layer.ff2B,
			&layer.ln2Gamma, &layer.ln2Beta,
		} {
			if err := dec.Decode(ptr); err != nil {
				return err
			}
		}
		t.layers[l] = layer
	}

	// Output head
	if err := dec.Decode(&t.wOut); err != nil {
		return err
	}
	if err := dec.Decode(&t.bOut); err != nil {
		return err
	}

	return nil
}

// ResetPassMemory resets pass memory (for compatibility with training loops).
func (t *SequenceTransformer) ResetPassMemory() {
	t.passMemory = [28]float64{}
}
