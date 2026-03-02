package nn

import (
	"math"

	"github.com/HowardDunn/go-dominos/dominos"
	jsdonline "github.com/HowardDunn/jsd-online-game/game"
)

// Token vocabulary constants
const (
	playerVocab = 5  // 0-3 players, 4=padding
	cardVocab   = 30 // 0-27 dominos, 28=PASS, 29=SEP
	sideVocab   = 4  // 0=Left, 1=Right, 2=Posed, 3=Pass
	modeVocab   = 2  // 0=cutthroat, 1=partner

	cardPASS  = 28
	cardSEP   = 29
	sidePosed = 2
	sidePass  = 3

	maxSeqLenDefault = 40 // 7 hand + 1 SEP + up to 31 moves + 1 query
)

// moveToken represents a single token in the game sequence.
type moveToken struct {
	playerID int // 0-3, or 4 for padding
	cardID   int // 0-27 dominos, 28=PASS, 29=SEP
	sideID   int // 0=Left, 1=Right, 2=Posed, 3=Pass
}

// embeddingTable holds learned embedding vectors.
type embeddingTable struct {
	data     []float64 // [vocabSize x dModel]
	vocabSize int
	dModel    int
}

func newEmbeddingTable(vocabSize, dModel int) *embeddingTable {
	data := make([]float64, vocabSize*dModel)
	fillRandom(data, float64(dModel))
	return &embeddingTable{data: data, vocabSize: vocabSize, dModel: dModel}
}

// lookup returns the embedding vector for the given index.
func (e *embeddingTable) lookup(idx int) []float64 {
	start := idx * e.dModel
	return e.data[start : start+e.dModel]
}

// buildSinusoidalPositions precomputes sinusoidal positional encodings [maxSeqLen x dModel].
func buildSinusoidalPositions(maxSeqLen, dModel int) []float64 {
	pe := make([]float64, maxSeqLen*dModel)
	for pos := 0; pos < maxSeqLen; pos++ {
		for i := 0; i < dModel; i++ {
			angle := float64(pos) / math.Pow(10000.0, float64(2*(i/2))/float64(dModel))
			if i%2 == 0 {
				pe[pos*dModel+i] = math.Sin(angle)
			} else {
				pe[pos*dModel+i] = math.Cos(angle)
			}
		}
	}
	return pe
}

// embedTokens converts a sequence of moveTokens to embedded representations.
// Returns [seqLen x dModel] embedding matrix.
func (t *SequenceTransformer) embedTokens(tokens []moveToken) []float64 {
	seqLen := len(tokens)
	dModel := t.dModel
	out := make([]float64, seqLen*dModel)

	modeIdx := 0
	if t.gameMode == "partner" {
		modeIdx = 1
	}
	modeEmb := t.modeEmbed.lookup(modeIdx)

	for i, tok := range tokens {
		base := i * dModel
		playerEmb := t.playerEmbed.lookup(tok.playerID)
		cardEmb := t.cardEmbed.lookup(tok.cardID)
		sideEmb := t.sideEmbed.lookup(tok.sideID)

		for j := 0; j < dModel; j++ {
			out[base+j] = playerEmb[j] + cardEmb[j] + sideEmb[j] + modeEmb[j] + t.posEncode[i*dModel+j]
		}
	}

	return out
}

// embedBackward accumulates gradients into embedding tables.
// dEmbed is [seqLen x dModel], tokens is the sequence used in forward.
func (t *SequenceTransformer) embedBackward(dEmbed []float64, tokens []moveToken, lr float64) {
	dModel := t.dModel

	modeIdx := 0
	if t.gameMode == "partner" {
		modeIdx = 1
	}

	for i, tok := range tokens {
		base := i * dModel
		// Update each embedding table with the gradient
		pBase := tok.playerID * dModel
		cBase := tok.cardID * dModel
		sBase := tok.sideID * dModel
		mBase := modeIdx * dModel

		for j := 0; j < dModel; j++ {
			grad := clipGrad(dEmbed[base+j])
			t.playerEmbed.data[pBase+j] = t.playerEmbed.data[pBase+j]*weightDecay + lr*grad
			t.cardEmbed.data[cBase+j] = t.cardEmbed.data[cBase+j]*weightDecay + lr*grad
			t.sideEmbed.data[sBase+j] = t.sideEmbed.data[sBase+j]*weightDecay + lr*grad
			t.modeEmbed.data[mBase+j] = t.modeEmbed.data[mBase+j]*weightDecay + lr*grad
		}
	}
}

// buildSequenceFromGameEvent constructs the input token sequence for a given game state.
// Sequence: [hand_card_1] ... [hand_card_N] [SEP] [history...] [QUERY]
func (t *SequenceTransformer) buildSequenceFromGameEvent(gameEvent *dominos.GameEvent) []moveToken {
	tokens := []moveToken{}

	// Hand tokens: each card in player's hand that is compatible
	for _, card := range gameEvent.PlayerHands[gameEvent.Player] {
		if card >= 28 {
			continue
		}
		tokens = append(tokens, moveToken{
			playerID: 0, // Self
			cardID:   int(card),
			sideID:   sidePosed,
		})
	}

	// SEP token
	tokens = append(tokens, moveToken{
		playerID: 0,
		cardID:   cardSEP,
		sideID:   sidePass,
	})

	// History tokens
	for _, histTok := range t.gameHistory {
		tokens = append(tokens, histTok)
	}

	// Query token
	tokens = append(tokens, moveToken{
		playerID: 0,
		cardID:   cardPASS,
		sideID:   sidePass,
	})

	// Truncate to maxSeqLen
	if len(tokens) > t.maxSeqLen {
		// Keep hand + SEP at start, truncate oldest history
		tokens = tokens[len(tokens)-t.maxSeqLen:]
	}

	return tokens
}

// observeGameEvent converts a game event to a moveToken and appends to history.
func (t *SequenceTransformer) observeGameEvent(gameEvent *dominos.GameEvent) {
	if gameEvent == nil {
		return
	}

	// Rotate to our perspective
	rotated := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
	player := rotated.Player

	switch gameEvent.EventType {
	case dominos.PlayedCard:
		sideID := 0 // Left
		if gameEvent.Side == dominos.Right {
			sideID = 1
		}
		t.gameHistory = append(t.gameHistory, moveToken{
			playerID: player,
			cardID:   int(gameEvent.Card),
			sideID:   sideID,
		})
	case dominos.PosedCard:
		t.gameHistory = append(t.gameHistory, moveToken{
			playerID: player,
			cardID:   int(gameEvent.Card),
			sideID:   sidePosed,
		})
	case dominos.Passed:
		t.gameHistory = append(t.gameHistory, moveToken{
			playerID: player,
			cardID:   cardPASS,
			sideID:   sidePass,
		})
	}
}
