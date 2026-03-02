package nn

import (
	"errors"
	"math"

	"github.com/HowardDunn/go-dominos/dominos"
	jsdonline "github.com/HowardDunn/jsd-online-game/game"
)

// backward runs the full backward pass through the transformer.
// outputGrad is [outputDim] — the gradient of the loss w.r.t. the output logits.
// Returns embedding gradient for embedBackward.
func (t *SequenceTransformer) backward(outputGrad []float64, cache *forwardCache, lr float64) {
	dModel := t.dModel
	seqLen := cache.seqLen

	// Output head backward: output = lastTokenOut @ wOut^T + bOut
	// dLastTokenOut = outputGrad @ wOut → [dModel]
	dLastTokenOut := make([]float64, dModel)
	for j := 0; j < dModel; j++ {
		sum := 0.0
		for i := 0; i < t.outputDim; i++ {
			sum += outputGrad[i] * t.wOut[i*dModel+j]
		}
		dLastTokenOut[j] = sum
	}

	// dwOut = outputGrad (outer) lastTokenOut → [outputDim x dModel]
	// bOut grad = outputGrad
	for i := 0; i < t.outputDim; i++ {
		grad := clipGrad(outputGrad[i])
		t.bOut[i] += lr * grad
		for j := 0; j < dModel; j++ {
			dwOut := clipGrad(outputGrad[i] * cache.lastTokenOut[j])
			t.wOut[i*dModel+j] = t.wOut[i*dModel+j]*weightDecay + lr*dwOut
		}
	}

	// Scatter dLastTokenOut back into full sequence gradient
	// Only the last token position has gradient
	dx := make([]float64, seqLen*dModel)
	lastBase := (seqLen - 1) * dModel
	copy(dx[lastBase:lastBase+dModel], dLastTokenOut)

	// Backward through transformer layers (reverse order)
	for l := t.nLayers - 1; l >= 0; l-- {
		var grads *transformerLayerGrads
		dx, grads = transformerLayerBackward(dx, cache.layerCaches[l], t.layers[l], cache.mask, seqLen)
		applyLayerGradients(t.layers[l], grads, lr)
	}

	// Backward through embeddings
	t.embedBackward(dx, cache.tokens, lr)
}

// applyLayerGradients updates a transformer layer's weights using computed gradients.
func applyLayerGradients(layer *transformerLayer, grads *transformerLayerGrads, lr float64) {
	updateWeightSlice := func(w []float64, dw []float64) {
		for i := range w {
			grad := clipGrad(dw[i])
			w[i] = w[i]*weightDecay + lr*grad
		}
	}

	updateBiasSlice := func(b []float64, db []float64) {
		for i := range b {
			grad := clipGrad(db[i])
			b[i] += lr * grad
		}
	}

	// MHA weights
	updateWeightSlice(layer.wQ.data, grads.dwQ)
	updateWeightSlice(layer.wK.data, grads.dwK)
	updateWeightSlice(layer.wV.data, grads.dwV)
	updateWeightSlice(layer.wO.data, grads.dwO)
	updateBiasSlice(layer.bQ, grads.dbQ)
	updateBiasSlice(layer.bK, grads.dbK)
	updateBiasSlice(layer.bV, grads.dbV)
	updateBiasSlice(layer.bO, grads.dbO)

	// Layer norm 1
	updateBiasSlice(layer.ln1Gamma, grads.dln1Gamma)
	updateBiasSlice(layer.ln1Beta, grads.dln1Beta)

	// FFN
	updateWeightSlice(layer.ff1W, grads.dff1W)
	updateBiasSlice(layer.ff1B, grads.dff1B)
	updateWeightSlice(layer.ff2W, grads.dff2W)
	updateBiasSlice(layer.ff2B, grads.dff2B)

	// Layer norm 2
	updateBiasSlice(layer.ln2Gamma, grads.dln2Gamma)
	updateBiasSlice(layer.ln2Beta, grads.dln2Beta)
}

// TrainReinforced trains the transformer on a single game event using reinforced learning.
// history is the accumulated game history tokens at this point.
// Returns the error signal magnitude.
func (t *SequenceTransformer) TrainReinforced(gameEvent *dominos.GameEvent, learnRate float64, nextEvents [16]*dominos.GameEvent) (float64, error) {
	if gameEvent.EventType != dominos.PlayedCard && gameEvent.EventType != dominos.PosedCard {
		return 0.0, errors.New("invalid game event to train with")
	}

	// Build target using existing reward computation
	target := t.computeRewardCutthroat(gameEvent, nextEvents)

	return t.trainOnEvent(gameEvent, target, learnRate)
}

// TrainReinforcedPartner trains using partner-aware rewards.
func (t *SequenceTransformer) TrainReinforcedPartner(gameEvent *dominos.GameEvent, learnRate float64, nextEvents [16]*dominos.GameEvent) (float64, error) {
	if gameEvent.EventType != dominos.PlayedCard && gameEvent.EventType != dominos.PosedCard {
		return 0.0, errors.New("invalid game event to train with")
	}

	target := t.computeRewardPartner(gameEvent, nextEvents)

	return t.trainOnEvent(gameEvent, target, learnRate)
}

// trainOnEvent runs forward and backward pass for a single training event.
func (t *SequenceTransformer) trainOnEvent(gameEvent *dominos.GameEvent, target [56]float64, learnRate float64) (float64, error) {
	// Build the token sequence
	tokens := t.buildSequenceFromGameEvent(gameEvent)
	if len(tokens) == 0 {
		return 0.0, nil
	}

	// Forward pass
	output, cache := t.forward(tokens)

	// Compute output error (only at the action index, masked)
	actionIndex := int(gameEvent.Card)
	if gameEvent.Side == dominos.Right {
		actionIndex += 28
	}

	outputGrad := make([]float64, t.outputDim)
	cost := 0.0

	// Only compute gradient at the action taken (masked update)
	err := target[actionIndex] - output[actionIndex]
	err = clipValue(err)
	outputGrad[actionIndex] = err
	cost = err

	// Backward pass
	t.backward(outputGrad, cache, learnRate)

	return cost, nil
}

// computeRewardCutthroat computes reward for cutthroat mode.
// Reuses the same logic as JSDNN.ConvertCardChoiceToTensorReinforced.
func (t *SequenceTransformer) computeRewardCutthroat(gameEvent *dominos.GameEvent, nextGameEvents [16]*dominos.GameEvent) [56]float64 {
	cardChoice := [56]float64{}
	index := gameEvent.Card
	suitPlayed := gameEvent.BoardState.LeftSuit
	if gameEvent.Side == dominos.Right {
		index = index + 28
		suitPlayed = gameEvent.BoardState.RightSuit
	}
	reward := 0.0
	chainBroken := false
	hasHardEnd := false
	boardSuitCount := gameEvent.BoardState.CountSuit(suitPlayed)
	handSuitCount := t.CountSuit(suitPlayed, dominos.GetCLIDominos())
	isDouble := dominos.GetCLIDominos()[gameEvent.Card].IsDouble()
	if boardSuitCount+handSuitCount == 6 {
		hasHardEnd = true
	}
	won := false
	wonByBlock := true
	for i, nextEvent := range nextGameEvents {
		if nextEvent == nil || i > 4 {
			break
		}
		switch nextEvent.EventType {
		case dominos.Passed:
			if nextEvent.Player == 0 {
				reward = -1.0
			} else if !chainBroken {
				reward = reward + 1.0
			}
		case dominos.PlayedCard:
			if nextEvent.Player != 0 {
				chainBroken = true
			}
		case dominos.RoundWin:
			if nextEvent.Player == 0 {
				reward = 7.0
				won = true
			} else {
				reward = -7.0
				wonByBlock = false
			}
			for k := 0; k < len(nextEvent.PlayerCardsRemaining); k++ {
				if nextEvent.PlayerCardsRemaining[k] == 0 {
					wonByBlock = false
					break
				}
			}
		case dominos.RoundDraw:
			wonByBlock = false
		}
	}
	if !won {
		wonByBlock = false
	}
	if hasHardEnd && !won {
		reward = -5.0
	} else if isDouble && !won {
		reward = reward + 3.0
	}
	if wonByBlock {
		reward = reward * 1.5
	}

	// Board control bonus
	suit1, suit2 := dominos.GetCLIDominos()[gameEvent.Card].GetSuits()
	newLeftSuit := gameEvent.BoardState.LeftSuit
	newRightSuit := gameEvent.BoardState.RightSuit
	if gameEvent.EventType == dominos.PosedCard || !gameEvent.BoardState.CardPosed {
		newLeftSuit = suit1
		newRightSuit = suit2
	} else if gameEvent.Side == dominos.Left {
		if suit1 == gameEvent.BoardState.LeftSuit {
			newLeftSuit = suit2
		} else {
			newLeftSuit = suit1
		}
	} else {
		if suit1 == gameEvent.BoardState.RightSuit {
			newRightSuit = suit2
		} else {
			newRightSuit = suit1
		}
	}
	playableCards := 0
	for _, card := range gameEvent.PlayerHands[0] {
		if card == gameEvent.Card || card >= 28 {
			continue
		}
		cs1, cs2 := dominos.GetCLIDominos()[card].GetSuits()
		if cs1 == newLeftSuit || cs2 == newLeftSuit || cs1 == newRightSuit || cs2 == newRightSuit {
			playableCards++
		}
	}
	reward += 0.3 * float64(playableCards)
	reward = (reward + 7.0) / 19.5

	cardChoice[index] = reward
	return cardChoice
}

// computeRewardPartner computes reward for partner mode.
// Reuses the same logic as JSDNN.ConvertCardChoiceToTensorReinforcedPartner.
func (t *SequenceTransformer) computeRewardPartner(gameEvent *dominos.GameEvent, nextGameEvents [16]*dominos.GameEvent) [56]float64 {
	cardChoice := [56]float64{}
	index := gameEvent.Card
	suitPlayed := gameEvent.BoardState.LeftSuit
	if gameEvent.Side == dominos.Right {
		index = index + 28
		suitPlayed = gameEvent.BoardState.RightSuit
	}
	reward := 0.0
	chainBroken := false
	hasHardEnd := false
	boardSuitCount := gameEvent.BoardState.CountSuit(suitPlayed)
	handSuitCount := t.CountSuit(suitPlayed, dominos.GetCLIDominos())
	isDouble := dominos.GetCLIDominos()[gameEvent.Card].IsDouble()
	if boardSuitCount+handSuitCount == 6 {
		hasHardEnd = true
	}
	won := false
	wonByBlock := true
	for i, nextEvent := range nextGameEvents {
		if nextEvent == nil || i > 4 {
			break
		}
		isPartner := nextEvent.Player == 2
		switch nextEvent.EventType {
		case dominos.Passed:
			if nextEvent.Player == 0 {
				reward = -1.0
			} else if isPartner {
				// Partner passing is neutral
			} else if !chainBroken {
				reward = reward + 1.0
			}
		case dominos.PlayedCard:
			if nextEvent.Player != 0 && !isPartner {
				chainBroken = true
			}
		case dominos.RoundWin:
			if nextEvent.Player == 0 || isPartner {
				reward = 7.0
				won = true
			} else {
				reward = -7.0
				wonByBlock = false
			}
			for k := 0; k < len(nextEvent.PlayerCardsRemaining); k++ {
				if nextEvent.PlayerCardsRemaining[k] == 0 {
					wonByBlock = false
					break
				}
			}
		case dominos.RoundDraw:
			wonByBlock = false
		}
	}
	if !won {
		wonByBlock = false
	}
	if hasHardEnd && !won {
		reward = -5.0
	} else if isDouble && !won {
		reward = reward + 3.0
	}
	if wonByBlock {
		reward = reward * 1.5
	}

	// Board control bonus
	suit1, suit2 := dominos.GetCLIDominos()[gameEvent.Card].GetSuits()
	newLeftSuit := gameEvent.BoardState.LeftSuit
	newRightSuit := gameEvent.BoardState.RightSuit
	if gameEvent.EventType == dominos.PosedCard || !gameEvent.BoardState.CardPosed {
		newLeftSuit = suit1
		newRightSuit = suit2
	} else if gameEvent.Side == dominos.Left {
		if suit1 == gameEvent.BoardState.LeftSuit {
			newLeftSuit = suit2
		} else {
			newLeftSuit = suit1
		}
	} else {
		if suit1 == gameEvent.BoardState.RightSuit {
			newRightSuit = suit2
		} else {
			newRightSuit = suit1
		}
	}
	playableCards := 0
	for _, card := range gameEvent.PlayerHands[0] {
		if card == gameEvent.Card || card >= 28 {
			continue
		}
		cs1, cs2 := dominos.GetCLIDominos()[card].GetSuits()
		if cs1 == newLeftSuit || cs2 == newLeftSuit || cs1 == newRightSuit || cs2 == newRightSuit {
			playableCards++
		}
	}
	reward += 0.3 * float64(playableCards)
	reward = (reward + 7.0) / 19.5

	cardChoice[index] = reward
	return cardChoice
}

// BuildHistoryFromRoundEvents reconstructs gameHistory tokens from round events
// for training purposes (when training master models post-game).
func (t *SequenceTransformer) BuildHistoryFromRoundEvents(events []dominos.GameEvent) {
	t.gameHistory = nil
	for _, evt := range events {
		t.observeGameEvent(&evt)
	}
}

// SetGameHistory sets the game history directly (for training from collected events).
func (t *SequenceTransformer) SetGameHistory(history []moveToken) {
	t.gameHistory = make([]moveToken, len(history))
	copy(t.gameHistory, history)
}

// GetGameHistory returns the current game history.
func (t *SequenceTransformer) GetGameHistory() []moveToken {
	return t.gameHistory
}

// softmax computes softmax over a 1D slice of logits (numerically stable).
func softmax(logits []float64) []float64 {
	n := len(logits)
	out := make([]float64, n)
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	for i, v := range logits {
		out[i] = math.Exp(v - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// TrainSupervised trains the transformer on a single game event using supervised learning.
// Uses softmax + cross-entropy loss. The gradient is (softmax_prob - one_hot_target).
func (t *SequenceTransformer) TrainSupervised(gameEvent *dominos.GameEvent, learnRate float64) (float64, error) {
	if gameEvent.EventType != dominos.PlayedCard && gameEvent.EventType != dominos.PosedCard {
		return 0.0, errors.New("invalid game event to train with")
	}

	actionIndex := int(gameEvent.Card)
	if gameEvent.Side == dominos.Right {
		actionIndex += 28
	}

	// Build the token sequence
	tokens := t.buildSequenceFromGameEvent(gameEvent)
	if len(tokens) == 0 {
		return 0.0, nil
	}

	// Forward pass (raw logits)
	logits, cache := t.forward(tokens)

	// Softmax
	probs := softmax(logits)

	// Cross-entropy loss: -log(prob[correct_class])
	cost := -math.Log(math.Max(probs[actionIndex], 1e-12))

	// Gradient of cross-entropy w.r.t. logits = prob - target
	// For softmax+CE this is simply: grad[i] = prob[i], grad[correct] = prob[correct] - 1
	// We negate because backward() expects (target - output) direction
	outputGrad := make([]float64, t.outputDim)
	for i := 0; i < t.outputDim; i++ {
		outputGrad[i] = -probs[i]
	}
	outputGrad[actionIndex] = 1.0 - probs[actionIndex]

	// Backward pass
	t.backward(outputGrad, cache, learnRate)

	return cost, nil
}

// Predict runs the transformer forward pass with softmax and returns the best valid card choice.
func (t *SequenceTransformer) Predict(gameEvent *dominos.GameEvent) (*dominos.CardChoice, error) {
	tokens := t.buildSequenceFromGameEvent(gameEvent)
	if len(tokens) == 0 {
		return nil, errors.New("empty token sequence")
	}

	logits, _ := t.forward(tokens)
	probs := softmax(logits)
	doms := dominos.GetCLIDominos()

	maxProb := -1.0
	bestCard := uint(0)
	bestSide := dominos.Left
	found := false

	hand := gameEvent.PlayerHands[gameEvent.Player]
	contained := func(card uint) bool {
		for _, h := range hand {
			if h == card {
				return true
			}
		}
		return false
	}

	for i, prob := range probs {
		side := dominos.Left
		card := i
		if i >= 28 {
			side = dominos.Right
			card -= 28
		}
		if card >= 28 {
			continue
		}
		if !contained(uint(card)) {
			continue
		}
		if gameEvent.BoardState != nil && gameEvent.BoardState.CardPosed {
			suit1, suit2 := doms[card].GetSuits()
			if side == dominos.Left {
				if suit1 != gameEvent.BoardState.LeftSuit && suit2 != gameEvent.BoardState.LeftSuit {
					continue
				}
			} else {
				if suit1 != gameEvent.BoardState.RightSuit && suit2 != gameEvent.BoardState.RightSuit {
					continue
				}
			}
		}
		if prob > maxProb {
			maxProb = prob
			bestCard = uint(card)
			bestSide = side
			found = true
		}
	}

	if !found {
		// Fallback: pick first valid card in hand
		for _, c := range hand {
			if c < 28 {
				return &dominos.CardChoice{Card: c, Side: dominos.Left}, nil
			}
		}
		return &dominos.CardChoice{Card: 0, Side: dominos.Left}, nil
	}

	return &dominos.CardChoice{Card: bestCard, Side: bestSide}, nil
}

// ConvertGameEventToHistoryTokens converts round events up to a given index into history tokens.
// This is used to reconstruct the history state at each training point.
func ConvertRoundEventsToHistory(roundGameEvents []*dominos.GameEvent, upToIndex int) []moveToken {
	tokens := []moveToken{}
	for i := 0; i < upToIndex; i++ {
		evt := roundGameEvents[i]
		if evt == nil {
			continue
		}
		rotated := jsdonline.CopyandRotateGameEvent(evt, evt.Player)
		player := rotated.Player

		switch evt.EventType {
		case dominos.PlayedCard:
			sideID := 0
			if evt.Side == dominos.Right {
				sideID = 1
			}
			tokens = append(tokens, moveToken{
				playerID: player,
				cardID:   int(evt.Card),
				sideID:   sideID,
			})
		case dominos.PosedCard:
			tokens = append(tokens, moveToken{
				playerID: player,
				cardID:   int(evt.Card),
				sideID:   sidePosed,
			})
		case dominos.Passed:
			tokens = append(tokens, moveToken{
				playerID: player,
				cardID:   cardPASS,
				sideID:   sidePass,
			})
		}
	}
	return tokens
}
