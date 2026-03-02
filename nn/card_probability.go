package nn

import (
	"math/rand"

	"github.com/HowardDunn/go-dominos/dominos"
)

// cardsForSuit maps each suit (0-6) to the card indices that contain that suit.
var cardsForSuit [7][]uint

func init() {
	doms := dominos.GetCLIDominos()
	for i := uint(0); i < uint(len(doms)); i++ {
		s1, s2 := doms[i].GetSuits()
		cardsForSuit[s1] = append(cardsForSuit[s1], i)
		if s1 != s2 {
			cardsForSuit[s2] = append(cardsForSuit[s2], i)
		}
	}
}

// initCardProbabilities sets up uniform priors for card probabilities.
// cardProb[p][c] = P(player p holds card c), zeroed for cards we know
// the location of (our hand, board, knownNotHaves).
func (j *JSDNN) initCardProbabilities(gameEvent *dominos.GameEvent) {
	// Zero everything
	j.cardProb = [4][28]float64{}

	// Build set of known cards (our hand + board)
	known := [28]bool{}
	for _, c := range gameEvent.PlayerHands[0] {
		if c < 28 {
			known[c] = true
		}
	}
	if gameEvent.BoardState.CardPosed {
		known[gameEvent.BoardState.PosedCard] = true
		for _, c := range gameEvent.BoardState.LeftBoard {
			if c < 28 {
				known[c] = true
			}
		}
		for _, c := range gameEvent.BoardState.RightBoard {
			if c < 28 {
				known[c] = true
			}
		}
	}

	// Total outstanding cards held by opponents
	totalOutstanding := 0
	for p := 1; p < 4; p++ {
		totalOutstanding += gameEvent.PlayerCardsRemaining[p]
	}
	if totalOutstanding == 0 {
		j.probInitialized = true
		return
	}

	// Set uniform priors for unknown cards, weighted by how many cards each player holds
	for c := uint(0); c < 28; c++ {
		if known[c] {
			continue
		}
		for p := 1; p < 4; p++ {
			if j.knownNotHaves[p][c] {
				continue
			}
			j.cardProb[p][c] = float64(gameEvent.PlayerCardsRemaining[p])
		}
	}

	// Renormalize so that for each card, probs across opponents sum to 1
	j.renormalizeProbabilities()
	j.probInitialized = true
}

// updateProbFromPass zeroes out probability for all cards matching the exposed
// board suits for the passing player, then renormalizes.
func (j *JSDNN) updateProbFromPass(player int, leftSuit, rightSuit uint) {
	if player == 0 || player > 3 {
		return
	}
	// A pass means the player has NO card matching either exposed suit
	for _, suit := range []uint{leftSuit, rightSuit} {
		for _, c := range cardsForSuit[suit] {
			j.cardProb[player][c] = 0
		}
	}
	j.renormalizeProbabilities()
}

// updateProbFromPlay removes the played card from all players and applies
// a soft discount to infer suit preferences.
func (j *JSDNN) updateProbFromPlay(player int, card uint, side dominos.BoardSide, boardState *dominos.Board) {
	if card >= 28 {
		return
	}
	// Hard: the played card is no longer held by anyone
	for p := 0; p < 4; p++ {
		j.cardProb[p][card] = 0
	}

	if player == 0 || player > 3 {
		j.renormalizeProbabilities()
		return
	}

	// Soft signal: if the card was played on one side and could NOT go on the
	// other side, it tells us less. But if it COULD go on both sides and the
	// player chose this side, it hints they may not have strong cards for the
	// other side's suit.
	s1, s2 := dominos.GetCLIDominos()[card].GetSuits()
	var playedOnSuit, otherSuit uint
	if side == dominos.Left {
		playedOnSuit = boardState.LeftSuit
		otherSuit = boardState.RightSuit
	} else {
		playedOnSuit = boardState.RightSuit
		otherSuit = boardState.LeftSuit
	}

	// Check if the card could have gone on the other side too
	couldPlayOtherSide := (s1 == otherSuit || s2 == otherSuit)
	_ = playedOnSuit

	if !couldPlayOtherSide {
		// Card could only go on the chosen side — no soft signal
		j.renormalizeProbabilities()
		return
	}

	// The player chose this side despite being able to play on the other.
	// Apply a mild discount to other-side-suit cards for this player.
	const discount = 0.85
	for _, c := range cardsForSuit[otherSuit] {
		j.cardProb[player][c] *= discount
	}

	j.renormalizeProbabilities()
}

// renormalizeProbabilities ensures that for each card, the probabilities
// across all opponent players sum to 1 (or 0 if the card is fully known).
func (j *JSDNN) renormalizeProbabilities() {
	for c := uint(0); c < 28; c++ {
		total := 0.0
		for p := 1; p < 4; p++ {
			total += j.cardProb[p][c]
		}
		if total > 0 {
			for p := 1; p < 4; p++ {
				j.cardProb[p][c] /= total
			}
		}
	}
}

// sampleOpponentHands uses probability-weighted sampling to distribute
// outstanding cards among opponents, respecting cardsRemaining counts.
func (j *JSDNN) sampleOpponentHands(outstandingCards []uint, gameEvent *dominos.GameEvent) [4][]uint {
	playerHands := [4][]uint{}
	playerHands[0] = make([]uint, len(j.GetHand()))
	copy(playerHands[0], j.GetHand())

	remaining := make([]uint, len(outstandingCards))
	copy(remaining, outstandingCards)

	// Assign cards to players 1, 2, 3 in order
	for p := 1; p < 4; p++ {
		need := gameEvent.PlayerCardsRemaining[p]
		hand := make([]uint, 0, need)

		for len(hand) < need && len(remaining) > 0 {
			// Build weights for remaining cards for this player
			weights := make([]float64, len(remaining))
			totalWeight := 0.0
			for i, c := range remaining {
				w := j.cardProb[p][c]
				// Respect hard constraints
				if j.knownNotHaves[p][c] {
					w = 0
				}
				weights[i] = w
				totalWeight += w
			}

			// If all weights are 0, fall back to uniform over non-excluded cards
			if totalWeight <= 0 {
				for i, c := range remaining {
					if !j.knownNotHaves[p][c] {
						weights[i] = 1.0
						totalWeight += 1.0
					}
				}
			}

			// If still 0 (all cards excluded), just take anything remaining
			if totalWeight <= 0 {
				for i := range remaining {
					weights[i] = 1.0
					totalWeight += 1.0
				}
			}

			// Weighted random pick
			r := rand.Float64() * totalWeight
			cumulative := 0.0
			picked := len(remaining) - 1
			for i, w := range weights {
				cumulative += w
				if r <= cumulative {
					picked = i
					break
				}
			}

			hand = append(hand, remaining[picked])
			// Remove picked card from remaining
			remaining[picked] = remaining[len(remaining)-1]
			remaining = remaining[:len(remaining)-1]
		}
		playerHands[p] = hand
	}

	return playerHands
}

// UpdateProbFromPlay is the exported method called from game loops.
// The gameEvent should already be rotated to this NN's perspective
// (i.e., this NN is position 0, opponents are 1-3).
func (j *JSDNN) UpdateProbFromPlay(gameEvent *dominos.GameEvent) {
	if gameEvent.BoardState == nil {
		return
	}
	if !j.probInitialized {
		return
	}
	j.updateProbFromPlay(gameEvent.Player, gameEvent.Card, gameEvent.Side, gameEvent.BoardState)
}
