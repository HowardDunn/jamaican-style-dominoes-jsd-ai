package nn

import (
	"math"
	"testing"

	"github.com/HowardDunn/go-dominos/dominos"
)

func makeTestGameEvent() *dominos.GameEvent {
	// Player 0 holds cards [0, 1, 2, 3, 4, 5, 6] (7 cards)
	// Board has posed card 7, left board [8], right board [9]
	// 10 cards known, 18 outstanding → opponents have 6 each
	return &dominos.GameEvent{
		EventType:            dominos.PlayedCard,
		Player:               0,
		PlayerCardsRemaining: [4]int{7, 6, 6, 6},
		PlayerHands: [4][]uint{
			{0, 1, 2, 3, 4, 5, 6},
			{},
			{},
			{},
		},
		Card: 0,
		Side: dominos.Left,
		BoardState: &dominos.Board{
			CardPosed:  true,
			PosedCard:  7,
			LeftBoard:  []uint{8},
			RightBoard: []uint{9},
			LeftSuit:   0,
			RightSuit:  1,
		},
	}
}

func TestInitCardProbabilities(t *testing.T) {
	j := newGenericNN(126, []int{64}, 28)
	ge := makeTestGameEvent()

	j.initCardProbabilities(ge)

	if !j.probInitialized {
		t.Fatal("expected probInitialized to be true")
	}

	// Known cards (hand: 0-6, board: 7, 8, 9) should have 0 probability
	knownCards := []uint{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	for _, c := range knownCards {
		for p := 1; p < 4; p++ {
			if j.cardProb[p][c] != 0 {
				t.Errorf("known card %d should have 0 prob for player %d, got %f", c, p, j.cardProb[p][c])
			}
		}
	}

	// Unknown cards (10-27) should have probs summing to ~1 across opponents
	for c := uint(10); c < 28; c++ {
		sum := 0.0
		for p := 1; p < 4; p++ {
			sum += j.cardProb[p][c]
		}
		if math.Abs(sum-1.0) > 1e-9 {
			t.Errorf("card %d: opponent probs should sum to 1.0, got %f", c, sum)
		}
	}

	// Player 0 should always have 0 probability
	for c := uint(0); c < 28; c++ {
		if j.cardProb[0][c] != 0 {
			t.Errorf("player 0 should always have 0 prob, card %d got %f", c, j.cardProb[0][c])
		}
	}

	// Since all opponents have 6 cards each, priors should be equal (~1/3 each)
	for c := uint(10); c < 28; c++ {
		expected := 1.0 / 3.0
		for p := 1; p < 4; p++ {
			if math.Abs(j.cardProb[p][c]-expected) > 1e-9 {
				t.Errorf("card %d, player %d: expected uniform %f, got %f", c, p, expected, j.cardProb[p][c])
			}
		}
	}
}

func TestInitCardProbabilitiesRespectsKnownNotHaves(t *testing.T) {
	j := newGenericNN(126, []int{64}, 28)
	ge := makeTestGameEvent()

	// Mark player 1 as known not to have card 10
	j.knownNotHaves[1][10] = true

	j.initCardProbabilities(ge)

	// Player 1 should have 0 prob for card 10
	if j.cardProb[1][10] != 0 {
		t.Errorf("player 1 should have 0 prob for card 10 (knownNotHave), got %f", j.cardProb[1][10])
	}

	// Players 2, 3 should split the probability for card 10
	sum := j.cardProb[2][10] + j.cardProb[3][10]
	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("probs for card 10 should sum to 1.0 across players 2,3, got %f", sum)
	}
}

func TestUpdateFromPass(t *testing.T) {
	j := newGenericNN(126, []int{64}, 28)
	ge := makeTestGameEvent()

	j.initCardProbabilities(ge)

	// Player 1 passes with left suit 0 and right suit 1
	j.updateProbFromPass(1, 0, 1)

	// All cards containing suit 0 or suit 1 should have 0 prob for player 1
	doms := dominos.GetCLIDominos()
	for c := uint(10); c < 28; c++ {
		s1, s2 := doms[c].GetSuits()
		hasSuit0 := s1 == 0 || s2 == 0
		hasSuit1 := s1 == 1 || s2 == 1
		if hasSuit0 || hasSuit1 {
			if j.cardProb[1][c] != 0 {
				t.Errorf("card %d (suits %d,%d) should have 0 prob for passing player 1, got %f",
					c, s1, s2, j.cardProb[1][c])
			}
		}
	}

	// Non-suit cards should still have valid probabilities summing to 1
	for c := uint(10); c < 28; c++ {
		sum := 0.0
		for p := 1; p < 4; p++ {
			sum += j.cardProb[p][c]
		}
		if sum > 0 && math.Abs(sum-1.0) > 1e-9 {
			t.Errorf("card %d: probs should sum to 1.0 (or 0 if all zeroed), got %f", c, sum)
		}
	}
}

func TestUpdateFromPlay(t *testing.T) {
	j := newGenericNN(126, []int{64}, 28)
	ge := makeTestGameEvent()

	j.initCardProbabilities(ge)

	// Player 2 plays card 15 on the left side
	j.updateProbFromPlay(2, 15, dominos.Left, ge.BoardState)

	// Played card should have 0 prob for all players
	for p := 0; p < 4; p++ {
		if j.cardProb[p][15] != 0 {
			t.Errorf("played card 15 should have 0 prob for player %d, got %f", p, j.cardProb[p][15])
		}
	}

	// Remaining cards should still sum to ~1 across opponents
	for c := uint(10); c < 28; c++ {
		if c == 15 {
			continue
		}
		sum := 0.0
		for p := 1; p < 4; p++ {
			sum += j.cardProb[p][c]
		}
		if sum > 0 && math.Abs(sum-1.0) > 1e-9 {
			t.Errorf("card %d: probs should sum to 1.0, got %f", c, sum)
		}
	}
}

func TestUpdateFromPlaySoftDiscount(t *testing.T) {
	j := newGenericNN(126, []int{64}, 28)

	// Find a card that matches two different suits (for both-side play)
	doms := dominos.GetCLIDominos()
	var bothSidesCard uint
	var leftSuit, rightSuit uint
	found := false
	for c := uint(10); c < 28; c++ {
		s1, s2 := doms[c].GetSuits()
		if s1 != s2 {
			bothSidesCard = c
			leftSuit = s1
			rightSuit = s2
			found = true
			break
		}
	}
	if !found {
		t.Skip("no non-double card found in test range")
	}

	// Set up board with these suits so the card can go on both sides
	ge := makeTestGameEvent()
	ge.BoardState.LeftSuit = leftSuit
	ge.BoardState.RightSuit = rightSuit

	j.initCardProbabilities(ge)

	// Record player 2's probs for right-suit cards before the play
	rightSuitProbsBefore := map[uint]float64{}
	for _, c := range cardsForSuit[rightSuit] {
		rightSuitProbsBefore[c] = j.cardProb[2][c]
	}

	// Player 2 plays the both-sides card on the left
	j.updateProbFromPlay(2, bothSidesCard, dominos.Left, ge.BoardState)

	// The soft discount should reduce player 2's right-suit card probs
	// (since they chose left despite being able to play right)
	discountApplied := false
	for _, c := range cardsForSuit[rightSuit] {
		if c == bothSidesCard {
			continue
		}
		before := rightSuitProbsBefore[c]
		after := j.cardProb[2][c]
		if before > 0 && after < before {
			discountApplied = true
		}
	}
	if !discountApplied {
		t.Error("expected soft discount to reduce right-suit probs for player 2")
	}
}

func TestSampleOpponentHands(t *testing.T) {
	j := newGenericNN(126, []int{64}, 28)
	ge := makeTestGameEvent()

	// Set player 0's hand
	j.SetHand([]uint{0, 1, 2, 3, 4, 5, 6})

	j.initCardProbabilities(ge)

	outstandingCards := []uint{}
	for c := uint(10); c < 28; c++ {
		outstandingCards = append(outstandingCards, c)
	}

	// Run multiple samples and verify constraints
	for trial := 0; trial < 50; trial++ {
		hands := j.sampleOpponentHands(outstandingCards, ge)

		// Player 0 should have its original hand
		if len(hands[0]) != 7 {
			t.Errorf("trial %d: player 0 should have 7 cards, got %d", trial, len(hands[0]))
		}

		// Each opponent should have the right number of cards
		for p := 1; p < 4; p++ {
			if len(hands[p]) != ge.PlayerCardsRemaining[p] {
				t.Errorf("trial %d: player %d should have %d cards, got %d",
					trial, p, ge.PlayerCardsRemaining[p], len(hands[p]))
			}
		}

		// No card should appear in multiple hands
		seen := map[uint]int{}
		for p := 1; p < 4; p++ {
			for _, c := range hands[p] {
				seen[c]++
				if seen[c] > 1 {
					t.Errorf("trial %d: card %d appears in multiple opponent hands", trial, c)
				}
			}
		}

		// All dealt cards should be from outstandingCards
		outstandingSet := map[uint]bool{}
		for _, c := range outstandingCards {
			outstandingSet[c] = true
		}
		for p := 1; p < 4; p++ {
			for _, c := range hands[p] {
				if !outstandingSet[c] {
					t.Errorf("trial %d: card %d dealt to player %d is not outstanding", trial, c, p)
				}
			}
		}
	}
}

func TestSampleRespectsKnownNotHaves(t *testing.T) {
	j := newGenericNN(126, []int{64}, 28)
	ge := makeTestGameEvent()

	j.SetHand([]uint{0, 1, 2, 3, 4, 5, 6})

	// Mark player 1 as known not to have cards 10-16
	for c := uint(10); c <= 16; c++ {
		j.knownNotHaves[1][c] = true
	}

	j.initCardProbabilities(ge)

	outstandingCards := []uint{}
	for c := uint(10); c < 28; c++ {
		outstandingCards = append(outstandingCards, c)
	}

	for trial := 0; trial < 50; trial++ {
		hands := j.sampleOpponentHands(outstandingCards, ge)

		// Player 1 should never have cards 10-16
		for _, c := range hands[1] {
			if c >= 10 && c <= 16 {
				t.Errorf("trial %d: player 1 has excluded card %d", trial, c)
			}
		}
	}
}

func TestSampleStatisticalDistribution(t *testing.T) {
	j := newGenericNN(126, []int{64}, 28)
	ge := makeTestGameEvent()

	j.SetHand([]uint{0, 1, 2, 3, 4, 5, 6})
	j.initCardProbabilities(ge)

	// Zero out player 1's prob for card 10 (simulate a pass)
	j.cardProb[1][10] = 0
	j.renormalizeProbabilities()

	outstandingCards := []uint{}
	for c := uint(10); c < 28; c++ {
		outstandingCards = append(outstandingCards, c)
	}

	// Count how many times card 10 ends up in each player's hand
	counts := [4]int{}
	numTrials := 2000
	for trial := 0; trial < numTrials; trial++ {
		hands := j.sampleOpponentHands(outstandingCards, ge)
		for p := 1; p < 4; p++ {
			for _, c := range hands[p] {
				if c == 10 {
					counts[p]++
				}
			}
		}
	}

	// Player 1 should get card 10 much less often than players 2 and 3
	// Since player 1's prob is 0, it should never get card 10
	if counts[1] > 0 {
		t.Errorf("player 1 should never get card 10 (prob=0), got it %d/%d times", counts[1], numTrials)
	}

	// Players 2 and 3 should each get it roughly half the time
	if counts[2] == 0 || counts[3] == 0 {
		t.Errorf("players 2 and 3 should both get card 10: p2=%d, p3=%d out of %d",
			counts[2], counts[3], numTrials)
	}
}

func TestResetClearsProb(t *testing.T) {
	j := newGenericNN(126, []int{64}, 28)
	ge := makeTestGameEvent()

	j.initCardProbabilities(ge)
	if !j.probInitialized {
		t.Fatal("should be initialized")
	}

	j.ResetPassMemory()
	if j.probInitialized {
		t.Error("probInitialized should be false after reset")
	}

	// All probs should be 0
	for p := 0; p < 4; p++ {
		for c := 0; c < 28; c++ {
			if j.cardProb[p][c] != 0 {
				t.Errorf("cardProb[%d][%d] should be 0 after reset, got %f", p, c, j.cardProb[p][c])
			}
		}
	}
}

func TestCardsForSuitInit(t *testing.T) {
	// Verify the cardsForSuit cache is populated
	totalEntries := 0
	for s := 0; s < 7; s++ {
		if len(cardsForSuit[s]) == 0 {
			t.Errorf("cardsForSuit[%d] is empty", s)
		}
		totalEntries += len(cardsForSuit[s])
	}
	// Each card has 2 suits (or 1 for doubles counted once), total should be reasonable
	// 28 cards, each appearing in 2 suit lists (doubles appear once) = 28 + 21 non-doubles = 49
	// Actually: 7 doubles (each in 1 list) + 21 non-doubles (each in 2 lists) = 7 + 42 = 49
	if totalEntries != 49 {
		t.Errorf("expected 49 total entries in cardsForSuit, got %d", totalEntries)
	}
}
