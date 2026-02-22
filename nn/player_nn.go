package nn

import (
	"encoding/gob"
	"errors"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/HowardDunn/go-dominos/dominos"
	jsdonline "github.com/HowardDunn/jsd-online-game/game"
	log "github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/stat/distuv"
	"gorgonia.org/tensor"
)

// The input layer consists of the player hand which is the 0,1 encoded of all 28 cards
// board state which is 0, 1 encoded of all 28 cards
// The suit state which is 0,1 encoded for both sides ( one-hot ), 14
// the known pass state of all 4 players which is 28
// how many cards each player has remaining which is 28
type JSDNN struct {
	hidden  []*tensor.Dense
	final   *tensor.Dense
	bHidden []*tensor.Dense
	bFinal  *tensor.Dense
	*dominos.ComputerPlayer
	passMemory    [28]float64
	TotalWins     int
	TotalWins2    int
	inputDim      int
	hiddenDim     []int
	outputDim     int
	gameType      string
	knownNotHaves [4]map[uint]bool
	Search           bool
	SearchNum        int
	OutputActivation string  // "relu" or "linear"
	Epsilon          float64 // epsilon-greedy exploration rate (0.0 = greedy, 0.1 = 10% random)
}

func fillRandom(a []float64, v float64) {
	dist := distuv.Uniform{
		Min: -math.Sqrt(6.0 / v),
		Max: math.Sqrt(6.0 / v),
	}
	for i := range a {
		a[i] = dist.Rand()
	}
}

func FillRandom(a []float64, v float64) {
	fillRandom(a, v)
}

func New(input int, hidden []int, output int) *JSDNN {
	if output != 56 {
		panic("Invalid output size")
	}

	if len(hidden) < 1 {
		panic("Invalid hidden size")
	}

	r := make([]float64, input*hidden[0])
	r2 := [][]float64{}
	r3 := [][]float64{}
	r4 := make([]float64, output)
	hiddenT := []*tensor.Dense{}
	bHidden := []*tensor.Dense{}
	rb := make([]float64, hidden[0])
	r3 = append(r3, rb)
	fillRandom(r, float64(input))
	fillRandom(r3[0], float64(len(r3[0])))
	hiddenT = append(hiddenT, tensor.New(tensor.WithShape(hidden[0], input), tensor.WithBacking(r)))
	bHidden = append(bHidden, tensor.New(tensor.WithShape(hidden[0]), tensor.WithBacking(r3[0])))

	for i := range hidden {
		next := output
		if (i + 1) < len(hidden) {
			next = hidden[i+1]
		}
		h := make([]float64, hidden[i]*next)

		fillRandom(h, float64(hidden[i]))

		r2 = append(r2, h)

		if (i + 1) < len(hidden) {
			hb := make([]float64, hidden[i+1])
			fillRandom(hb, float64(len(hb)))
			r3 = append(r3, hb)
			hiddenT = append(hiddenT, tensor.New(tensor.WithShape(next, hidden[i]), tensor.WithBacking(r2[i])))
			bHidden = append(bHidden, tensor.New(tensor.WithShape(hidden[i+1]), tensor.WithBacking(r3[i+1])))
		}
	}

	finalT := tensor.New(tensor.WithShape(output, hidden[len(hidden)-1]), tensor.WithBacking(r2[len(r2)-1]))
	bFinal := tensor.New(tensor.WithShape(output), tensor.WithBacking(r4))
	knownNotHaves := [4]map[uint]bool{}
	for i := 0; i < len(knownNotHaves); i++ {
		knownNotHaves[i] = make(map[uint]bool)
	}

	return &JSDNN{
		hidden:           hiddenT,
		final:            finalT,
		bHidden:          bHidden,
		bFinal:           bFinal,
		ComputerPlayer:   &dominos.ComputerPlayer{},
		inputDim:         input,
		hiddenDim:        hidden,
		outputDim:        output,
		gameType:         "cutthroat",
		knownNotHaves:    knownNotHaves,
		OutputActivation: "relu",
	}
}

func relu(x float64) float64 {
	// if x > 1 {
	// 	return 1
	// }

	if x > 0 {
		return x
	}
	return 0
}

func reluPrime(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1.0
}

// sigmoid implements the sigmoid function
// for use in activation functions.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}

func clipValue(x float64) float64 {
	if x > 1.0 {
		return 1.0
	}
	if x < -1.0 {
		return -1.0
	}
	return x
}

func sum(x []float64) float64 {
	s := 0.0
	for i := range x {
		s += x[i]
	}
	return s
}

func (*JSDNN) GetPlayerType() string {
	return "localHumanPlayer"
}

func (j *JSDNN) GetNumHidden() int {
	return len(j.hidden)
}

func (j *JSDNN) UpdatePassMemory(gameEvent *dominos.GameEvent) {
	gameEvent = jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
	card := gameEvent.Card
	player := gameEvent.Player
	doms := dominos.GetCLIDominos()
	dom := doms[card]
	suit1, suit2 := dom.GetSuits()
	suit1Index := uint(player*7) + suit1
	suit2Index := uint(player*7) + suit2
	j.passMemory[suit1Index] = 1.0
	j.passMemory[suit2Index] = 1.0
	for i := 0; i < 28; i++ {
		suit11, suit22 := doms[i].GetSuits()

		if suit11 == suit1 || suit11 == suit2 {
			j.knownNotHaves[player][uint(i)] = true
		} else if suit22 == suit1 || suit22 == suit2 {
			j.knownNotHaves[player][uint(i)] = true
		}
	}
}

func (j *JSDNN) ResetPassMemory() {
	j.passMemory = [28]float64{}
	j.knownNotHaves = [4]map[uint]bool{}
	for i := 0; i < len(j.knownNotHaves); i++ {
		j.knownNotHaves[i] = make(map[uint]bool)
	}
	for i := 0; i < len(j.knownNotHaves); i++ {
		for k := 0; k < 28; k++ {
			j.knownNotHaves[i][uint(k)] = false
		}
	}
}

func (j *JSDNN) ConvertGameEventToTensor(gameEvent *dominos.GameEvent) (tensor.Tensor, int, int) {
	type JSDNNGameState struct {
		playerHand    [28]float64
		boardState    [28]float64
		suitState     [14]float64
		playerPass    [28]float64
		cardRemaining [28]float64
	}

	jsdgameState := &JSDNNGameState{}
	jsdgameState.playerPass = j.passMemory
	compatibleCardsL := 0
	compatibleCardsR := 0
	for i := range gameEvent.PlayerHands[gameEvent.Player] {
		card := gameEvent.PlayerHands[gameEvent.Player][i]

		if card < 28 {
			cardCompatible := false
			suit1, suit2 := dominos.GetCLIDominos()[card].GetSuits()
			if suit1 == gameEvent.BoardState.LeftSuit || suit2 == gameEvent.BoardState.LeftSuit {
				cardCompatible = true
				compatibleCardsL++
			}

			if suit1 == gameEvent.BoardState.RightSuit || suit2 == gameEvent.BoardState.RightSuit {
				cardCompatible = true
				compatibleCardsR++
			}

			if cardCompatible {
				jsdgameState.playerHand[card] = 1.0
			}

		}
	}

	if gameEvent.EventType == dominos.PlayedCard || gameEvent.EventType == dominos.PosedCard {
		if gameEvent.BoardState.CardPosed {
			jsdgameState.playerHand[gameEvent.Card] = 1.0
		}
	}

	if gameEvent.BoardState.CardPosed {
		jsdgameState.boardState[gameEvent.BoardState.PosedCard] = 1.0
		for i := range gameEvent.BoardState.LeftBoard {
			card := gameEvent.BoardState.LeftBoard[i]
			jsdgameState.boardState[card] = 1.0
		}

		for i := range gameEvent.BoardState.RightBoard {
			card := gameEvent.BoardState.RightBoard[i]
			jsdgameState.boardState[card] = 1.0
		}

		leftSuit := gameEvent.BoardState.LeftSuit
		rightSuit := gameEvent.BoardState.RightSuit + 7.0
		jsdgameState.suitState[leftSuit] = 1.0
		jsdgameState.suitState[rightSuit] = 1.0
	}

	cardsRemaining0 := gameEvent.PlayerCardsRemaining[0] - 1
	cardsRemaining1 := gameEvent.PlayerCardsRemaining[1] + 6
	cardsRemaining2 := gameEvent.PlayerCardsRemaining[2] + 13
	cardsRemaining3 := gameEvent.PlayerCardsRemaining[3] + 20
	if cardsRemaining0 < 0 {
		cardsRemaining0 = 0
	}
	if cardsRemaining1 < 0 {
		cardsRemaining1 = 0
	}
	if cardsRemaining2 < 0 {
		cardsRemaining2 = 0
	}
	if cardsRemaining3 < 0 {
		cardsRemaining3 = 0
	}
	jsdgameState.cardRemaining[cardsRemaining0] = 1.0
	jsdgameState.cardRemaining[cardsRemaining1] = 1.0
	jsdgameState.cardRemaining[cardsRemaining2] = 1.0
	jsdgameState.cardRemaining[cardsRemaining3] = 1.0

	flattened := []float64{}
	flattened = append(flattened, jsdgameState.playerHand[:]...)
	flattened = append(flattened, jsdgameState.boardState[:]...)
	flattened = append(flattened, jsdgameState.suitState[:]...)
	flattened = append(flattened, jsdgameState.playerPass[:]...)
	flattened = append(flattened, jsdgameState.cardRemaining[:]...)
	res := tensor.New(tensor.WithShape(126), tensor.WithBacking(flattened))
	return res, compatibleCardsL, compatibleCardsR
}

func (j *JSDNN) ConvertCardChoiceToTensor(gameEvent *dominos.GameEvent) tensor.Tensor {
	cardChoice := [56]float64{}
	index := gameEvent.Card
	if gameEvent.Side == dominos.Right {
		index = index + 28
	}
	cardChoice[index] = 1.0
	res := tensor.New(tensor.WithShape(56), tensor.WithBacking(cardChoice[:]))
	return res
}

func (j *JSDNN) ConvertCardChoiceToTensorReinforced(gameEvent *dominos.GameEvent, nextGameEvents [16]*dominos.GameEvent) tensor.Tensor {
	// Since the agent will want to maximize the reward, lets reward for every time it is able to play the next turn
	// I think it makes sense to look at the 4 events, the relevant events, pass, play card, round win

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
	handSuitCount := j.CountSuit(suitPlayed, dominos.GetCLIDominos())
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
		modded := (i + 1) % 4
		if nextEvent.Player != modded && nextEvent.EventType != dominos.RoundWin && nextEvent.EventType != dominos.RoundDraw {
			log.Fatal("Not what you expected: ", gameEvent.Player, " ", nextEvent.Player, " ", gameEvent.EventType, " ", nextEvent.EventType, " ", gameEvent.Card, " ", nextEvent.Card)
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

			for j := 0; j < len(nextEvent.PlayerCardsRemaining); j++ {
				if nextEvent.PlayerCardsRemaining[j] == 0 {
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

	cardChoice[index] = reward
	res := tensor.New(tensor.WithShape(56), tensor.WithBacking(cardChoice[:]))
	return res
}

func (j *JSDNN) GetOutputMask(gameEvent *dominos.GameEvent) tensor.Tensor {
	playerHandMask := [56]float64{}
	for i := range gameEvent.PlayerHands[gameEvent.Player] {
		card := gameEvent.PlayerHands[gameEvent.Player][i]

		if card < 28 {
			cardCompatible := false
			suit1, suit2 := dominos.GetCLIDominos()[card].GetSuits()
			if suit1 == gameEvent.BoardState.LeftSuit || suit1 == gameEvent.BoardState.RightSuit {
				cardCompatible = true
			}

			if suit2 == gameEvent.BoardState.LeftSuit || suit2 == gameEvent.BoardState.RightSuit {
				cardCompatible = true
			}

			if cardCompatible {
				playerHandMask[card] = 1.0
				playerHandMask[card+28] = 1.0
			}
		}
	}
	if gameEvent.EventType == dominos.PlayedCard || gameEvent.EventType == dominos.PosedCard {
		if gameEvent.Card < 28 {
			playerHandMask[gameEvent.Card] = 1.0
			playerHandMask[gameEvent.Card+28] = 1.0
		}
	}
	res := tensor.New(tensor.WithShape(56), tensor.WithBacking(playerHandMask[:]))
	return res
}

func (j *JSDNN) predict(a tensor.Tensor) (tensor.Tensor, error) {
	hidden, err := j.hidden[0].MatVecMul(a)
	if err != nil {
		return nil, err
	}

	_, err = tensor.Add(hidden, j.bHidden[0], tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "add1")
		return nil, err
	}

	hiddenActivation, err := hidden.Apply(relu, tensor.UseUnsafe())
	if err != nil {
		return nil, err
	}

	for i := 1; i < len(j.hidden); i++ {
		hidden, err = j.hidden[i].MatVecMul(hiddenActivation)
		if err != nil {
			log.Fatal(err, " ", "1")
			return nil, err
		}

		_, err = tensor.Add(hidden, j.bHidden[i], tensor.UseUnsafe())
		if err != nil {
			log.Fatal(err, " ", "add10")
			return nil, err
		}

		hiddenActivation, err = hidden.Apply(relu, tensor.UseUnsafe())
		if err != nil {
			log.Fatal(err, " ", "2")
			return nil, err
		}

		err = hiddenActivation.Reshape(hiddenActivation.Shape()[0], 1)
		if err != nil {
			log.Fatal(err, " ", "hA")
			return nil, err
		}

	}

	final, err := tensor.MatVecMul(j.final, hiddenActivation)
	if err != nil {
		return nil, err
	}

	_, err = tensor.Add(final, j.bFinal, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "add2")
		return nil, err
	}

	if j.OutputActivation != "linear" {
		prediction, err := final.Apply(relu, tensor.UseSafe())
		if err != nil {
			return nil, err
		}
		return prediction, nil
	}

	return final, nil
}

func (j *JSDNN) Clone() *JSDNN {
	clone := New(j.inputDim, j.hiddenDim, j.outputDim)
	clone.final = j.final.Clone().(*tensor.Dense)
	clone.bFinal = j.bFinal.Clone().(*tensor.Dense)
	clone.OutputActivation = j.OutputActivation

	for i := range clone.hidden {
		clone.hidden[i] = j.hidden[i].Clone().(*tensor.Dense)
	}

	for i := range clone.bHidden {
		clone.bHidden[i] = j.bHidden[i].Clone().(*tensor.Dense)
	}

	return clone
}

func (j *JSDNN) PredictSearch(gameEvent *dominos.GameEvent, searchNum int, learnRate float64) (*dominos.CardChoice, float32, error) {
	if gameEvent.EventType == "" {
		panic("event type nil")
	}

	if !gameEvent.BoardState.CardPosed {
		return &dominos.CardChoice{}, 0.0, nil
	}

	cardStatesKnown := make([]uint, len(j.GetHand()))
	copy(cardStatesKnown, j.GetHand())
	if gameEvent.BoardState.CardPosed {
		cardStatesKnown = append(cardStatesKnown, gameEvent.BoardState.PosedCard)
	}
	cardStatesKnown = append(cardStatesKnown, gameEvent.BoardState.LeftBoard...)
	cardStatesKnown = append(cardStatesKnown, gameEvent.BoardState.RightBoard...)
	cardStatesKnownMap := map[uint]bool{}
	outstandingCards := []uint{}
	for i := 0; i < 28; i++ {
		cardStatesKnownMap[uint(i)] = false
	}
	for _, card := range cardStatesKnown {
		cardStatesKnownMap[card] = true
	}
	for card, present := range cardStatesKnownMap {
		if !present {
			outstandingCards = append(outstandingCards, card)
		}
	}
	// Assign Cards Based on known information

	randomSeed := time.Now().UnixNano()
	rand.Seed(randomSeed)

	winners := []int{}
	cardChoicesForWin := []*dominos.CardChoice{}
	clone2 := j.Clone()
	clone3 := j.Clone()
	clone4 := j.Clone()
	for i := 0; i < searchNum; i++ {

		randomSeed := time.Now().UnixNano()
		rand.Seed(randomSeed)

		rn := rand.Int31n(100)
		clone1 := &dominos.ComputerPlayer{RandomMode: true}

		players := [4]dominos.Player{clone1, clone2, clone3, clone4}
		if rn > 55 {
			players = [4]dominos.Player{clone1, &dominos.ComputerPlayer{RandomMode: true}, &dominos.ComputerPlayer{RandomMode: true}, &dominos.ComputerPlayer{RandomMode: true}}
		}
		localGame := dominos.NewLocalGame(players, 0, j.gameType)
		localGame.SetBoard(gameEvent.BoardState)
		localGame.SetState(dominos.ExecutePlayerTurn)
		rand.Shuffle(len(outstandingCards), func(i, j int) { outstandingCards[i], outstandingCards[j] = outstandingCards[j], outstandingCards[i] })
		playerHands := [4][]uint{}
		playerHands[0] = make([]uint, len(j.GetHand()))
		copy(playerHands[0], j.GetHand())
		o2 := make([]uint, len(outstandingCards))
		copy(o2, outstandingCards)
		for i := 1; i < 4; i++ {
			playerHand := []uint{}
			index := 0
			for len(playerHand) < gameEvent.PlayerCardsRemaining[i] {
				if !j.knownNotHaves[i][o2[index]] {
					playerHand = append(playerHand, o2[index])
					o2 = append(o2[:index], o2[index+1:]...)
				} else {
					index++
				}
			}
			playerHands[i] = playerHand
		}
		localGame.SetPlayerHands(playerHands)

		var firstCardChoice *dominos.CardChoice
		for {
			lastGameEvent := localGame.AdvanceGameIteration()
			if lastGameEvent.EventType == dominos.RoundInvalid {
				log.Info("Invalid Game")
				log.Infof("HERE: %+#v Board: %+#v", lastGameEvent, lastGameEvent.BoardState)
				break
			} else if lastGameEvent.EventType == dominos.PlayedCard && lastGameEvent.Player == 0 {
				if firstCardChoice == nil {
					firstCardChoice = &dominos.CardChoice{Card: lastGameEvent.Card, Side: lastGameEvent.Side}
				}
			} else if lastGameEvent.EventType == dominos.Passed {
				clone2.UpdatePassMemory(lastGameEvent)
				clone3.UpdatePassMemory(lastGameEvent)
				clone4.UpdatePassMemory(lastGameEvent)

			} else if lastGameEvent.EventType == dominos.RoundWin || lastGameEvent.EventType == dominos.RoundDraw {

				if firstCardChoice != nil && lastGameEvent.Player == 0 {
					cardChoicesForWin = append(cardChoicesForWin, firstCardChoice)
				}
				winners = append(winners, lastGameEvent.Player)
				clone2.ResetPassMemory()
				clone3.ResetPassMemory()
				clone4.ResetPassMemory()
				break
			}
		}

	}

	count := 0
	for i := 0; i < len(winners); i++ {
		if winners[i] == 0 {
			count++
		}
	}

	popularChoice := map[uint]int{}
	for i := 0; i < 56; i++ {
		popularChoice[uint(i)] = 0
	}

	maxCount := 0
	bestCard := &dominos.CardChoice{}

	for i := 0; i < len(cardChoicesForWin); i++ {
		card := cardChoicesForWin[i].Card
		if cardChoicesForWin[i].Side == dominos.Right {
			card += 28
		}
		popularChoice[card]++
		if popularChoice[card] > maxCount {
			maxCount = popularChoice[card]
			bestCard = &dominos.CardChoice{Card: cardChoicesForWin[i].Card, Side: cardChoicesForWin[i].Side}
		}
	}
	probab := float32(maxCount) / float32(len(winners))

	return bestCard, probab, nil
}

func (j *JSDNN) Predict(gameEvent *dominos.GameEvent) (*dominos.CardChoice, error) {
	a, _, _ := j.ConvertGameEventToTensor(gameEvent)
	if a.Dims() != 1 {
		return nil, errors.New("expected a vector")
	}

	prediction, err := j.predict(a)
	if err != nil {
		return &dominos.CardChoice{}, err
	}

	card, side := j.ComputerPlayer.PlayCard(gameEvent, dominos.GetCLIDominos())
	choice := &dominos.CardChoice{Card: card, Side: side}
	cardConfidences, ok := prediction.Data().([]float64)
	maxConfidence := -100000000000.0
	contained := func(card uint) bool {
		hand := gameEvent.PlayerHands[gameEvent.Player]
		if len(hand) == 0 {
			return true
		}
		if gameEvent.EventType == dominos.PlayedCard {
			if card == gameEvent.Card {
				return true
			}
		}
		for i := range hand {
			if hand[i] == card {
				return true
			}
		}
		return false
	}

	compatible := func(card uint, side dominos.BoardSide) bool {
		compatibleCard := false
		if card > 27 {
			log.Error("Too big")
			return false
		}
		suit1, suit2 := dominos.GetCLIDominos()[card].GetSuits()
		if side == dominos.Left {
			if suit1 == gameEvent.BoardState.LeftSuit || suit2 == gameEvent.BoardState.LeftSuit {
				compatibleCard = true
			}
		} else {
			if suit1 == gameEvent.BoardState.RightSuit || suit2 == gameEvent.BoardState.RightSuit {
				compatibleCard = true
			}
		}
		return compatibleCard
	}
	somethingFound := false
	if ok {
		for i, cardConfidence := range cardConfidences {
			if cardConfidence > maxConfidence {
				side := dominos.Left
				card := i
				if i > 27 {
					side = dominos.Right
					card = card - 28
				}
				if contained(uint(card)) && compatible(uint(card), side) {
					choice.Card = uint(card)
					choice.Side = side
					maxConfidence = cardConfidence
					somethingFound = true
				}

			}
		}
	}
	if !ok {
		log.Fatal("Failure to get prediction")
	}

	if !somethingFound {
		log.Errorf("No card to play found. Max confidence: %.7f Player: %d Hand: %#v Hand2: %#v Confidences: %#v Event: %s", maxConfidence, gameEvent.Player, gameEvent.PlayerHands[gameEvent.Player], j.GetHand(), cardConfidences, gameEvent.EventType)
		hand := gameEvent.PlayerHands[gameEvent.Player]
		for i := range hand {
			log.Warn(" ", hand[i], " ", compatible(hand[i], "left"), " ", compatible(hand[i], "right"))
		}
	}
	// if maxConfidence < 0 {
	// 	log.Infof("Card Choice: %+#v, Confidence: %.5f Confidences %+#v", choice, maxConfidence, cardConfidences)
	// } else {
	// 	log.Infof("Card Choice: %+#v, Confidence: %.5f Confidences %+#v", choice, maxConfidence, cardConfidences)
	// }

	return choice, nil
}

func (j *JSDNN) getLearnRate(learnRates []float64, layerIndex int) float64 {
	if len(learnRates) == 1 {
		return learnRates[0]
	}
	if layerIndex < len(learnRates) {
		return learnRates[layerIndex]
	}
	return learnRates[len(learnRates)-1]
}

func (j *JSDNN) train(x, y tensor.Tensor, gameEvent *dominos.GameEvent, learnRates []float64, mask tensor.Tensor) (float64, error) {
	err := x.Reshape(x.Shape()[0], 1)
	if err != nil {
		log.Fatal(err, " ", "x")
		return 0.0, err
	}

	err = y.Reshape(y.Shape()[0], 1)
	if err != nil {
		log.Fatal(err, " ", "y")
		return 0.0, err
	}

	hiddenActs := []tensor.Tensor{x}
	hidden, err := j.hidden[0].MatVecMul(x)
	if err != nil {
		log.Fatal(err, " ", "1")
		return 0.0, err
	}

	_, err = tensor.Add(hidden, j.bHidden[0], tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "add3")
		return 0.0, err
	}

	hiddenActivation, err := hidden.Apply(relu, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "2")
		return 0.0, err
	}

	err = hiddenActivation.Reshape(hiddenActivation.Shape()[0], 1)
	if err != nil {
		log.Fatal(err, " ", "hA")
		return 0.0, err
	}

	hiddenActs = append(hiddenActs, hiddenActivation.Clone().(tensor.Tensor))

	for i := 1; i < len(j.hidden); i++ {
		hidden, err = j.hidden[i].MatVecMul(hiddenActivation)
		if err != nil {
			log.Fatal(err, " ", "1")
			return 0.0, err
		}

		_, err = tensor.Add(hidden, j.bHidden[i], tensor.UseUnsafe())
		if err != nil {
			log.Fatal(err, " ", "add4")
			return 0.0, err
		}

		hiddenActivation, err = hidden.Apply(relu, tensor.UseUnsafe())
		if err != nil {
			log.Fatal(err, " ", "2")
			return 0.0, err
		}

		err = hiddenActivation.Reshape(hiddenActivation.Shape()[0], 1)
		if err != nil {
			log.Fatal(err, " ", "hA")
			return 0.0, err
		}
		hiddenActs = append(hiddenActs, hiddenActivation.Clone().(tensor.Tensor))
	}

	final, err := tensor.MatVecMul(j.final, hiddenActivation)
	if err != nil {
		log.Fatal(err, " ", "3")
		return 0.0, err
	}

	_, err = tensor.Add(final, j.bFinal, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "add5")
		return 0.0, err
	}

	var pred *tensor.Dense
	if j.OutputActivation != "linear" {
		predT, applyErr := final.Apply(relu, tensor.UseSafe())
		if applyErr != nil {
			log.Fatal(applyErr, " ", "4")
			return 0.0, applyErr
		}
		pred = predT.(*tensor.Dense)
	} else {
		pred = final.Clone().(*tensor.Dense)
	}

	err = pred.Reshape(pred.Shape()[0], 1)
	if err != nil {
		log.Fatal(err, " ", "pred")
		return 0.0, err
	}

	// Backpropagate
	// Equivalent to the derivative of the cost with respect to the activation
	outputErrors, err := tensor.Sub(y, pred)
	if err != nil {
		log.Fatal(err, " ", "5")
		return 0.0, err
	}

	// Apply mask to only update the taken action's error (if mask provided)
	if mask != nil {
		err = mask.Reshape(mask.Shape()[0], 1)
		if err != nil {
			log.Fatal(err, " ", "mask reshape")
			return 0.0, err
		}
		_, err = tensor.Mul(outputErrors, mask, tensor.UseUnsafe())
		if err != nil {
			log.Fatal(err, " ", "mask mul")
			return 0.0, err
		}
	}

	// Clip gradients to prevent explosion
	_, err = outputErrors.Apply(clipValue, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "clip")
		return 0.0, err
	}

	// Calculate the derivative of the output activation with respect to the hidden activaton layer
	var dpred *tensor.Dense
	if j.OutputActivation != "linear" {
		dpredT, applyErr := pred.Apply(reluPrime, tensor.UseUnsafe())
		if applyErr != nil {
			log.Fatal(applyErr, " ", "6")
			return 0.0, applyErr
		}
		dpred = dpredT.(*tensor.Dense)
	} else {
		// Linear activation: derivative is 1.0 everywhere
		ones := make([]float64, pred.Shape()[0])
		for i := range ones {
			ones[i] = 1.0
		}
		dpred = tensor.New(tensor.WithShape(pred.Shape()...), tensor.WithBacking(ones))
	}

	j.final.T()
	hiddErrs, err := tensor.MatMul(j.final, outputErrors)
	if err != nil {
		log.Fatal(err, " ", "7")
		return 0.0, err
	}
	j.final.UT()

	cost := sum(outputErrors.Data().([]float64))

	// Multiply the prediction by the output errors
	_, err = tensor.Mul(dpred, outputErrors, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "8")
		return 0.0, err
	}

	// The Bias is just the same term without the hidden activation
	dcost_dbiasfinal := dpred.Clone().(tensor.Tensor)

	hiddenActivation.T()
	dcost_dfinal, err := tensor.MatMul(dpred, hiddenActivation) /// was output errs
	if err != nil {
		log.Fatal(err, " ", "9")
		return 0.0, err
	}
	hiddenActivation.UT()

	dHiddenActivation, err := hiddenActivation.Apply(reluPrime)
	if err != nil {
		log.Fatal(err, " ", "10")
		return 0.0, err
	}

	_, err = tensor.Mul(hiddErrs, dHiddenActivation)
	if err != nil {
		log.Fatal(err, " ", "11")
		return 0.0, err
	}

	err = hiddErrs.Reshape(hiddErrs.Shape()[0], 1)
	if err != nil {
		log.Fatal(err, " ", "12")
		return 0.0, err
	}

	// The Bias is just the same term without the hidden activation
	dcost_dbiashidden := hiddErrs.Clone().(tensor.Tensor)
	hErrs := hiddErrs.Clone().(tensor.Tensor)

	hiddenActs[len(hiddenActs)-2].T()
	dcost_dhidden, err := tensor.MatMul(hiddErrs, hiddenActs[len(hiddenActs)-2])
	if err != nil {
		log.Fatal(err, " ", "13")
		return 0.0, err
	}
	hiddenActs[len(hiddenActs)-2].UT()

	// Update the gradients
	finalLR := j.getLearnRate(learnRates, len(j.hidden))
	_, err = tensor.Mul(dcost_dfinal, finalLR, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "14")
		return 0.0, err
	}

	_, err = tensor.Mul(dcost_dbiasfinal, finalLR, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "15")
		return 0.0, err
	}

	_, err = tensor.Add(j.final, dcost_dfinal, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "add6")
		log.Fatal(err, " ", "16")
		return 0.0, err
	}

	_, err = tensor.Add(j.bFinal, dcost_dbiasfinal, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "add19")
		log.Fatal(err, " ", "17")
		return 0.0, err
	}

	lastHiddenLR := j.getLearnRate(learnRates, len(j.hidden)-1)
	_, err = tensor.Mul(dcost_dbiashidden, lastHiddenLR, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "15")
		return 0.0, err
	}

	_, err = tensor.Mul(dcost_dhidden, lastHiddenLR, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "15")
		return 0.0, err
	}

	_, err = tensor.Add(j.hidden[len(j.hidden)-1], dcost_dhidden, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "add111")
		return 0.0, err
	}

	_, err = tensor.Add(j.bHidden[len(j.bHidden)-1], dcost_dbiashidden, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "add7")
		return 0.0, err
	}

	// Calculate the update for the rest of the hidden layers
	j.hidden[len(j.hidden)-1].T()
	hErrs, err = tensor.MatMul(j.hidden[len(j.hidden)-1], hErrs)
	if err != nil {
		log.Fatal(err, " ", "7")
		return 0.0, err
	}
	j.hidden[len(j.hidden)-1].UT()

	for i := (len(j.hidden) - 1); i > 0; i-- {

		dHiddenActivation, err := hiddenActs[i].Apply(reluPrime)
		if err != nil {
			log.Fatal(err, " ", "10")
			return 0.0, err
		}

		_, err = tensor.Mul(hErrs, dHiddenActivation)
		if err != nil {
			log.Fatal(err, " ", "11", " ", len(j.hidden))
			return 0.0, err
		}

		err = hErrs.Reshape(hErrs.Shape()[0], 1)
		if err != nil {
			log.Fatal(err, " ", "12")
			return 0.0, err
		}

		// The Bias is just the same term without the hidden activation
		dcost_dbiashidden = hErrs.Clone().(tensor.Tensor)
		hErrsCache := hErrs.Clone().(tensor.Tensor)

		hiddenActs[i-1].T()
		dcost_dhidden, err = tensor.MatMul(hErrs, hiddenActs[i-1])
		if err != nil {
			log.Fatal(err, " ", "13")
			return 0.0, err
		}
		hiddenActs[i-1].UT()

		hiddenLR := j.getLearnRate(learnRates, i-1)
		_, err = tensor.Mul(dcost_dbiashidden, hiddenLR, tensor.UseUnsafe())
		if err != nil {
			log.Fatal(err, " ", "15")
			return 0.0, err
		}

		_, err = tensor.Mul(dcost_dhidden, hiddenLR, tensor.UseUnsafe())
		if err != nil {
			log.Fatal(err, " ", "15")
			return 0.0, err
		}

		_, err = tensor.Add(j.hidden[i-1], dcost_dhidden, tensor.UseUnsafe())
		if err != nil {
			log.Fatal(err, " ", "add8")
			log.Fatal(err, " ", "17")
			return 0.0, err
		}

		_, err = tensor.Add(j.bHidden[i-1], dcost_dbiashidden, tensor.UseUnsafe())
		if err != nil {
			log.Fatal(err, " ", "add9")
			log.Fatal(err, " ", "17")
			return 0.0, err
		}

		j.hidden[i-1].T()
		hErrs, err = tensor.MatMul(j.hidden[i-1], hErrsCache)
		if err != nil {
			log.Fatal(err, " ", "7")
			return 0.0, err
		}
		j.hidden[i-1].UT()

	}

	return cost, nil
}

func (j *JSDNN) Train(gameEvent *dominos.GameEvent, learnRates []float64) (float64, error) {
	if gameEvent.EventType != dominos.PlayedCard && gameEvent.EventType != dominos.PosedCard {
		return 0.0, errors.New("Invalid game event to train with")
	}
	x, compatL, compatR := j.ConvertGameEventToTensor(gameEvent)
	if (compatL + compatR) == 1 {
		log.Debug("If there is only one choice, no need to train it")
		return 0.0, nil
	}
	y := j.ConvertCardChoiceToTensor(gameEvent)

	return j.train(x, y, gameEvent, learnRates, nil)
}

func (j *JSDNN) TrainReinforced(gameEvent *dominos.GameEvent, learnRates []float64, nextGameEvents [16]*dominos.GameEvent) (float64, error) {
	if gameEvent.EventType != dominos.PlayedCard && gameEvent.EventType != dominos.PosedCard {
		return 0.0, errors.New("Invalid game event to train with")
	}
	x, compatL, compatR := j.ConvertGameEventToTensor(gameEvent)
	if (compatL + compatR) == 1 {
		log.Debug("If there is only one choice, no need to train it")
		return 0.0, nil
	}
	y := j.ConvertCardChoiceToTensorReinforced(gameEvent, nextGameEvents)

	// Build a mask: 1.0 only at the index of the action taken, 0.0 elsewhere
	actionMask := [56]float64{}
	actionIndex := gameEvent.Card
	if gameEvent.Side == dominos.Right {
		actionIndex = actionIndex + 28
	}
	actionMask[actionIndex] = 1.0
	mask := tensor.New(tensor.WithShape(56), tensor.WithBacking(actionMask[:]))

	return j.train(x, y, gameEvent, learnRates, mask)
}

func (j *JSDNN) Save(fileName string) error {
	f, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)

	for i := range j.hidden {
		err = enc.Encode(j.hidden[i])
		if err != nil {
			return err
		}

	}

	err = enc.Encode(j.final)
	if err != nil {
		return err
	}

	for i := range j.bHidden {
		err = enc.Encode(j.bHidden[i])
		if err != nil {
			return err
		}
	}

	err = enc.Encode(j.bFinal)
	if err != nil {
		return err
	}

	return nil
}

func (j *JSDNN) Load(fileName string) error {
	f, err := os.Open(fileName)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	for i := range j.hidden {
		err = dec.Decode(&j.hidden[i])
		if err != nil {
			return err
		}
	}

	err = dec.Decode(&j.final)
	if err != nil {
		return err
	}

	for i := range j.bHidden {
		err = dec.Decode(&j.bHidden[i])
		if err != nil {
			return err
		}

	}

	err = dec.Decode(&j.bFinal)
	if err != nil {
		return err
	}

	return nil
}

func (j *JSDNN) PoseCard(doms []dominos.Domino) uint {
	hand := j.ComputerPlayer.GetHand()
	suitCount := [7]int{}
	for i := range hand {
		c := hand[i]
		if int(c) >= len(dominos.GetCLIDominos()) {
			continue
		}
		dom := dominos.GetCLIDominos()[c]
		suit1, _ := dom.GetSuits()
		if dom.IsDouble() {
			suitCount[suit1]++
		}
	}

	for i := range hand {
		c := hand[i]
		if int(c) >= len(dominos.GetCLIDominos()) {
			continue
		}
		dom := dominos.GetCLIDominos()[c]
		suit1, suit2 := dom.GetSuits()
		suitSum := suitCount[suit1] + suitCount[suit2]
		if !dom.IsDouble() && suitSum > 0 {
			suitCount[suit1]++
			suitCount[suit2]++
		}
	}

	maxSuit := 0
	maxCount := 0
	for i := range suitCount {
		if suitCount[i] >= maxCount {
			maxCount = suitCount[i]
			maxSuit = i
		}
	}

	for i := range hand {
		dom := dominos.GetCLIDominos()[hand[i]]
		suit1, _ := dom.GetSuits()
		if dom.IsDouble() && suit1 == uint(maxSuit) {
			return hand[i]
		}
	}

	return j.ComputerPlayer.PoseCard(doms)
}

func (j *JSDNN) PlayCard(gameEvent *dominos.GameEvent, doms []dominos.Domino) (uint, dominos.BoardSide) {
	gameEvent = jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)

	// Epsilon-greedy exploration: pick a random compatible card
	if j.Epsilon > 0 && rand.Float64() < j.Epsilon {
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

	cardChoice, err := j.Predict(gameEvent)
	if err != nil || cardChoice.Card > 27 {
		log.Warn("Error predicting using AI: ", err)
		return j.ComputerPlayer.PlayCard(gameEvent, doms)
	}
	_, compatL, compatR := j.ConvertGameEventToTensor(gameEvent)

	choices := compatL + compatR
	typ := "predict"
	if j.Search && choices > 1 && gameEvent.PlayerCardsRemaining[0] < 5 {
		cardChoiceSearch, probability, err := j.PredictSearch(gameEvent, j.SearchNum, 0.001)
		if err != nil {
			log.Warn("Error predicting Search using AI")
			return j.ComputerPlayer.PlayCard(gameEvent, doms)
		}

		sameChoice := true
		if cardChoiceSearch.Card != cardChoice.Card || cardChoiceSearch.Side != cardChoice.Side {
			sameChoice = false
		}

		if sameChoice && probability < 0.01 && !doms[cardChoice.Card].IsDouble() {
			explorer := New(j.inputDim, j.hiddenDim, j.outputDim)
			cardChoiceExplorer, err := explorer.Predict(gameEvent)
			if err == nil {
				compatibleCard := false
				suit1, suit2 := doms[cardChoice.Card].GetSuits()
				if cardChoice.Side == dominos.Left {
					if suit1 == gameEvent.BoardState.LeftSuit || suit2 == gameEvent.BoardState.LeftSuit {
						compatibleCard = true
					}
				} else {
					if suit1 == gameEvent.BoardState.RightSuit || suit2 == gameEvent.BoardState.RightSuit {
						compatibleCard = true
					}
				}
				if compatibleCard {
					cardChoice = cardChoiceExplorer
				}
				typ = "explored"
			}
		} else if probability > 0.90 && !sameChoice {
			cardChoice = cardChoiceSearch
			log.Infof("Better choice %+#v Probab %.2f", cardChoiceSearch, probability)
			typ = "better"
		}

	}

	compatibleCard := false
	suit1, suit2 := doms[cardChoice.Card].GetSuits()
	if cardChoice.Side == dominos.Left {
		if suit1 == gameEvent.BoardState.LeftSuit || suit2 == gameEvent.BoardState.LeftSuit {
			compatibleCard = true
		}
	} else {
		if suit1 == gameEvent.BoardState.RightSuit || suit2 == gameEvent.BoardState.RightSuit {
			compatibleCard = true
		}
	}

	if !compatibleCard {
		log.Warnf("AI picked incompatible card: %+#v %s", cardChoice, typ)
		return j.ComputerPlayer.PlayCard(gameEvent, doms)
	}
	// log.Infof("Precicted: %+#v", cardChoice)
	return cardChoice.Card, cardChoice.Side
}
