package nn

import (
	"encoding/gob"
	"errors"
	"math"
	"os"

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
	passMemory [28]float64
	TotalWins  int
	TotalWins2 int
}

func fillRandom(a []float64, v float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
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

	for i := range hidden {
		previous := input
		backing := r
		next := output
		if (i + 1) < len(hidden) {
			next = hidden[i+1]
		}
		if i > 0 {
			previous = hidden[i-1]
			backing = r2[i-1]
		}
		h := make([]float64, hidden[i]*next)
		hb := make([]float64, hidden[i])
		fillRandom(h, float64(len(h)))
		fillRandom(hb, float64(len(hb)))
		r2 = append(r2, h)
		r3 = append(r3, hb)
		hiddenT = append(hiddenT, tensor.New(tensor.WithShape(hidden[i], previous), tensor.WithBacking(backing)))
		bHidden = append(bHidden, tensor.New(tensor.WithShape(hidden[i]), tensor.WithBacking(r3[i])))
	}

	fillRandom(r, float64(len(r)))

	finalT := tensor.New(tensor.WithShape(output, hidden[len(hidden)-1]), tensor.WithBacking(r2[len(r2)-1]))
	bFinal := tensor.New(tensor.WithShape(output), tensor.WithBacking(r4))

	return &JSDNN{
		hidden:         hiddenT,
		final:          finalT,
		bHidden:        bHidden,
		bFinal:         bFinal,
		ComputerPlayer: &dominos.ComputerPlayer{},
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

func (j *JSDNN) UpdatePassMemory(gameEvent *dominos.GameEvent) {
	gameEvent = jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
	card := gameEvent.Card
	player := gameEvent.Player
	dom := dominos.GetCLIDominos()[card]
	suit1, suit2 := dom.GetSuits()
	suit1Index := uint(player*7) + suit1
	suit2Index := uint(player*7) + suit2
	j.passMemory[suit1Index] = 1.0
	j.passMemory[suit2Index] = 1.0
}

func (j *JSDNN) ResetPassMemory() {
	j.passMemory = [28]float64{}
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

func (j *JSDNN) ConvertCardChoiceToTensorReinforced(gameEvent *dominos.GameEvent, lastGameEvent *dominos.GameEvent) tensor.Tensor {
	cardChoice := [56]float64{}
	index := gameEvent.Card
	if gameEvent.Side == dominos.Right {
		index = index + 28
	}
	rew := float64(7-lastGameEvent.PlayerCardsRemaining[0]) / 7.0
	isDouble := dominos.GetCLIDominos()[gameEvent.Card].IsDouble()
	if rew == 1 {
		rew = rew * 12.5
		reward := rew + float64((3*lastGameEvent.PlayerCardsRemaining[1])+lastGameEvent.PlayerCardsRemaining[2]+lastGameEvent.PlayerCardsRemaining[3])/21.0
		if isDouble {
			reward = reward * 5.0
		}
		countRatio := float64(dominos.GetCLIDominos()[gameEvent.Card].GetCount()) / 12.0
		cardChoice[index] = 1.0*reward + countRatio
	} else {
		reward := -1.0 * float64(lastGameEvent.PlayerCardsRemaining[0])
		if !isDouble {
			reward = 2.5
		}

		countRatio := float64(dominos.GetCLIDominos()[gameEvent.Card].GetCount()) / 12.0
		cardChoice[index] = 1.0*reward + countRatio
	}

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
		return nil, err
	}

	prediction, err := final.Apply(relu, tensor.UseSafe())
	if err != nil {
		return nil, err
	}

	return prediction, nil
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

	// Wef cant predict dominoes not in our hand
	outputMask := j.GetOutputMask(gameEvent)
	_, err = tensor.Mul(prediction, outputMask, tensor.UseUnsafe())
	if err != nil {
		return &dominos.CardChoice{}, err
	}

	choice := &dominos.CardChoice{}
	cardConfidences, ok := prediction.Data().([]float64)
	maxConfidence := -1.0
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
				}

			}
		}
	}
	// log.Infof("Card Choice: %+#v, Confidence: %.5f", choice, maxConfidence)
	return choice, nil
}

func (j *JSDNN) train(x, y tensor.Tensor, gameEvent *dominos.GameEvent, learnRate float64) (float64, error) {
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

	for i := 1; i < len(j.hidden); i++ {
		hidden, err = j.hidden[i].MatVecMul(hiddenActivation)
		if err != nil {
			log.Fatal(err, " ", "1")
			return 0.0, err
		}

		_, err = tensor.Add(hidden, j.bHidden[i], tensor.UseUnsafe())
		if err != nil {
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
		return 0.0, err
	}

	pred, err := final.Apply(relu, tensor.UseSafe())
	if err != nil {
		log.Fatal(err, " ", "4")
		return 0.0, err
	}

	// We cant predict dominoes not in our hand
	outputMask := j.GetOutputMask(gameEvent)
	_, err = tensor.Mul(pred, outputMask, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "4")
		return 0.0, err
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

	// Calculate the derivative of the output activation with respect to the hidden activaton layer
	dpred, err := pred.Apply(reluPrime, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "6")
		return 0.0, err
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

	hiddenActs[len(hiddenActs)-1].T()
	dcost_dhidden, err := tensor.MatMul(hiddErrs, hiddenActs[len(hiddenActs)-1])
	if err != nil {
		log.Fatal(err, " ", "13")
		return 0.0, err
	}
	hiddenActs[len(hiddenActs)-1].UT()

	// Update the gradients
	_, err = tensor.Mul(dcost_dfinal, learnRate, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "14")
		return 0.0, err
	}

	_, err = tensor.Mul(dcost_dbiasfinal, learnRate, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "15")
		return 0.0, err
	}

	_, err = tensor.Add(j.final, dcost_dfinal, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "16")
		return 0.0, err
	}

	_, err = tensor.Add(j.bFinal, dcost_dbiasfinal, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "17")
		return 0.0, err
	}

	_, err = tensor.Mul(dcost_dbiashidden, learnRate, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "15")
		return 0.0, err
	}

	_, err = tensor.Mul(dcost_dhidden, learnRate, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "15")
		return 0.0, err
	}

	_, err = tensor.Add(j.hidden[len(j.hidden)-1], dcost_dhidden, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "17")
		return 0.0, err
	}

	_, err = tensor.Add(j.bHidden[len(j.bHidden)-1], dcost_dbiashidden, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "17")
		return 0.0, err
	}

	return cost, nil
}

func (j *JSDNN) Train(gameEvent *dominos.GameEvent, learnRate float64) (float64, error) {
	if gameEvent.EventType != dominos.PlayedCard && gameEvent.EventType != dominos.PosedCard {
		return 0.0, errors.New("Invalid game event to train with")
	}
	x, compatL, compatR := j.ConvertGameEventToTensor(gameEvent)
	if (compatL + compatR) == 1 {
		log.Debug("If there is only one choice, no need to train it")
		return 0.0, nil
	}
	y := j.ConvertCardChoiceToTensor(gameEvent)

	return j.train(x, y, gameEvent, learnRate)
}

func (j *JSDNN) TrainReinforced(gameEvent *dominos.GameEvent, learnRate float64, finalGameEvent *dominos.GameEvent) (float64, error) {
	if gameEvent.EventType != dominos.PlayedCard && gameEvent.EventType != dominos.PosedCard {
		return 0.0, errors.New("Invalid game event to train with")
	}
	x, compatL, compatR := j.ConvertGameEventToTensor(gameEvent)
	if (compatL + compatR) == 1 {
		log.Debug("If there is only one choice, no need to train it")
		return 0.0, nil
	}
	y := j.ConvertCardChoiceToTensorReinforced(gameEvent, finalGameEvent)

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

	return j.train(x, y, gameEvent, learnRate)
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

	gameEvent := &dominos.GameEvent{
		EventType: dominos.PlayerTurn, Player: 0,
		BoardState: &dominos.Board{}, PlayerWins: [4]int{},
		PoseModeDoubleSix:    false,
		PlayerCardsRemaining: [4]int{7, 7, 7, 7}, GameState: dominos.ExecutePlayerPose,
		PlayerHands: [4][]uint{j.ComputerPlayer.GetHand(), {}, {}, {}},
	}

	cardChoice, err := j.Predict(gameEvent)
	if err != nil {
		log.Warn("Error predicting using AI")
		return j.ComputerPlayer.PoseCard(doms)
	}

	cardChoiceInHand := false
	for i := range j.GetHand() {
		if hand[i] == cardChoice.Card {
			cardChoiceInHand = true
		}
	}
	if !cardChoiceInHand {
		return j.ComputerPlayer.PoseCard(doms)
	}

	return cardChoice.Card
}

func (j *JSDNN) PlayCard(gameEvent *dominos.GameEvent, doms []dominos.Domino) (uint, dominos.BoardSide) {
	gameEvent = jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
	cardChoice, err := j.Predict(gameEvent)
	if err != nil {
		log.Warn("Error predicting using AI")
		return j.ComputerPlayer.PlayCard(gameEvent, doms)
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
		log.Warnf("AI picked incompatible card: %+#v", cardChoice)
		return j.ComputerPlayer.PlayCard(gameEvent, doms)
	}
	// log.Infof("Precicted: %+#v", cardChoice)
	return cardChoice.Card, cardChoice.Side
}
