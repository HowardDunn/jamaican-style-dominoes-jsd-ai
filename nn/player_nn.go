package nn

import (
	"encoding/gob"
	"errors"
	"math"
	"os"

	"github.com/HowardDunn/go-dominos/dominos"
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
	hidden *tensor.Dense
	final  *tensor.Dense
	b0, b1 float64
	*dominos.ComputerPlayer
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

func New(input, hidden, output int) *JSDNN {
	if output != 56 {
		panic("Invalid output size")
	}

	r := make([]float64, input*hidden)
	r2 := make([]float64, output*hidden)
	fillRandom(r, float64(len(r)))
	fillRandom(r2, float64(len(r2)))
	hiddenT := tensor.New(tensor.WithShape(hidden, input), tensor.WithBacking(r))
	finalT := tensor.New(tensor.WithShape(output, hidden), tensor.WithBacking(r2))

	return &JSDNN{
		hidden:         hiddenT,
		final:          finalT,
		ComputerPlayer: &dominos.ComputerPlayer{},
	}
}

func relu(x float64) float64 {
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

func (j *JSDNN) ConvertGameEventToTensor(gameEvent *dominos.GameEvent) tensor.Tensor {
	type JSDNNGameState struct {
		playerHand    [28]float64
		boardState    [28]float64
		suitState     [14]float64
		playerPass    [28]float64
		cardRemaining [28]float64
	}

	jsdgameState := &JSDNNGameState{}

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
				jsdgameState.playerHand[card] = 1.0
			}

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
	return res
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
	hidden, err := j.hidden.MatVecMul(a)
	if err != nil {
		return nil, err
	}
	hiddenActivation, err := hidden.Apply(relu, tensor.UseUnsafe())
	if err != nil {
		return nil, err
	}
	final, err := tensor.MatVecMul(j.final, hiddenActivation)
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
	a := j.ConvertGameEventToTensor(gameEvent)
	if a.Dims() != 1 {
		return nil, errors.New("expected a vector")
	}

	prediction, err := j.predict(a)
	if err != nil {
		return &dominos.CardChoice{}, err
	}

	// We cant predict dominoes not in our hand
	outputMask := j.GetOutputMask(gameEvent)
	_, err = tensor.Mul(prediction, outputMask, tensor.UseUnsafe())
	if err != nil {
		return &dominos.CardChoice{}, err
	}

	choice := &dominos.CardChoice{}
	cardConfidences, ok := prediction.Data().([]float64)
	maxConfidence := 0.0
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
	if ok {
		for i, cardConfidence := range cardConfidences {
			if cardConfidence > maxConfidence {

				side := dominos.Left
				card := i
				if i > 27 {
					side = dominos.Right
					card = card - 28
				}
				if contained(uint(card)) {
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

func (j *JSDNN) Train(gameEvent *dominos.GameEvent, learnRate float64) (float64, error) {
	if gameEvent.EventType != dominos.PlayedCard && gameEvent.EventType != dominos.PosedCard {
		return 0.0, errors.New("Invalid game event to train with")
	}
	x := j.ConvertGameEventToTensor(gameEvent)
	y := j.ConvertCardChoiceToTensor(gameEvent)

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

	hidden, err := j.hidden.MatVecMul(x)
	if err != nil {
		log.Fatal(err, " ", "1")
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

	final, err := tensor.MatVecMul(j.final, hiddenActivation)
	if err != nil {
		log.Fatal(err, " ", "3")
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

	x.T()
	dcost_dhidden, err := tensor.MatMul(hiddErrs, x)
	if err != nil {
		log.Fatal(err, " ", "13")
		return 0.0, err
	}
	x.UT()

	// Update the gradients
	_, err = tensor.Mul(dcost_dfinal, learnRate, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "14")
		return 0.0, err
	}

	_, err = tensor.Mul(dcost_dhidden, learnRate, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "15")
		return 0.0, err
	}

	_, err = tensor.Add(j.final, dcost_dfinal, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "16")
		return 0.0, err
	}

	_, err = tensor.Add(j.hidden, dcost_dhidden, tensor.UseUnsafe())
	if err != nil {
		log.Fatal(err, " ", "17")
		return 0.0, err
	}

	return cost, nil
}

func (j *JSDNN) Save(fileName string) error {
	f, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	err = enc.Encode(j.hidden)
	if err != nil {
		return err
	}
	err = enc.Encode(j.final)
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
	err = dec.Decode(&j.hidden)
	if err != nil {
		return err
	}
	err = dec.Decode(&j.final)
	if err != nil {
		return err
	}

	return nil
}

func (j *JSDNN) PlayCard(gameEvent *dominos.GameEvent, doms []dominos.Domino) (uint, dominos.BoardSide) {
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
		log.Warn("AI picked incompatible card")
		return j.ComputerPlayer.PlayCard(gameEvent, doms)
	}
	log.Infof("Precicted: %+#v", cardChoice)
	return cardChoice.Card, cardChoice.Side
}
