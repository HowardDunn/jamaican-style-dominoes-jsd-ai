package nn

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"

	"github.com/HowardDunn/go-dominos/dominos"
	jsdonline "github.com/HowardDunn/jsd-online-game/game"
	log "github.com/sirupsen/logrus"
)

// KerasPlayer implements dominos.Player using a remote Keras/TF server for
// GPU-accelerated training, with fast in-process Go inference.
type KerasPlayer struct {
	*dominos.ComputerPlayer
	serverURL     string
	modelID       string
	hiddenDims    []int
	passMemory    [28]float64
	knownNotHaves [4]map[uint]bool
	TotalWins     int
	TotalWins2    int
	Epsilon       float64
	httpClient    *http.Client

	// Local weight copies for fast Go-side inference.
	// Keras Dense layers: each has a kernel (rows=in, cols=out) and bias (out).
	// Stored as flat row-major slices with their shapes.
	localWeights [][]float64 // flat weight arrays from Keras model.get_weights()
	localShapes  [][]int     // shapes corresponding to each weight array
	weightsReady bool        // true after first SyncWeights or TrainBatch
}

func NewKerasPlayer(serverURL, modelID string, hiddenDims []int) *KerasPlayer {
	knownNotHaves := [4]map[uint]bool{}
	for i := 0; i < 4; i++ {
		knownNotHaves[i] = make(map[uint]bool)
	}
	return &KerasPlayer{
		ComputerPlayer: &dominos.ComputerPlayer{},
		serverURL:      serverURL,
		modelID:        modelID,
		hiddenDims:     hiddenDims,
		knownNotHaves:  knownNotHaves,
		httpClient:     &http.Client{},
	}
}

func (*KerasPlayer) GetPlayerType() string {
	return "localHumanPlayer"
}

func (k *KerasPlayer) GetHiddenDims() []int {
	return k.hiddenDims
}

// --- HTTP helpers ---

type trainSample struct {
	Features   []float64 `json:"features"`
	Target     []float64 `json:"target"`
	ActionMask []float64 `json:"action_mask"`
}

type trainReq struct {
	ModelID      string        `json:"model_id"`
	Samples      []trainSample `json:"samples"`
	LearningRate float64       `json:"learning_rate"`
	HiddenDims   []int         `json:"hidden_dims,omitempty"`
}

type trainResp struct {
	AvgLoss float64       `json:"avg_loss"`
	Weights [][]float64   `json:"weights"`
}

type saveLoadReq struct {
	ModelID    string `json:"model_id"`
	Path       string `json:"path,omitempty"`
	HiddenDims []int  `json:"hidden_dims,omitempty"`
}

type modelIDReq struct {
	ModelID    string `json:"model_id"`
	HiddenDims []int  `json:"hidden_dims,omitempty"`
}

type getWeightsResp struct {
	Weights [][]float64 `json:"weights"`
	Shapes  [][]int     `json:"shapes"`
}

func (k *KerasPlayer) post(endpoint string, reqBody any, respBody any) error {
	data, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	resp, err := k.httpClient.Post(k.serverURL+endpoint, "application/json", bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("post %s: %w", endpoint, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		var errBody map[string]any
		json.NewDecoder(resp.Body).Decode(&errBody)
		return fmt.Errorf("server %s returned %d: %v", endpoint, resp.StatusCode, errBody)
	}
	if respBody != nil {
		return json.NewDecoder(resp.Body).Decode(respBody)
	}
	return nil
}

// --- Local forward pass (no HTTP) ---

// predictLocal runs inference using locally cached weights.
// Keras Dense layout: kernel shape [in, out] (row-major), bias shape [out].
// Weights order: [hidden0_kernel, hidden0_bias, hidden1_kernel, hidden1_bias, ..., output_kernel, output_bias]
func (k *KerasPlayer) predictLocal(features []float64, validMask []float64) (uint, dominos.BoardSide) {
	activation := make([]float64, len(features))
	copy(activation, features)

	numLayers := len(k.localWeights) / 2

	for layer := 0; layer < numLayers; layer++ {
		kernel := k.localWeights[layer*2]
		bias := k.localWeights[layer*2+1]
		shape := k.localShapes[layer*2] // [in, out]
		inDim := shape[0]
		outDim := shape[1]
		isLast := layer == numLayers-1

		output := make([]float64, outDim)
		// matrix-vector multiply: output[j] = sum_i(kernel[i*outDim + j] * activation[i]) + bias[j]
		for j := 0; j < outDim; j++ {
			sum := bias[j]
			for i := 0; i < inDim; i++ {
				sum += kernel[i*outDim+j] * activation[i]
			}
			if !isLast {
				// ReLU for hidden layers
				if sum < 0 {
					sum = 0
				}
			}
			output[j] = sum
		}
		activation = output
	}

	// Apply mask and pick best action
	bestIdx := -1
	bestVal := -math.MaxFloat64
	for i := 0; i < 56; i++ {
		if validMask[i] > 0 && activation[i] > bestVal {
			bestVal = activation[i]
			bestIdx = i
		}
	}

	if bestIdx < 0 {
		return 0, dominos.Left
	}
	if bestIdx >= 28 {
		return uint(bestIdx - 28), dominos.Right
	}
	return uint(bestIdx), dominos.Left
}

// SyncWeights fetches the current model weights from the server for local inference.
func (k *KerasPlayer) SyncWeights() error {
	req := modelIDReq{ModelID: k.modelID, HiddenDims: k.hiddenDims}
	var resp getWeightsResp
	if err := k.post("/get_weights", req, &resp); err != nil {
		return fmt.Errorf("sync weights: %w", err)
	}
	k.localWeights = resp.Weights
	k.localShapes = resp.Shapes
	k.weightsReady = true
	return nil
}

// --- Pass memory (same logic as JSDNN, kept in Go) ---

func (k *KerasPlayer) UpdatePassMemory(gameEvent *dominos.GameEvent) {
	gameEvent = jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
	card := gameEvent.Card
	player := gameEvent.Player
	doms := dominos.GetCLIDominos()
	dom := doms[card]
	suit1, suit2 := dom.GetSuits()
	suit1Index := uint(player*7) + suit1
	suit2Index := uint(player*7) + suit2
	k.passMemory[suit1Index] = 1.0
	k.passMemory[suit2Index] = 1.0
	for i := 0; i < 28; i++ {
		suit11, suit22 := doms[i].GetSuits()
		if suit11 == suit1 || suit11 == suit2 {
			k.knownNotHaves[player][uint(i)] = true
		} else if suit22 == suit1 || suit22 == suit2 {
			k.knownNotHaves[player][uint(i)] = true
		}
	}
}

func (k *KerasPlayer) ResetPassMemory() {
	k.passMemory = [28]float64{}
	k.knownNotHaves = [4]map[uint]bool{}
	for i := 0; i < 4; i++ {
		k.knownNotHaves[i] = make(map[uint]bool)
		for j := 0; j < 28; j++ {
			k.knownNotHaves[i][uint(j)] = false
		}
	}
}

// --- Tensor conversion (same logic as JSDNN, reused here) ---

func (k *KerasPlayer) convertGameEventToFeatures(gameEvent *dominos.GameEvent) ([]float64, int, int) {
	playerHand := [28]float64{}
	boardState := [28]float64{}
	suitState := [14]float64{}
	cardRemaining := [28]float64{}

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
				playerHand[card] = 1.0
			}
		}
	}

	if gameEvent.EventType == dominos.PlayedCard || gameEvent.EventType == dominos.PosedCard {
		if gameEvent.BoardState.CardPosed {
			suit1, suit2 := dominos.GetCLIDominos()[gameEvent.Card].GetSuits()
			if suit1 == gameEvent.BoardState.LeftSuit || suit2 == gameEvent.BoardState.LeftSuit {
				playerHand[gameEvent.Card] = 1.0
			}
			if suit1 == gameEvent.BoardState.RightSuit || suit2 == gameEvent.BoardState.RightSuit {
				playerHand[gameEvent.Card] = 1.0
			}
		}
	}

	if gameEvent.BoardState.CardPosed {
		boardState[gameEvent.BoardState.PosedCard] = 1.0
		for i := range gameEvent.BoardState.LeftBoard {
			boardState[gameEvent.BoardState.LeftBoard[i]] = 1.0
		}
		for i := range gameEvent.BoardState.RightBoard {
			boardState[gameEvent.BoardState.RightBoard[i]] = 1.0
		}
		suitState[gameEvent.BoardState.LeftSuit] = 1.0
		suitState[gameEvent.BoardState.RightSuit+7] = 1.0
	}

	cr0 := gameEvent.PlayerCardsRemaining[0] - 1
	cr1 := gameEvent.PlayerCardsRemaining[1] + 6
	cr2 := gameEvent.PlayerCardsRemaining[2] + 13
	cr3 := gameEvent.PlayerCardsRemaining[3] + 20
	if cr0 < 0 {
		cr0 = 0
	}
	if cr1 < 0 {
		cr1 = 0
	}
	if cr2 < 0 {
		cr2 = 0
	}
	if cr3 < 0 {
		cr3 = 0
	}
	cardRemaining[cr0] = 1.0
	cardRemaining[cr1] = 1.0
	cardRemaining[cr2] = 1.0
	cardRemaining[cr3] = 1.0

	features := make([]float64, 0, 126)
	features = append(features, playerHand[:]...)
	features = append(features, boardState[:]...)
	features = append(features, suitState[:]...)
	features = append(features, k.passMemory[:]...)
	features = append(features, cardRemaining[:]...)

	return features, compatibleCardsL, compatibleCardsR
}

func (k *KerasPlayer) getOutputMask(gameEvent *dominos.GameEvent) []float64 {
	mask := [56]float64{}
	for i := range gameEvent.PlayerHands[gameEvent.Player] {
		card := gameEvent.PlayerHands[gameEvent.Player][i]
		if card < 28 {
			suit1, suit2 := dominos.GetCLIDominos()[card].GetSuits()
			// Left side: index 0-27
			if suit1 == gameEvent.BoardState.LeftSuit || suit2 == gameEvent.BoardState.LeftSuit {
				mask[card] = 1.0
			}
			// Right side: index 28-55
			if suit1 == gameEvent.BoardState.RightSuit || suit2 == gameEvent.BoardState.RightSuit {
				mask[card+28] = 1.0
			}
		}
	}
	if gameEvent.EventType == dominos.PlayedCard || gameEvent.EventType == dominos.PosedCard {
		if gameEvent.Card < 28 {
			suit1, suit2 := dominos.GetCLIDominos()[gameEvent.Card].GetSuits()
			if suit1 == gameEvent.BoardState.LeftSuit || suit2 == gameEvent.BoardState.LeftSuit {
				mask[gameEvent.Card] = 1.0
			}
			if suit1 == gameEvent.BoardState.RightSuit || suit2 == gameEvent.BoardState.RightSuit {
				mask[gameEvent.Card+28] = 1.0
			}
		}
	}
	return mask[:]
}

func (k *KerasPlayer) convertCardChoiceToTargetReinforced(gameEvent *dominos.GameEvent, nextGameEvents [16]*dominos.GameEvent) []float64 {
	target := [56]float64{}
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
	handSuitCount := k.CountSuit(suitPlayed, dominos.GetCLIDominos())
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
	target[index] = reward
	return target[:]
}

// PrepareTrainSample builds the feature/target/mask arrays for a single training
// event without making any HTTP calls. Returns skip=true if only one valid action.
func (k *KerasPlayer) PrepareTrainSample(gameEvent *dominos.GameEvent, nextGameEvents [16]*dominos.GameEvent) (features []float64, target []float64, actionMask [56]float64, skip bool) {
	features, compatL, compatR := k.convertGameEventToFeatures(gameEvent)
	if (compatL + compatR) == 1 {
		return nil, nil, actionMask, true
	}
	target = k.convertCardChoiceToTargetReinforced(gameEvent, nextGameEvents)

	actionIndex := gameEvent.Card
	if gameEvent.Side == dominos.Right {
		actionIndex = actionIndex + 28
	}
	actionMask[actionIndex] = 1.0
	return features, target, actionMask, false
}

// --- Player interface ---

func (k *KerasPlayer) PoseCard(doms []dominos.Domino) uint {
	hand := k.ComputerPlayer.GetHand()
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
	return k.ComputerPlayer.PoseCard(doms)
}

func (k *KerasPlayer) PlayCard(gameEvent *dominos.GameEvent, doms []dominos.Domino) (uint, dominos.BoardSide) {
	gameEvent = jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)

	// Epsilon-greedy exploration
	if k.Epsilon > 0 && rand.Float64() < k.Epsilon {
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

	features, _, _ := k.convertGameEventToFeatures(gameEvent)
	validMask := k.getOutputMask(gameEvent)

	// Use fast local inference if weights are synced
	if k.weightsReady {
		card, side := k.predictLocal(features, validMask)
		// Validate compatibility
		if card > 27 {
			return k.ComputerPlayer.PlayCard(gameEvent, doms)
		}
		suit1, suit2 := doms[card].GetSuits()
		compatible := false
		if side == dominos.Left {
			if suit1 == gameEvent.BoardState.LeftSuit || suit2 == gameEvent.BoardState.LeftSuit {
				compatible = true
			}
		} else {
			if suit1 == gameEvent.BoardState.RightSuit || suit2 == gameEvent.BoardState.RightSuit {
				compatible = true
			}
		}
		if !compatible {
			log.Warnf("Local predict picked incompatible card %d side %s, falling back", card, side)
			return k.ComputerPlayer.PlayCard(gameEvent, doms)
		}
		return card, side
	}

	// Fallback: no weights synced yet
	return k.ComputerPlayer.PlayCard(gameEvent, doms)
}

// --- Training ---

func (k *KerasPlayer) TrainReinforced(gameEvent *dominos.GameEvent, learnRate float64, nextGameEvents [16]*dominos.GameEvent) (float64, error) {
	if gameEvent.EventType != dominos.PlayedCard && gameEvent.EventType != dominos.PosedCard {
		return 0.0, fmt.Errorf("invalid game event to train with")
	}
	features, compatL, compatR := k.convertGameEventToFeatures(gameEvent)
	if (compatL + compatR) == 1 {
		return 0.0, nil
	}
	target := k.convertCardChoiceToTargetReinforced(gameEvent, nextGameEvents)

	actionMask := [56]float64{}
	actionIndex := gameEvent.Card
	if gameEvent.Side == dominos.Right {
		actionIndex = actionIndex + 28
	}
	actionMask[actionIndex] = 1.0

	return k.TrainBatch(
		[][]float64{features},
		[][]float64{target},
		[][56]float64{actionMask},
		learnRate,
	)
}

// TrainBatch sends a batch of samples to the Keras server for GPU training,
// then syncs the updated weights back for local inference.
func (k *KerasPlayer) TrainBatch(featuresBatch [][]float64, targetBatch [][]float64, maskBatch [][56]float64, learnRate float64) (float64, error) {
	samples := make([]trainSample, len(featuresBatch))
	for i := range featuresBatch {
		samples[i] = trainSample{
			Features:   featuresBatch[i],
			Target:     targetBatch[i],
			ActionMask: maskBatch[i][:],
		}
	}
	req := trainReq{
		ModelID:      k.modelID,
		Samples:      samples,
		LearningRate: learnRate,
		HiddenDims:   k.hiddenDims,
	}
	var resp trainResp
	if err := k.post("/train", req, &resp); err != nil {
		return 0.0, fmt.Errorf("train: %w", err)
	}

	// Update local weights from server response
	if len(resp.Weights) > 0 {
		k.localWeights = resp.Weights
		// Rebuild shapes from hiddenDims
		k.localShapes = k.buildShapes()
		k.weightsReady = true
	}

	return resp.AvgLoss, nil
}

// buildShapes reconstructs the expected Keras weight shapes from the network architecture.
func (k *KerasPlayer) buildShapes() [][]int {
	shapes := [][]int{}
	prevDim := 126
	for _, h := range k.hiddenDims {
		shapes = append(shapes, []int{prevDim, h}) // kernel
		shapes = append(shapes, []int{h})           // bias
		prevDim = h
	}
	shapes = append(shapes, []int{prevDim, 56}) // output kernel
	shapes = append(shapes, []int{56})           // output bias
	return shapes
}

// Save persists model weights on the server.
func (k *KerasPlayer) Save(path string) error {
	req := saveLoadReq{ModelID: k.modelID, Path: path, HiddenDims: k.hiddenDims}
	return k.post("/save", req, nil)
}

// Load restores model weights from the server, then syncs locally.
func (k *KerasPlayer) Load(path string) error {
	req := saveLoadReq{ModelID: k.modelID, Path: path, HiddenDims: k.hiddenDims}
	if err := k.post("/load", req, nil); err != nil {
		// File might not exist yet, that's ok
		log.Infof("Load %s: %v (may be first run)", k.modelID, err)
	}
	return k.SyncWeights()
}
