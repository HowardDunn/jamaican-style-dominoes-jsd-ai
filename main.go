package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/HowardDunn/go-dominos/dominos"
	"github.com/HowardDunn/jsd-ai/duppy"
	"github.com/HowardDunn/jsd-ai/mongodb"
	"github.com/HowardDunn/jsd-ai/nn"
	jsdonline "github.com/HowardDunn/jsd-online-game/game"
	"github.com/schollz/progressbar/v3"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

type trainConfig struct {
	resultsDir string  // --results-dir (default: "./results")
	maxGames   int     // --max-games (default: 100000, 0 = unlimited)
	maxDays    float64 // --max-days (default: 0 = disabled)
}

func (c trainConfig) shouldStop(iteration int, startTime time.Time) bool {
	if c.maxGames > 0 && iteration >= c.maxGames {
		return true
	}
	if c.maxDays > 0 && time.Since(startTime).Hours() >= c.maxDays*24 {
		return true
	}
	return false
}

// resultsPath joins the results directory with a filename.
func (c trainConfig) resultsPath(name string) string {
	return filepath.Join(c.resultsDir, name)
}

type gameCache struct {
	UUID       string               `bson:"uuid", json:"uuid"`
	GameEvents []*dominos.GameEvent `bson:"gameEvents", json:"gameEvents"`
	GameType   string               `bson:"gameType", json:"gameType"`
	TimeCreate time.Time            `bson:"timeCreated", json:"timeCreated"`
}

func LoadData(path string) []*gameCache {
	dat, err := os.ReadFile(path)
	if err != nil {
		log.Fatal("Unable to read file")
		return nil
	}
	res := make([]*gameCache, 0)
	err = json.Unmarshal(dat, &res)
	if err != nil {
		log.Fatal("Unable to unmarshal")
		return nil
	}
	return res
}

func calculateAverageSquaredCost(x []float64) float64 {
	s := 0.0
	for i := range x {
		s += x[i] * x[i]
	}
	return s / float64(len(x))
}

func userIn(users []string, filteredUsers map[string]any) bool {
	for _, usr := range users {
		_, ok := filteredUsers[usr]
		if ok {
			return true
		}
	}
	return false
}

func trainHuman(cfg trainConfig, modelName string, mongoURI string, gameMode string) {
	path := cfg.resultsPath(time.Now().Format(time.RFC3339))
	if _, err := os.Stat(path); os.IsNotExist(err) {
		err := os.Mkdir(path, 0o644)
		if err != nil {
			log.Fatal(err)
		}
	}

	// variables
	learnRate := 0.0005
	filteredUsers := map[string]any{"deigodon201": nil}
	filterUsers := true
	epochs := 5

	// Create transformer model (same architecture as trainReinforcedTransformer)
	transformer := nn.NewSequenceTransformer(64, 2, 2, 128, 40, 56)

	// Try loading existing model and metadata
	modelPath := cfg.resultsPath(modelName)
	metaPath := modelPath + ".meta.json"
	if _, err := os.Stat(modelPath); err == nil {
		if err := transformer.Load(modelPath); err != nil {
			log.Warnf("Failed to load existing model %s: %v — starting fresh", modelPath, err)
		} else {
			log.Infof("Loaded existing model: %s", modelPath)
		}
	}
	meta := loadHumanTrainMeta(metaPath)
	seenGameIDs := map[string]bool{}
	for _, id := range meta.TrainedGameIDs {
		seenGameIDs[id] = true
	}
	log.Infof("Loaded meta: %d iterations, %d games already trained on", meta.Iterations, len(seenGameIDs))

	// Connect to MongoDB and fetch game data
	log.Info("Connecting to MongoDB...")
	client, err := mongodb.Connect(mongoURI)
	if err != nil {
		log.Fatal("Failed to connect to MongoDB: ", err)
	}
	defer client.Disconnect()

	if err := client.Ping(); err != nil {
		log.Fatal("MongoDB ping failed: ", err)
	}
	log.Info("MongoDB connected successfully")

	count, err := client.GameCount(gameMode)
	if err != nil {
		log.Fatal("Failed to count games: ", err)
	}
	log.Infof("Found %d games in MongoDB (mode: %s)", count, gameMode)

	docs, err := client.FetchGames(gameMode)
	if err != nil {
		log.Fatal("Failed to fetch games: ", err)
	}

	skipped := 0
	gameCaches := make([]*gameCache, 0, len(docs))
	for _, doc := range docs {
		if len(doc.GameEvents) == 0 {
			continue
		}
		if seenGameIDs[doc.UUID] {
			skipped++
			continue
		}
		gameCaches = append(gameCaches, &gameCache{
			UUID:       doc.UUID,
			GameEvents: doc.GameEvents,
			GameType:   doc.GameType,
			TimeCreate: doc.TimeCreate,
		})
	}
	log.Infof("Loaded %d new games from MongoDB (%d skipped as already trained)", len(gameCaches), skipped)

	randomSeed := time.Now().UnixNano()
	log.Info("Using random Seed: ", randomSeed)
	rand.Seed(randomSeed)
	rand.Shuffle(len(gameCaches), func(i, j int) { gameCaches[i], gameCaches[j] = gameCaches[j], gameCaches[i] })

	trainingGamesIndex := int(float64(len(gameCaches)) * 0.8)
	trainingGames := gameCaches[:trainingGamesIndex]
	validationGames := gameCaches[trainingGamesIndex:]
	correctGameEventTime := time.Date(2022, 12, 9, 0, 0, 0, 0, time.Local)

	// predictionEvaluation
	evaluate := func() float64 {
		accuracies := 0.0
		inaccuracies := 0.0
		compatibles := 0.0
		incompatibles := 0.0
		bar := progressbar.Default(int64(len(validationGames)))
		for _, gameCache := range validationGames {
			bar.Add(1)
			if !userIn(gameCache.GameEvents[0].PlayerNames[:], filteredUsers) && filterUsers {
				continue
			}

			if gameCache.TimeCreate.Before(correctGameEventTime) {
				continue
			}
			if len(gameCache.GameEvents) < 1 {
				continue
			}

			transformer.ResetHistory()
			for _, gameEvent := range gameCache.GameEvents {
				if gameEvent.EventType == dominos.PlayedCard {
					rotatedGameEvent := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
					_, ok := filteredUsers[rotatedGameEvent.PlayerNames[0]]
					if !ok && filterUsers {
						transformer.ObserveEvent(gameEvent)
						continue
					}
					cardChoice, err := transformer.Predict(rotatedGameEvent)
					if err != nil {
						log.Fatal("Error predicting: ", err)
					}
					if cardChoice.Card == gameEvent.Card && cardChoice.Side == gameEvent.Side {
						accuracies++
					} else {
						inaccuracies++
					}
					suit1, suit2 := dominos.GetCLIDominos()[cardChoice.Card].GetSuits()
					if cardChoice.Side == dominos.Left {
						if suit1 == rotatedGameEvent.BoardState.LeftSuit || suit2 == rotatedGameEvent.BoardState.LeftSuit {
							compatibles++
						} else {
							incompatibles++
						}
					} else {
						if suit1 == rotatedGameEvent.BoardState.RightSuit || suit2 == rotatedGameEvent.BoardState.RightSuit {
							compatibles++
						} else {
							incompatibles++
						}
					}
				}
				// Update history for all events
				transformer.ObserveEvent(gameEvent)
				if gameEvent.EventType == dominos.RoundWin || gameEvent.EventType == dominos.RoundDraw {
					transformer.ResetHistory()
				}
			}
		}

		acc := 0.0
		if accuracies+inaccuracies > 0 {
			acc = float64(accuracies) / float64(accuracies+inaccuracies)
		}
		log.Infof("Accuracy of validation: %.5f", acc)
		if compatibles+incompatibles > 0 {
			log.Infof("Compatible of validation data: %.5f", float64(compatibles)/float64(incompatibles+compatibles))
		}
		return acc
	}

	averageCosts := []float64{}
	newGameIDs := []string{}
	train := func() {
		bar := progressbar.Default(int64(len(trainingGames)))
		for _, gameCache := range trainingGames {
			bar.Add(1)
			if gameCache.TimeCreate.Before(correctGameEventTime) {
				continue
			}

			if !userIn(gameCache.GameEvents[0].PlayerNames[:], filteredUsers) && filterUsers {
				continue
			}
			gameCost := []float64{}
			transformer.ResetHistory()
			for _, gameEvent := range gameCache.GameEvents {
				if gameEvent.EventType == dominos.PosedCard || gameEvent.EventType == dominos.PlayedCard {
					rotatedGameEvent := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
					_, ok := filteredUsers[rotatedGameEvent.PlayerNames[0]]
					if !ok && filterUsers {
						transformer.ObserveEvent(gameEvent)
						continue
					}
					cost, err := transformer.TrainSupervised(rotatedGameEvent, learnRate)
					if err != nil {
						log.Fatal("Error training: ", err)
					}
					gameCost = append(gameCost, cost)
					meta.Iterations++
				}
				// Update history for all events
				transformer.ObserveEvent(gameEvent)
				if gameEvent.EventType == dominos.RoundWin || gameEvent.EventType == dominos.RoundDraw {
					transformer.ResetHistory()
				}
			}
			if len(gameCost) > 0 {
				averageCosts = append(averageCosts, calculateAverageSquaredCost(gameCost))
			}
			if gameCache.UUID != "" {
				newGameIDs = append(newGameIDs, gameCache.UUID)
			}
		}
	}

	saveMeta := func() {
		meta.TrainedGameIDs = append(meta.TrainedGameIDs, newGameIDs...)
		meta.GamesTrainedOn = len(meta.TrainedGameIDs)
		if err := saveHumanTrainMeta(metaPath, meta); err != nil {
			log.Error("Failed to save meta: ", err)
		}
		// Clear newGameIDs since they're now persisted
		newGameIDs = nil
	}

	accuracies := []float64{}
	accuracies = append(accuracies, evaluate())
	for i := 0; i < epochs; i++ {
		log.Infof("Epoch: %d out of %d", i+1, epochs)
		train()
		accuracies = append(accuracies, evaluate())
		// Save model and meta after each epoch
		transformer.Save(path + modelName)
		saveMeta()
		log.Infof("Model + meta saved to %s (iterations: %d, games: %d)", path+modelName, meta.Iterations, meta.GamesTrainedOn)
	}

	pts := make(plotter.XYs, len(averageCosts))
	acPts := make(plotter.XYs, len(accuracies))
	for i := range averageCosts {
		pts[i].Y = averageCosts[i]
		pts[i].X = float64(i)
	}

	for i := range accuracies {
		acPts[i].Y = accuracies[i]
		acPts[i].X = float64(i)
	}

	p := plot.New()
	p.Title.Text = "Cost vs iteration"
	p.X.Label.Text = "iteration"
	p.Y.Label.Text = "Cost"
	err = plotutil.AddLinePoints(p, "JSD", pts)
	if err != nil {
		log.Fatal(err)
	}

	aP := plot.New()
	aP.Title.Text = "Accuracy vs iteration"
	aP.X.Label.Text = "iteration"
	aP.Y.Label.Text = "Accuracy"
	err = plotutil.AddLinePoints(aP, "JSD", acPts)
	if err != nil {
		log.Fatal(err)
	}

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, path+"costs.png"); err != nil {
		panic(err)
	}

	// Save the plot to a PNG file.
	if err := aP.Save(4*vg.Inch, 4*vg.Inch, path+"accuracies.png"); err != nil {
		panic(err)
	}

	transformer.Save(path + modelName)
	saveMeta()
	log.Infof("Final model saved to %s (iterations: %d, games: %d)", path+modelName, meta.Iterations, meta.GamesTrainedOn)
}

type modelMeta struct {
	Iterations int     `json:"iterations"`
	TotalWins  int     `json:"total_wins"`
	TotalWins2 int     `json:"total_wins_shuffled"`
	EloNN      float64 `json:"elo_nn"`
	EloBench   float64 `json:"elo_bench"`
}

const (
	eloK            = 32.0
	eloMaxRoundWins = 6.0
	eloRandomPlayer = 1000.0
)

// updateEloBench calculates a new Elo rating for a model after a benchmark game
// against 3 random players (each at eloRandomPlayer). Matches backend logic.
func updateEloBench(currentElo float64, roundWins int) float64 {
	playerScore := float64(roundWins) / eloMaxRoundWins
	if roundWins >= 6 {
		playerScore = 1.0
	}
	// Sum pairwise expected scores vs 3 random opponents
	probabilityTotal := 0.0
	for j := 0; j < 3; j++ {
		probabilityTotal += 1.0 / (1.0 + math.Pow(10, (eloRandomPlayer-currentElo)/400.0))
	}
	expected := probabilityTotal / eloMaxRoundWins
	return currentElo + eloK*(playerScore-expected)
}

// updateEloMultiplayer updates Elo ratings for a 4-player cutthroat game.
// Matches the backend logic: score = roundWins/6, expected = sum of pairwise probs / 6.
func updateEloMultiplayer(ratings [4]float64, playerWins [4]int) [4]float64 {
	newRatings := ratings
	for i := 0; i < 4; i++ {
		playerScore := float64(playerWins[i]) / eloMaxRoundWins
		if playerWins[i] >= 6 {
			playerScore = 1.0
		}
		// Sum pairwise expected scores vs each opponent
		probabilityTotal := 0.0
		for j := 0; j < 4; j++ {
			if i == j {
				continue
			}
			probabilityTotal += 1.0 / (1.0 + math.Pow(10, (ratings[j]-ratings[i])/400.0))
		}
		expected := probabilityTotal / eloMaxRoundWins
		newRatings[i] = ratings[i] + eloK*(playerScore-expected)
	}
	return newRatings
}

func saveModelMeta(path string, meta modelMeta) error {
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func loadModelMeta(path string) modelMeta {
	data, err := os.ReadFile(path)
	if err != nil {
		return modelMeta{}
	}
	var meta modelMeta
	if err := json.Unmarshal(data, &meta); err != nil {
		return modelMeta{}
	}
	return meta
}

type humanTrainMeta struct {
	Iterations    int      `json:"iterations"`
	GamesTrainedOn int     `json:"games_trained_on"`
	TrainedGameIDs []string `json:"trained_game_ids"`
}

func loadHumanTrainMeta(path string) humanTrainMeta {
	data, err := os.ReadFile(path)
	if err != nil {
		return humanTrainMeta{}
	}
	var meta humanTrainMeta
	if err := json.Unmarshal(data, &meta); err != nil {
		return humanTrainMeta{}
	}
	return meta
}

func saveHumanTrainMeta(path string, meta humanTrainMeta) error {
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

type h2hMeta struct {
	Iterations    int `json:"iterations"`
	MlpVsAttnMlp  int `json:"mlp_vs_attn_mlp_wins"`
	MlpVsAttnAttn int `json:"mlp_vs_attn_attn_wins"`
	MlpVsTFMlp    int `json:"mlp_vs_tf_mlp_wins"`
	MlpVsTFTF     int `json:"mlp_vs_tf_tf_wins"`
	AttnVsTFAttn  int `json:"attn_vs_tf_attn_wins"`
	AttnVsTFTF    int `json:"attn_vs_tf_tf_wins"`
}

func saveH2HMeta(path string, meta h2hMeta) error {
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func loadH2HMeta(path string) h2hMeta {
	data, err := os.ReadFile(path)
	if err != nil {
		return h2hMeta{}
	}
	var meta h2hMeta
	if err := json.Unmarshal(data, &meta); err != nil {
		return h2hMeta{}
	}
	return meta
}

type roundEvent struct {
	eventType  string // "train", "pass", or "reset"
	player     int
	gameEvent  *dominos.GameEvent
	nextEvents [16]*dominos.GameEvent // only used for "train"
}

type gameResult struct {
	rounds     [][]roundEvent
	playerWins [4]int
}

// cloneNNs creates deep copies of the master NNs for concurrent gameplay.
// Copies Epsilon, Search, and SearchNum so clones behave identically during play.
func cloneNNs(masters [4]*nn.JSDNN) [4]*nn.JSDNN {
	var clones [4]*nn.JSDNN
	for i, m := range masters {
		clones[i] = m.Clone()
		clones[i].Epsilon = m.Epsilon
		clones[i].Search = m.Search
		clones[i].SearchNum = m.SearchNum
	}
	return clones
}

// playGame runs a full domino game using the provided players and NN references.
// It collects round events without training. The nnPlayers are used to determine
// which player index maps to which NN for event collection.
func playGame(dp [4]dominos.Player, nnPlayers [4]*nn.JSDNN, seed int64) gameResult {
	dominosGame := dominos.NewLocalGame(dp, seed, "cutthroat")
	roundGameEvents := []*dominos.GameEvent{}
	lastGameEvent := &dominos.GameEvent{}
	result := gameResult{}

	for lastGameEvent != nil && lastGameEvent.EventType != dominos.GameWin {
		lastGameEvent = dominosGame.AdvanceGameIteration()
		lGE := jsdonline.CopyandRotateGameEvent(lastGameEvent, 0)

		if lGE.EventType == dominos.PlayedCard ||
			lGE.EventType == dominos.Passed || lastGameEvent.EventType == dominos.RoundWin ||
			lastGameEvent.EventType == dominos.RoundDraw {
			roundGameEvents = append(roundGameEvents, lGE)
		}

		// Notify EventObserver players (e.g., transformer) about game events
		if lGE.EventType == dominos.PlayedCard || lGE.EventType == dominos.Passed || lGE.EventType == dominos.PosedCard {
			for k := 0; k < 4; k++ {
				if obs, ok := dp[k].(nn.EventObserver); ok {
					obs.ObserveEvent(lGE)
				}
			}
		}

		if lastGameEvent.EventType == dominos.RoundWin || lastGameEvent.EventType == dominos.RoundDraw {
			// Reset pass memory on the cloned NNs used for gameplay
			for k := 0; k < 4; k++ {
				nnPlayers[k].ResetPassMemory()
				if obs, ok := dp[k].(nn.EventObserver); ok {
					obs.ResetHistory()
				}
			}

			var round []roundEvent
			for i, gameEvent := range roundGameEvents {
				if gameEvent.EventType == dominos.PlayedCard {
					rotatedGameEvent := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
					nextRelevant := [16]*dominos.GameEvent{}
					for j := 1; j < 17; j++ {
						if (i + j) >= len(roundGameEvents) {
							break
						}
						nextRelevant[j-1] = jsdonline.CopyandRotateGameEvent(roundGameEvents[i+j], gameEvent.Player)
					}
					round = append(round, roundEvent{
						eventType:  "train",
						player:     gameEvent.Player,
						gameEvent:  rotatedGameEvent,
						nextEvents: nextRelevant,
					})
				} else if gameEvent.EventType == dominos.Passed {
					round = append(round, roundEvent{
						eventType: "pass",
						player:    gameEvent.Player,
						gameEvent: gameEvent,
					})
				} else if gameEvent.EventType == dominos.RoundWin || gameEvent.EventType == dominos.RoundDraw {
					round = append(round, roundEvent{
						eventType: "reset",
						player:    gameEvent.Player,
						gameEvent: gameEvent,
					})
				}
			}
			result.rounds = append(result.rounds, round)
			roundGameEvents = []*dominos.GameEvent{}
		}
	}

	result.playerWins = lastGameEvent.PlayerWins
	return result
}

// applyTraining processes collected game results on the master NNs.
// Each NN only processes its own player's events, so all 4 NNs train in parallel.
// nnPlayers maps position index to master NN for training.
func applyTraining(nnPlayers [4]*nn.JSDNN, results []gameResult) {
	// Build per-player event sequences across all games/rounds.
	// Each NN's events are independent: pass memory and training only depend
	// on that player's own prior events, not other players' state.
	playerEvents := [4][]roundEvent{}

	for _, result := range results {
		for _, round := range result.rounds {
			// At round start, all NNs reset pass memory
			for p := 0; p < 4; p++ {
				playerEvents[p] = append(playerEvents[p], roundEvent{eventType: "reset"})
			}
			// Distribute events to their player's NN
			for _, evt := range round {
				playerEvents[evt.player] = append(playerEvents[evt.player], evt)
			}
		}
	}

	// Train each NN in parallel
	var wg sync.WaitGroup
	for p := 0; p < 4; p++ {
		if len(playerEvents[p]) == 0 {
			continue
		}
		wg.Add(1)
		go func(playerIdx int) {
			defer wg.Done()
			jsdai := nnPlayers[playerIdx]
			for _, evt := range playerEvents[playerIdx] {
				switch evt.eventType {
				case "train":
					var learnRates []float64
					if jsdai.GetNumHidden() == 1 {
						learnRates = []float64{0.001}
					} else {
						// First hidden layer at full rate, deeper layers 10x slower
						learnRates = []float64{0.001, 0.0001}
					}
					_, err := jsdai.TrainReinforced(evt.gameEvent, learnRates, evt.nextEvents)
					if err != nil {
						log.Fatal("Error training: ", err)
					}
				case "pass":
					jsdai.UpdatePassMemory(evt.gameEvent)
				case "reset":
					jsdai.ResetPassMemory()
				}
			}
		}(p)
	}
	wg.Wait()
}

// playGamePartner runs a full domino game in partner mode using the provided players.
// Identical to playGame but uses "partner" game type where positions 0&2 and 1&3 are partners.
func playGamePartner(dp [4]dominos.Player, nnPlayers [4]*nn.JSDNN, seed int64) gameResult {
	dominosGame := dominos.NewLocalGame(dp, seed, "partner")
	roundGameEvents := []*dominos.GameEvent{}
	lastGameEvent := &dominos.GameEvent{}
	result := gameResult{}

	for lastGameEvent != nil && lastGameEvent.EventType != dominos.GameWin {
		lastGameEvent = dominosGame.AdvanceGameIteration()
		lGE := jsdonline.CopyandRotateGameEvent(lastGameEvent, 0)

		if lGE.EventType == dominos.PlayedCard ||
			lGE.EventType == dominos.Passed || lastGameEvent.EventType == dominos.RoundWin ||
			lastGameEvent.EventType == dominos.RoundDraw {
			roundGameEvents = append(roundGameEvents, lGE)
		}

		// Notify EventObserver players (e.g., transformer) about game events
		if lGE.EventType == dominos.PlayedCard || lGE.EventType == dominos.Passed || lGE.EventType == dominos.PosedCard {
			for k := 0; k < 4; k++ {
				if obs, ok := dp[k].(nn.EventObserver); ok {
					obs.ObserveEvent(lGE)
				}
			}
		}

		if lastGameEvent.EventType == dominos.RoundWin || lastGameEvent.EventType == dominos.RoundDraw {
			for k := 0; k < 4; k++ {
				nnPlayers[k].ResetPassMemory()
				if obs, ok := dp[k].(nn.EventObserver); ok {
					obs.ResetHistory()
				}
			}

			var round []roundEvent
			for i, gameEvent := range roundGameEvents {
				if gameEvent.EventType == dominos.PlayedCard {
					rotatedGameEvent := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
					nextRelevant := [16]*dominos.GameEvent{}
					for j := 1; j < 17; j++ {
						if (i + j) >= len(roundGameEvents) {
							break
						}
						nextRelevant[j-1] = jsdonline.CopyandRotateGameEvent(roundGameEvents[i+j], gameEvent.Player)
					}
					round = append(round, roundEvent{
						eventType:  "train",
						player:     gameEvent.Player,
						gameEvent:  rotatedGameEvent,
						nextEvents: nextRelevant,
					})
				} else if gameEvent.EventType == dominos.Passed {
					round = append(round, roundEvent{
						eventType: "pass",
						player:    gameEvent.Player,
						gameEvent: gameEvent,
					})
				} else if gameEvent.EventType == dominos.RoundWin || gameEvent.EventType == dominos.RoundDraw {
					round = append(round, roundEvent{
						eventType: "reset",
						player:    gameEvent.Player,
						gameEvent: gameEvent,
					})
				}
			}
			result.rounds = append(result.rounds, round)
			roundGameEvents = []*dominos.GameEvent{}
		}
	}

	result.playerWins = lastGameEvent.PlayerWins
	return result
}

// applyTrainingPartner processes collected game results using partner-aware reward.
// Identical to applyTraining but calls TrainReinforcedPartner instead of TrainReinforced.
func applyTrainingPartner(nnPlayers [4]*nn.JSDNN, results []gameResult) {
	playerEvents := [4][]roundEvent{}

	for _, result := range results {
		for _, round := range result.rounds {
			for p := 0; p < 4; p++ {
				playerEvents[p] = append(playerEvents[p], roundEvent{eventType: "reset"})
			}
			for _, evt := range round {
				playerEvents[evt.player] = append(playerEvents[evt.player], evt)
			}
		}
	}

	var wg sync.WaitGroup
	for p := 0; p < 4; p++ {
		if len(playerEvents[p]) == 0 {
			continue
		}
		wg.Add(1)
		go func(playerIdx int) {
			defer wg.Done()
			jsdai := nnPlayers[playerIdx]
			for _, evt := range playerEvents[playerIdx] {
				switch evt.eventType {
				case "train":
					var learnRates []float64
					if jsdai.GetNumHidden() == 1 {
						learnRates = []float64{0.001}
					} else {
						learnRates = []float64{0.001, 0.0001}
					}
					_, err := jsdai.TrainReinforcedPartner(evt.gameEvent, learnRates, evt.nextEvents)
					if err != nil {
						log.Fatal("Error training: ", err)
					}
				case "pass":
					jsdai.UpdatePassMemory(evt.gameEvent)
				case "reset":
					jsdai.ResetPassMemory()
				}
			}
		}(p)
	}
	wg.Wait()
}

func trainReinforced(cfg trainConfig) {
	// Train Reinforcement
	start := time.Now()

	sameGameIterations := 8
	numWorkers := runtime.NumCPU()
	totalRoundWins := [4]int{}  // per-model wins against random
	totalBenchGames := [4]int{} // per-model total games in benchmark
	jsdai1 := nn.New(126, []int{202}, 56)
	jsdai2 := nn.New(126, []int{102, 56}, 56)
	jsdai3 := nn.New(126, []int{204, 102}, 56)
	jsdai4 := nn.New(126, []int{128, 64}, 56)
	jsdai1.OutputActivation = "linear"
	jsdai2.OutputActivation = "linear"
	jsdai3.OutputActivation = "linear"
	jsdai4.OutputActivation = "linear"
	// NOTE: Do not load models trained with ReLU output when using linear output.
	// The pre-trained weights assume ReLU clips negatives, so linear will cause
	// gradient explosion. Start fresh or use OutputActivation = "relu" to resume.
	jsdai1.Load(cfg.resultsPath("jasai1.mdl"))
	jsdai2.Load(cfg.resultsPath("jasai2.mdl"))
	jsdai3.Load(cfg.resultsPath("jasai3.mdl"))
	jsdai4.Load(cfg.resultsPath("jasai4.mdl"))

	// Load metadata to resume iteration counts and win totals
	modelNames := [4]string{"jasai1", "jasai2", "jasai3", "jasai4"}
	metas := [4]modelMeta{}
	for i, name := range modelNames {
		metas[i] = loadModelMeta(cfg.resultsPath(name + ".meta.json"))
	}
	iterationCount := metas[0].Iterations
	jsdai1.TotalWins = metas[0].TotalWins
	jsdai2.TotalWins = metas[1].TotalWins
	jsdai3.TotalWins = metas[2].TotalWins
	jsdai4.TotalWins = metas[3].TotalWins
	jsdai1.TotalWins2 = metas[0].TotalWins2
	jsdai2.TotalWins2 = metas[1].TotalWins2
	jsdai3.TotalWins2 = metas[2].TotalWins2
	jsdai4.TotalWins2 = metas[3].TotalWins2
	elosNN := [4]float64{}
	elosBench := [4]float64{}
	for i := range elosNN {
		if metas[i].EloNN > 0 {
			elosNN[i] = metas[i].EloNN
		} else {
			elosNN[i] = eloRandomPlayer
		}
		if metas[i].EloBench > 0 {
			elosBench[i] = metas[i].EloBench
		} else {
			elosBench[i] = eloRandomPlayer
		}
	}
	log.Info("Resuming from iteration: ", iterationCount)
	log.Infof("Starting NN Elos: [%.1f, %.1f, %.1f, %.1f]", elosNN[0], elosNN[1], elosNN[2], elosNN[3])
	log.Infof("Starting Bench Elos: [%.1f, %.1f, %.1f, %.1f]", elosBench[0], elosBench[1], elosBench[2], elosBench[3])

	jsdai1.Epsilon = 0.1
	jsdai2.Epsilon = 0.1
	jsdai3.Epsilon = 0.1
	jsdai4.Epsilon = 0.1

	masters := [4]*nn.JSDNN{jsdai1, jsdai2, jsdai3, jsdai4}

	// Rolling window buffers (last 1000 iterations) for win tracking
	const rollingSize = 1000
	rollingNN := [4][]int{}         // Phase 1: NN vs NN wins per iteration
	rollingNN2 := [4][]int{}        // Phase 2: shuffled NN vs NN wins per iteration
	rollingBench := [4][]int{}      // Phase 3: per-model wins vs random per iteration
	rollingBenchGames := [4][]int{} // Phase 3: per-model total games per iteration
	rollingIdx := 0
	rollingCount := 0

	rollingRatios := func(buf [4][]int, count int) []float64 {
		total := 0
		sums := [4]int{}
		n := count
		if n > rollingSize {
			n = rollingSize
		}
		for p := 0; p < 4; p++ {
			for i := 0; i < n; i++ {
				sums[p] += buf[p][i]
			}
			total += sums[p]
		}
		if total == 0 {
			return []float64{0, 0, 0, 0}
		}
		return []float64{
			float64(sums[0]) / float64(total),
			float64(sums[1]) / float64(total),
			float64(sums[2]) / float64(total),
			float64(sums[3]) / float64(total),
		}
	}

	// Pre-allocate ring buffers
	for p := 0; p < 4; p++ {
		rollingNN[p] = make([]int, rollingSize)
		rollingNN2[p] = make([]int, rollingSize)
		rollingBench[p] = make([]int, rollingSize)
		rollingBenchGames[p] = make([]int, rollingSize)
	}

	reinForceMentLearn := func(r int64) {
		// Phase 1: Fixed-order NN players, parallel game simulation
		seeds := make([]int64, sameGameIterations)
		for i := range seeds {
			seeds[i] = rand.Int63n(9223372036854775607)
		}

		results := make([]gameResult, sameGameIterations)
		var wg sync.WaitGroup
		sem := make(chan struct{}, numWorkers)

		for i := 0; i < sameGameIterations; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()
				clones := cloneNNs(masters)
				gp := [4]dominos.Player{clones[0], clones[1], clones[2], clones[3]}
				results[idx] = playGame(gp, clones, seeds[idx])
			}(i)
		}
		wg.Wait()

		// Apply training sequentially on master NNs
		applyTraining(masters, results)
		// Accumulate wins and record rolling window
		for p := 0; p < 4; p++ {
			rollingNN[p][rollingIdx] = 0
		}
		for _, r := range results {
			for k := 0; k < 4; k++ {
				masters[k].TotalWins += r.playerWins[k]
				rollingNN[k][rollingIdx] += r.playerWins[k]
			}
			// Update NN Elo (fixed order: position k = masters[k])
			elosNN = updateEloMultiplayer(elosNN, r.playerWins)
		}

		// Phase 2: Shuffled NN players, parallel game simulation
		// Pre-generate shuffled orderings and seeds on the main goroutine
		type shuffledGame struct {
			order [4]int
			seed  int64
		}
		shuffledGames := make([]shuffledGame, sameGameIterations)
		for i := range shuffledGames {
			order := [4]int{0, 1, 2, 3}
			rand.Shuffle(4, func(a, b int) { order[a], order[b] = order[b], order[a] })
			shuffledGames[i] = shuffledGame{
				order: order,
				seed:  rand.Int63n(9223372036854775607),
			}
		}

		results2 := make([]gameResult, sameGameIterations)
		// Store the player ordering so we can map wins back to correct master NNs
		for i := 0; i < sameGameIterations; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()
				sg := shuffledGames[idx]
				clones := cloneNNs(masters)
				// Shuffle clones to match the pre-generated order
				shuffled := [4]*nn.JSDNN{clones[sg.order[0]], clones[sg.order[1]], clones[sg.order[2]], clones[sg.order[3]]}
				gp := [4]dominos.Player{shuffled[0], shuffled[1], shuffled[2], shuffled[3]}
				results2[idx] = playGame(gp, shuffled, sg.seed)
			}(i)
		}
		wg.Wait()

		// Apply training with shuffled ordering (map back to master NNs)
		for p := 0; p < 4; p++ {
			rollingNN2[p][rollingIdx] = 0
		}
		for ri, result := range results2 {
			sg := shuffledGames[ri]
			shuffledMasters := [4]*nn.JSDNN{masters[sg.order[0]], masters[sg.order[1]], masters[sg.order[2]], masters[sg.order[3]]}
			applyTraining(shuffledMasters, []gameResult{result})
			// Accumulate wins to the correct master's TotalWins2
			for k := 0; k < 4; k++ {
				shuffledMasters[k].TotalWins2 += result.playerWins[k]
				rollingNN2[sg.order[k]][rollingIdx] += result.playerWins[k]
			}
			// Update NN Elo: remap position wins to master Elo indices
			shuffledElos := [4]float64{elosNN[sg.order[0]], elosNN[sg.order[1]], elosNN[sg.order[2]], elosNN[sg.order[3]]}
			newShuffledElos := updateEloMultiplayer(shuffledElos, result.playerWins)
			for k := 0; k < 4; k++ {
				elosNN[sg.order[k]] = newShuffledElos[k]
			}
		}

		// Phase 3: Benchmark each model individually against 3 random players (no exploration)
		savedEpsilon := [4]float64{jsdai1.Epsilon, jsdai2.Epsilon, jsdai3.Epsilon, jsdai4.Epsilon}
		jsdai1.Epsilon = 0
		jsdai2.Epsilon = 0
		jsdai3.Epsilon = 0
		jsdai4.Epsilon = 0

		for p := 0; p < 4; p++ {
			rollingBench[p][rollingIdx] = 0
			rollingBenchGames[p][rollingIdx] = 0
		}

		// Benchmark each of the 4 models: 2 games each against 3 random players
		type benchResult struct {
			nnWins    int
			totalWins int
		}
		benchOut := make([]benchResult, 4*2) // 4 models × 2 games
		benchSeeds := make([]int64, 4*2)
		benchPositions := make([]int, 4*2)
		for i := range benchSeeds {
			benchSeeds[i] = rand.Int63n(9223372036854775607)
			benchPositions[i] = rand.Intn(4)
		}

		for i := 0; i < 4*2; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()
				modelIdx := idx / 2
				pos := benchPositions[idx]
				clone := masters[modelIdx].Clone()
				clone.Epsilon = 0
				clone.Search = masters[modelIdx].Search
				clone.SearchNum = masters[modelIdx].SearchNum
				var gp [4]dominos.Player
				// playGame needs nnPlayers for event collection; use a dummy set
				dummyNNs := cloneNNs(masters)
				for k := 0; k < 4; k++ {
					dummyNNs[k].Epsilon = 0
				}
				for k := 0; k < 4; k++ {
					if k == pos {
						gp[k] = clone
					} else {
						gp[k] = &dominos.ComputerPlayer{RandomMode: true}
					}
				}
				result := playGame(gp, dummyNNs, benchSeeds[idx])
				total := 0
				for k := 0; k < 4; k++ {
					total += result.playerWins[k]
				}
				benchOut[idx] = benchResult{nnWins: result.playerWins[pos], totalWins: total}
			}(i)
		}
		wg.Wait()

		// Accumulate per-model benchmark wins and update Elo per game
		for m := 0; m < 4; m++ {
			for g := 0; g < 2; g++ {
				totalRoundWins[m] += benchOut[m*2+g].nnWins
				totalBenchGames[m] += benchOut[m*2+g].totalWins
				rollingBench[m][rollingIdx] += benchOut[m*2+g].nnWins
				rollingBenchGames[m][rollingIdx] += benchOut[m*2+g].totalWins
				// Update bench Elo once per benchmark game (matches backend logic)
				elosBench[m] = updateEloBench(elosBench[m], benchOut[m*2+g].nnWins)
			}
		}

		// Restore exploration
		jsdai1.Epsilon = savedEpsilon[0]
		jsdai2.Epsilon = savedEpsilon[1]
		jsdai3.Epsilon = savedEpsilon[2]
		jsdai4.Epsilon = savedEpsilon[3]

		// Advance rolling window
		rollingIdx = (rollingIdx + 1) % rollingSize
		rollingCount++

		// Save models and metadata once per batch
		iterationCount++
		for i, name := range modelNames {
			masters[i].Save(cfg.resultsPath(name + ".mdl"))
			saveModelMeta(cfg.resultsPath(name+".meta.json"), modelMeta{
				Iterations: iterationCount,
				TotalWins:  masters[i].TotalWins,
				TotalWins2: masters[i].TotalWins2,
				EloNN:      elosNN[i],
				EloBench:   elosBench[i],
			})
		}
	}

	jsdai1.Search = true
	jsdai1.SearchNum = 100
	jsdai2.Search = true
	jsdai2.SearchNum = 100
	jsdai3.Search = true
	jsdai3.SearchNum = 100
	jsdai4.Search = true
	jsdai4.SearchNum = 100

	// Open CSV for Elo tracking (append mode)
	eloCSVPath := cfg.resultsPath("elo_history.csv")
	eloFileExists := false
	if _, err := os.Stat(eloCSVPath); err == nil {
		eloFileExists = true
	}
	eloFile, err := os.OpenFile(eloCSVPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatal("Failed to open elo CSV: ", err)
	}
	defer eloFile.Close()
	eloWriter := csv.NewWriter(eloFile)
	defer eloWriter.Flush()
	if !eloFileExists {
		eloWriter.Write([]string{
			"iteration",
			modelNames[0] + "_nn", modelNames[1] + "_nn", modelNames[2] + "_nn", modelNames[3] + "_nn",
			modelNames[0] + "_bench", modelNames[1] + "_bench", modelNames[2] + "_bench", modelNames[3] + "_bench",
		})
		eloWriter.Flush()
	}

	randomSeed := time.Now().UnixNano()
	log.Info("Using random Seed: ", randomSeed)
	rand.Seed(randomSeed)
	for j := 0; !cfg.shouldStop(j, start); j++ {
		randNum := rand.Int63n(9223372036854775607)
		reinForceMentLearn(randNum)
		log.Info("Games Seen: ", j+1)
		// Per-model benchmark win rates (each model vs 3 random players, 25% = random baseline)
		for m := 0; m < 4; m++ {
			rate := 0.0
			if totalBenchGames[m] > 0 {
				rate = float64(totalRoundWins[m]) / float64(totalBenchGames[m])
			}
			rollingWins := 0
			rollingGames := 0
			n := rollingCount
			if n > rollingSize {
				n = rollingSize
			}
			for i := 0; i < n; i++ {
				rollingWins += rollingBench[m][i]
				rollingGames += rollingBenchGames[m][i]
			}
			rollingRate := 0.0
			if rollingGames > 0 {
				rollingRate = float64(rollingWins) / float64(rollingGames)
			}
			log.Infof("Bench %s: total=%.4f (%d/%d)  rolling1k=%.4f (%d/%d)  eloNN=%.1f  eloBench=%.1f",
				modelNames[m], rate, totalRoundWins[m], totalBenchGames[m],
				rollingRate, rollingWins, rollingGames, elosNN[m], elosBench[m])
		}
		// Write Elo to CSV
		eloWriter.Write([]string{
			fmt.Sprintf("%d", iterationCount),
			fmt.Sprintf("%.1f", elosNN[0]), fmt.Sprintf("%.1f", elosNN[1]),
			fmt.Sprintf("%.1f", elosNN[2]), fmt.Sprintf("%.1f", elosNN[3]),
			fmt.Sprintf("%.1f", elosBench[0]), fmt.Sprintf("%.1f", elosBench[1]),
			fmt.Sprintf("%.1f", elosBench[2]), fmt.Sprintf("%.1f", elosBench[3]),
		})
		eloWriter.Flush()
		log.Info("NN Wins: ", []int{jsdai1.TotalWins, jsdai2.TotalWins, jsdai3.TotalWins, jsdai4.TotalWins})
		log.Info("NN Wins2: ", []int{jsdai1.TotalWins2, jsdai2.TotalWins2, jsdai3.TotalWins2, jsdai4.TotalWins2})
		n1 := float64(jsdai1.TotalWins) / float64(jsdai1.TotalWins+jsdai2.TotalWins+jsdai3.TotalWins+jsdai4.TotalWins)
		n2 := float64(jsdai2.TotalWins) / float64(jsdai1.TotalWins+jsdai2.TotalWins+jsdai3.TotalWins+jsdai4.TotalWins)
		n3 := float64(jsdai3.TotalWins) / float64(jsdai1.TotalWins+jsdai2.TotalWins+jsdai3.TotalWins+jsdai4.TotalWins)
		n4 := float64(jsdai4.TotalWins) / float64(jsdai1.TotalWins+jsdai2.TotalWins+jsdai3.TotalWins+jsdai4.TotalWins)
		log.Info("NN Ratios: ", []float64{n1, n2, n3, n4})
		log.Info("NN Ratios Rolling 1k: ", rollingRatios(rollingNN, rollingCount))
		n21 := float64(jsdai1.TotalWins2) / float64(jsdai1.TotalWins2+jsdai2.TotalWins2+jsdai3.TotalWins2+jsdai4.TotalWins2)
		n22 := float64(jsdai2.TotalWins2) / float64(jsdai1.TotalWins2+jsdai2.TotalWins2+jsdai3.TotalWins2+jsdai4.TotalWins2)
		n23 := float64(jsdai3.TotalWins2) / float64(jsdai1.TotalWins2+jsdai2.TotalWins2+jsdai3.TotalWins2+jsdai4.TotalWins2)
		n24 := float64(jsdai4.TotalWins2) / float64(jsdai1.TotalWins2+jsdai2.TotalWins2+jsdai3.TotalWins2+jsdai4.TotalWins2)
		log.Info("NN Ratios2: ", []float64{n21, n22, n23, n24})
		log.Info("NN Ratios2 Rolling 1k: ", rollingRatios(rollingNN2, rollingCount))
		if j > 1 {
			jsdai1.Search = false
			jsdai2.Search = false
			jsdai3.Search = false
			jsdai4.Search = false

		}
	}

	for i, name := range modelNames {
		masters[i].Save(cfg.resultsPath(name + ".mdl"))
		saveModelMeta(cfg.resultsPath(name+".meta.json"), modelMeta{
			Iterations: iterationCount,
			TotalWins:  masters[i].TotalWins,
			TotalWins2: masters[i].TotalWins2,
			EloNN:      elosNN[i],
			EloBench:   elosBench[i],
		})
	}
	log.Info("Took : ", time.Now().Sub(start))
}

func trainReinforcedAttention(cfg trainConfig) {
	// Train Reinforcement with Attention
	start := time.Now()

	sameGameIterations := 8
	numWorkers := runtime.NumCPU()
	totalRoundWins := [4]int{}
	totalBenchGames := [4]int{}
	jsdai1 := nn.New(126, []int{256}, 56)
	jsdai2 := nn.New(126, []int{512, 256}, 56)
	jsdai3 := nn.New(126, []int{384, 192}, 56)
	jsdai4 := nn.New(126, []int{256, 128}, 56)
	jsdai1.OutputActivation = "linear"
	jsdai2.OutputActivation = "linear"
	jsdai3.OutputActivation = "linear"
	jsdai4.OutputActivation = "linear"
	jsdai1.EnableAttention(28)
	jsdai2.EnableAttention(28)
	jsdai3.EnableAttention(28)
	jsdai4.EnableAttention(28)
	jsdai1.Load(cfg.resultsPath("jasai_attn1.mdl"))
	jsdai2.Load(cfg.resultsPath("jasai_attn2.mdl"))
	jsdai3.Load(cfg.resultsPath("jasai_attn3.mdl"))
	jsdai4.Load(cfg.resultsPath("jasai_attn4.mdl"))

	// Load metadata to resume iteration counts and win totals
	modelNames := [4]string{"jasai_attn1", "jasai_attn2", "jasai_attn3", "jasai_attn4"}
	metas := [4]modelMeta{}
	for i, name := range modelNames {
		metas[i] = loadModelMeta(cfg.resultsPath(name + ".meta.json"))
	}
	iterationCount := metas[0].Iterations
	jsdai1.TotalWins = metas[0].TotalWins
	jsdai2.TotalWins = metas[1].TotalWins
	jsdai3.TotalWins = metas[2].TotalWins
	jsdai4.TotalWins = metas[3].TotalWins
	jsdai1.TotalWins2 = metas[0].TotalWins2
	jsdai2.TotalWins2 = metas[1].TotalWins2
	jsdai3.TotalWins2 = metas[2].TotalWins2
	jsdai4.TotalWins2 = metas[3].TotalWins2
	elosNN := [4]float64{}
	elosBench := [4]float64{}
	for i := range elosNN {
		if metas[i].EloNN > 0 {
			elosNN[i] = metas[i].EloNN
		} else {
			elosNN[i] = eloRandomPlayer
		}
		if metas[i].EloBench > 0 {
			elosBench[i] = metas[i].EloBench
		} else {
			elosBench[i] = eloRandomPlayer
		}
	}
	log.Info("Resuming from iteration: ", iterationCount)
	log.Infof("Starting NN Elos: [%.1f, %.1f, %.1f, %.1f]", elosNN[0], elosNN[1], elosNN[2], elosNN[3])
	log.Infof("Starting Bench Elos: [%.1f, %.1f, %.1f, %.1f]", elosBench[0], elosBench[1], elosBench[2], elosBench[3])

	jsdai1.Epsilon = 0.1
	jsdai2.Epsilon = 0.1
	jsdai3.Epsilon = 0.1
	jsdai4.Epsilon = 0.1

	masters := [4]*nn.JSDNN{jsdai1, jsdai2, jsdai3, jsdai4}

	// Rolling window buffers (last 1000 iterations) for win tracking
	const rollingSize = 1000
	rollingNN := [4][]int{}
	rollingNN2 := [4][]int{}
	rollingBench := [4][]int{}
	rollingBenchGames := [4][]int{}
	rollingIdx := 0
	rollingCount := 0

	rollingRatios := func(buf [4][]int, count int) []float64 {
		total := 0
		sums := [4]int{}
		n := count
		if n > rollingSize {
			n = rollingSize
		}
		for p := 0; p < 4; p++ {
			for i := 0; i < n; i++ {
				sums[p] += buf[p][i]
			}
			total += sums[p]
		}
		if total == 0 {
			return []float64{0, 0, 0, 0}
		}
		return []float64{
			float64(sums[0]) / float64(total),
			float64(sums[1]) / float64(total),
			float64(sums[2]) / float64(total),
			float64(sums[3]) / float64(total),
		}
	}

	// Pre-allocate ring buffers
	for p := 0; p < 4; p++ {
		rollingNN[p] = make([]int, rollingSize)
		rollingNN2[p] = make([]int, rollingSize)
		rollingBench[p] = make([]int, rollingSize)
		rollingBenchGames[p] = make([]int, rollingSize)
	}

	reinForceMentLearn := func(r int64) {
		// Phase 1: Fixed-order NN players, parallel game simulation
		seeds := make([]int64, sameGameIterations)
		for i := range seeds {
			seeds[i] = rand.Int63n(9223372036854775607)
		}

		results := make([]gameResult, sameGameIterations)
		var wg sync.WaitGroup
		sem := make(chan struct{}, numWorkers)

		for i := 0; i < sameGameIterations; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()
				clones := cloneNNs(masters)
				gp := [4]dominos.Player{clones[0], clones[1], clones[2], clones[3]}
				results[idx] = playGame(gp, clones, seeds[idx])
			}(i)
		}
		wg.Wait()

		// Apply training sequentially on master NNs
		applyTraining(masters, results)
		// Accumulate wins and record rolling window
		for p := 0; p < 4; p++ {
			rollingNN[p][rollingIdx] = 0
		}
		for _, r := range results {
			for k := 0; k < 4; k++ {
				masters[k].TotalWins += r.playerWins[k]
				rollingNN[k][rollingIdx] += r.playerWins[k]
			}
			// Update NN Elo (fixed order: position k = masters[k])
			elosNN = updateEloMultiplayer(elosNN, r.playerWins)
		}

		// Phase 2: Shuffled NN players, parallel game simulation
		type shuffledGame struct {
			order [4]int
			seed  int64
		}
		shuffledGames := make([]shuffledGame, sameGameIterations)
		for i := range shuffledGames {
			order := [4]int{0, 1, 2, 3}
			rand.Shuffle(4, func(a, b int) { order[a], order[b] = order[b], order[a] })
			shuffledGames[i] = shuffledGame{
				order: order,
				seed:  rand.Int63n(9223372036854775607),
			}
		}

		results2 := make([]gameResult, sameGameIterations)
		for i := 0; i < sameGameIterations; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()
				sg := shuffledGames[idx]
				clones := cloneNNs(masters)
				shuffled := [4]*nn.JSDNN{clones[sg.order[0]], clones[sg.order[1]], clones[sg.order[2]], clones[sg.order[3]]}
				gp := [4]dominos.Player{shuffled[0], shuffled[1], shuffled[2], shuffled[3]}
				results2[idx] = playGame(gp, shuffled, sg.seed)
			}(i)
		}
		wg.Wait()

		// Apply training with shuffled ordering (map back to master NNs)
		for p := 0; p < 4; p++ {
			rollingNN2[p][rollingIdx] = 0
		}
		for ri, result := range results2 {
			sg := shuffledGames[ri]
			shuffledMasters := [4]*nn.JSDNN{masters[sg.order[0]], masters[sg.order[1]], masters[sg.order[2]], masters[sg.order[3]]}
			applyTraining(shuffledMasters, []gameResult{result})
			for k := 0; k < 4; k++ {
				shuffledMasters[k].TotalWins2 += result.playerWins[k]
				rollingNN2[sg.order[k]][rollingIdx] += result.playerWins[k]
			}
			shuffledElos := [4]float64{elosNN[sg.order[0]], elosNN[sg.order[1]], elosNN[sg.order[2]], elosNN[sg.order[3]]}
			newShuffledElos := updateEloMultiplayer(shuffledElos, result.playerWins)
			for k := 0; k < 4; k++ {
				elosNN[sg.order[k]] = newShuffledElos[k]
			}
		}

		// Phase 3: Benchmark each model individually against 3 random players (no exploration)
		savedEpsilon := [4]float64{jsdai1.Epsilon, jsdai2.Epsilon, jsdai3.Epsilon, jsdai4.Epsilon}
		jsdai1.Epsilon = 0
		jsdai2.Epsilon = 0
		jsdai3.Epsilon = 0
		jsdai4.Epsilon = 0

		for p := 0; p < 4; p++ {
			rollingBench[p][rollingIdx] = 0
			rollingBenchGames[p][rollingIdx] = 0
		}

		type benchResult struct {
			nnWins    int
			totalWins int
		}
		benchOut := make([]benchResult, 4*2)
		benchSeeds := make([]int64, 4*2)
		benchPositions := make([]int, 4*2)
		for i := range benchSeeds {
			benchSeeds[i] = rand.Int63n(9223372036854775607)
			benchPositions[i] = rand.Intn(4)
		}

		for i := 0; i < 4*2; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()
				modelIdx := idx / 2
				pos := benchPositions[idx]
				clone := masters[modelIdx].Clone()
				clone.Epsilon = 0
				clone.Search = masters[modelIdx].Search
				clone.SearchNum = masters[modelIdx].SearchNum
				var gp [4]dominos.Player
				dummyNNs := cloneNNs(masters)
				for k := 0; k < 4; k++ {
					dummyNNs[k].Epsilon = 0
				}
				for k := 0; k < 4; k++ {
					if k == pos {
						gp[k] = clone
					} else {
						gp[k] = &dominos.ComputerPlayer{RandomMode: true}
					}
				}
				result := playGame(gp, dummyNNs, benchSeeds[idx])
				total := 0
				for k := 0; k < 4; k++ {
					total += result.playerWins[k]
				}
				benchOut[idx] = benchResult{nnWins: result.playerWins[pos], totalWins: total}
			}(i)
		}
		wg.Wait()

		// Accumulate per-model benchmark wins and update Elo per game
		for m := 0; m < 4; m++ {
			for g := 0; g < 2; g++ {
				totalRoundWins[m] += benchOut[m*2+g].nnWins
				totalBenchGames[m] += benchOut[m*2+g].totalWins
				rollingBench[m][rollingIdx] += benchOut[m*2+g].nnWins
				rollingBenchGames[m][rollingIdx] += benchOut[m*2+g].totalWins
				elosBench[m] = updateEloBench(elosBench[m], benchOut[m*2+g].nnWins)
			}
		}

		// Restore exploration
		jsdai1.Epsilon = savedEpsilon[0]
		jsdai2.Epsilon = savedEpsilon[1]
		jsdai3.Epsilon = savedEpsilon[2]
		jsdai4.Epsilon = savedEpsilon[3]

		// Advance rolling window
		rollingIdx = (rollingIdx + 1) % rollingSize
		rollingCount++

		// Save models and metadata once per batch
		iterationCount++
		for i, name := range modelNames {
			masters[i].Save(cfg.resultsPath(name + ".mdl"))
			saveModelMeta(cfg.resultsPath(name+".meta.json"), modelMeta{
				Iterations: iterationCount,
				TotalWins:  masters[i].TotalWins,
				TotalWins2: masters[i].TotalWins2,
				EloNN:      elosNN[i],
				EloBench:   elosBench[i],
			})
		}
	}

	jsdai1.Search = true
	jsdai1.SearchNum = 100
	jsdai2.Search = true
	jsdai2.SearchNum = 100
	jsdai3.Search = true
	jsdai3.SearchNum = 100
	jsdai4.Search = true
	jsdai4.SearchNum = 100

	// Open CSV for Elo tracking (append mode)
	eloCSVPath := cfg.resultsPath("elo_history_attn.csv")
	eloFileExists := false
	if _, err := os.Stat(eloCSVPath); err == nil {
		eloFileExists = true
	}
	eloFile, err := os.OpenFile(eloCSVPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatal("Failed to open elo CSV: ", err)
	}
	defer eloFile.Close()
	eloWriter := csv.NewWriter(eloFile)
	defer eloWriter.Flush()
	if !eloFileExists {
		eloWriter.Write([]string{
			"iteration",
			modelNames[0] + "_nn", modelNames[1] + "_nn", modelNames[2] + "_nn", modelNames[3] + "_nn",
			modelNames[0] + "_bench", modelNames[1] + "_bench", modelNames[2] + "_bench", modelNames[3] + "_bench",
		})
		eloWriter.Flush()
	}

	randomSeed := time.Now().UnixNano()
	log.Info("Using random Seed: ", randomSeed)
	rand.Seed(randomSeed)
	for j := 0; !cfg.shouldStop(j, start); j++ {
		randNum := rand.Int63n(9223372036854775607)
		reinForceMentLearn(randNum)
		log.Info("Games Seen: ", j+1)
		// Per-model benchmark win rates (each model vs 3 random players, 25% = random baseline)
		for m := 0; m < 4; m++ {
			rate := 0.0
			if totalBenchGames[m] > 0 {
				rate = float64(totalRoundWins[m]) / float64(totalBenchGames[m])
			}
			rollingWins := 0
			rollingGames := 0
			n := rollingCount
			if n > rollingSize {
				n = rollingSize
			}
			for i := 0; i < n; i++ {
				rollingWins += rollingBench[m][i]
				rollingGames += rollingBenchGames[m][i]
			}
			rollingRate := 0.0
			if rollingGames > 0 {
				rollingRate = float64(rollingWins) / float64(rollingGames)
			}
			log.Infof("Bench %s: total=%.4f (%d/%d)  rolling1k=%.4f (%d/%d)  eloNN=%.1f  eloBench=%.1f",
				modelNames[m], rate, totalRoundWins[m], totalBenchGames[m],
				rollingRate, rollingWins, rollingGames, elosNN[m], elosBench[m])
		}
		// Write Elo to CSV
		eloWriter.Write([]string{
			fmt.Sprintf("%d", iterationCount),
			fmt.Sprintf("%.1f", elosNN[0]), fmt.Sprintf("%.1f", elosNN[1]),
			fmt.Sprintf("%.1f", elosNN[2]), fmt.Sprintf("%.1f", elosNN[3]),
			fmt.Sprintf("%.1f", elosBench[0]), fmt.Sprintf("%.1f", elosBench[1]),
			fmt.Sprintf("%.1f", elosBench[2]), fmt.Sprintf("%.1f", elosBench[3]),
		})
		eloWriter.Flush()
		log.Info("NN Wins: ", []int{jsdai1.TotalWins, jsdai2.TotalWins, jsdai3.TotalWins, jsdai4.TotalWins})
		log.Info("NN Wins2: ", []int{jsdai1.TotalWins2, jsdai2.TotalWins2, jsdai3.TotalWins2, jsdai4.TotalWins2})
		n1 := float64(jsdai1.TotalWins) / float64(jsdai1.TotalWins+jsdai2.TotalWins+jsdai3.TotalWins+jsdai4.TotalWins)
		n2 := float64(jsdai2.TotalWins) / float64(jsdai1.TotalWins+jsdai2.TotalWins+jsdai3.TotalWins+jsdai4.TotalWins)
		n3 := float64(jsdai3.TotalWins) / float64(jsdai1.TotalWins+jsdai2.TotalWins+jsdai3.TotalWins+jsdai4.TotalWins)
		n4 := float64(jsdai4.TotalWins) / float64(jsdai1.TotalWins+jsdai2.TotalWins+jsdai3.TotalWins+jsdai4.TotalWins)
		log.Info("NN Ratios: ", []float64{n1, n2, n3, n4})
		log.Info("NN Ratios Rolling 1k: ", rollingRatios(rollingNN, rollingCount))
		n21 := float64(jsdai1.TotalWins2) / float64(jsdai1.TotalWins2+jsdai2.TotalWins2+jsdai3.TotalWins2+jsdai4.TotalWins2)
		n22 := float64(jsdai2.TotalWins2) / float64(jsdai1.TotalWins2+jsdai2.TotalWins2+jsdai3.TotalWins2+jsdai4.TotalWins2)
		n23 := float64(jsdai3.TotalWins2) / float64(jsdai1.TotalWins2+jsdai2.TotalWins2+jsdai3.TotalWins2+jsdai4.TotalWins2)
		n24 := float64(jsdai4.TotalWins2) / float64(jsdai1.TotalWins2+jsdai2.TotalWins2+jsdai3.TotalWins2+jsdai4.TotalWins2)
		log.Info("NN Ratios2: ", []float64{n21, n22, n23, n24})
		log.Info("NN Ratios2 Rolling 1k: ", rollingRatios(rollingNN2, rollingCount))
		if j > 1 {
			jsdai1.Search = false
			jsdai2.Search = false
			jsdai3.Search = false
			jsdai4.Search = false

		}
	}

	for i, name := range modelNames {
		masters[i].Save(cfg.resultsPath(name + ".mdl"))
		saveModelMeta(cfg.resultsPath(name+".meta.json"), modelMeta{
			Iterations: iterationCount,
			TotalWins:  masters[i].TotalWins,
			TotalWins2: masters[i].TotalWins2,
			EloNN:      elosNN[i],
			EloBench:   elosBench[i],
		})
	}
	log.Info("Took : ", time.Now().Sub(start))
}

// func trainWinProbability() {
// 	hiddenLayerNodes1 := 150
// 	learnRate := 0.001
// 	sameGameIterations := 300
// 	totalRoundWins := [4]int{}
// 	jsdai1 := nn.New(126, []int{150}, 1)
// 	jsdai2 := nn.New(126, []int{hiddenLayerNodes1}, 1)
// 	jsdai3 := nn.New(126, []int{150}, 1)
// 	jsdai4 := nn.New(126, []int{150}, 1)
// 	for i := 0; i < sameGameIterations; i++ {
// 		// TODO: bar
// 		nnPlayers := [4]*nn.JSDNN{jsdai1, jsdai2, jsdai3, jsdai4}
// 		gp := [4]dominos.Player{&dominos.ComputerPlayer{RandomMode: true}, &dominos.ComputerPlayer{RandomMode: true}, &dominos.ComputerPlayer{RandomMode: true}, &dominos.ComputerPlayer{RandomMode: true}}
// 		tots := [4]*int{&totalRoundWins[0], &totalRoundWins[1], &totalRoundWins[2], &totalRoundWins[3]}
// 		// rand.Shuffle(len(nnPlayers), func(i, j int) {
// 		// 	gp[i], gp[j] = gp[j], gp[i]
// 		// 	tots[i], tots[j] = tots[j], tots[i]
// 		// })
// 		iter(gp, nnPlayers, tots, r)
// 	}
// }

func trainKerasReinforced(cfg trainConfig) {
	start := time.Now()
	serverURL := "http://localhost:8777"

	sameGameIterations := 8
	totalRoundWins := [4]int{}

	kp1 := nn.NewKerasPlayer(serverURL, "keras1", []int{150})
	kp2 := nn.NewKerasPlayer(serverURL, "keras2", []int{64, 64})
	kp3 := nn.NewKerasPlayer(serverURL, "keras3", []int{32, 32})
	kp4 := nn.NewKerasPlayer(serverURL, "keras4", []int{128, 128})

	// Try loading existing weights (ignore errors if no saved weights yet)
	kp1.Load("")
	kp2.Load("")
	kp3.Load("")
	kp4.Load("")

	kp1.Epsilon = 0.1
	kp2.Epsilon = 0.1
	kp3.Epsilon = 0.1
	kp4.Epsilon = 0.1

	reinForceMentLearn := func(r int64) {
		iter := func(dp [4]dominos.Player, kpPlayers [4]*nn.KerasPlayer, totWins [4]*int, rand int64) {
			dominosGame := dominos.NewLocalGame(dp, int64(rand), "cutthroat")
			roundGameEvents := []*dominos.GameEvent{}
			lastGameEvent := &dominos.GameEvent{}

			for lastGameEvent != nil && lastGameEvent.EventType != dominos.GameWin {
				lastGameEvent = dominosGame.AdvanceGameIteration()
				lGE := jsdonline.CopyandRotateGameEvent(lastGameEvent, 0)

				if lGE.EventType == dominos.PlayedCard ||
					lGE.EventType == dominos.Passed || lastGameEvent.EventType == dominos.RoundWin ||
					lastGameEvent.EventType == dominos.RoundDraw {
					roundGameEvents = append(roundGameEvents, lGE)
				}

				if lastGameEvent.EventType == dominos.RoundWin || lastGameEvent.EventType == dominos.RoundDraw {
					kp1.ResetPassMemory()
					kp2.ResetPassMemory()
					kp3.ResetPassMemory()
					kp4.ResetPassMemory()

					// Collect batches per player, then send one train call each
					type sample struct {
						features   []float64
						target     []float64
						actionMask [56]float64
					}
					playerBatches := [4][]sample{}
					playerLearnRates := [4]float64{}
					for p := 0; p < 4; p++ {
						playerLearnRates[p] = 0.0001
						if len(kpPlayers[p].GetHiddenDims()) == 1 {
							playerLearnRates[p] = 0.001
						}
					}

					for i, gameEvent := range roundGameEvents {
						if gameEvent.EventType == dominos.PlayedCard {
							kp := kpPlayers[gameEvent.Player]
							rotatedGameEvent := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
							nextRelevant := [16]*dominos.GameEvent{}
							for j := 1; j < 17; j++ {
								if (i + j) >= len(roundGameEvents) {
									break
								}
								nextRelevant[j-1] = jsdonline.CopyandRotateGameEvent(roundGameEvents[i+j], gameEvent.Player)
							}
							features, target, actionMask, skip := kp.PrepareTrainSample(rotatedGameEvent, nextRelevant)
							if !skip {
								playerBatches[gameEvent.Player] = append(playerBatches[gameEvent.Player], sample{features, target, actionMask})
							}
						} else if gameEvent.EventType == dominos.Passed {
							kp := kpPlayers[gameEvent.Player]
							kp.UpdatePassMemory(gameEvent)
						} else if gameEvent.EventType == dominos.RoundWin || gameEvent.EventType == dominos.RoundDraw {
							kp := kpPlayers[gameEvent.Player]
							kp.ResetPassMemory()
						}
					}

					// Send one batch per player
					for p := 0; p < 4; p++ {
						if len(playerBatches[p]) == 0 {
							continue
						}
						fb := make([][]float64, len(playerBatches[p]))
						tb := make([][]float64, len(playerBatches[p]))
						mb := make([][56]float64, len(playerBatches[p]))
						for s := range playerBatches[p] {
							fb[s] = playerBatches[p][s].features
							tb[s] = playerBatches[p][s].target
							mb[s] = playerBatches[p][s].actionMask
						}
						_, err := kpPlayers[p].TrainBatch(fb, tb, mb, playerLearnRates[p])
						if err != nil {
							log.Fatal("Error training keras batch: ", err)
						}
					}

					roundGameEvents = []*dominos.GameEvent{}
				}
			}
			*totWins[0] += lastGameEvent.PlayerWins[0]
			*totWins[1] += lastGameEvent.PlayerWins[1]
			*totWins[2] += lastGameEvent.PlayerWins[2]
			*totWins[3] += lastGameEvent.PlayerWins[3]

			kp1.Save("")
			kp2.Save("")
			kp3.Save("")
			kp4.Save("")
		}

		for i := 0; i < sameGameIterations; i++ {
			kpPlayers := [4]*nn.KerasPlayer{kp1, kp2, kp3, kp4}
			gp := [4]dominos.Player{kpPlayers[0], kpPlayers[1], kpPlayers[2], kpPlayers[3]}
			tots := [4]*int{&kpPlayers[0].TotalWins, &kpPlayers[1].TotalWins, &kpPlayers[2].TotalWins, &kpPlayers[3].TotalWins}
			iterSeed := rand.Int63n(9223372036854775607)
			iter(gp, kpPlayers, tots, iterSeed)
		}

		for i := 0; i < sameGameIterations; i++ {
			kpPlayers := [4]*nn.KerasPlayer{kp1, kp2, kp3, kp4}
			rand.Shuffle(len(kpPlayers), func(i, j int) {
				kpPlayers[i], kpPlayers[j] = kpPlayers[j], kpPlayers[i]
			})
			gp := [4]dominos.Player{kpPlayers[0], kpPlayers[1], kpPlayers[2], kpPlayers[3]}
			tots := [4]*int{&kpPlayers[0].TotalWins2, &kpPlayers[1].TotalWins2, &kpPlayers[2].TotalWins2, &kpPlayers[3].TotalWins2}
			randNum := rand.Int63n(9223372036854775607)
			iter(gp, kpPlayers, tots, randNum)
		}

		// Benchmark against random players (no exploration)
		savedEpsilon := [4]float64{kp1.Epsilon, kp2.Epsilon, kp3.Epsilon, kp4.Epsilon}
		kp1.Epsilon = 0
		kp2.Epsilon = 0
		kp3.Epsilon = 0
		kp4.Epsilon = 0

		kpPlayers := [4]*nn.KerasPlayer{kp1, kp2, kp3, kp4}
		gp := [4]dominos.Player{&dominos.ComputerPlayer{RandomMode: true}, &dominos.ComputerPlayer{RandomMode: true}, kpPlayers[2], &dominos.ComputerPlayer{RandomMode: true}}
		tots := [4]*int{&totalRoundWins[0], &totalRoundWins[1], &totalRoundWins[2], &totalRoundWins[3]}
		rand.Shuffle(len(kpPlayers), func(i, j int) {
			gp[i], gp[j] = gp[j], gp[i]
			tots[i], tots[j] = tots[j], tots[i]
		})

		for i := 0; i < sameGameIterations; i++ {
			benchSeed := rand.Int63n(9223372036854775607)
			iter(gp, kpPlayers, tots, benchSeed)
		}

		kp1.Epsilon = savedEpsilon[0]
		kp2.Epsilon = savedEpsilon[1]
		kp3.Epsilon = savedEpsilon[2]
		kp4.Epsilon = savedEpsilon[3]
	}

	randomSeed := time.Now().UnixNano()
	log.Info("Using random Seed: ", randomSeed)
	rand.Seed(randomSeed)
	for j := 0; !cfg.shouldStop(j, start); j++ {
		randNum := rand.Int63n(9223372036854775607)
		reinForceMentLearn(randNum)
		log.Info("Games Seen: ", j+1)
		log.Info("Total Player Wins 3rd: ", totalRoundWins, " Total Rounds: ", totalRoundWins[0]+totalRoundWins[1]+totalRoundWins[2]+totalRoundWins[3])
		r1 := float64(totalRoundWins[0]) / float64(totalRoundWins[0]+totalRoundWins[1]+totalRoundWins[2]+totalRoundWins[3])
		r2 := float64(totalRoundWins[1]) / float64(totalRoundWins[0]+totalRoundWins[1]+totalRoundWins[2]+totalRoundWins[3])
		r3 := float64(totalRoundWins[2]) / float64(totalRoundWins[0]+totalRoundWins[1]+totalRoundWins[2]+totalRoundWins[3])
		r4 := float64(totalRoundWins[3]) / float64(totalRoundWins[0]+totalRoundWins[1]+totalRoundWins[2]+totalRoundWins[3])
		log.Info("Player Ratios: ", []float64{r1, r2, r3, r4})
		log.Info("Keras NN Wins: ", []int{kp1.TotalWins, kp2.TotalWins, kp3.TotalWins, kp4.TotalWins})
		log.Info("Keras NN Wins2: ", []int{kp1.TotalWins2, kp2.TotalWins2, kp3.TotalWins2, kp4.TotalWins2})
		n1 := float64(kp1.TotalWins) / float64(kp1.TotalWins+kp2.TotalWins+kp3.TotalWins+kp4.TotalWins)
		n2 := float64(kp2.TotalWins) / float64(kp1.TotalWins+kp2.TotalWins+kp3.TotalWins+kp4.TotalWins)
		n3 := float64(kp3.TotalWins) / float64(kp1.TotalWins+kp2.TotalWins+kp3.TotalWins+kp4.TotalWins)
		n4 := float64(kp4.TotalWins) / float64(kp1.TotalWins+kp2.TotalWins+kp3.TotalWins+kp4.TotalWins)
		log.Info("Keras NN Ratios: ", []float64{n1, n2, n3, n4})
		n21 := float64(kp1.TotalWins2) / float64(kp1.TotalWins2+kp2.TotalWins2+kp3.TotalWins2+kp4.TotalWins2)
		n22 := float64(kp2.TotalWins2) / float64(kp1.TotalWins2+kp2.TotalWins2+kp3.TotalWins2+kp4.TotalWins2)
		n23 := float64(kp3.TotalWins2) / float64(kp1.TotalWins2+kp2.TotalWins2+kp3.TotalWins2+kp4.TotalWins2)
		n24 := float64(kp4.TotalWins2) / float64(kp1.TotalWins2+kp2.TotalWins2+kp3.TotalWins2+kp4.TotalWins2)
		log.Info("Keras NN Ratios2: ", []float64{n21, n22, n23, n24})
	}

	kp1.Save("")
	kp2.Save("")
	kp3.Save("")
	kp4.Save("")
	log.Info("Took : ", time.Now().Sub(start))
}

func duppyPlay(cfg trainConfig, modelPath string, mode string, statsPath string) {
	token, err := duppy.UserLogin("duppy", "@ecIN)A*$Y93UhZ*")
	if err != nil {
		panic(err)
	}
	stats := duppy.LoadStats(statsPath)
	log.Infof("Loaded stats: %d games played, %d won (%.1f%%)", stats.GamesPlayed, stats.GamesWon, stats.GameWinPct)

	joinedGame := false
	playedGames := map[string]bool{}
	for {
		onlineTables, err := duppy.GetOnlineTableList(token)
		if err != nil {
			log.Error(err)
			continue
		}

		for _, table := range onlineTables {
			if table.PlayerInvolved && table.GameState == "running" && !playedGames[table.GameID] {
				log.Info("Playing game: ", table.GameID)
				record := duppy.PlayGame(table, token, modelPath, mode)
				playedGames[table.GameID] = true

				stats.Games = append(stats.Games, record)
				stats.GamesPlayed++
				if record.DuppyWon {
					stats.GamesWon++
				}
				for i := 0; i < 4; i++ {
					stats.TotalRounds += record.RoundWins[i]
				}
				stats.RoundsWon += record.RoundWins[0]
				if stats.GamesPlayed > 0 {
					stats.GameWinPct = float64(stats.GamesWon) / float64(stats.GamesPlayed) * 100
				}
				if stats.TotalRounds > 0 {
					stats.RoundWinPct = float64(stats.RoundsWon) / float64(stats.TotalRounds) * 100
				}
				if err := duppy.SaveStats(statsPath, stats); err != nil {
					log.Error("Failed to save stats: ", err)
				}
				log.Infof("Game %d done | won=%v | rounds=%v | game-win%%=%.1f | round-win%%=%.1f",
					stats.GamesPlayed, record.DuppyWon, record.RoundWins, stats.GameWinPct, stats.RoundWinPct)

				joinedGame = false
				onlineTables = []*duppy.OnlineGameTable{}
				break
			}
		}

		for _, table := range onlineTables {
			if !joinedGame {
				if table.GameType == mode && table.GameState == "waiting" {
					log.Infof("%+#v", table)
					openPositions := []int{}
					users := []string{table.Player1.Username, table.Player2.Username, table.Player3.Username, table.Player4.Username}
					for i, user := range users {
						if user == "" {
							openPositions = append(openPositions, i)
						}
					}
					randomSeed := time.Now().UnixNano()
					rand.Seed(randomSeed)
					rand.Shuffle(len(openPositions), func(i, j int) { openPositions[i], openPositions[j] = openPositions[j], openPositions[i] })
					if len(openPositions) > 0 {
						gamePayload := &duppy.OnlineGamePayload{
							GameID:   table.GameID,
							Position: openPositions[0],
						}

						err := duppy.JoinOnlineGame(gamePayload, token)
						if err != nil {
							log.Error(err)
							continue
						}
						joinedGame = true
					}

					break
				}
			}
		}

		time.Sleep(2 * time.Second)
	}
}

func trainReinforcedPartner(cfg trainConfig) {
	// Train attention models vs regular models in partner mode
	start := time.Now()

	sameGameIterations := 8
	numWorkers := runtime.NumCPU()

	// Partner-specific model names (separate from cutthroat training)
	attnNames := [4]string{"jasai_attn_partner1", "jasai_attn_partner2", "jasai_attn_partner3", "jasai_attn_partner4"}
	regNames := [4]string{"jasai_partner1", "jasai_partner2", "jasai_partner3", "jasai_partner4"}
	// Source models to load initial weights from (if partner models don't exist yet)
	attnSourceNames := [4]string{"jasai_attn1", "jasai_attn2", "jasai_attn3", "jasai_attn4"}
	regSourceNames := [4]string{"jasai1", "jasai2", "jasai3", "jasai4"}

	// Load 4 attention models
	attnModels := [4]*nn.JSDNN{
		nn.New(126, []int{256}, 56),
		nn.New(126, []int{512, 256}, 56),
		nn.New(126, []int{384, 192}, 56),
		nn.New(126, []int{256, 128}, 56),
	}
	for i := range attnModels {
		attnModels[i].OutputActivation = "linear"
		attnModels[i].EnableAttention(28)
		// Try partner model first, fall back to source model
		if _, err := os.Stat(cfg.resultsPath(attnNames[i] + ".mdl")); err == nil {
			attnModels[i].Load(cfg.resultsPath(attnNames[i] + ".mdl"))
		} else {
			attnModels[i].Load(cfg.resultsPath(attnSourceNames[i] + ".mdl"))
			log.Infof("Loaded %s from source %s", attnNames[i], attnSourceNames[i])
		}
		attnModels[i].Epsilon = 0.1
	}

	// Load 4 regular models
	regModels := [4]*nn.JSDNN{
		nn.New(126, []int{202}, 56),
		nn.New(126, []int{102, 56}, 56),
		nn.New(126, []int{204, 102}, 56),
		nn.New(126, []int{128, 64}, 56),
	}
	for i := range regModels {
		regModels[i].OutputActivation = "linear"
		// Try partner model first, fall back to source model
		if _, err := os.Stat(cfg.resultsPath(regNames[i] + ".mdl")); err == nil {
			regModels[i].Load(cfg.resultsPath(regNames[i] + ".mdl"))
		} else {
			regModels[i].Load(cfg.resultsPath(regSourceNames[i] + ".mdl"))
			log.Infof("Loaded %s from source %s", regNames[i], regSourceNames[i])
		}
		regModels[i].Epsilon = 0.1
	}

	// Load metadata for all 8 partner models
	attnMetas := [4]modelMeta{}
	regMetas := [4]modelMeta{}
	for i := range attnNames {
		attnMetas[i] = loadModelMeta(cfg.resultsPath(attnNames[i] + ".meta.json"))
		regMetas[i] = loadModelMeta(cfg.resultsPath(regNames[i] + ".meta.json"))
	}
	iterationCount := attnMetas[0].Iterations
	for i := range attnModels {
		attnModels[i].TotalWins = attnMetas[i].TotalWins
		attnModels[i].TotalWins2 = attnMetas[i].TotalWins2
		regModels[i].TotalWins = regMetas[i].TotalWins
		regModels[i].TotalWins2 = regMetas[i].TotalWins2
	}

	// Separate Elo arrays
	elosAttn := [4]float64{}
	elosReg := [4]float64{}
	elosBenchAttn := [4]float64{}
	elosBenchReg := [4]float64{}
	for i := range elosAttn {
		if attnMetas[i].EloNN > 0 {
			elosAttn[i] = attnMetas[i].EloNN
		} else {
			elosAttn[i] = eloRandomPlayer
		}
		if attnMetas[i].EloBench > 0 {
			elosBenchAttn[i] = attnMetas[i].EloBench
		} else {
			elosBenchAttn[i] = eloRandomPlayer
		}
		if regMetas[i].EloNN > 0 {
			elosReg[i] = regMetas[i].EloNN
		} else {
			elosReg[i] = eloRandomPlayer
		}
		if regMetas[i].EloBench > 0 {
			elosBenchReg[i] = regMetas[i].EloBench
		} else {
			elosBenchReg[i] = eloRandomPlayer
		}
	}
	log.Info("Resuming from iteration: ", iterationCount)
	log.Infof("Starting Attn Elos: [%.1f, %.1f, %.1f, %.1f]", elosAttn[0], elosAttn[1], elosAttn[2], elosAttn[3])
	log.Infof("Starting Reg Elos: [%.1f, %.1f, %.1f, %.1f]", elosReg[0], elosReg[1], elosReg[2], elosReg[3])
	log.Infof("Starting Attn Bench Elos: [%.1f, %.1f, %.1f, %.1f]", elosBenchAttn[0], elosBenchAttn[1], elosBenchAttn[2], elosBenchAttn[3])
	log.Infof("Starting Reg Bench Elos: [%.1f, %.1f, %.1f, %.1f]", elosBenchReg[0], elosBenchReg[1], elosBenchReg[2], elosBenchReg[3])

	// Rolling window buffers
	const rollingSize = 1000
	rollingAttnWins := [4][]int{}
	rollingRegWins := [4][]int{}
	rollingBenchAttn := [4][]int{}
	rollingBenchReg := [4][]int{}
	rollingBenchAttnGames := [4][]int{}
	rollingBenchRegGames := [4][]int{}
	rollingIdx := 0
	rollingCount := 0
	totalBenchAttn := [4]int{}
	totalBenchReg := [4]int{}
	totalBenchAttnGames := [4]int{}
	totalBenchRegGames := [4]int{}

	for p := 0; p < 4; p++ {
		rollingAttnWins[p] = make([]int, rollingSize)
		rollingRegWins[p] = make([]int, rollingSize)
		rollingBenchAttn[p] = make([]int, rollingSize)
		rollingBenchReg[p] = make([]int, rollingSize)
		rollingBenchAttnGames[p] = make([]int, rollingSize)
		rollingBenchRegGames[p] = make([]int, rollingSize)
	}

	// Open CSV for Elo tracking
	eloCSVPath := cfg.resultsPath("elo_history_partner.csv")
	eloFileExists := false
	if _, err := os.Stat(eloCSVPath); err == nil {
		eloFileExists = true
	}
	eloFile, err := os.OpenFile(eloCSVPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatal("Failed to open elo CSV: ", err)
	}
	defer eloFile.Close()
	eloWriter := csv.NewWriter(eloFile)
	defer eloWriter.Flush()
	if !eloFileExists {
		header := []string{"iteration"}
		for _, name := range attnNames {
			header = append(header, name+"_nn", name+"_bench")
		}
		for _, name := range regNames {
			header = append(header, name+"_nn", name+"_bench")
		}
		eloWriter.Write(header)
		eloWriter.Flush()
	}

	randomSeed := time.Now().UnixNano()
	log.Info("Using random Seed: ", randomSeed)
	rand.Seed(randomSeed)

	for j := 0; !cfg.shouldStop(j, start); j++ {
		// Pick 2 random attention and 2 random regular models
		attnPerm := rand.Perm(4)
		regPerm := rand.Perm(4)
		attnA, attnB := attnPerm[0], attnPerm[1]
		regA, regB := regPerm[0], regPerm[1]

		seeds := make([]int64, sameGameIterations)
		for i := range seeds {
			seeds[i] = rand.Int63n(9223372036854775607)
		}
		var wg sync.WaitGroup
		sem := make(chan struct{}, numWorkers)

		// Phase 1: Attention pair at positions 0&2 (partners), Regular pair at 1&3 (partners)
		// Game positions: {attn_a, reg_a, attn_b, reg_b}
		mixedMasters1 := [4]*nn.JSDNN{attnModels[attnA], regModels[regA], attnModels[attnB], regModels[regB]}
		results1 := make([]gameResult, sameGameIterations)

		for i := 0; i < sameGameIterations; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()
				clones := cloneNNs(mixedMasters1)
				gp := [4]dominos.Player{clones[0], clones[1], clones[2], clones[3]}
				results1[idx] = playGamePartner(gp, clones, seeds[idx])
			}(i)
		}
		wg.Wait()

		applyTrainingPartner(mixedMasters1, results1)

		// Track wins back to respective pools
		for p := 0; p < 4; p++ {
			rollingAttnWins[p][rollingIdx] = 0
			rollingRegWins[p][rollingIdx] = 0
		}
		for _, r := range results1 {
			// Position 0 = attnA, position 2 = attnB
			attnModels[attnA].TotalWins += r.playerWins[0]
			attnModels[attnB].TotalWins += r.playerWins[2]
			rollingAttnWins[attnA][rollingIdx] += r.playerWins[0]
			rollingAttnWins[attnB][rollingIdx] += r.playerWins[2]
			// Position 1 = regA, position 3 = regB
			regModels[regA].TotalWins += r.playerWins[1]
			regModels[regB].TotalWins += r.playerWins[3]
			rollingRegWins[regA][rollingIdx] += r.playerWins[1]
			rollingRegWins[regB][rollingIdx] += r.playerWins[3]
		}

		// Phase 2: Swap sides — Regular pair at positions 0&2, Attention pair at 1&3
		// Game positions: {reg_a, attn_a, reg_b, attn_b}
		seeds2 := make([]int64, sameGameIterations)
		for i := range seeds2 {
			seeds2[i] = rand.Int63n(9223372036854775607)
		}
		mixedMasters2 := [4]*nn.JSDNN{regModels[regA], attnModels[attnA], regModels[regB], attnModels[attnB]}
		results2 := make([]gameResult, sameGameIterations)

		for i := 0; i < sameGameIterations; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()
				clones := cloneNNs(mixedMasters2)
				gp := [4]dominos.Player{clones[0], clones[1], clones[2], clones[3]}
				results2[idx] = playGamePartner(gp, clones, seeds2[idx])
			}(i)
		}
		wg.Wait()

		applyTrainingPartner(mixedMasters2, results2)

		for _, r := range results2 {
			// Position 0 = regA, position 2 = regB
			regModels[regA].TotalWins2 += r.playerWins[0]
			regModels[regB].TotalWins2 += r.playerWins[2]
			rollingRegWins[regA][rollingIdx] += r.playerWins[0]
			rollingRegWins[regB][rollingIdx] += r.playerWins[2]
			// Position 1 = attnA, position 3 = attnB
			attnModels[attnA].TotalWins2 += r.playerWins[1]
			attnModels[attnB].TotalWins2 += r.playerWins[3]
			rollingAttnWins[attnA][rollingIdx] += r.playerWins[1]
			rollingAttnWins[attnB][rollingIdx] += r.playerWins[3]
		}

		// Phase 3: Benchmark each of the 8 models individually vs 3 random players (cutthroat)
		savedAttnEpsilon := [4]float64{}
		savedRegEpsilon := [4]float64{}
		for i := range attnModels {
			savedAttnEpsilon[i] = attnModels[i].Epsilon
			savedRegEpsilon[i] = regModels[i].Epsilon
			attnModels[i].Epsilon = 0
			regModels[i].Epsilon = 0
		}

		for p := 0; p < 4; p++ {
			rollingBenchAttn[p][rollingIdx] = 0
			rollingBenchReg[p][rollingIdx] = 0
			rollingBenchAttnGames[p][rollingIdx] = 0
			rollingBenchRegGames[p][rollingIdx] = 0
		}

		type benchResult struct {
			nnWins    int
			totalWins int
		}
		benchOut := make([]benchResult, 8) // 4 attn + 4 reg, 1 game each
		benchSeeds := make([]int64, 8)
		benchPositions := make([]int, 8)
		for i := range benchSeeds {
			benchSeeds[i] = rand.Int63n(9223372036854775607)
			benchPositions[i] = rand.Intn(4)
		}

		for i := 0; i < 8; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()
				pos := benchPositions[idx]
				var clone *nn.JSDNN
				if idx < 4 {
					clone = attnModels[idx].Clone()
				} else {
					clone = regModels[idx-4].Clone()
				}
				clone.Epsilon = 0
				var gp [4]dominos.Player
				// Use dummy NNs for event collection
				allModels := [4]*nn.JSDNN{attnModels[0], attnModels[1], attnModels[2], attnModels[3]}
				dummyNNs := cloneNNs(allModels)
				for k := 0; k < 4; k++ {
					dummyNNs[k].Epsilon = 0
				}
				for k := 0; k < 4; k++ {
					if k == pos {
						gp[k] = clone
					} else {
						gp[k] = &dominos.ComputerPlayer{RandomMode: true}
					}
				}
				result := playGame(gp, dummyNNs, benchSeeds[idx])
				total := 0
				for k := 0; k < 4; k++ {
					total += result.playerWins[k]
				}
				benchOut[idx] = benchResult{nnWins: result.playerWins[pos], totalWins: total}
			}(i)
		}
		wg.Wait()

		// Accumulate benchmark results
		for m := 0; m < 4; m++ {
			// Attention models
			totalBenchAttn[m] += benchOut[m].nnWins
			totalBenchAttnGames[m] += benchOut[m].totalWins
			rollingBenchAttn[m][rollingIdx] += benchOut[m].nnWins
			rollingBenchAttnGames[m][rollingIdx] += benchOut[m].totalWins
			elosBenchAttn[m] = updateEloBench(elosBenchAttn[m], benchOut[m].nnWins)
			// Regular models
			totalBenchReg[m] += benchOut[m+4].nnWins
			totalBenchRegGames[m] += benchOut[m+4].totalWins
			rollingBenchReg[m][rollingIdx] += benchOut[m+4].nnWins
			rollingBenchRegGames[m][rollingIdx] += benchOut[m+4].totalWins
			elosBenchReg[m] = updateEloBench(elosBenchReg[m], benchOut[m+4].nnWins)
		}

		// Restore exploration
		for i := range attnModels {
			attnModels[i].Epsilon = savedAttnEpsilon[i]
			regModels[i].Epsilon = savedRegEpsilon[i]
		}

		// Advance rolling window
		rollingIdx = (rollingIdx + 1) % rollingSize
		rollingCount++

		// Save all 8 models and metadata
		iterationCount++
		for i, name := range attnNames {
			attnModels[i].Save(cfg.resultsPath(name + ".mdl"))
			saveModelMeta(cfg.resultsPath(name+".meta.json"), modelMeta{
				Iterations: iterationCount,
				TotalWins:  attnModels[i].TotalWins,
				TotalWins2: attnModels[i].TotalWins2,
				EloNN:      elosAttn[i],
				EloBench:   elosBenchAttn[i],
			})
		}
		for i, name := range regNames {
			regModels[i].Save(cfg.resultsPath(name + ".mdl"))
			saveModelMeta(cfg.resultsPath(name+".meta.json"), modelMeta{
				Iterations: iterationCount,
				TotalWins:  regModels[i].TotalWins,
				TotalWins2: regModels[i].TotalWins2,
				EloNN:      elosReg[i],
				EloBench:   elosBenchReg[i],
			})
		}

		// Write Elo to CSV
		row := []string{fmt.Sprintf("%d", iterationCount)}
		for i := range attnNames {
			row = append(row, fmt.Sprintf("%.1f", elosAttn[i]), fmt.Sprintf("%.1f", elosBenchAttn[i]))
		}
		for i := range regNames {
			row = append(row, fmt.Sprintf("%.1f", elosReg[i]), fmt.Sprintf("%.1f", elosBenchReg[i]))
		}
		eloWriter.Write(row)
		eloWriter.Flush()

		// Logging
		log.Info("Partner Games Seen: ", j+1)
		for m := 0; m < 4; m++ {
			rollingWins := 0
			rollingGames := 0
			n := rollingCount
			if n > rollingSize {
				n = rollingSize
			}
			for i := 0; i < n; i++ {
				rollingWins += rollingBenchAttn[m][i]
				rollingGames += rollingBenchAttnGames[m][i]
			}
			rollingRate := 0.0
			if rollingGames > 0 {
				rollingRate = float64(rollingWins) / float64(rollingGames)
			}
			totalRate := 0.0
			if totalBenchAttnGames[m] > 0 {
				totalRate = float64(totalBenchAttn[m]) / float64(totalBenchAttnGames[m])
			}
			log.Infof("Bench %s: total=%.4f (%d/%d)  rolling1k=%.4f (%d/%d)  eloBench=%.1f",
				attnNames[m], totalRate, totalBenchAttn[m], totalBenchAttnGames[m],
				rollingRate, rollingWins, rollingGames, elosBenchAttn[m])
		}
		for m := 0; m < 4; m++ {
			rollingWins := 0
			rollingGames := 0
			n := rollingCount
			if n > rollingSize {
				n = rollingSize
			}
			for i := 0; i < n; i++ {
				rollingWins += rollingBenchReg[m][i]
				rollingGames += rollingBenchRegGames[m][i]
			}
			rollingRate := 0.0
			if rollingGames > 0 {
				rollingRate = float64(rollingWins) / float64(rollingGames)
			}
			totalRate := 0.0
			if totalBenchRegGames[m] > 0 {
				totalRate = float64(totalBenchReg[m]) / float64(totalBenchRegGames[m])
			}
			log.Infof("Bench %s: total=%.4f (%d/%d)  rolling1k=%.4f (%d/%d)  eloBench=%.1f",
				regNames[m], totalRate, totalBenchReg[m], totalBenchRegGames[m],
				rollingRate, rollingWins, rollingGames, elosBenchReg[m])
		}
		log.Infof("Attn Wins: [%d, %d, %d, %d]", attnModels[0].TotalWins, attnModels[1].TotalWins, attnModels[2].TotalWins, attnModels[3].TotalWins)
		log.Infof("Reg Wins: [%d, %d, %d, %d]", regModels[0].TotalWins, regModels[1].TotalWins, regModels[2].TotalWins, regModels[3].TotalWins)
	}

	log.Info("Took : ", time.Now().Sub(start))
}

// cloneTransformers creates deep copies of the master transformers for concurrent gameplay.
func cloneTransformers(masters [4]*nn.SequenceTransformer) [4]*nn.SequenceTransformer {
	var clones [4]*nn.SequenceTransformer
	for i, m := range masters {
		clones[i] = m.Clone()
		clones[i].Epsilon = m.Epsilon
	}
	return clones
}

// playGamePartnerTransformer runs a full partner game with transformer players at specific positions.
// tfPositions maps game position → transformer clone (nil for non-transformer positions).
// Returns game result with round events for training.
func playGamePartnerTransformer(dp [4]dominos.Player, tfPositions [4]*nn.SequenceTransformer, seed int64) gameResult {
	dominosGame := dominos.NewLocalGame(dp, seed, "partner")
	roundGameEvents := []*dominos.GameEvent{}
	lastGameEvent := &dominos.GameEvent{}
	result := gameResult{}

	for lastGameEvent != nil && lastGameEvent.EventType != dominos.GameWin {
		lastGameEvent = dominosGame.AdvanceGameIteration()
		lGE := jsdonline.CopyandRotateGameEvent(lastGameEvent, 0)

		if lGE.EventType == dominos.PlayedCard ||
			lGE.EventType == dominos.Passed || lastGameEvent.EventType == dominos.RoundWin ||
			lastGameEvent.EventType == dominos.RoundDraw {
			roundGameEvents = append(roundGameEvents, lGE)
		}

		// Notify transformer players about game events for history tracking
		if lGE.EventType == dominos.PlayedCard || lGE.EventType == dominos.Passed || lGE.EventType == dominos.PosedCard {
			for k := 0; k < 4; k++ {
				if tfPositions[k] != nil {
					tfPositions[k].ObserveEvent(lGE)
				}
			}
		}

		if lastGameEvent.EventType == dominos.RoundWin || lastGameEvent.EventType == dominos.RoundDraw {
			for k := 0; k < 4; k++ {
				if tfPositions[k] != nil {
					tfPositions[k].ResetHistory()
					tfPositions[k].ResetPassMemory()
				}
			}

			var round []roundEvent
			for i, gameEvent := range roundGameEvents {
				if gameEvent.EventType == dominos.PlayedCard {
					rotatedGameEvent := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
					nextRelevant := [16]*dominos.GameEvent{}
					for j := 1; j < 17; j++ {
						if (i + j) >= len(roundGameEvents) {
							break
						}
						nextRelevant[j-1] = jsdonline.CopyandRotateGameEvent(roundGameEvents[i+j], gameEvent.Player)
					}
					round = append(round, roundEvent{
						eventType:  "train",
						player:     gameEvent.Player,
						gameEvent:  rotatedGameEvent,
						nextEvents: nextRelevant,
					})
				} else if gameEvent.EventType == dominos.Passed {
					round = append(round, roundEvent{
						eventType: "pass",
						player:    gameEvent.Player,
						gameEvent: gameEvent,
					})
				} else if gameEvent.EventType == dominos.RoundWin || gameEvent.EventType == dominos.RoundDraw {
					round = append(round, roundEvent{
						eventType: "reset",
						player:    gameEvent.Player,
						gameEvent: gameEvent,
					})
				}
			}
			result.rounds = append(result.rounds, round)
			roundGameEvents = []*dominos.GameEvent{}
		}
	}

	result.playerWins = lastGameEvent.PlayerWins
	return result
}

// playGamePartnerMixed runs a full partner game with any mix of player types.
// Works with JSDNN, SequenceTransformer, and ComputerPlayer via interface type assertions.
func playGamePartnerMixed(dp [4]dominos.Player, seed int64) gameResult {
	dominosGame := dominos.NewLocalGame(dp, seed, "partner")
	roundGameEvents := []*dominos.GameEvent{}
	lastGameEvent := &dominos.GameEvent{}
	result := gameResult{}

	for lastGameEvent != nil && lastGameEvent.EventType != dominos.GameWin {
		lastGameEvent = dominosGame.AdvanceGameIteration()
		lGE := jsdonline.CopyandRotateGameEvent(lastGameEvent, 0)

		if lGE.EventType == dominos.PlayedCard ||
			lGE.EventType == dominos.Passed || lastGameEvent.EventType == dominos.RoundWin ||
			lastGameEvent.EventType == dominos.RoundDraw {
			roundGameEvents = append(roundGameEvents, lGE)
		}

		// Notify EventObserver players (e.g., transformer) about game events
		if lGE.EventType == dominos.PlayedCard || lGE.EventType == dominos.Passed || lGE.EventType == dominos.PosedCard {
			for k := 0; k < 4; k++ {
				if obs, ok := dp[k].(nn.EventObserver); ok {
					obs.ObserveEvent(lGE)
				}
			}
		}

		if lastGameEvent.EventType == dominos.RoundWin || lastGameEvent.EventType == dominos.RoundDraw {
			// Reset pass memory and history per player type
			for k := 0; k < 4; k++ {
				switch p := dp[k].(type) {
				case *nn.JSDNN:
					p.ResetPassMemory()
				case *nn.SequenceTransformer:
					p.ResetHistory()
					p.ResetPassMemory()
				}
			}

			var round []roundEvent
			for i, gameEvent := range roundGameEvents {
				if gameEvent.EventType == dominos.PlayedCard {
					rotatedGameEvent := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
					nextRelevant := [16]*dominos.GameEvent{}
					for j := 1; j < 17; j++ {
						if (i + j) >= len(roundGameEvents) {
							break
						}
						nextRelevant[j-1] = jsdonline.CopyandRotateGameEvent(roundGameEvents[i+j], gameEvent.Player)
					}
					round = append(round, roundEvent{
						eventType:  "train",
						player:     gameEvent.Player,
						gameEvent:  rotatedGameEvent,
						nextEvents: nextRelevant,
					})
				} else if gameEvent.EventType == dominos.Passed {
					round = append(round, roundEvent{
						eventType: "pass",
						player:    gameEvent.Player,
						gameEvent: gameEvent,
					})
				} else if gameEvent.EventType == dominos.RoundWin || gameEvent.EventType == dominos.RoundDraw {
					round = append(round, roundEvent{
						eventType: "reset",
						player:    gameEvent.Player,
						gameEvent: gameEvent,
					})
				}
			}
			result.rounds = append(result.rounds, round)
			roundGameEvents = []*dominos.GameEvent{}
		}
	}

	result.playerWins = lastGameEvent.PlayerWins
	return result
}

// applyTrainingTransformer processes collected game results on master transformers.
// tfPlayers maps position → master transformer (nil for non-transformer positions).
// Each transformer trains on its own player's events with proper history context.
func applyTrainingTransformer(tfPlayers [4]*nn.SequenceTransformer, results []gameResult) {
	playerEvents := [4][]roundEvent{}

	for _, result := range results {
		for _, round := range result.rounds {
			for p := 0; p < 4; p++ {
				playerEvents[p] = append(playerEvents[p], roundEvent{eventType: "reset"})
			}
			for _, evt := range round {
				playerEvents[evt.player] = append(playerEvents[evt.player], evt)
			}
		}
	}

	// Train each transformer in parallel (skip nil positions)
	var wg sync.WaitGroup
	for p := 0; p < 4; p++ {
		if tfPlayers[p] == nil || len(playerEvents[p]) == 0 {
			continue
		}
		wg.Add(1)
		go func(playerIdx int) {
			defer wg.Done()
			tf := tfPlayers[playerIdx]
			// Rebuild history as we process events sequentially
			for _, evt := range playerEvents[playerIdx] {
				switch evt.eventType {
				case "train":
					_, err := tf.TrainReinforcedPartner(evt.gameEvent, 0.0005, evt.nextEvents)
					if err != nil {
						log.Error("Error training transformer: ", err)
					}
					// Observe the event so history builds up
					tf.ObserveEvent(evt.gameEvent)
				case "pass":
					tf.UpdatePassMemory(evt.gameEvent)
					tf.ObserveEvent(evt.gameEvent)
				case "reset":
					tf.ResetHistory()
					tf.ResetPassMemory()
				}
			}
		}(p)
	}
	wg.Wait()
}

// applyTrainingMixed processes collected game results for any mix of player types.
// Type-switches per player position to call the correct training method.
func applyTrainingMixed(players [4]dominos.Player, results []gameResult) {
	playerEvents := [4][]roundEvent{}

	for _, result := range results {
		for _, round := range result.rounds {
			for p := 0; p < 4; p++ {
				playerEvents[p] = append(playerEvents[p], roundEvent{eventType: "reset"})
			}
			for _, evt := range round {
				playerEvents[evt.player] = append(playerEvents[evt.player], evt)
			}
		}
	}

	var wg sync.WaitGroup
	for p := 0; p < 4; p++ {
		if len(playerEvents[p]) == 0 {
			continue
		}
		wg.Add(1)
		go func(playerIdx int) {
			defer wg.Done()
			player := players[playerIdx]
			for _, evt := range playerEvents[playerIdx] {
				switch pl := player.(type) {
				case *nn.JSDNN:
					switch evt.eventType {
					case "train":
						var learnRates []float64
						if pl.GetNumHidden() == 1 {
							learnRates = []float64{0.001}
						} else {
							learnRates = []float64{0.001, 0.0001}
						}
						_, err := pl.TrainReinforcedPartner(evt.gameEvent, learnRates, evt.nextEvents)
						if err != nil {
							log.Fatal("Error training JSDNN: ", err)
						}
					case "pass":
						pl.UpdatePassMemory(evt.gameEvent)
					case "reset":
						pl.ResetPassMemory()
					}
				case *nn.SequenceTransformer:
					switch evt.eventType {
					case "train":
						_, err := pl.TrainReinforcedPartner(evt.gameEvent, 0.0005, evt.nextEvents)
						if err != nil {
							log.Error("Error training transformer: ", err)
						}
						pl.ObserveEvent(evt.gameEvent)
					case "pass":
						pl.UpdatePassMemory(evt.gameEvent)
						pl.ObserveEvent(evt.gameEvent)
					case "reset":
						pl.ResetHistory()
						pl.ResetPassMemory()
					}
				}
			}
		}(p)
	}
	wg.Wait()
}

func trainReinforcedTransformer(cfg trainConfig) {
	// Train transformer models as partners vs random computer AI
	start := time.Now()

	sameGameIterations := 8
	numWorkers := runtime.NumCPU()

	modelNames := [4]string{"jasai_transformer1", "jasai_transformer2", "jasai_transformer3", "jasai_transformer4"}

	// Create 4 transformer models
	models := [4]*nn.SequenceTransformer{
		nn.NewSequenceTransformer(64, 2, 2, 128, 40, 56),
		nn.NewSequenceTransformer(64, 2, 2, 128, 40, 56),
		nn.NewSequenceTransformer(64, 2, 2, 128, 40, 56),
		nn.NewSequenceTransformer(64, 2, 2, 128, 40, 56),
	}

	// Try loading existing models
	for i, name := range modelNames {
		if _, err := os.Stat(cfg.resultsPath(name + ".mdl")); err == nil {
			models[i].Load(cfg.resultsPath(name + ".mdl"))
			log.Infof("Loaded %s", name)
		}
		models[i].Epsilon = 0.1
	}

	// Load metadata
	metas := [4]modelMeta{}
	for i, name := range modelNames {
		metas[i] = loadModelMeta(cfg.resultsPath(name + ".meta.json"))
	}
	iterationCount := metas[0].Iterations
	for i := range models {
		models[i].TotalWins = metas[i].TotalWins
		models[i].TotalWins2 = metas[i].TotalWins2
	}

	elosBench := [4]float64{}
	for i := range elosBench {
		if metas[i].EloBench > 0 {
			elosBench[i] = metas[i].EloBench
		} else {
			elosBench[i] = eloRandomPlayer
		}
	}
	log.Info("Transformer: Resuming from iteration: ", iterationCount)
	log.Infof("Starting Bench Elos: [%.1f, %.1f, %.1f, %.1f]", elosBench[0], elosBench[1], elosBench[2], elosBench[3])

	// Rolling window buffers
	const rollingSize = 1000
	rollingWins := [4][]int{}
	rollingBench := [4][]int{}
	rollingBenchGames := [4][]int{}
	rollingIdx := 0
	rollingCount := 0
	totalBenchWins := [4]int{}
	totalBenchGames := [4]int{}

	// Transformer vs random win tracking
	totalTFWins := 0     // transformer team round wins
	totalRandomWins := 0 // random team round wins
	rollingTFWins := make([]int, rollingSize)
	rollingRandomWins := make([]int, rollingSize)

	for p := 0; p < 4; p++ {
		rollingWins[p] = make([]int, rollingSize)
		rollingBench[p] = make([]int, rollingSize)
		rollingBenchGames[p] = make([]int, rollingSize)
	}

	// Open CSV for Elo tracking
	eloCSVPath := cfg.resultsPath("elo_history_transformer.csv")
	eloFileExists := false
	if _, err := os.Stat(eloCSVPath); err == nil {
		eloFileExists = true
	}
	eloFile, err := os.OpenFile(eloCSVPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatal("Failed to open elo CSV: ", err)
	}
	defer eloFile.Close()
	eloWriter := csv.NewWriter(eloFile)
	defer eloWriter.Flush()
	if !eloFileExists {
		header := []string{"iteration", "games_seen", "tf_win_pct", "tf_wins", "random_wins"}
		for _, name := range modelNames {
			header = append(header, name+"_bench")
		}
		eloWriter.Write(header)
		eloWriter.Flush()
	}

	randomSeed := time.Now().UnixNano()
	log.Info("Using random Seed: ", randomSeed)
	rand.Seed(randomSeed)

	bar := progressbar.Default(int64(cfg.maxGames), "Transformer Training")

	for j := 0; !cfg.shouldStop(j, start); j++ {
		// Pick 2 random transformer models to be partners
		perm := rand.Perm(4)
		tfA, tfB := perm[0], perm[1]

		var wg sync.WaitGroup
		sem := make(chan struct{}, numWorkers)

		// Phase 1: Transformers at positions 0&2 (partners) vs random AI at 1&3
		seeds := make([]int64, sameGameIterations)
		for i := range seeds {
			seeds[i] = rand.Int63n(9223372036854775607)
		}
		results1 := make([]gameResult, sameGameIterations)
		for i := 0; i < sameGameIterations; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()
				cloneA := models[tfA].Clone()
				cloneA.Epsilon = models[tfA].Epsilon
				cloneB := models[tfB].Clone()
				cloneB.Epsilon = models[tfB].Epsilon
				gp := [4]dominos.Player{
					cloneA,
					&dominos.ComputerPlayer{RandomMode: true},
					cloneB,
					&dominos.ComputerPlayer{RandomMode: true},
				}
				tfPositions := [4]*nn.SequenceTransformer{cloneA, nil, cloneB, nil}
				results1[idx] = playGamePartnerTransformer(gp, tfPositions, seeds[idx])
			}(i)
		}
		wg.Wait()

		// Apply training only to the transformer positions
		masterPositions1 := [4]*nn.SequenceTransformer{models[tfA], nil, models[tfB], nil}
		applyTrainingTransformer(masterPositions1, results1)

		// Track wins for phase 1
		for p := 0; p < 4; p++ {
			rollingWins[p][rollingIdx] = 0
		}
		rollingTFWins[rollingIdx] = 0
		rollingRandomWins[rollingIdx] = 0
		for _, r := range results1 {
			// Positions 0&2 = transformers, 1&3 = random
			tfRoundWins := r.playerWins[0]
			randomRoundWins := r.playerWins[1]
			totalTFWins += tfRoundWins
			totalRandomWins += randomRoundWins
			rollingTFWins[rollingIdx] += tfRoundWins
			rollingRandomWins[rollingIdx] += randomRoundWins
			models[tfA].TotalWins += r.playerWins[0]
			models[tfB].TotalWins += r.playerWins[2]
			rollingWins[tfA][rollingIdx] += r.playerWins[0]
			rollingWins[tfB][rollingIdx] += r.playerWins[2]
		}

		// Phase 2: Swap sides — Transformers at positions 1&3, random AI at 0&2
		seeds2 := make([]int64, sameGameIterations)
		for i := range seeds2 {
			seeds2[i] = rand.Int63n(9223372036854775607)
		}
		results2 := make([]gameResult, sameGameIterations)
		for i := 0; i < sameGameIterations; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()
				cloneA := models[tfA].Clone()
				cloneA.Epsilon = models[tfA].Epsilon
				cloneB := models[tfB].Clone()
				cloneB.Epsilon = models[tfB].Epsilon
				gp := [4]dominos.Player{
					&dominos.ComputerPlayer{RandomMode: true},
					cloneA,
					&dominos.ComputerPlayer{RandomMode: true},
					cloneB,
				}
				tfPositions := [4]*nn.SequenceTransformer{nil, cloneA, nil, cloneB}
				results2[idx] = playGamePartnerTransformer(gp, tfPositions, seeds2[idx])
			}(i)
		}
		wg.Wait()

		masterPositions2 := [4]*nn.SequenceTransformer{nil, models[tfA], nil, models[tfB]}
		applyTrainingTransformer(masterPositions2, results2)

		for _, r := range results2 {
			// Positions 1&3 = transformers, 0&2 = random
			tfRoundWins := r.playerWins[1]
			randomRoundWins := r.playerWins[0]
			totalTFWins += tfRoundWins
			totalRandomWins += randomRoundWins
			rollingTFWins[rollingIdx] += tfRoundWins
			rollingRandomWins[rollingIdx] += randomRoundWins
			models[tfA].TotalWins2 += r.playerWins[1]
			models[tfB].TotalWins2 += r.playerWins[3]
			rollingWins[tfA][rollingIdx] += r.playerWins[1]
			rollingWins[tfB][rollingIdx] += r.playerWins[3]
		}

		// Phase 3: Benchmark each model individually vs 3 random players (cutthroat)
		savedEpsilon := [4]float64{}
		for i := range models {
			savedEpsilon[i] = models[i].Epsilon
			models[i].Epsilon = 0
		}

		for p := 0; p < 4; p++ {
			rollingBench[p][rollingIdx] = 0
			rollingBenchGames[p][rollingIdx] = 0
		}

		type benchResultT struct {
			nnWins    int
			totalWins int
		}
		benchOut := make([]benchResultT, 4)
		benchSeeds := make([]int64, 4)
		benchPositions := make([]int, 4)
		for i := range benchSeeds {
			benchSeeds[i] = rand.Int63n(9223372036854775607)
			benchPositions[i] = rand.Intn(4)
		}

		for i := 0; i < 4; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()
				pos := benchPositions[idx]
				clone := models[idx].Clone()
				clone.Epsilon = 0
				var gp [4]dominos.Player
				for k := 0; k < 4; k++ {
					if k == pos {
						gp[k] = clone
					} else {
						gp[k] = &dominos.ComputerPlayer{RandomMode: true}
					}
				}
				dominosGame := dominos.NewLocalGame(gp, benchSeeds[idx], "cutthroat")
				lastGameEvent := &dominos.GameEvent{}
				for lastGameEvent != nil && lastGameEvent.EventType != dominos.GameWin {
					lastGameEvent = dominosGame.AdvanceGameIteration()
					lGE := jsdonline.CopyandRotateGameEvent(lastGameEvent, 0)
					if lGE.EventType == dominos.PlayedCard || lGE.EventType == dominos.Passed || lGE.EventType == dominos.PosedCard {
						clone.ObserveEvent(lGE)
					}
					if lastGameEvent.EventType == dominos.RoundWin || lastGameEvent.EventType == dominos.RoundDraw {
						clone.ResetHistory()
					}
				}
				total := 0
				for k := 0; k < 4; k++ {
					total += lastGameEvent.PlayerWins[k]
				}
				benchOut[idx] = benchResultT{nnWins: lastGameEvent.PlayerWins[pos], totalWins: total}
			}(i)
		}
		wg.Wait()

		for m := 0; m < 4; m++ {
			totalBenchWins[m] += benchOut[m].nnWins
			totalBenchGames[m] += benchOut[m].totalWins
			rollingBench[m][rollingIdx] += benchOut[m].nnWins
			rollingBenchGames[m][rollingIdx] += benchOut[m].totalWins
			elosBench[m] = updateEloBench(elosBench[m], benchOut[m].nnWins)
		}

		// Restore exploration
		for i := range models {
			models[i].Epsilon = savedEpsilon[i]
		}

		// Advance rolling window
		rollingIdx = (rollingIdx + 1) % rollingSize
		rollingCount++

		// Save models and metadata
		iterationCount++
		for i, name := range modelNames {
			models[i].Save(cfg.resultsPath(name + ".mdl"))
			saveModelMeta(cfg.resultsPath(name+".meta.json"), modelMeta{
				Iterations: iterationCount,
				TotalWins:  models[i].TotalWins,
				TotalWins2: models[i].TotalWins2,
				EloBench:   elosBench[i],
			})
		}

		// Write Elo + win% to CSV
		csvTotalGames := totalTFWins + totalRandomWins
		csvWinPct := 0.0
		if csvTotalGames > 0 {
			csvWinPct = float64(totalTFWins) / float64(csvTotalGames) * 100
		}
		row := []string{
			fmt.Sprintf("%d", iterationCount),
			fmt.Sprintf("%d", j+1),
			fmt.Sprintf("%.2f", csvWinPct),
			fmt.Sprintf("%d", totalTFWins),
			fmt.Sprintf("%d", totalRandomWins),
		}
		for i := range modelNames {
			row = append(row, fmt.Sprintf("%.1f", elosBench[i]))
		}
		eloWriter.Write(row)
		eloWriter.Flush()

		// Logging
		bar.Add(1)
		log.Info("Transformer Games Seen: ", j+1)
		for m := 0; m < 4; m++ {
			rWins := 0
			rGames := 0
			n := rollingCount
			if n > rollingSize {
				n = rollingSize
			}
			for i := 0; i < n; i++ {
				rWins += rollingBench[m][i]
				rGames += rollingBenchGames[m][i]
			}
			rollingRate := 0.0
			if rGames > 0 {
				rollingRate = float64(rWins) / float64(rGames)
			}
			totalRate := 0.0
			if totalBenchGames[m] > 0 {
				totalRate = float64(totalBenchWins[m]) / float64(totalBenchGames[m])
			}
			log.Infof("Bench %s: total=%.4f (%d/%d)  rolling1k=%.4f (%d/%d)  eloBench=%.1f",
				modelNames[m], totalRate, totalBenchWins[m], totalBenchGames[m],
				rollingRate, rWins, rGames, elosBench[m])
		}
		// Transformer vs Random win %
		totalGamesPlayed := totalTFWins + totalRandomWins
		tfWinPct := 0.0
		if totalGamesPlayed > 0 {
			tfWinPct = float64(totalTFWins) / float64(totalGamesPlayed) * 100
		}
		// Rolling win %
		rollingTF := 0
		rollingRandom := 0
		n := rollingCount
		if n > rollingSize {
			n = rollingSize
		}
		for i := 0; i < n; i++ {
			rollingTF += rollingTFWins[i]
			rollingRandom += rollingRandomWins[i]
		}
		rollingPct := 0.0
		if rollingTF+rollingRandom > 0 {
			rollingPct = float64(rollingTF) / float64(rollingTF+rollingRandom) * 100
		}
		log.Infof("TF vs Random: total=%.1f%% (%d/%d)  rolling1k=%.1f%% (%d/%d)",
			tfWinPct, totalTFWins, totalGamesPlayed, rollingPct, rollingTF, rollingTF+rollingRandom)
		log.Infof("TF Wins (vs random): [%d, %d, %d, %d]", models[0].TotalWins, models[1].TotalWins, models[2].TotalWins, models[3].TotalWins)
		log.Infof("TF Wins2 (swapped): [%d, %d, %d, %d]", models[0].TotalWins2, models[1].TotalWins2, models[2].TotalWins2, models[3].TotalWins2)
	}

	for i, name := range modelNames {
		models[i].Save(cfg.resultsPath(name + ".mdl"))
		saveModelMeta(cfg.resultsPath(name+".meta.json"), modelMeta{
			Iterations: iterationCount,
			TotalWins:  models[i].TotalWins,
			TotalWins2: models[i].TotalWins2,
			EloBench:   elosBench[i],
		})
	}
	log.Info("Took : ", time.Now().Sub(start))
}

// cloneMixedPlayer creates a deep copy of a dominos.Player (JSDNN or SequenceTransformer).
func cloneMixedPlayer(p dominos.Player) dominos.Player {
	switch pl := p.(type) {
	case *nn.JSDNN:
		c := pl.Clone()
		c.Epsilon = pl.Epsilon
		c.Search = pl.Search
		c.SearchNum = pl.SearchNum
		return c
	case *nn.SequenceTransformer:
		c := pl.Clone()
		c.Epsilon = pl.Epsilon
		return c
	default:
		return p
	}
}

func trainHeadToHead(cfg trainConfig) {
	start := time.Now()

	sameGameIterations := 8
	numWorkers := runtime.NumCPU()

	// Model file names
	mlpNames := [2]string{"jasai_h2h_mlp1", "jasai_h2h_mlp2"}
	attnNames := [2]string{"jasai_h2h_attn1", "jasai_h2h_attn2"}
	tfNames := [2]string{"jasai_h2h_tf1", "jasai_h2h_tf2"}

	// Create 6 models (2 per type)
	mlpModels := [2]*nn.JSDNN{
		nn.New(126, []int{256, 128}, 56),
		nn.New(126, []int{256, 128}, 56),
	}
	for i := range mlpModels {
		mlpModels[i].OutputActivation = "linear"
		if _, err := os.Stat(cfg.resultsPath(mlpNames[i] + ".mdl")); err == nil {
			mlpModels[i].Load(cfg.resultsPath(mlpNames[i] + ".mdl"))
			log.Infof("Loaded %s", mlpNames[i])
		}
		mlpModels[i].Epsilon = 0.1
	}

	attnModels := [2]*nn.JSDNN{
		nn.New(126, []int{256, 128}, 56),
		nn.New(126, []int{256, 128}, 56),
	}
	for i := range attnModels {
		attnModels[i].OutputActivation = "linear"
		attnModels[i].EnableAttention(28)
		if _, err := os.Stat(cfg.resultsPath(attnNames[i] + ".mdl")); err == nil {
			attnModels[i].Load(cfg.resultsPath(attnNames[i] + ".mdl"))
			log.Infof("Loaded %s", attnNames[i])
		}
		attnModels[i].Epsilon = 0.1
	}

	tfModels := [2]*nn.SequenceTransformer{
		nn.NewSequenceTransformer(64, 2, 2, 128, 40, 56),
		nn.NewSequenceTransformer(64, 2, 2, 128, 40, 56),
	}
	for i := range tfModels {
		if _, err := os.Stat(cfg.resultsPath(tfNames[i] + ".mdl")); err == nil {
			tfModels[i].Load(cfg.resultsPath(tfNames[i] + ".mdl"))
			log.Infof("Loaded %s", tfNames[i])
		}
		tfModels[i].Epsilon = 0.1
	}

	// Load metadata to resume progress
	h2hMetaPath := cfg.resultsPath("h2h_meta.json")
	meta := loadH2HMeta(h2hMetaPath)
	iterationCount := meta.Iterations

	// Win tracking counters (cumulative, resumed from metadata)
	mlpVsAttn_mlpWins := meta.MlpVsAttnMlp
	mlpVsAttn_attnWins := meta.MlpVsAttnAttn
	mlpVsTF_mlpWins := meta.MlpVsTFMlp
	mlpVsTF_tfWins := meta.MlpVsTFTF
	attnVsTF_attnWins := meta.AttnVsTFAttn
	attnVsTF_tfWins := meta.AttnVsTFTF

	log.Info("H2H: Resuming from iteration: ", iterationCount)

	// Open CSV for win% tracking
	eloCSVPath := cfg.resultsPath("elo_history_headtohead.csv")
	eloFileExists := false
	if _, err := os.Stat(eloCSVPath); err == nil {
		eloFileExists = true
	}
	eloFile, err := os.OpenFile(eloCSVPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatal("Failed to open h2h CSV: ", err)
	}
	defer eloFile.Close()
	eloWriter := csv.NewWriter(eloFile)
	defer eloWriter.Flush()
	if !eloFileExists {
		header := []string{
			"iteration", "games_seen",
			"mlp_vs_attn_mlp_pct", "mlp_vs_attn_attn_pct",
			"mlp_vs_tf_mlp_pct", "mlp_vs_tf_tf_pct",
			"attn_vs_tf_attn_pct", "attn_vs_tf_tf_pct",
		}
		eloWriter.Write(header)
		eloWriter.Flush()
	}

	randomSeed := time.Now().UnixNano()
	log.Info("Using random Seed: ", randomSeed)
	rand.Seed(randomSeed)

	bar := progressbar.Default(int64(cfg.maxGames), "Head-to-Head Training")

	for j := 0; !cfg.shouldStop(j, start); j++ {
		var wg sync.WaitGroup
		sem := make(chan struct{}, numWorkers)

		// === Matchup 1: MLP vs Attention+MLP ===
		{
			// Phase 1: MLP at 0&2, Attn at 1&3
			seeds := make([]int64, sameGameIterations)
			for i := range seeds {
				seeds[i] = rand.Int63n(9223372036854775607)
			}
			results := make([]gameResult, sameGameIterations)
			for i := 0; i < sameGameIterations; i++ {
				wg.Add(1)
				sem <- struct{}{}
				go func(idx int) {
					defer wg.Done()
					defer func() { <-sem }()
					dp := [4]dominos.Player{
						cloneMixedPlayer(mlpModels[0]),
						cloneMixedPlayer(attnModels[0]),
						cloneMixedPlayer(mlpModels[1]),
						cloneMixedPlayer(attnModels[1]),
					}
					results[idx] = playGamePartnerMixed(dp, seeds[idx])
				}(i)
			}
			wg.Wait()
			masters := [4]dominos.Player{mlpModels[0], attnModels[0], mlpModels[1], attnModels[1]}
			applyTrainingMixed(masters, results)
			for _, r := range results {
				mlpVsAttn_mlpWins += r.playerWins[0]  // team 0&2 = MLP
				mlpVsAttn_attnWins += r.playerWins[1] // team 1&3 = Attn
			}

			// Phase 2: Swap — Attn at 0&2, MLP at 1&3
			seeds2 := make([]int64, sameGameIterations)
			for i := range seeds2 {
				seeds2[i] = rand.Int63n(9223372036854775607)
			}
			results2 := make([]gameResult, sameGameIterations)
			for i := 0; i < sameGameIterations; i++ {
				wg.Add(1)
				sem <- struct{}{}
				go func(idx int) {
					defer wg.Done()
					defer func() { <-sem }()
					dp := [4]dominos.Player{
						cloneMixedPlayer(attnModels[0]),
						cloneMixedPlayer(mlpModels[0]),
						cloneMixedPlayer(attnModels[1]),
						cloneMixedPlayer(mlpModels[1]),
					}
					results2[idx] = playGamePartnerMixed(dp, seeds2[idx])
				}(i)
			}
			wg.Wait()
			masters2 := [4]dominos.Player{attnModels[0], mlpModels[0], attnModels[1], mlpModels[1]}
			applyTrainingMixed(masters2, results2)
			for _, r := range results2 {
				mlpVsAttn_attnWins += r.playerWins[0] // team 0&2 = Attn (swapped)
				mlpVsAttn_mlpWins += r.playerWins[1]  // team 1&3 = MLP (swapped)
			}
		}

		// === Matchup 2: MLP vs Transformer ===
		{
			// Phase 1: MLP at 0&2, TF at 1&3
			seeds := make([]int64, sameGameIterations)
			for i := range seeds {
				seeds[i] = rand.Int63n(9223372036854775607)
			}
			results := make([]gameResult, sameGameIterations)
			for i := 0; i < sameGameIterations; i++ {
				wg.Add(1)
				sem <- struct{}{}
				go func(idx int) {
					defer wg.Done()
					defer func() { <-sem }()
					dp := [4]dominos.Player{
						cloneMixedPlayer(mlpModels[0]),
						cloneMixedPlayer(tfModels[0]),
						cloneMixedPlayer(mlpModels[1]),
						cloneMixedPlayer(tfModels[1]),
					}
					results[idx] = playGamePartnerMixed(dp, seeds[idx])
				}(i)
			}
			wg.Wait()
			masters := [4]dominos.Player{mlpModels[0], tfModels[0], mlpModels[1], tfModels[1]}
			applyTrainingMixed(masters, results)
			for _, r := range results {
				mlpVsTF_mlpWins += r.playerWins[0] // team 0&2 = MLP
				mlpVsTF_tfWins += r.playerWins[1]  // team 1&3 = TF
			}

			// Phase 2: Swap — TF at 0&2, MLP at 1&3
			seeds2 := make([]int64, sameGameIterations)
			for i := range seeds2 {
				seeds2[i] = rand.Int63n(9223372036854775607)
			}
			results2 := make([]gameResult, sameGameIterations)
			for i := 0; i < sameGameIterations; i++ {
				wg.Add(1)
				sem <- struct{}{}
				go func(idx int) {
					defer wg.Done()
					defer func() { <-sem }()
					dp := [4]dominos.Player{
						cloneMixedPlayer(tfModels[0]),
						cloneMixedPlayer(mlpModels[0]),
						cloneMixedPlayer(tfModels[1]),
						cloneMixedPlayer(mlpModels[1]),
					}
					results2[idx] = playGamePartnerMixed(dp, seeds2[idx])
				}(i)
			}
			wg.Wait()
			masters2 := [4]dominos.Player{tfModels[0], mlpModels[0], tfModels[1], mlpModels[1]}
			applyTrainingMixed(masters2, results2)
			for _, r := range results2 {
				mlpVsTF_tfWins += r.playerWins[0]  // team 0&2 = TF (swapped)
				mlpVsTF_mlpWins += r.playerWins[1] // team 1&3 = MLP (swapped)
			}
		}

		// === Matchup 3: Attention+MLP vs Transformer ===
		{
			// Phase 1: Attn at 0&2, TF at 1&3
			seeds := make([]int64, sameGameIterations)
			for i := range seeds {
				seeds[i] = rand.Int63n(9223372036854775607)
			}
			results := make([]gameResult, sameGameIterations)
			for i := 0; i < sameGameIterations; i++ {
				wg.Add(1)
				sem <- struct{}{}
				go func(idx int) {
					defer wg.Done()
					defer func() { <-sem }()
					dp := [4]dominos.Player{
						cloneMixedPlayer(attnModels[0]),
						cloneMixedPlayer(tfModels[0]),
						cloneMixedPlayer(attnModels[1]),
						cloneMixedPlayer(tfModels[1]),
					}
					results[idx] = playGamePartnerMixed(dp, seeds[idx])
				}(i)
			}
			wg.Wait()
			masters := [4]dominos.Player{attnModels[0], tfModels[0], attnModels[1], tfModels[1]}
			applyTrainingMixed(masters, results)
			for _, r := range results {
				attnVsTF_attnWins += r.playerWins[0] // team 0&2 = Attn
				attnVsTF_tfWins += r.playerWins[1]   // team 1&3 = TF
			}

			// Phase 2: Swap — TF at 0&2, Attn at 1&3
			seeds2 := make([]int64, sameGameIterations)
			for i := range seeds2 {
				seeds2[i] = rand.Int63n(9223372036854775607)
			}
			results2 := make([]gameResult, sameGameIterations)
			for i := 0; i < sameGameIterations; i++ {
				wg.Add(1)
				sem <- struct{}{}
				go func(idx int) {
					defer wg.Done()
					defer func() { <-sem }()
					dp := [4]dominos.Player{
						cloneMixedPlayer(tfModels[0]),
						cloneMixedPlayer(attnModels[0]),
						cloneMixedPlayer(tfModels[1]),
						cloneMixedPlayer(attnModels[1]),
					}
					results2[idx] = playGamePartnerMixed(dp, seeds2[idx])
				}(i)
			}
			wg.Wait()
			masters2 := [4]dominos.Player{tfModels[0], attnModels[0], tfModels[1], attnModels[1]}
			applyTrainingMixed(masters2, results2)
			for _, r := range results2 {
				attnVsTF_tfWins += r.playerWins[0]   // team 0&2 = TF (swapped)
				attnVsTF_attnWins += r.playerWins[1] // team 1&3 = Attn (swapped)
			}
		}

		// Save all 6 models and metadata
		iterationCount++
		for i, name := range mlpNames {
			mlpModels[i].Save(cfg.resultsPath(name + ".mdl"))
		}
		for i, name := range attnNames {
			attnModels[i].Save(cfg.resultsPath(name + ".mdl"))
		}
		for i, name := range tfNames {
			tfModels[i].Save(cfg.resultsPath(name + ".mdl"))
		}
		saveH2HMeta(h2hMetaPath, h2hMeta{
			Iterations:    iterationCount,
			MlpVsAttnMlp:  mlpVsAttn_mlpWins,
			MlpVsAttnAttn: mlpVsAttn_attnWins,
			MlpVsTFMlp:    mlpVsTF_mlpWins,
			MlpVsTFTF:     mlpVsTF_tfWins,
			AttnVsTFAttn:  attnVsTF_attnWins,
			AttnVsTFTF:    attnVsTF_tfWins,
		})

		// Compute win percentages
		winPct := func(wins, oppWins int) float64 {
			total := wins + oppWins
			if total == 0 {
				return 50.0
			}
			return float64(wins) / float64(total) * 100
		}

		mlpVsAttnMlpPct := winPct(mlpVsAttn_mlpWins, mlpVsAttn_attnWins)
		mlpVsAttnAttnPct := winPct(mlpVsAttn_attnWins, mlpVsAttn_mlpWins)
		mlpVsTFMlpPct := winPct(mlpVsTF_mlpWins, mlpVsTF_tfWins)
		mlpVsTFTFPct := winPct(mlpVsTF_tfWins, mlpVsTF_mlpWins)
		attnVsTFAttnPct := winPct(attnVsTF_attnWins, attnVsTF_tfWins)
		attnVsTFTFPct := winPct(attnVsTF_tfWins, attnVsTF_attnWins)

		// Write CSV row
		row := []string{
			fmt.Sprintf("%d", iterationCount),
			fmt.Sprintf("%d", iterationCount*sameGameIterations*6), // 3 matchups × 2 phases × sameGameIterations
			fmt.Sprintf("%.2f", mlpVsAttnMlpPct),
			fmt.Sprintf("%.2f", mlpVsAttnAttnPct),
			fmt.Sprintf("%.2f", mlpVsTFMlpPct),
			fmt.Sprintf("%.2f", mlpVsTFTFPct),
			fmt.Sprintf("%.2f", attnVsTFAttnPct),
			fmt.Sprintf("%.2f", attnVsTFTFPct),
		}
		eloWriter.Write(row)
		eloWriter.Flush()

		// Logging
		bar.Add(1)
		log.Infof("H2H Iteration %d | MLP vs Attn: %.1f%% / %.1f%% | MLP vs TF: %.1f%% / %.1f%% | Attn vs TF: %.1f%% / %.1f%%",
			iterationCount, mlpVsAttnMlpPct, mlpVsAttnAttnPct, mlpVsTFMlpPct, mlpVsTFTFPct, attnVsTFAttnPct, attnVsTFTFPct)
	}

	log.Info("Took : ", time.Now().Sub(start))
}

func main() {
	rootCmd := &cobra.Command{
		Use:   "jsd-ai",
		Short: "Jamaican Style Dominoes AI trainer",
	}

	var cfg trainConfig

	rootCmd.PersistentFlags().StringVar(&cfg.resultsDir, "results-dir", "./results", "directory for models/CSV/metadata")
	rootCmd.PersistentFlags().IntVar(&cfg.maxGames, "max-games", 100000, "max training iterations (0=unlimited)")
	rootCmd.PersistentFlags().Float64Var(&cfg.maxDays, "max-days", 0, "max training duration in days (0=disabled)")

	var duppyModel string
	var duppyMode string
	var duppyStatsPath string
	duppyCmd := &cobra.Command{
		Use:   "duppy-play",
		Short: "Play games online as duppy",
		Run:   func(cmd *cobra.Command, args []string) { duppyPlay(cfg, duppyModel, duppyMode, duppyStatsPath) },
	}
	duppyCmd.Flags().StringVar(&duppyModel, "model", "duppy.mdl", "path to model file")
	duppyCmd.Flags().StringVar(&duppyMode, "mode", "cutthroat", "game mode (cutthroat or partner)")
	duppyCmd.Flags().StringVar(&duppyStatsPath, "stats", "duppy_stats.json", "path to stats JSON file")

	var mongoURI string
	rootCmd.PersistentFlags().StringVar(&mongoURI, "mongo-uri", "", "MongoDB connection URI (default: production)")

	checkMongoCmd := &cobra.Command{
		Use:   "check-mongo",
		Short: "Test MongoDB connection and show game counts",
		Run: func(cmd *cobra.Command, args []string) {
			client, err := mongodb.Connect(mongoURI)
			if err != nil {
				log.Fatalf("Connect failed: %v", err)
			}
			defer client.Disconnect()

			if err := client.Ping(); err != nil {
				log.Fatalf("Ping failed: %v", err)
			}
			log.Info("MongoDB connection OK")

			for _, mode := range []string{"", "partner", "cutthroat"} {
				count, err := client.GameCount(mode)
				if err != nil {
					log.Errorf("GameCount(%s) failed: %v", mode, err)
					continue
				}
				label := "all"
				if mode != "" {
					label = mode
				}
				log.Infof("Games (%s): %d", label, count)
			}
		},
	}

	var humanModelName string
	var humanGameMode string
	trainHumanCmd := &cobra.Command{
		Use:   "train-human",
		Short: "Train on human game data from MongoDB",
		Run:   func(cmd *cobra.Command, args []string) { trainHuman(cfg, humanModelName, mongoURI, humanGameMode) },
	}
	trainHumanCmd.Flags().StringVar(&humanModelName, "model", "human_trained.mdl", "output model filename")
	trainHumanCmd.Flags().StringVar(&humanGameMode, "game-mode", "", "filter by game mode (partner, cutthroat, or empty for all)")

	rootCmd.AddCommand(
		&cobra.Command{
			Use:   "train-reinforced",
			Short: "Train standard MLP models (cutthroat)",
			Run:   func(cmd *cobra.Command, args []string) { trainReinforced(cfg) },
		},
		&cobra.Command{
			Use:   "train-attention",
			Short: "Train attention+MLP models (cutthroat)",
			Run:   func(cmd *cobra.Command, args []string) { trainReinforcedAttention(cfg) },
		},
		&cobra.Command{
			Use:   "train-partner",
			Short: "Train attention vs MLP (partner mode)",
			Run:   func(cmd *cobra.Command, args []string) { trainReinforcedPartner(cfg) },
		},
		&cobra.Command{
			Use:   "train-transformer",
			Short: "Train sequence transformer (partner mode)",
			Run:   func(cmd *cobra.Command, args []string) { trainReinforcedTransformer(cfg) },
		},
		&cobra.Command{
			Use:   "train-h2h",
			Short: "Head-to-head: MLP vs Attention vs Transformer",
			Run:   func(cmd *cobra.Command, args []string) { trainHeadToHead(cfg) },
		},
		&cobra.Command{
			Use:   "train-keras",
			Short: "Train via Keras server",
			Run:   func(cmd *cobra.Command, args []string) { trainKerasReinforced(cfg) },
		},
		trainHumanCmd,
		checkMongoCmd,
		duppyCmd,
	)

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
