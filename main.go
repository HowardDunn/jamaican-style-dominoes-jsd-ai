package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/HowardDunn/go-dominos/dominos"
	"github.com/HowardDunn/jsd-ai/duppy"
	"github.com/HowardDunn/jsd-ai/nn"
	jsdonline "github.com/HowardDunn/jsd-online-game/game"
	"github.com/schollz/progressbar/v3"
	log "github.com/sirupsen/logrus"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

type gameCache struct {
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

func trainHuman() {
	path := "./results/" + time.Now().Format(time.RFC3339)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		err := os.Mkdir(path, 0o644)
		if err != nil {
			log.Fatal(err)
		}
	}

	// variables
	learnRate := 0.001
	filteredUsers := map[string]any{"deigodon201": nil}
	filterUsers := true
	epochs := 5

	jsdai := nn.New(126, []int{204, 102}, 56)
	r := make([]float64, 96)
	nn.FillRandom(r, float64(len(r)))

	gameCaches := LoadData("./data/games_partner.json")
	// gameCaches := LoadData("./data/ozark.games.json")
	log.Info("Successfully Loaded data")

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

			for _, gameEvent := range gameCache.GameEvents {
				if gameEvent.EventType == dominos.PlayedCard {
					rotatedGameEvent := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
					_, ok := filteredUsers[rotatedGameEvent.PlayerNames[0]]
					if !ok && filterUsers {
						continue
					}
					cardChoice, err := jsdai.Predict(rotatedGameEvent)
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
			}
		}

		acc := float64(accuracies) / float64(accuracies+inaccuracies)
		log.Infof("Accuracy of validation: %.5f", float64(accuracies)/float64(accuracies+inaccuracies))
		log.Infof("Compatible of validation data: %.5f", float64(compatibles)/float64(incompatibles+compatibles))
		return acc
	}

	averageCosts := []float64{}
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
			for _, gameEvent := range gameCache.GameEvents {
				if gameEvent.EventType == dominos.PosedCard || gameEvent.EventType == dominos.PlayedCard {
					rotatedGameEvent := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
					_, ok := filteredUsers[rotatedGameEvent.PlayerNames[0]]
					if !ok && filterUsers {
						continue
					}
					cost, err := jsdai.Train(rotatedGameEvent, []float64{learnRate})
					if err != nil {
						log.Fatal("Error training: ", err)
					}
					gameCost = append(gameCost, cost)
				} else if gameEvent.EventType == dominos.Passed {
					jsdai.UpdatePassMemory(gameEvent)
				} else if gameEvent.EventType == dominos.RoundWin || gameEvent.EventType == dominos.RoundDraw {
					jsdai.ResetPassMemory()
				}
			}
			averageCosts = append(averageCosts, calculateAverageSquaredCost(gameCost))
		}
	}

	accuracies := []float64{}
	accuracies = append(accuracies, evaluate())
	for i := 0; i < epochs; i++ {
		log.Infof("Epoch: %d out of %d", i+1, epochs)
		train()
		accuracies = append(accuracies, evaluate())
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
	err := plotutil.AddLinePoints(p, "JSD", pts)
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

	jsdai.Save(path + "model.mdl")
}

type modelMeta struct {
	Iterations int     `json:"iterations"`
	TotalWins  int     `json:"total_wins"`
	TotalWins2 int     `json:"total_wins_shuffled"`
	Elo        float64 `json:"elo"`
}

const eloK = 32.0
const eloMaxRoundWins = 6.0
const eloRandomPlayer = 1000.0

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

		if lastGameEvent.EventType == dominos.RoundWin || lastGameEvent.EventType == dominos.RoundDraw {
			// Reset pass memory on the cloned NNs used for gameplay
			for k := 0; k < 4; k++ {
				nnPlayers[k].ResetPassMemory()
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

func trainReinforced() {
	// Train Reinforcement
	start := time.Now()

	sameGameIterations := 8
	maxGames := 100000
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
	jsdai1.Load("./results/" + "jasai1.mdl")
	jsdai2.Load("./results/" + "jasai2.mdl")
	jsdai3.Load("./results/" + "jasai3.mdl")
	jsdai4.Load("./results/" + "jasai4.mdl")

	// Load metadata to resume iteration counts and win totals
	modelNames := [4]string{"jasai1", "jasai2", "jasai3", "jasai4"}
	metas := [4]modelMeta{}
	for i, name := range modelNames {
		metas[i] = loadModelMeta("./results/" + name + ".meta.json")
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
	elos := [4]float64{}
	for i := range elos {
		if metas[i].Elo > 0 {
			elos[i] = metas[i].Elo
		} else {
			elos[i] = eloRandomPlayer // start at random player baseline
		}
	}
	log.Info("Resuming from iteration: ", iterationCount)
	log.Infof("Starting Elos: [%.1f, %.1f, %.1f, %.1f]", elos[0], elos[1], elos[2], elos[3])

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
			// Update Elo from NN-vs-NN game (fixed order: position k = masters[k])
			elos = updateEloMultiplayer(elos, r.playerWins)
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
			// Update Elo: remap position wins to master Elo indices
			shuffledElos := [4]float64{elos[sg.order[0]], elos[sg.order[1]], elos[sg.order[2]], elos[sg.order[3]]}
			newShuffledElos := updateEloMultiplayer(shuffledElos, result.playerWins)
			for k := 0; k < 4; k++ {
				elos[sg.order[k]] = newShuffledElos[k]
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
				// Update Elo once per benchmark game (matches backend logic)
				elos[m] = updateEloBench(elos[m], benchOut[m*2+g].nnWins)
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
			masters[i].Save("./results/" + name + ".mdl")
			saveModelMeta("./results/"+name+".meta.json", modelMeta{
				Iterations: iterationCount,
				TotalWins:  masters[i].TotalWins,
				TotalWins2: masters[i].TotalWins2,
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
	eloCSVPath := "./results/elo_history.csv"
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
		eloWriter.Write([]string{"iteration", modelNames[0], modelNames[1], modelNames[2], modelNames[3]})
		eloWriter.Flush()
	}

	randomSeed := time.Now().UnixNano()
	log.Info("Using random Seed: ", randomSeed)
	rand.Seed(randomSeed)
	for j := 0; j < maxGames; j++ {
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
			log.Infof("Bench %s: total=%.4f (%d/%d)  rolling1k=%.4f (%d/%d)  elo=%.1f",
				modelNames[m], rate, totalRoundWins[m], totalBenchGames[m],
				rollingRate, rollingWins, rollingGames, elos[m])
		}
		// Write Elo to CSV
		eloWriter.Write([]string{
			fmt.Sprintf("%d", iterationCount),
			fmt.Sprintf("%.1f", elos[0]),
			fmt.Sprintf("%.1f", elos[1]),
			fmt.Sprintf("%.1f", elos[2]),
			fmt.Sprintf("%.1f", elos[3]),
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
		masters[i].Save("./results/" + name + ".mdl")
		saveModelMeta("./results/"+name+".meta.json", modelMeta{
			Iterations: iterationCount,
			TotalWins:  masters[i].TotalWins,
			TotalWins2: masters[i].TotalWins2,
			Elo:        elos[i],
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

func trainKerasReinforced() {
	start := time.Now()
	serverURL := "http://localhost:8777"

	sameGameIterations := 8
	maxGames := 100
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
	for j := 0; j < maxGames; j++ {
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

func duppyPlay() {
	token, err := duppy.UserLogin("duppy", "@ecIN)A*$Y93UhZ*")
	if err != nil {
		panic(err)
	}
	joinedGame := false
	gamesPlayed := 0
	for {
		onlineTables, err := duppy.GetOnlineTableList(token)
		if err != nil {
			log.Error(err)
			continue
		}

		for _, table := range onlineTables {
			if table.PlayerInvolved && table.GameState == "running" {
				log.Info("Playing game: ", table.GameID)
				duppy.PlayGame(table, token)
				gamesPlayed++
				log.Info("Finished playing game: ", table.GameID, " Num Played: ", gamesPlayed)
				joinedGame = false
				onlineTables = []*duppy.OnlineGameTable{}
				break
			}
		}

		for _, table := range onlineTables {
			if !joinedGame {
				if table.GameType == "cutthroat" && table.GameState == "waiting" {
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

func main() {
	trainReinforced()
	// trainKerasReinforced()
	// trainHuman()
	// 69 / 333, 80/367, 194,997. 205,1085
	// duppyPlay()
}
