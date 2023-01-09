package main

import (
	"encoding/json"
	"math/rand"
	"os"
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
	epochs := 10

	jsdai := nn.New(126, []int{20, 30, 25}, 56)
	r := make([]float64, 96)
	nn.FillRandom(r, float64(len(r)))

	gameCaches := LoadData("./data/deigodon201.json")
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
					cost, err := jsdai.Train(rotatedGameEvent, learnRate)
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

func trainReinforced() {
	// Train Reinforcement
	start := time.Now()
	hiddenLayerNodes1 := 150
	learnRate := 0.001
	sameGameIterations := 1
	totalRoundWins := [4]int{}
	jsdai1 := nn.New(126, []int{150, 150}, 56)
	jsdai2 := nn.New(126, []int{hiddenLayerNodes1}, 56)
	jsdai3 := nn.New(126, []int{20, 25}, 56)
	jsdai4 := nn.New(126, []int{56, 56}, 56)
	jc := nn.New(126, []int{1000}, 56)
	jc.Load("./results/" + "johncanoe.mdl")
	jsdai1.Load("./results/" + "jasai1.mdl")
	jsdai2.Load("./results/" + "jasai2.mdl")
	jsdai3.Load("./results/" + "jasai3.mdl")
	jsdai4.Load("./results/" + "jasai4.mdl")

	jsdai3.Search = false
	jsdai3.SearchNum = 8000
	reinForceMentLearn := func(r int64) {
		iter := func(dp [4]dominos.Player, nnPlayers [4]*nn.JSDNN, totWins [4]*int, rand int64) {
			dominosGame := dominos.NewLocalGame(dp, int64(rand), "cutthroat")
			roundGameEvents := []*dominos.GameEvent{}
			lastGameEvent := &dominos.GameEvent{}
			gameCost := []float64{}
			for lastGameEvent != nil && lastGameEvent.EventType != dominos.GameWin {
				lastGameEvent = dominosGame.AdvanceGameIteration()
				// log.Infof("LastGame Event: %+#v", lastGameEvent)
				lGE := jsdonline.CopyandRotateGameEvent(lastGameEvent, 0)

				if lGE.EventType == dominos.PosedCard || lGE.EventType == dominos.PlayedCard ||
					lGE.EventType == dominos.Passed || lastGameEvent.EventType == dominos.RoundWin ||
					lastGameEvent.EventType == dominos.RoundDraw {
					roundGameEvents = append(roundGameEvents, lGE)
				}

				if lastGameEvent.EventType == dominos.RoundWin {
					for i, gameEvent := range roundGameEvents {
						if gameEvent.EventType == dominos.PosedCard || gameEvent.EventType == dominos.PlayedCard {

							jsdai := nnPlayers[gameEvent.Player]
							rotatedGameEvent := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
							// Get the next 4 relevant events
							nextFourRelevant := [4]*dominos.GameEvent{}
							for j := 1; j < 5; j++ {
								if (i + j) > len(roundGameEvents) {
									break
								}
								nextFourRelevant[j-1] = jsdonline.CopyandRotateGameEvent(roundGameEvents[i], gameEvent.Player)
							}
							cost, err := jsdai.TrainReinforced(rotatedGameEvent, learnRate, nextFourRelevant, lastGameEvent)
							if err != nil {
								log.Fatal("Error training: ", err)
							}
							gameCost = append(gameCost, cost)
						} else if gameEvent.EventType == dominos.Passed {
							jsdai := nnPlayers[gameEvent.Player]
							jsdai.UpdatePassMemory(gameEvent)
						} else if gameEvent.EventType == dominos.RoundWin || gameEvent.EventType == dominos.RoundDraw {
							jsdai := nnPlayers[gameEvent.Player]
							jsdai.ResetPassMemory()
						}
					}
					roundGameEvents = []*dominos.GameEvent{}
				} else if lastGameEvent.EventType == dominos.RoundDraw {
					roundGameEvents = []*dominos.GameEvent{}
					jsdai1.ResetPassMemory()
					jsdai2.ResetPassMemory()
					jsdai3.ResetPassMemory()
					jsdai4.ResetPassMemory()
				}

			}
			*totWins[0] += lastGameEvent.PlayerWins[0]
			*totWins[1] += lastGameEvent.PlayerWins[1]
			*totWins[2] += lastGameEvent.PlayerWins[2]
			*totWins[3] += lastGameEvent.PlayerWins[3]

			jsdai1.Save("./results/" + "jasai1.mdl")
			jsdai2.Save("./results/" + "jasai2.mdl")
			jsdai3.Save("./results/" + "jasai3.mdl")
			jsdai4.Save("./results/" + "jasai4.mdl")
		}

		for i := 0; i < sameGameIterations; i++ {
			// TODO: bar
			nnPlayers := [4]*nn.JSDNN{jsdai1, jsdai2, jsdai3, jsdai4}
			gp := [4]dominos.Player{nnPlayers[0], nnPlayers[1], nnPlayers[2], nnPlayers[3]}
			tots := [4]*int{&nnPlayers[0].TotalWins, &nnPlayers[1].TotalWins, &nnPlayers[2].TotalWins, &nnPlayers[3].TotalWins}
			iter(gp, nnPlayers, tots, r)
		}

		// for i := 0; i < sameGameIterations; i++ {
		// 	// TODO: bar
		// 	log.Info("ITER: ", i)
		// 	nnPlayers := [4]*nn.JSDNN{jsdai1, jsdai2, jsdai3, jsdai4}
		// 	rand.Shuffle(len(nnPlayers), func(i, j int) { nnPlayers[i], nnPlayers[j] = nnPlayers[j], nnPlayers[i] })
		// 	gp := [4]dominos.Player{nnPlayers[0], nnPlayers[1], nnPlayers[2], nnPlayers[3]}
		// 	tots := [4]*int{&nnPlayers[0].TotalWins2, &nnPlayers[1].TotalWins2, &nnPlayers[2].TotalWins2, &nnPlayers[3].TotalWins2}
		// 	randNum := rand.Int63n(9223372036854775607)
		// 	iter(gp, nnPlayers, tots, randNum)
		// }

		// for i := 0; i < sameGameIterations; i++ {
		// 	// TODO: bar
		// 	nnPlayers := [4]*nn.JSDNN{jsdai1, jsdai2, jsdai3, jsdai4}
		// 	gp := [4]dominos.Player{nnPlayers[0], &dominos.ComputerPlayer{}, nnPlayers[3], jc}
		// 	tots := [4]*int{&totalRoundWins[0], &totalRoundWins[1], &totalRoundWins[2], &totalRoundWins[3]}
		// 	iter(gp, nnPlayers, tots)
		// }
	}

	randomSeed := time.Now().UnixNano()
	log.Info("Using random Seed: ", randomSeed)
	rand.Seed(randomSeed)
	for j := 0; j < 10000000; j++ {
		randNum := rand.Int63n(9223372036854775607)
		reinForceMentLearn(randNum)
		log.Info("Games Seen: ", j+1)
		log.Debug("Player Wins: ", totalRoundWins, " Total Rounds: ", totalRoundWins[0]+totalRoundWins[1]+totalRoundWins[2]+totalRoundWins[3])
		// r1 := float64(totalRoundWins[0]) / float64(totalRoundWins[0]+totalRoundWins[1]+totalRoundWins[2]+totalRoundWins[3])
		// r2 := float64(totalRoundWins[1]) / float64(totalRoundWins[0]+totalRoundWins[1]+totalRoundWins[2]+totalRoundWins[3])
		// r3 := float64(totalRoundWins[2]) / float64(totalRoundWins[0]+totalRoundWins[1]+totalRoundWins[2]+totalRoundWins[3])
		// r4 := float64(totalRoundWins[3]) / float64(totalRoundWins[0]+totalRoundWins[1]+totalRoundWins[2]+totalRoundWins[3])
		// log.Info("Player Ratios: ", []float64{r1, r2, r3, r4})
		log.Info("NN Wins: ", []int{jsdai1.TotalWins, jsdai2.TotalWins, jsdai3.TotalWins, jsdai4.TotalWins})
		log.Info("NN Wins2: ", []int{jsdai1.TotalWins2, jsdai2.TotalWins2, jsdai3.TotalWins2, jsdai4.TotalWins2})
		n1 := float64(jsdai1.TotalWins) / float64(jsdai1.TotalWins+jsdai2.TotalWins+jsdai3.TotalWins+jsdai4.TotalWins)
		n2 := float64(jsdai2.TotalWins) / float64(jsdai1.TotalWins+jsdai2.TotalWins+jsdai3.TotalWins+jsdai4.TotalWins)
		n3 := float64(jsdai3.TotalWins) / float64(jsdai1.TotalWins+jsdai2.TotalWins+jsdai3.TotalWins+jsdai4.TotalWins)
		n4 := float64(jsdai4.TotalWins) / float64(jsdai1.TotalWins+jsdai2.TotalWins+jsdai3.TotalWins+jsdai4.TotalWins)
		log.Info("NN Ratios: ", []float64{n1, n2, n3, n4})
		n21 := float64(jsdai1.TotalWins2) / float64(jsdai1.TotalWins2+jsdai2.TotalWins2+jsdai3.TotalWins2+jsdai4.TotalWins2)
		n22 := float64(jsdai2.TotalWins2) / float64(jsdai1.TotalWins2+jsdai2.TotalWins2+jsdai3.TotalWins2+jsdai4.TotalWins2)
		n23 := float64(jsdai3.TotalWins2) / float64(jsdai1.TotalWins2+jsdai2.TotalWins2+jsdai3.TotalWins2+jsdai4.TotalWins2)
		n24 := float64(jsdai4.TotalWins2) / float64(jsdai1.TotalWins2+jsdai2.TotalWins2+jsdai3.TotalWins2+jsdai4.TotalWins2)
		log.Info("NN Ratios2: ", []float64{n21, n22, n23, n24})
	}

	jsdai1.Save("./results/" + "jasai1.mdl")
	jsdai2.Save("./results/" + "jasai2.mdl")
	jsdai3.Save("./results/" + "jasai3.mdl")
	jsdai4.Save("./results/" + "jasai4.mdl")
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
					position := 0
					users := []string{table.Player1.Username, table.Player2.Username, table.Player3.Username, table.Player4.Username}
					for i, user := range users {
						if user == "" {
							position = i
							break
						}
					}
					gamePayload := &duppy.OnlineGamePayload{
						GameID:   table.GameID,
						Position: position,
					}

					err := duppy.JoinOnlineGame(gamePayload, token)
					if err != nil {
						log.Error(err)
						continue
					}
					joinedGame = true
					break
				}
			}
		}

		time.Sleep(2 * time.Second)
	}
}

func main() {
	trainReinforced()
	// trainHuman()
	// 69 / 333, 80/367
	// duppyPlay()
}
