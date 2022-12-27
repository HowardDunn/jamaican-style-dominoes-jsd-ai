package main

import (
	"encoding/json"
	"math/rand"
	"os"
	"time"

	"github.com/HowardDunn/go-dominos/dominos"
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

func LoadData() []*gameCache {
	dat, err := os.ReadFile("./data/december_25_games_partner.json")
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

func userIn(users []string, user string) bool {
	for _, usr := range users {
		if usr == user {
			return true
		}
	}
	return false
}

func main() {
	path := "./results/" + time.Now().Format(time.RFC3339)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		err := os.Mkdir(path, 0o644)
		if err != nil {
			log.Fatal(err)
		}
	}

	// variables
	learnRate := 0.001
	filteredUser := "ozark.games"
	filterUsers := false
	hiddenLayerNodes := 1000
	epochs := 3

	jsdai := nn.New(126, hiddenLayerNodes, 56)
	r := make([]float64, 96)
	nn.FillRandom(r, float64(len(r)))

	gameCaches := LoadData()
	log.Info("Successfully Loaded data")

	randomSeed := time.Now().UnixNano()
	log.Info("Using random Seed: ", randomSeed)
	rand.Seed(randomSeed)
	rand.Shuffle(len(gameCaches), func(i, j int) { gameCaches[i], gameCaches[j] = gameCaches[j], gameCaches[i] })

	trainingGamesIndex := int(float64(len(gameCaches)) * 0.8)
	trainingGames := gameCaches[:trainingGamesIndex]
	validationGames := gameCaches[trainingGamesIndex:]

	// predictionEvaluation
	evaluate := func() float64 {
		accuracies := 0.0
		inaccuracies := 0.0
		compatibles := 0.0
		incompatibles := 0.0
		bar := progressbar.Default(int64(len(validationGames)))
		for _, gameCache := range validationGames {
			if !userIn(gameCache.GameEvents[0].PlayerNames[:], filteredUser) && filterUsers {
				continue
			}
			bar.Add(1)
			for _, gameEvent := range gameCache.GameEvents {
				if gameEvent.EventType == dominos.PlayedCard {
					rotatedGameEvent := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
					if filteredUser != rotatedGameEvent.PlayerNames[0] && filterUsers {
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
			if !userIn(gameCache.GameEvents[0].PlayerNames[:], filteredUser) && filterUsers {
				continue
			}
			bar.Add(1)
			gameCost := []float64{}
			for _, gameEvent := range gameCache.GameEvents {
				if gameEvent.EventType == dominos.PosedCard || gameEvent.EventType == dominos.PlayedCard {
					rotatedGameEvent := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
					if filteredUser != rotatedGameEvent.PlayerNames[0] && filterUsers {
						continue
					}
					cost, err := jsdai.Train(rotatedGameEvent, learnRate)
					if err != nil {
						log.Fatal("Error training: ", err)
					}
					gameCost = append(gameCost, cost)
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
