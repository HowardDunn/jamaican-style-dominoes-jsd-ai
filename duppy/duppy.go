package duppy

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io/ioutil"
	"net/http"
	"strings"
	"time"

	"github.com/HowardDunn/go-dominos/dominos"
	"github.com/HowardDunn/jsd-ai/nn"
	jsdonline "github.com/HowardDunn/jsd-online-game/game"
	log "github.com/sirupsen/logrus"
)

const (
	jsdUrl   = "https://api.jamaicanstyledominoes.com/"
	wsJSDUrl = "wss://api.jamaicanstyledominoes.com/"
)

var userData map[string]interface{}

type OnlineGamePlayer struct {
	Username     string `json:"username"`
	SpotOccupied bool   `json:"spotOccupied"`
	Wins         int    `json:"wins"`
	ProfPicURL   string `json:"profilePicURL"`
}

type OnlineGameTable struct {
	GameID         string            `json:"id"`
	GameType       string            `json:"gameType"`
	GameState      string            `json:"gameState"`
	PlayerInvolved bool              `json:"playerInvolved"`
	Player1        *OnlineGamePlayer `json:"player1"`
	Player2        *OnlineGamePlayer `json:"player2"`
	Player3        *OnlineGamePlayer `json:"player3"`
	Player4        *OnlineGamePlayer `json:"player4"`
	AppGame        bool              `json:"appGame"`
}

func getTableList(endpoint string, token string) ([]*OnlineGameTable, error) {
	values := map[string]interface{}{}
	jsdonData, err := json.Marshal(values)
	if err != nil {
		log.Error(err)
		return nil, err
	}

	netClient := &http.Client{
		Timeout: time.Second * 15,
	}

	req, err := http.NewRequest("POST", jsdUrl+endpoint, bytes.NewBuffer(jsdonData))
	req.Header = http.Header{
		"jsd-auth":     {token},
		"Content-Type": {"application/json"},
	}
	if err != nil {
		log.Error(err)
		return nil, err
	}
	resp, err := netClient.Do(req)
	if err != nil {
		log.Error(err)
		return nil, err
	}
	if resp.StatusCode >= 400 {
		log.Error("Not 200 status code: ", resp)
		return nil, err
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Error(err)
		return nil, err
	}

	onlineGames := []*OnlineGameTable{}
	err = json.Unmarshal(body, &onlineGames)
	if err != nil {
		log.Error(err)
		return nil, err
	}

	return onlineGames, nil
}

func UserLogin(userIdentifier string, password string) (string, error) {
	values := map[string]string{"identifier": userIdentifier, "password": password}
	jsdonData, err := json.Marshal(values)
	if err != nil {
		log.Error(err)
		return "", err
	}

	netClient := &http.Client{
		Timeout: time.Second * 15,
	}

	resp, err := netClient.Post(jsdUrl+"user/login/", "application/json",
		bytes.NewBuffer(jsdonData))
	if err != nil {
		log.Error(err)
		return "", err
	}

	if resp.StatusCode > 400 {
		log.Error("Error logging in status code: ", resp.StatusCode)
		return "", errors.New("Error code > 400")
	}

	var res map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&res)

	token, tokenOk := res["token"]
	if !tokenOk {
		return "", errors.New("Did not receive auth token")
	}

	tokenString, tokenOk := token.(string)
	if !tokenOk {
		return "", errors.New("can't convert token to string")
	}
	authToken := tokenString
	jwtPart := strings.Split(authToken, ".")
	rawDecodedText, err := base64.RawStdEncoding.DecodeString(jwtPart[1])
	if err != nil {
		return "", errors.New("Cand decode jwt")
	}
	err = json.Unmarshal(rawDecodedText, &userData)
	if err != nil {
		return "", errors.New("Cand decode jwt: " + err.Error())
	}
	return authToken, nil
}

func GetOnlineTableList(token string) ([]*OnlineGameTable, error) {
	return getTableList("online/list/", token)
}

func PlayGame(game *OnlineGameTable, token string) {
	jsdAI := nn.New(126, []int{56, 56}, 56)
	jsdAI.Search = true
	jsdAI.SearchNum = 5000
	err := jsdAI.Load("./results/duppy.mdl")
	if err != nil {
		log.Error("Error: ", err)
		panic(err)
	}
	players := [4]dominos.Player{jsdAI, &dominos.ComputerPlayer{}, &dominos.ComputerPlayer{}, &dominos.ComputerPlayer{}}
	gameURL := wsJSDUrl + "online/play/" + game.GameID
	log.Info(gameURL)
	g := dominos.NewWebClientGame(gameURL, token, players)
	roundGameEvents := []*dominos.GameEvent{}
	lastGameEvent := &dominos.GameEvent{}
	for lastGameEvent.GameState != dominos.Idle {
		lastGameEvent = g.AdvanceGameIteration()
		if lastGameEvent.EventType != dominos.NullEvent {
			lGE := jsdonline.CopyandRotateGameEvent(lastGameEvent, 0)
			roundGameEvents = append(roundGameEvents, lGE)
		}
		if lastGameEvent.EventType == dominos.PlayerTurn && lastGameEvent.Player == 0 {
			if lastGameEvent.BoardState.CardPosed {
				log.Info("AI Player Turn")
				card, side := jsdAI.PlayCard(lastGameEvent, dominos.GetCLIDominos())
				cardChoice := &dominos.CardChoice{
					Card: card,
					Side: side,
				}
				g.PlayHumanCard(cardChoice)
				log.Infof("Played Card: %+#v", cardChoice)
			} else {
				card := jsdAI.PoseCard(dominos.GetCLIDominos())
				cardChoice := &dominos.CardChoice{
					Card: card,
					Side: dominos.Posed,
				}
				g.PlayHumanCard(cardChoice)
				log.Infof("Posed Card: %+#v", cardChoice)
			}
		} else if lastGameEvent.EventType == dominos.Passed {
			jsdAI.UpdatePassMemory(lastGameEvent)
		} else if lastGameEvent.EventType == dominos.RoundWin || lastGameEvent.EventType == dominos.RoundDraw {
			jsdAI.ResetPassMemory()
		} else if lastGameEvent.EventType == dominos.GameWin || g.ConnectionCount > 100 {
			g.Reset(0, "cutthroat")
			jsdAI.ResetPassMemory()
			// for _, gameEvent := range roundGameEvents {
			// 	if gameEvent.EventType == dominos.PosedCard || gameEvent.EventType == dominos.PlayedCard {
			// 		rotatedGameEvent := jsdonline.CopyandRotateGameEvent(gameEvent, gameEvent.Player)
			// 		_, err := jsdAI.TrainReinforced(rotatedGameEvent, 0.001, lastGameEvent)
			// 		if err != nil {
			// 			log.Error("Error training: ", err)
			// 			return
			// 		}
			// 	} else if gameEvent.EventType == dominos.Passed {
			// 		jsdAI.UpdatePassMemory(gameEvent)
			// 	} else if gameEvent.EventType == dominos.RoundWin || gameEvent.EventType == dominos.RoundDraw {
			// 		jsdAI.ResetPassMemory()
			// 	}
			// }
			// roundGameEvents = []*dominos.GameEvent{}
			// jsdAI.Save("duppy2.mdl")
			return
		}

		time.Sleep(300 * time.Millisecond)

	}
}

type OnlineGamePayload struct {
	GameID   string `json:"gameId"`
	Position int    `json:"position"`
	Type     string `json:"type,omitempty"`
}

func JoinOnlineGame(payload *OnlineGamePayload, token string) error {
	endpoint := "online/join"
	jsdonData, err := json.Marshal(payload)
	if err != nil {
		log.Error(err)
		return err
	}

	netClient := &http.Client{
		Timeout: time.Second * 15,
	}

	req, err := http.NewRequest("POST", jsdUrl+endpoint, bytes.NewBuffer(jsdonData))
	req.Header = http.Header{
		"jsd-auth":     {token},
		"Content-Type": {"application/json"},
	}
	if err != nil {
		log.Error(err)

		return err
	}
	resp, err := netClient.Do(req)
	if err != nil {
		log.Error(err)
		body, errr := ioutil.ReadAll(resp.Body)
		if errr != nil {
			log.Error(err)
		}
		log.Error(string(body))
		return err
	}
	if resp.StatusCode >= 400 {
		log.Error("Not 200 status code: ", resp)
		body, errr := ioutil.ReadAll(resp.Body)
		if errr != nil {
			log.Error(err)
		}
		log.Error(string(body))
		return err
	}

	return nil
}
