package mongodb

import (
	"testing"

	"github.com/HowardDunn/go-dominos/dominos"
)

// connectOrSkip creates a client and pings MongoDB.
// Skips the test if the database is unreachable (e.g. IP not allowlisted).
func connectOrSkip(t *testing.T) *Client {
	t.Helper()
	client, err := Connect("")
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}
	if err := client.Ping(); err != nil {
		client.Disconnect()
		t.Skipf("MongoDB unreachable (skip integration test): %v", err)
	}
	return client
}

func TestConnect(t *testing.T) {
	client, err := Connect("")
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}
	defer client.Disconnect()
}

func TestPing(t *testing.T) {
	client := connectOrSkip(t)
	defer client.Disconnect()
	// If we got here, ping already succeeded in connectOrSkip
	t.Log("MongoDB ping OK")
}

func TestGameCount(t *testing.T) {
	client := connectOrSkip(t)
	defer client.Disconnect()

	count, err := client.GameCount("")
	if err != nil {
		t.Fatalf("GameCount failed: %v", err)
	}
	t.Logf("Total games in MongoDB: %d", count)

	if count == 0 {
		t.Log("Warning: no games found — database may be empty")
	}
}

func TestFetchGames(t *testing.T) {
	client := connectOrSkip(t)
	defer client.Disconnect()

	docs, err := client.FetchGames("")
	if err != nil {
		t.Fatalf("FetchGames failed: %v", err)
	}
	t.Logf("Fetched %d game documents", len(docs))

	if len(docs) == 0 {
		t.Log("Warning: no games fetched — database may be empty")
		return
	}

	doc := docs[0]
	if doc.UUID == "" {
		t.Error("First document has empty UUID")
	}
	if len(doc.GameEvents) == 0 {
		t.Error("First document has no game events")
	}

	// Verify game events have expected structure
	hasPlayedCard := false
	for _, ev := range doc.GameEvents {
		if ev.EventType == dominos.PlayedCard || ev.EventType == dominos.PosedCard {
			hasPlayedCard = true
			break
		}
	}
	if !hasPlayedCard {
		t.Log("Warning: first game has no PlayedCard/PosedCard events")
	}
}

func TestFetchGamesFiltered(t *testing.T) {
	client := connectOrSkip(t)
	defer client.Disconnect()

	for _, mode := range []string{"partner", "cutthroat"} {
		count, err := client.GameCount(mode)
		if err != nil {
			t.Fatalf("GameCount(%s) failed: %v", mode, err)
		}
		t.Logf("Games with gameType=%s: %d", mode, count)
	}
}
