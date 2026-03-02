package mongodb

import (
	"context"
	"time"

	"github.com/HowardDunn/go-dominos/dominos"
	log "github.com/sirupsen/logrus"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"
)

const (
	defaultURI    = "mongodb+srv://doadmin:Bt3Gq295a0y16n8p@jsd-mongodb-1f08e1ea.mongo.ondigitalocean.com/admin?tls=true&authSource=admin&replicaSet=jsd-mongodb"
	dbName        = "jamaican-style-dominoes"
	gamesCollName = "onlinegames"
)

type Client struct {
	mongo *mongo.Client
}

type GameDocument struct {
	UUID        string               `bson:"uuid"`
	GameType    string               `bson:"gameType"`
	GameEvents  []*dominos.GameEvent `bson:"gameEvents"`
	GameQuality float64              `bson:"gameQuality"`
	TimeCreate  time.Time            `bson:"timeCreated"`
}

func Connect(uri string) (*Client, error) {
	if uri == "" {
		uri = defaultURI
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		return nil, err
	}
	return &Client{mongo: client}, nil
}

func (c *Client) Disconnect() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return c.mongo.Disconnect(ctx)
}

func (c *Client) Ping() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return c.mongo.Ping(ctx, readpref.Primary())
}

func (c *Client) GameCount(gameType string) (int64, error) {
	col := c.mongo.Database(dbName).Collection(gamesCollName)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	filter := bson.M{}
	if gameType != "" {
		filter["gameType"] = gameType
	}
	return col.CountDocuments(ctx, filter)
}

func (c *Client) FetchGames(gameType string) ([]*GameDocument, error) {
	col := c.mongo.Database(dbName).Collection(gamesCollName)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	filter := bson.M{}
	if gameType != "" {
		filter["gameType"] = gameType
	}

	log.Info("Downloading games from MongoDB (this may take a few minutes)...")
	cursor, err := col.Find(ctx, filter)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var docs []*GameDocument
	for cursor.Next(ctx) {
		var doc GameDocument
		if err := cursor.Decode(&doc); err != nil {
			return nil, err
		}
		docs = append(docs, &doc)
		if len(docs)%100 == 0 {
			log.Infof("  ...fetched %d games so far", len(docs))
		}
	}
	if err := cursor.Err(); err != nil {
		return nil, err
	}

	log.Infof("Fetched %d games from MongoDB (filter: %s)", len(docs), gameType)
	return docs, nil
}

// FetchGamesChan starts downloading games in the background and streams them
// through a channel. The caller can begin processing while the download continues.
// The error channel receives at most one error (or is closed on success).
// skip allows resuming a partial download by skipping the first N documents.
func (c *Client) FetchGamesChan(gameType string, skip int64) (<-chan *GameDocument, <-chan error) {
	ch := make(chan *GameDocument, 64)
	errCh := make(chan error, 1)

	go func() {
		defer close(ch)
		defer close(errCh)

		col := c.mongo.Database(dbName).Collection(gamesCollName)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()

		filter := bson.M{}
		if gameType != "" {
			filter["gameType"] = gameType
		}

		opts := options.Find()
		if skip > 0 {
			opts.SetSkip(skip)
			log.Infof("Resuming download from game %d...", skip)
		} else {
			log.Info("Downloading games from MongoDB in background...")
		}

		cursor, err := col.Find(ctx, filter, opts)
		if err != nil {
			errCh <- err
			return
		}
		defer cursor.Close(ctx)

		count := int64(0)
		for cursor.Next(ctx) {
			var doc GameDocument
			if err := cursor.Decode(&doc); err != nil {
				errCh <- err
				return
			}
			ch <- &doc
			count++
			if count%100 == 0 {
				log.Infof("  ...fetched %d games so far (total: %d)", count, skip+count)
			}
		}
		if err := cursor.Err(); err != nil {
			errCh <- err
			return
		}
		log.Infof("Download complete: %d new games fetched (total: %d)", count, skip+count)
	}()

	return ch, errCh
}
