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
	UUID       string               `bson:"uuid"`
	GameType   string               `bson:"gameType"`
	GameEvents []*dominos.GameEvent `bson:"gameEvents"`
	TimeCreate time.Time            `bson:"timeCreated"`
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
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	filter := bson.M{}
	if gameType != "" {
		filter["gameType"] = gameType
	}

	cursor, err := col.Find(ctx, filter)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var docs []*GameDocument
	if err := cursor.All(ctx, &docs); err != nil {
		return nil, err
	}

	log.Infof("Fetched %d games from MongoDB (filter: %s)", len(docs), gameType)
	return docs, nil
}
