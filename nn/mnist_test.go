package nn

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/HowardDunn/go-dominos/dominos"
	"gorgonia.org/tensor"
)

const mnistBaseURL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

func downloadFile(url, dest string) error {
	if _, err := os.Stat(dest); err == nil {
		return nil
	}
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("download failed with status %d", resp.StatusCode)
	}
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

func loadMNISTImages(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	var magic, numImages, rows, cols int32
	binary.Read(gz, binary.BigEndian, &magic)
	binary.Read(gz, binary.BigEndian, &numImages)
	binary.Read(gz, binary.BigEndian, &rows)
	binary.Read(gz, binary.BigEndian, &cols)

	images := make([][]float64, numImages)
	for i := int32(0); i < numImages; i++ {
		img := make([]byte, rows*cols)
		_, err := io.ReadFull(gz, img)
		if err != nil {
			return nil, err
		}
		imgFloat := make([]float64, rows*cols)
		for j, b := range img {
			imgFloat[j] = float64(b) / 255.0
		}
		images[i] = imgFloat
	}
	return images, nil
}

func loadMNISTLabels(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	var magic, numLabels int32
	binary.Read(gz, binary.BigEndian, &magic)
	binary.Read(gz, binary.BigEndian, &numLabels)

	labels := make([]int, numLabels)
	rawLabels := make([]byte, numLabels)
	_, err = io.ReadFull(gz, rawLabels)
	if err != nil {
		return nil, err
	}
	for i, b := range rawLabels {
		labels[i] = int(b)
	}
	return labels, nil
}

// newGenericNN creates a JSDNN without the output==56 restriction,
// allowing us to test the core forward/backward pass against MNIST.
func newGenericNN(input int, hidden []int, output int) *JSDNN {
	if len(hidden) < 1 {
		panic("Invalid hidden size")
	}

	r := make([]float64, input*hidden[0])
	r2 := [][]float64{}
	r3 := [][]float64{}
	r4 := make([]float64, output)
	hiddenT := []*tensor.Dense{}
	bHidden := []*tensor.Dense{}
	rb := make([]float64, hidden[0])
	r3 = append(r3, rb)
	fillRandom(r, float64(input))
	fillRandom(r3[0], float64(len(r3[0])))
	hiddenT = append(hiddenT, tensor.New(tensor.WithShape(hidden[0], input), tensor.WithBacking(r)))
	bHidden = append(bHidden, tensor.New(tensor.WithShape(hidden[0]), tensor.WithBacking(r3[0])))

	for i := range hidden {
		next := output
		if (i + 1) < len(hidden) {
			next = hidden[i+1]
		}
		h := make([]float64, hidden[i]*next)
		fillRandom(h, float64(hidden[i]))
		r2 = append(r2, h)
		if (i + 1) < len(hidden) {
			hb := make([]float64, hidden[i+1])
			fillRandom(hb, float64(len(hb)))
			r3 = append(r3, hb)
			hiddenT = append(hiddenT, tensor.New(tensor.WithShape(next, hidden[i]), tensor.WithBacking(r2[i])))
			bHidden = append(bHidden, tensor.New(tensor.WithShape(hidden[i+1]), tensor.WithBacking(r3[i+1])))
		}
	}

	finalT := tensor.New(tensor.WithShape(output, hidden[len(hidden)-1]), tensor.WithBacking(r2[len(r2)-1]))
	bFinal := tensor.New(tensor.WithShape(output), tensor.WithBacking(r4))
	knownNotHaves := [4]map[uint]bool{}
	for i := 0; i < len(knownNotHaves); i++ {
		knownNotHaves[i] = make(map[uint]bool)
	}

	return &JSDNN{
		hidden:         hiddenT,
		final:          finalT,
		bHidden:        bHidden,
		bFinal:         bFinal,
		ComputerPlayer: &dominos.ComputerPlayer{},
		inputDim:       input,
		hiddenDim:      hidden,
		outputDim:      output,
		gameType:       "cutthroat",
		knownNotHaves:  knownNotHaves,
	}
}

func argmax(data []float64) int {
	best := 0
	for i := 1; i < len(data); i++ {
		if data[i] > data[best] {
			best = i
		}
	}
	return best
}

func evaluateAccuracy(t *testing.T, net *JSDNN, images [][]float64, labels []int) float64 {
	correct := 0
	for i := range images {
		imgCopy := make([]float64, len(images[i]))
		copy(imgCopy, images[i])
		x := tensor.New(tensor.WithShape(len(imgCopy)), tensor.WithBacking(imgCopy))
		pred, err := net.predict(x)
		if err != nil {
			t.Fatalf("prediction error at sample %d: %v", i, err)
		}
		predData := pred.Data().([]float64)
		if argmax(predData) == labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(images)) * 100.0
}

func downloadMNIST(t *testing.T) (trainImages, testImages [][]float64, trainLabels, testLabels []int) {
	tmpDir := filepath.Join(os.TempDir(), "mnist_test_data")
	os.MkdirAll(tmpDir, 0755)

	files := []struct {
		name string
		url  string
	}{
		{"train-images-idx3-ubyte.gz", mnistBaseURL + "train-images-idx3-ubyte.gz"},
		{"train-labels-idx1-ubyte.gz", mnistBaseURL + "train-labels-idx1-ubyte.gz"},
		{"t10k-images-idx3-ubyte.gz", mnistBaseURL + "t10k-images-idx3-ubyte.gz"},
		{"t10k-labels-idx1-ubyte.gz", mnistBaseURL + "t10k-labels-idx1-ubyte.gz"},
	}

	for _, f := range files {
		dest := filepath.Join(tmpDir, f.name)
		if err := downloadFile(f.url, dest); err != nil {
			t.Fatalf("failed to download %s: %v", f.name, err)
		}
	}

	var err error
	trainImages, err = loadMNISTImages(filepath.Join(tmpDir, "train-images-idx3-ubyte.gz"))
	if err != nil {
		t.Fatalf("failed to load training images: %v", err)
	}
	trainLabels, err = loadMNISTLabels(filepath.Join(tmpDir, "train-labels-idx1-ubyte.gz"))
	if err != nil {
		t.Fatalf("failed to load training labels: %v", err)
	}
	testImages, err = loadMNISTImages(filepath.Join(tmpDir, "t10k-images-idx3-ubyte.gz"))
	if err != nil {
		t.Fatalf("failed to load test images: %v", err)
	}
	testLabels, err = loadMNISTLabels(filepath.Join(tmpDir, "t10k-labels-idx1-ubyte.gz"))
	if err != nil {
		t.Fatalf("failed to load test labels: %v", err)
	}

	t.Logf("MNIST loaded: %d training, %d test images", len(trainImages), len(testImages))
	return
}

// TestMNISTSingleHiddenLayer tests the neural network's backpropagation and inference
// using MNIST digit classification with a single hidden layer (784 -> 128 -> 10).
func TestMNISTSingleHiddenLayer(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping MNIST test in short mode")
	}

	trainImages, testImages, trainLabels, testLabels := downloadMNIST(t)

	rand.Seed(42)
	net := newGenericNN(784, []int{256}, 10)

	learnRate := 0.001
	epochs := 10

	for epoch := 0; epoch < epochs; epoch++ {
		indices := rand.Perm(len(trainImages))
		totalCost := 0.0

		for _, idx := range indices {
			imgCopy := make([]float64, 784)
			copy(imgCopy, trainImages[idx])
			x := tensor.New(tensor.WithShape(784), tensor.WithBacking(imgCopy))

			yData := make([]float64, 10)
			yData[trainLabels[idx]] = 1.0
			y := tensor.New(tensor.WithShape(10), tensor.WithBacking(yData))

			cost, err := net.train(x, y, nil, []float64{learnRate}, nil)
			if err != nil {
				t.Fatalf("training error: %v", err)
			}
			totalCost += cost * cost
		}

		avgCost := totalCost / float64(len(trainImages))
		trainAcc := evaluateAccuracy(t, net, trainImages[:1000], trainLabels[:1000])
		t.Logf("Epoch %d: AvgCost=%.6f, TrainAcc(1k sample)=%.2f%%", epoch+1, avgCost, trainAcc)
	}

	testAcc := evaluateAccuracy(t, net, testImages, testLabels)
	t.Logf("Test Accuracy: %.2f%% (%d samples)", testAcc, len(testImages))

	if testAcc < 70.0 {
		t.Errorf("test accuracy %.2f%% is below 70%% - backpropagation may not be working correctly", testAcc)
	} else {
		t.Logf("PASS: Neural network learned MNIST with %.2f%% accuracy (single hidden layer)", testAcc)
	}
}

// TestMNISTMultiHiddenLayer tests with multiple hidden layers (784 -> 128 -> 64 -> 10)
// to verify that the multi-layer backpropagation loop works correctly.
func TestMNISTMultiHiddenLayer(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping MNIST test in short mode")
	}

	trainImages, testImages, trainLabels, testLabels := downloadMNIST(t)

	rand.Seed(42)
	net := newGenericNN(784, []int{256, 128}, 10)

	baseRates := []float64{0.001, 0.0007, 0.0005}
	epochs := 10

	for epoch := 0; epoch < epochs; epoch++ {
		decay := 1.0 / (1.0 + 0.1*float64(epoch))
		currentRates := make([]float64, len(baseRates))
		for i, r := range baseRates {
			currentRates[i] = r * decay
		}
		indices := rand.Perm(len(trainImages))
		totalCost := 0.0

		for _, idx := range indices {
			imgCopy := make([]float64, 784)
			copy(imgCopy, trainImages[idx])
			x := tensor.New(tensor.WithShape(784), tensor.WithBacking(imgCopy))

			yData := make([]float64, 10)
			yData[trainLabels[idx]] = 1.0
			y := tensor.New(tensor.WithShape(10), tensor.WithBacking(yData))

			cost, err := net.train(x, y, nil, currentRates, nil)
			if err != nil {
				t.Fatalf("training error: %v", err)
			}
			totalCost += cost * cost
		}

		avgCost := totalCost / float64(len(trainImages))
		trainAcc := evaluateAccuracy(t, net, trainImages[:1000], trainLabels[:1000])
		t.Logf("Epoch %d (lr_decay=%.3f): AvgCost=%.6f, TrainAcc(1k sample)=%.2f%%", epoch+1, decay, avgCost, trainAcc)
	}

	testAcc := evaluateAccuracy(t, net, testImages, testLabels)
	t.Logf("Test Accuracy: %.2f%% (%d samples)", testAcc, len(testImages))

	if testAcc < 70.0 {
		t.Errorf("test accuracy %.2f%% is below 70%% - multi-layer backpropagation may not be working correctly", testAcc)
	} else {
		t.Logf("PASS: Neural network learned MNIST with %.2f%% accuracy (multi hidden layer)", testAcc)
	}
}

// TestGradientFlow verifies that gradients flow correctly through multiple layers
// by checking that weights change in each layer during training.
func TestGradientFlow(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping gradient flow test in short mode")
	}

	rand.Seed(42)
	net := newGenericNN(4, []int{8, 6}, 3)

	// Snapshot initial weights
	initialHidden0 := net.hidden[0].Clone().(*tensor.Dense)
	initialHidden1 := net.hidden[1].Clone().(*tensor.Dense)
	initialFinal := net.final.Clone().(*tensor.Dense)

	// Train on a simple sample
	xData := []float64{0.5, 0.3, 0.8, 0.1}
	yData := []float64{1.0, 0.0, 0.0}
	for i := 0; i < 100; i++ {
		xCopy := make([]float64, 4)
		copy(xCopy, xData)
		yCopy := make([]float64, 3)
		copy(yCopy, yData)
		x := tensor.New(tensor.WithShape(4), tensor.WithBacking(xCopy))
		y := tensor.New(tensor.WithShape(3), tensor.WithBacking(yCopy))
		_, err := net.train(x, y, nil, []float64{0.01}, nil)
		if err != nil {
			t.Fatalf("training error: %v", err)
		}
	}

	// Check that all layer weights changed
	diffHidden0 := weightDiff(initialHidden0, net.hidden[0])
	diffHidden1 := weightDiff(initialHidden1, net.hidden[1])
	diffFinal := weightDiff(initialFinal, net.final)

	t.Logf("Weight changes - hidden[0]: %.8f, hidden[1]: %.8f, final: %.8f",
		diffHidden0, diffHidden1, diffFinal)

	if diffHidden0 < 1e-10 {
		t.Errorf("hidden[0] weights did not change - gradients not flowing to first layer")
	}
	if diffHidden1 < 1e-10 {
		t.Errorf("hidden[1] weights did not change - gradients not flowing to second layer")
	}
	if diffFinal < 1e-10 {
		t.Errorf("final weights did not change - gradients not flowing to output layer")
	}

	// Verify the network learned the training sample
	xCheck := tensor.New(tensor.WithShape(4), tensor.WithBacking([]float64{0.5, 0.3, 0.8, 0.1}))
	pred, err := net.predict(xCheck)
	if err != nil {
		t.Fatalf("prediction error: %v", err)
	}
	predData := pred.Data().([]float64)
	t.Logf("After training, prediction: %v (expected class 0)", predData)
	if argmax(predData) != 0 {
		t.Errorf("network did not learn simple training sample: predicted class %d, expected 0", argmax(predData))
	}
}

func weightDiff(a, b *tensor.Dense) float64 {
	aData := a.Data().([]float64)
	bData := b.Data().([]float64)
	diff := 0.0
	for i := range aData {
		d := aData[i] - bData[i]
		diff += d * d
	}
	return diff
}

// TestPerLayerLearningRates verifies that per-layer learning rates apply correctly
// by checking that a frozen layer (lr=0) doesn't change while others do.
func TestPerLayerLearningRates(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping per-layer LR test in short mode")
	}

	rand.Seed(42)
	net := newGenericNN(4, []int{8, 6}, 3)

	// Snapshot initial weights
	initialHidden0 := net.hidden[0].Clone().(*tensor.Dense)
	initialHidden1 := net.hidden[1].Clone().(*tensor.Dense)
	initialFinal := net.final.Clone().(*tensor.Dense)

	// Train with hidden[0] frozen (lr=0), hidden[1] lr=0.01, final lr=0.01
	xData := []float64{0.5, 0.3, 0.8, 0.1}
	yData := []float64{1.0, 0.0, 0.0}
	for i := 0; i < 100; i++ {
		xCopy := make([]float64, 4)
		copy(xCopy, xData)
		yCopy := make([]float64, 3)
		copy(yCopy, yData)
		x := tensor.New(tensor.WithShape(4), tensor.WithBacking(xCopy))
		y := tensor.New(tensor.WithShape(3), tensor.WithBacking(yCopy))
		// learnRates: [hidden0=0.0, hidden1=0.01, final=0.01]
		_, err := net.train(x, y, nil, []float64{0.0, 0.01, 0.01}, nil)
		if err != nil {
			t.Fatalf("training error: %v", err)
		}
	}

	diffHidden0 := weightDiff(initialHidden0, net.hidden[0])
	diffHidden1 := weightDiff(initialHidden1, net.hidden[1])
	diffFinal := weightDiff(initialFinal, net.final)

	t.Logf("Weight changes - hidden[0]: %.10f, hidden[1]: %.8f, final: %.8f",
		diffHidden0, diffHidden1, diffFinal)

	if diffHidden0 > 1e-15 {
		t.Errorf("hidden[0] should be frozen (lr=0) but changed by %.10f", diffHidden0)
	}
	if diffHidden1 < 1e-10 {
		t.Errorf("hidden[1] should have updated (lr=0.01) but didn't change")
	}
	if diffFinal < 1e-10 {
		t.Errorf("final should have updated (lr=0.01) but didn't change")
	}
	t.Log("PASS: Per-layer learning rates correctly applied")
}

// TestMNISTPerLayerLR tests multi-layer MNIST with per-layer learning rates.
func TestMNISTPerLayerLR(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping MNIST per-layer LR test in short mode")
	}

	trainImages, testImages, trainLabels, testLabels := downloadMNIST(t)

	rand.Seed(42)
	net := newGenericNN(784, []int{128, 64}, 10)

	// Per-layer rates: higher for earlier layers (they receive weaker gradients)
	perLayerRates := []float64{0.001, 0.0005, 0.0003}
	trainSize := 20000
	epochs := 3

	for epoch := 0; epoch < epochs; epoch++ {
		indices := rand.Perm(len(trainImages))[:trainSize]
		totalCost := 0.0

		for _, idx := range indices {
			imgCopy := make([]float64, 784)
			copy(imgCopy, trainImages[idx])
			x := tensor.New(tensor.WithShape(784), tensor.WithBacking(imgCopy))

			yData := make([]float64, 10)
			yData[trainLabels[idx]] = 1.0
			y := tensor.New(tensor.WithShape(10), tensor.WithBacking(yData))

			cost, err := net.train(x, y, nil, perLayerRates, nil)
			if err != nil {
				t.Fatalf("training error: %v", err)
			}
			totalCost += cost * cost
		}

		avgCost := totalCost / float64(trainSize)
		trainAcc := evaluateAccuracy(t, net, trainImages[:1000], trainLabels[:1000])
		t.Logf("Epoch %d: AvgCost=%.6f, TrainAcc(1k sample)=%.2f%%", epoch+1, avgCost, trainAcc)
	}

	testAcc := evaluateAccuracy(t, net, testImages, testLabels)
	t.Logf("Test Accuracy with per-layer LR: %.2f%% (%d samples)", testAcc, len(testImages))

	if testAcc < 70.0 {
		t.Errorf("test accuracy %.2f%% is below 70%%", testAcc)
	} else {
		t.Logf("PASS: Per-layer LR MNIST achieved %.2f%% accuracy", testAcc)
	}
}

// ========================================
// Attention Tests
// ========================================

// TestAttentionTokenize verifies that tokenize() produces correct shape and values
// for a known input vector.
func TestAttentionTokenize(t *testing.T) {
	// Build a synthetic 126-dim input
	flat := make([]float64, 126)

	// Set playerHand[0] = 1 (domino 0: suits 0,0)
	flat[0] = 1.0
	// Set boardState[1] = 1 (domino 1: suits 1,0)
	flat[29] = 1.0
	// Set left suit = 3 (index 56+3 = 59)
	flat[59] = 1.0
	// Set right suit = 5 (index 63+5 = 68)
	flat[68] = 1.0
	// Set playerPass[2] = 1
	flat[72] = 1.0
	// Set cardRemaining[3] = 1
	flat[101] = 1.0

	tokens := tokenize(flat, 5)
	if len(tokens) != 28*5 {
		t.Fatalf("expected 140 tokens, got %d", len(tokens))
	}

	// Check token 0: playerHand=1, boardState=0, playerPass=0, cardRemaining=0
	// Domino 0 has suits (0,0). Left=3, Right=5. Neither matches → suitRelevance=0
	if tokens[0] != 1.0 {
		t.Errorf("token[0][0] (playerHand) = %.1f, want 1.0", tokens[0])
	}
	if tokens[4] != 0.0 {
		t.Errorf("token[0][4] (suitRelevance) = %.1f, want 0.0", tokens[4])
	}

	// Check token 1: playerHand=0, boardState=1, playerPass=0, cardRemaining=0
	// Domino 1 has suits (1,0). Left=3, Right=5. Neither matches → suitRelevance=0
	if tokens[5+1] != 1.0 {
		t.Errorf("token[1][1] (boardState) = %.1f, want 1.0", tokens[5+1])
	}

	// Check token 2: playerPass=1
	if tokens[10+2] != 1.0 {
		t.Errorf("token[2][2] (playerPass) = %.1f, want 1.0", tokens[10+2])
	}

	// Check token 3: cardRemaining=1
	if tokens[15+3] != 1.0 {
		t.Errorf("token[3][3] (cardRemaining) = %.1f, want 1.0", tokens[15+3])
	}

	// Check a domino with suit relevance: domino 6 has suits (3,0).
	// Left suit=3 matches! suitRelevance should be 1.0
	if tokens[6*5+4] != 1.0 {
		t.Errorf("token[6][4] (suitRelevance for domino 3|0 with left=3) = %.1f, want 1.0", tokens[6*5+4])
	}

	// Check a domino with suit relevance: domino 20 has suits (5,5).
	// Right suit=5 matches! suitRelevance should be 1.0
	if tokens[20*5+4] != 1.0 {
		t.Errorf("token[20][4] (suitRelevance for domino 5|5 with right=5) = %.1f, want 1.0", tokens[20*5+4])
	}

	t.Log("PASS: Tokenization produces correct shape and values")
}

// TestAttentionForwardShapes verifies that the attention forward pass produces
// the correct output dimensions.
func TestAttentionForwardShapes(t *testing.T) {
	rand.Seed(42)
	net := New(126, []int{64}, 56)
	net.EnableAttention(28)

	flat := make([]float64, 126)
	for i := range flat {
		flat[i] = rand.Float64()
	}

	projected, cache := net.attentionForward(flat)

	expectedLen := 28 * 28
	if len(projected) != expectedLen {
		t.Fatalf("attention output length = %d, want %d", len(projected), expectedLen)
	}

	if cache == nil {
		t.Fatal("attention cache is nil")
	}
	if len(cache.tokens) != 28*5 {
		t.Errorf("cache.tokens length = %d, want %d", len(cache.tokens), 28*5)
	}
	if len(cache.Q) != 28*28 {
		t.Errorf("cache.Q length = %d, want %d", len(cache.Q), 28*28)
	}
	if len(cache.attnWeights) != 28*28 {
		t.Errorf("cache.attnWeights length = %d, want %d", len(cache.attnWeights), 28*28)
	}

	// Verify attention weights sum to ~1.0 per row
	for r := 0; r < 28; r++ {
		rowSum := 0.0
		for c := 0; c < 28; c++ {
			rowSum += cache.attnWeights[r*28+c]
		}
		if math.Abs(rowSum-1.0) > 1e-6 {
			t.Errorf("attention weights row %d sum = %.8f, want ~1.0", r, rowSum)
		}
	}

	// Verify MLP can process the attention output
	input := tensor.New(tensor.WithShape(expectedLen), tensor.WithBacking(projected))
	_, err := net.predict(input)
	if err != nil {
		t.Fatalf("MLP predict after attention failed: %v", err)
	}

	t.Log("PASS: Attention forward produces correct shapes")
}

// TestAttentionGradientFlow verifies that all attention weights change during training.
func TestAttentionGradientFlow(t *testing.T) {
	rand.Seed(42)
	net := newGenericNN(126, []int{64}, 56)
	net.useAttention = false // will enable manually to control inputDim
	net.EnableAttention(28)

	// Snapshot initial attention weights
	initWQ := net.wQ.Clone().(*tensor.Dense)
	initWK := net.wK.Clone().(*tensor.Dense)
	initWV := net.wV.Clone().(*tensor.Dense)
	initWO := net.wO.Clone().(*tensor.Dense)
	initBQ := net.bQ.Clone().(*tensor.Dense)

	// Also snapshot MLP hidden[0]
	initH0 := net.hidden[0].Clone().(*tensor.Dense)

	// Train for a few iterations
	for i := 0; i < 50; i++ {
		xData := make([]float64, 126)
		for k := range xData {
			xData[k] = rand.Float64()
		}
		yData := make([]float64, 56)
		yData[rand.Intn(56)] = 1.0

		x := tensor.New(tensor.WithShape(126), tensor.WithBacking(xData))
		y := tensor.New(tensor.WithShape(56), tensor.WithBacking(yData))
		_, err := net.train(x, y, nil, []float64{0.001}, nil)
		if err != nil {
			t.Fatalf("training error: %v", err)
		}
	}

	// Check all attention weights changed
	dWQ := weightDiff(initWQ, net.wQ)
	dWK := weightDiff(initWK, net.wK)
	dWV := weightDiff(initWV, net.wV)
	dWO := weightDiff(initWO, net.wO)
	dBQ := weightDiff(initBQ, net.bQ)
	dH0 := weightDiff(initH0, net.hidden[0])

	t.Logf("Weight changes - wQ: %.8f, wK: %.8f, wV: %.8f, wO: %.8f, bQ: %.8f, hidden[0]: %.8f",
		dWQ, dWK, dWV, dWO, dBQ, dH0)

	if dWQ < 1e-10 {
		t.Error("wQ did not change — gradients not flowing through Q projection")
	}
	if dWK < 1e-10 {
		t.Error("wK did not change — gradients not flowing through K projection")
	}
	if dWV < 1e-10 {
		t.Error("wV did not change — gradients not flowing through V projection")
	}
	if dWO < 1e-10 {
		t.Error("wO did not change — gradients not flowing through output projection")
	}
	if dBQ < 1e-10 {
		t.Error("bQ did not change — gradients not flowing through Q bias")
	}
	if dH0 < 1e-10 {
		t.Error("hidden[0] did not change — MLP not training with attention input")
	}

	t.Log("PASS: Attention gradient flow verified")
}

// TestAttentionSaveLoad tests save/load round-trip with attention enabled.
func TestAttentionSaveLoad(t *testing.T) {
	rand.Seed(42)
	net := New(126, []int{64}, 56)
	net.EnableAttention(28)

	// Train a little to get non-initial weights
	for i := 0; i < 10; i++ {
		xData := make([]float64, 126)
		for k := range xData {
			xData[k] = rand.Float64()
		}
		yData := make([]float64, 56)
		yData[0] = 1.0
		x := tensor.New(tensor.WithShape(126), tensor.WithBacking(xData))
		y := tensor.New(tensor.WithShape(56), tensor.WithBacking(yData))
		net.train(x, y, nil, []float64{0.001}, nil)
	}

	// Save
	tmpFile := filepath.Join(t.TempDir(), "test_attn.model")
	if err := net.Save(tmpFile); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Load into a fresh net
	net2 := New(126, []int{64}, 56)
	net2.EnableAttention(28) // must have same architecture
	if err := net2.Load(tmpFile); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Compare weights
	if !net2.useAttention {
		t.Fatal("loaded model should have useAttention=true")
	}

	dWQ := weightDiff(net.wQ, net2.wQ)
	dWO := weightDiff(net.wO, net2.wO)
	dH0 := weightDiff(net.hidden[0], net2.hidden[0])
	dFinal := weightDiff(net.final, net2.final)

	if dWQ > 1e-15 {
		t.Errorf("wQ mismatch after save/load: diff=%.15f", dWQ)
	}
	if dWO > 1e-15 {
		t.Errorf("wO mismatch after save/load: diff=%.15f", dWO)
	}
	if dH0 > 1e-15 {
		t.Errorf("hidden[0] mismatch after save/load: diff=%.15f", dH0)
	}
	if dFinal > 1e-15 {
		t.Errorf("final mismatch after save/load: diff=%.15f", dFinal)
	}

	// Verify predictions match
	xData := make([]float64, 126)
	for k := range xData {
		xData[k] = rand.Float64()
	}
	x1 := tensor.New(tensor.WithShape(126), tensor.WithBacking(append([]float64{}, xData...)))
	x2 := tensor.New(tensor.WithShape(126), tensor.WithBacking(append([]float64{}, xData...)))

	pred1, err := net.predict(x1)
	if err != nil {
		t.Fatalf("predict on original failed: %v", err)
	}
	pred2, err := net2.predict(x2)
	if err != nil {
		t.Fatalf("predict on loaded failed: %v", err)
	}

	p1 := pred1.Data().([]float64)
	p2 := pred2.Data().([]float64)
	maxDiff := 0.0
	for i := range p1 {
		d := math.Abs(p1[i] - p2[i])
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-10 {
		t.Errorf("predictions differ after save/load: max diff = %.15f", maxDiff)
	}

	t.Log("PASS: Attention save/load round-trip verified")
}

// TestAttentionBackwardCompat verifies that loading a model saved without attention
// still works (old format).
func TestAttentionBackwardCompat(t *testing.T) {
	rand.Seed(42)
	// Save a model without attention
	net := New(126, []int{64}, 56)
	tmpFile := filepath.Join(t.TempDir(), "test_no_attn.model")
	if err := net.Save(tmpFile); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Load into a fresh net (no attention)
	net2 := New(126, []int{64}, 56)
	if err := net2.Load(tmpFile); err != nil {
		t.Fatalf("Load old-format failed: %v", err)
	}

	if net2.useAttention {
		t.Error("loaded old-format model should not have attention enabled")
	}

	// Predictions should match
	xData := make([]float64, 126)
	for k := range xData {
		xData[k] = rand.Float64()
	}
	x1 := tensor.New(tensor.WithShape(126), tensor.WithBacking(append([]float64{}, xData...)))
	x2 := tensor.New(tensor.WithShape(126), tensor.WithBacking(append([]float64{}, xData...)))

	pred1, _ := net.predict(x1)
	pred2, _ := net2.predict(x2)

	p1 := pred1.Data().([]float64)
	p2 := pred2.Data().([]float64)
	for i := range p1 {
		if math.Abs(p1[i]-p2[i]) > 1e-10 {
			t.Fatalf("predictions differ at index %d: %.10f vs %.10f", i, p1[i], p2[i])
		}
	}

	t.Log("PASS: Backward compatibility with old save format verified")
}

// TestAttentionPredict verifies end-to-end predict with attention enabled.
func TestAttentionPredict(t *testing.T) {
	rand.Seed(42)
	net := New(126, []int{64}, 56)
	net.EnableAttention(28)

	// Create a valid 126-dim input
	xData := make([]float64, 126)
	for i := range xData {
		xData[i] = rand.Float64()
	}
	x := tensor.New(tensor.WithShape(126), tensor.WithBacking(xData))

	pred, err := net.predict(x)
	if err != nil {
		t.Fatalf("predict with attention failed: %v", err)
	}

	predData := pred.Data().([]float64)
	if len(predData) != 56 {
		t.Fatalf("prediction length = %d, want 56", len(predData))
	}

	// Verify output is not all zeros (network should produce non-trivial output)
	allZero := true
	for _, v := range predData {
		if v != 0.0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("prediction is all zeros — attention + MLP may not be working")
	}

	t.Log("PASS: End-to-end attention predict works")
}

// ========================================
// Autograd Tests
// ========================================

// TestAutogradMLP verifies that autograd training produces weight updates
// (weights change in all layers after training).
func TestAutogradMLP(t *testing.T) {
	rand.Seed(42)
	net := newGenericNN(4, []int{8, 6}, 3)
	net.TrainMode = "autograd"
	net.OutputActivation = "linear"

	// Snapshot initial weights
	initH0 := net.hidden[0].Clone().(*tensor.Dense)
	initH1 := net.hidden[1].Clone().(*tensor.Dense)
	initFinal := net.final.Clone().(*tensor.Dense)

	// Train with higher LR and more iterations (autograd MSE gradient is 2x scale)
	xData := []float64{0.5, 0.3, 0.8, 0.1}
	yData := []float64{1.0, 0.0, 0.0}
	for i := 0; i < 500; i++ {
		xCopy := make([]float64, 4)
		copy(xCopy, xData)
		yCopy := make([]float64, 3)
		copy(yCopy, yData)
		x := tensor.New(tensor.WithShape(4), tensor.WithBacking(xCopy))
		y := tensor.New(tensor.WithShape(3), tensor.WithBacking(yCopy))
		_, err := net.train(x, y, nil, []float64{0.005}, nil)
		if err != nil {
			t.Fatalf("autograd training error: %v", err)
		}
	}

	dH0 := weightDiff(initH0, net.hidden[0])
	dH1 := weightDiff(initH1, net.hidden[1])
	dFinal := weightDiff(initFinal, net.final)

	t.Logf("Autograd weight changes - hidden[0]: %.8f, hidden[1]: %.8f, final: %.8f",
		dH0, dH1, dFinal)

	if dH0 < 1e-10 {
		t.Error("hidden[0] did not change with autograd training")
	}
	if dH1 < 1e-10 {
		t.Error("hidden[1] did not change with autograd training")
	}
	if dFinal < 1e-10 {
		t.Error("final did not change with autograd training")
	}

	// Verify the network learned
	xCheck := tensor.New(tensor.WithShape(4), tensor.WithBacking([]float64{0.5, 0.3, 0.8, 0.1}))
	pred, err := net.predict(xCheck)
	if err != nil {
		t.Fatalf("prediction error: %v", err)
	}
	predData := pred.Data().([]float64)
	t.Logf("Autograd prediction: %v (expected class 0)", predData)
	if argmax(predData) != 0 {
		t.Errorf("autograd network did not learn: predicted class %d, expected 0", argmax(predData))
	}

	t.Log("PASS: Autograd MLP training produces correct weight updates")
}

// TestAutogradWithAttention verifies autograd training works with attention enabled.
func TestAutogradWithAttention(t *testing.T) {
	rand.Seed(42)
	net := newGenericNN(126, []int{64}, 56)
	net.EnableAttention(28)
	net.TrainMode = "autograd"

	initH0 := net.hidden[0].Clone().(*tensor.Dense)
	initFinal := net.final.Clone().(*tensor.Dense)

	for i := 0; i < 20; i++ {
		xData := make([]float64, 126)
		for k := range xData {
			xData[k] = rand.Float64()
		}
		yData := make([]float64, 56)
		yData[rand.Intn(56)] = 1.0
		x := tensor.New(tensor.WithShape(126), tensor.WithBacking(xData))
		y := tensor.New(tensor.WithShape(56), tensor.WithBacking(yData))
		_, err := net.train(x, y, nil, []float64{0.001}, nil)
		if err != nil {
			t.Fatalf("autograd+attention training error: %v", err)
		}
	}

	dH0 := weightDiff(initH0, net.hidden[0])
	dFinal := weightDiff(initFinal, net.final)

	t.Logf("Autograd+attention weight changes - hidden[0]: %.8f, final: %.8f", dH0, dFinal)

	if dH0 < 1e-10 {
		t.Error("hidden[0] did not change with autograd+attention")
	}
	if dFinal < 1e-10 {
		t.Error("final did not change with autograd+attention")
	}

	t.Log("PASS: Autograd with attention training works")
}

// TestSoftmax verifies the softmax implementation for correctness.
func TestSoftmax(t *testing.T) {
	// Single row test
	data := []float64{1.0, 2.0, 3.0, 4.0}
	result := rowSoftmax(data, 1, 4)

	// Check sums to 1
	sum := 0.0
	for _, v := range result {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("softmax sum = %.15f, want 1.0", sum)
	}

	// Check ordering preserved (larger input → larger probability)
	for i := 0; i < 3; i++ {
		if result[i] >= result[i+1] {
			t.Errorf("softmax ordering wrong: result[%d]=%.6f >= result[%d]=%.6f", i, result[i], i+1, result[i+1])
		}
	}

	// Test numerical stability with large values
	bigData := []float64{1000.0, 1001.0, 1002.0}
	bigResult := rowSoftmax(bigData, 1, 3)
	bigSum := 0.0
	for _, v := range bigResult {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatal("softmax produced NaN/Inf with large values")
		}
		bigSum += v
	}
	if math.Abs(bigSum-1.0) > 1e-10 {
		t.Errorf("softmax sum with large values = %.15f, want 1.0", bigSum)
	}

	t.Log("PASS: Softmax correctness and numerical stability verified")
}
