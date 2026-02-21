package nn

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
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

			cost, err := net.train(x, y, nil, []float64{learnRate})
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

			cost, err := net.train(x, y, nil, currentRates)
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
		_, err := net.train(x, y, nil, []float64{0.01})
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
		_, err := net.train(x, y, nil, []float64{0.0, 0.01, 0.01})
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

			cost, err := net.train(x, y, nil, perLayerRates)
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
