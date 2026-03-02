package nn

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

func TestLayerNormForwardShape(t *testing.T) {
	seqLen := 5
	dModel := 8
	x := make([]float64, seqLen*dModel)
	for i := range x {
		x[i] = rand.NormFloat64()
	}
	gamma := make([]float64, dModel)
	beta := make([]float64, dModel)
	for i := range gamma {
		gamma[i] = 1.0
	}

	out, cache := layerNormForward(x, gamma, beta, seqLen, dModel)
	if len(out) != seqLen*dModel {
		t.Fatalf("expected output len %d, got %d", seqLen*dModel, len(out))
	}
	if cache == nil {
		t.Fatal("expected non-nil cache")
	}

	// Each row should be approximately mean=0, var=1
	for i := 0; i < seqLen; i++ {
		mean := 0.0
		for j := 0; j < dModel; j++ {
			mean += out[i*dModel+j]
		}
		mean /= float64(dModel)
		if math.Abs(mean) > 1e-5 {
			t.Errorf("row %d: mean = %f, expected ~0", i, mean)
		}
	}
}

func TestLayerNormBackward(t *testing.T) {
	seqLen := 3
	dModel := 4
	x := make([]float64, seqLen*dModel)
	for i := range x {
		x[i] = rand.NormFloat64()
	}
	gamma := make([]float64, dModel)
	beta := make([]float64, dModel)
	for i := range gamma {
		gamma[i] = 1.0
	}

	_, cache := layerNormForward(x, gamma, beta, seqLen, dModel)
	dOut := make([]float64, seqLen*dModel)
	for i := range dOut {
		dOut[i] = rand.NormFloat64() * 0.1
	}

	dx, dGamma, dBeta := layerNormBackward(dOut, cache, gamma, seqLen, dModel)
	if len(dx) != len(x) {
		t.Fatalf("dx len mismatch: %d vs %d", len(dx), len(x))
	}
	if len(dGamma) != dModel || len(dBeta) != dModel {
		t.Fatal("dGamma/dBeta length mismatch")
	}
}

func TestCausalMask(t *testing.T) {
	mask := buildCausalMask(4)
	// Position (0,1) should be -inf (can't attend to future)
	if !math.IsInf(mask[0*4+1], -1) {
		t.Error("expected -inf at (0,1)")
	}
	// Position (1,0) should be 0 (can attend to past)
	if mask[1*4+0] != 0 {
		t.Error("expected 0 at (1,0)")
	}
	// Diagonal should be 0 (can attend to self)
	if mask[2*4+2] != 0 {
		t.Error("expected 0 at (2,2)")
	}
}

func TestMultiHeadAttentionForwardShape(t *testing.T) {
	dModel := 8
	nHeads := 2
	dHead := 4
	dFF := 16
	seqLen := 5

	layer := newTransformerLayer(dModel, nHeads, dHead, dFF)
	x := make([]float64, seqLen*dModel)
	for i := range x {
		x[i] = rand.NormFloat64() * 0.1
	}
	mask := buildCausalMask(seqLen)

	out, cache := multiHeadAttentionForward(x, layer, mask, seqLen)
	if len(out) != seqLen*dModel {
		t.Fatalf("MHA output len: expected %d, got %d", seqLen*dModel, len(out))
	}
	if cache == nil {
		t.Fatal("expected non-nil cache")
	}
}

func TestFFNForwardShape(t *testing.T) {
	dModel := 8
	dFF := 16
	seqLen := 5

	layer := newTransformerLayer(dModel, 2, 4, dFF)
	x := make([]float64, seqLen*dModel)
	for i := range x {
		x[i] = rand.NormFloat64() * 0.1
	}

	out, cache := ffnForward(x, layer, seqLen)
	if len(out) != seqLen*dModel {
		t.Fatalf("FFN output len: expected %d, got %d", seqLen*dModel, len(out))
	}
	if cache == nil {
		t.Fatal("expected non-nil cache")
	}
}

func TestTransformerLayerForward(t *testing.T) {
	dModel := 8
	nHeads := 2
	dHead := 4
	dFF := 16
	seqLen := 5

	layer := newTransformerLayer(dModel, nHeads, dHead, dFF)
	x := make([]float64, seqLen*dModel)
	for i := range x {
		x[i] = rand.NormFloat64() * 0.1
	}
	mask := buildCausalMask(seqLen)

	out, cache := transformerLayerForward(x, layer, mask, seqLen)
	if len(out) != seqLen*dModel {
		t.Fatalf("layer output len: expected %d, got %d", seqLen*dModel, len(out))
	}
	if cache == nil {
		t.Fatal("expected non-nil cache")
	}
}

func TestTransformerForwardShape(t *testing.T) {
	tr := NewSequenceTransformer(64, 2, 2, 128, 40, 56)

	tokens := []moveToken{
		{playerID: 0, cardID: 5, sideID: sidePosed},
		{playerID: 0, cardID: 12, sideID: sidePosed},
		{playerID: 0, cardID: cardSEP, sideID: sidePass},
		{playerID: 1, cardID: 3, sideID: 0},
		{playerID: 2, cardID: 10, sideID: 1},
		{playerID: 0, cardID: cardPASS, sideID: sidePass}, // query
	}

	output, cache := tr.forward(tokens)
	if len(output) != 56 {
		t.Fatalf("expected 56-dim output, got %d", len(output))
	}
	if cache == nil {
		t.Fatal("expected non-nil cache")
	}
	if cache.seqLen != len(tokens) {
		t.Fatalf("cache seqLen: expected %d, got %d", len(tokens), cache.seqLen)
	}
}

func TestTransformerTrainingLossDecreases(t *testing.T) {
	rand.Seed(42)
	tr := NewSequenceTransformer(32, 2, 1, 64, 40, 56)

	// Fixed training sample
	tokens := []moveToken{
		{playerID: 0, cardID: 5, sideID: sidePosed},
		{playerID: 0, cardID: 12, sideID: sidePosed},
		{playerID: 0, cardID: cardSEP, sideID: sidePass},
		{playerID: 1, cardID: 3, sideID: 0},
		{playerID: 0, cardID: cardPASS, sideID: sidePass},
	}

	// Target: high reward at action index 3 (card 3, left side)
	target := [56]float64{}
	target[3] = 0.8
	actionIndex := 3

	lr := 0.001
	var firstLoss, lastLoss float64

	for iter := 0; iter < 200; iter++ {
		tr.gameHistory = nil
		tr.SetGameHistory([]moveToken{
			{playerID: 1, cardID: 3, sideID: 0},
		})

		output, cache := tr.forward(tokens)

		outputGrad := make([]float64, 56)
		err := target[actionIndex] - output[actionIndex]
		loss := err * err
		outputGrad[actionIndex] = clipValue(err)

		tr.backward(outputGrad, cache, lr)

		if iter == 0 {
			firstLoss = loss
		}
		lastLoss = loss
	}

	if lastLoss >= firstLoss {
		t.Errorf("loss did not decrease: first=%.6f, last=%.6f", firstLoss, lastLoss)
	}
	t.Logf("Loss: %.6f → %.6f (%.1f%% reduction)", firstLoss, lastLoss, (1-lastLoss/firstLoss)*100)
}

func TestTransformerSaveLoad(t *testing.T) {
	tr := NewSequenceTransformer(32, 2, 1, 64, 40, 56)

	// Run a forward pass to get baseline
	tokens := []moveToken{
		{playerID: 0, cardID: 5, sideID: sidePosed},
		{playerID: 0, cardID: cardSEP, sideID: sidePass},
		{playerID: 0, cardID: cardPASS, sideID: sidePass},
	}
	output1, _ := tr.forward(tokens)

	// Save
	tmpFile := "/tmp/test_transformer_save.mdl"
	defer os.Remove(tmpFile)
	if err := tr.Save(tmpFile); err != nil {
		t.Fatal("save error:", err)
	}

	// Load into new transformer
	tr2 := NewSequenceTransformer(32, 2, 1, 64, 40, 56)
	if err := tr2.Load(tmpFile); err != nil {
		t.Fatal("load error:", err)
	}

	// Forward pass should match
	output2, _ := tr2.forward(tokens)
	for i := range output1 {
		if math.Abs(output1[i]-output2[i]) > 1e-10 {
			t.Fatalf("output mismatch at %d: %.10f vs %.10f", i, output1[i], output2[i])
		}
	}
}

func TestTransformerClone(t *testing.T) {
	tr := NewSequenceTransformer(32, 2, 1, 64, 40, 56)

	tokens := []moveToken{
		{playerID: 0, cardID: 5, sideID: sidePosed},
		{playerID: 0, cardID: cardSEP, sideID: sidePass},
		{playerID: 0, cardID: cardPASS, sideID: sidePass},
	}
	output1, _ := tr.forward(tokens)

	clone := tr.Clone()
	output2, _ := clone.forward(tokens)

	for i := range output1 {
		if math.Abs(output1[i]-output2[i]) > 1e-10 {
			t.Fatalf("clone output mismatch at %d: %.10f vs %.10f", i, output1[i], output2[i])
		}
	}

	// Modify clone, original should be unchanged
	clone.wOut[0] += 1.0
	output3, _ := tr.forward(tokens)
	for i := range output1 {
		if math.Abs(output1[i]-output3[i]) > 1e-10 {
			t.Fatalf("original changed after clone modification at %d", i)
		}
	}
}

func TestEmbedTokens(t *testing.T) {
	tr := NewSequenceTransformer(16, 2, 1, 32, 40, 56)
	tokens := []moveToken{
		{playerID: 0, cardID: 5, sideID: 0},
		{playerID: 1, cardID: 10, sideID: 1},
	}

	embedded := tr.embedTokens(tokens)
	if len(embedded) != 2*16 {
		t.Fatalf("expected %d, got %d", 2*16, len(embedded))
	}

	// Verify embedding is not all zeros
	allZero := true
	for _, v := range embedded {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("embeddings should not be all zero")
	}
}

func TestSequenceConstruction(t *testing.T) {
	tr := NewSequenceTransformer(16, 2, 1, 32, 40, 56)

	// Add some history
	tr.gameHistory = []moveToken{
		{playerID: 1, cardID: 3, sideID: 0},
		{playerID: 2, cardID: cardPASS, sideID: sidePass},
		{playerID: 3, cardID: 15, sideID: 1},
	}

	// Verify max sequence length truncation
	tr2 := NewSequenceTransformer(16, 2, 1, 32, 10, 56) // maxSeqLen = 10
	tr2.gameHistory = make([]moveToken, 20)
	for i := range tr2.gameHistory {
		tr2.gameHistory[i] = moveToken{playerID: 1, cardID: i % 28, sideID: 0}
	}
	// Manually build a minimal gameEvent-like scenario
	// The sequence should be truncated to maxSeqLen
}

func TestExtractAndScatterHead(t *testing.T) {
	seqLen := 3
	dModel := 8
	dHead := 4

	x := make([]float64, seqLen*dModel)
	for i := range x {
		x[i] = float64(i)
	}

	// Extract head at offset 0
	h0 := extractHead(x, seqLen, dModel, 0, dHead)
	if len(h0) != seqLen*dHead {
		t.Fatalf("expected %d, got %d", seqLen*dHead, len(h0))
	}
	// First 4 values of first row should be 0,1,2,3
	for i := 0; i < dHead; i++ {
		if h0[i] != float64(i) {
			t.Errorf("h0[%d] = %f, expected %f", i, h0[i], float64(i))
		}
	}

	// Extract head at offset 4
	h1 := extractHead(x, seqLen, dModel, 4, dHead)
	// First 4 values should be 4,5,6,7
	for i := 0; i < dHead; i++ {
		if h1[i] != float64(4+i) {
			t.Errorf("h1[%d] = %f, expected %f", i, h1[i], float64(4+i))
		}
	}

	// Scatter back and verify roundtrip
	dst := make([]float64, seqLen*dModel)
	scatterHead(dst, h0, seqLen, dModel, 0, dHead)
	scatterHead(dst, h1, seqLen, dModel, 4, dHead)
	for i := range x {
		if math.Abs(dst[i]-x[i]) > 1e-10 {
			t.Errorf("roundtrip mismatch at %d: %f vs %f", i, dst[i], x[i])
		}
	}
}
