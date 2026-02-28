package nn

import (
	"math"

	"github.com/HowardDunn/go-dominos/dominos"
	"gonum.org/v1/gonum/stat/distuv"
	"gorgonia.org/tensor"
)

const (
	numTokens    = 28
	defaultDToken = 5
)

// attentionCache holds intermediate values from the forward pass needed for backprop.
type attentionCache struct {
	tokens     []float64 // [numTokens * dToken] flattened
	Q          []float64 // [numTokens * dModel]
	K          []float64 // [numTokens * dModel]
	V          []float64 // [numTokens * dModel]
	attnWeights []float64 // [numTokens * numTokens] after softmax
	attnOut    []float64 // [numTokens * dModel] = weights @ V
}

// tokenize converts a flat 126-dim input into a [28, 5] token matrix.
// Per-token features:
//   [0] playerHand[i]       (indices 0-27)
//   [1] boardState[i]       (indices 28-55)
//   [2] playerPass[i]       (indices 70-97)
//   [3] cardRemaining[i]    (indices 98-125)
//   [4] suit relevance: 1.0 if either suit of this domino matches left OR right board suit
func tokenize(flat []float64, dToken int) []float64 {
	tokens := make([]float64, numTokens*dToken)

	// Decode board suits from suitState (indices 56-69)
	// Left suit: one-hot at 56-62, Right suit: one-hot at 63-69
	leftSuit := -1
	rightSuit := -1
	for s := 0; s < 7; s++ {
		if flat[56+s] > 0.5 {
			leftSuit = s
		}
		if flat[63+s] > 0.5 {
			rightSuit = s
		}
	}

	for i := 0; i < numTokens; i++ {
		base := i * dToken
		tokens[base+0] = flat[i]      // playerHand
		tokens[base+1] = flat[28+i]   // boardState
		tokens[base+2] = flat[70+i]   // playerPass
		tokens[base+3] = flat[98+i]   // cardRemaining

		// Suit relevance: check if either suit of domino i matches board suits
		suitRelevance := 0.0
		suitInfo := dominos.IndexSuitMap[i]
		if suitInfo != nil {
			s1 := int(suitInfo.Suit1)
			s2 := int(suitInfo.Suit2)
			if leftSuit >= 0 && (s1 == leftSuit || s2 == leftSuit) {
				suitRelevance = 1.0
			}
			if rightSuit >= 0 && (s1 == rightSuit || s2 == rightSuit) {
				suitRelevance = 1.0
			}
		}
		tokens[base+4] = suitRelevance
	}

	return tokens
}

// rowSoftmax computes softmax independently for each row of an [nRows x nCols] matrix.
// Uses the max-subtraction trick for numerical stability.
func rowSoftmax(data []float64, nRows, nCols int) []float64 {
	out := make([]float64, len(data))
	for r := 0; r < nRows; r++ {
		base := r * nCols
		// Find max for numerical stability
		maxVal := data[base]
		for c := 1; c < nCols; c++ {
			if data[base+c] > maxVal {
				maxVal = data[base+c]
			}
		}
		// Exp and sum
		sum := 0.0
		for c := 0; c < nCols; c++ {
			out[base+c] = math.Exp(data[base+c] - maxVal)
			sum += out[base+c]
		}
		// Normalize
		for c := 0; c < nCols; c++ {
			out[base+c] /= sum
		}
	}
	return out
}

// softmaxBackward computes the gradient through the softmax for each row.
// For row i: dScores[i] = softmax[i] * (dAttnWeights[i] - dot(dAttnWeights[i], softmax[i]))
func softmaxBackward(dAttnWeights, softmaxOut []float64, nRows, nCols int) []float64 {
	dScores := make([]float64, len(dAttnWeights))
	for r := 0; r < nRows; r++ {
		base := r * nCols
		// dot(dAttnWeights[r], softmax[r])
		dot := 0.0
		for c := 0; c < nCols; c++ {
			dot += dAttnWeights[base+c] * softmaxOut[base+c]
		}
		for c := 0; c < nCols; c++ {
			dScores[base+c] = softmaxOut[base+c] * (dAttnWeights[base+c] - dot)
		}
	}
	return dScores
}

// matMul2D computes C = A @ B where A is [m x k] and B is [k x n], result is [m x n].
func matMul2D(a []float64, m, k int, b []float64, n int) []float64 {
	c := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for l := 0; l < k; l++ {
				sum += a[i*k+l] * b[l*n+j]
			}
			c[i*n+j] = sum
		}
	}
	return c
}

// matMulTransB computes C = A @ B^T where A is [m x k] and B is [n x k], result is [m x n].
func matMulTransB(a []float64, m, k int, b []float64, n int) []float64 {
	c := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for l := 0; l < k; l++ {
				sum += a[i*k+l] * b[j*k+l]
			}
			c[i*n+j] = sum
		}
	}
	return c
}

// matMulTransA computes C = A^T @ B where A is [m x k] (transposed to [k x m]) and B is [m x n], result is [k x n].
func matMulTransA(a []float64, m, k int, b []float64, n int) []float64 {
	c := make([]float64, k*n)
	for i := 0; i < k; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for l := 0; l < m; l++ {
				sum += a[l*k+i] * b[l*n+j]
			}
			c[i*n+j] = sum
		}
	}
	return c
}

// colSum computes the column-wise sum of an [m x n] matrix, returning [n].
func colSum(data []float64, m, n int) []float64 {
	sums := make([]float64, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sums[j] += data[i*n+j]
		}
	}
	return sums
}

// addBias adds a [dModel] bias to each row of an [nRows x dModel] matrix in-place.
func addBias(data []float64, bias []float64, nRows, dModel int) {
	for i := 0; i < nRows; i++ {
		for j := 0; j < dModel; j++ {
			data[i*dModel+j] += bias[j]
		}
	}
}

// EnableAttention activates the self-attention preprocessing layer.
// dModel controls the attention dimension; the MLP input becomes numTokens * dModel.
func (j *JSDNN) EnableAttention(dModel int) {
	j.useAttention = true
	j.attnNumTokens = numTokens
	j.attnDToken = defaultDToken
	j.attnDModel = dModel
	j.AttentionLR = 0.001

	// Initialize Q, K, V projections: [dToken x dModel]
	// Xavier uniform with fan_in = dToken
	initProjection := func(fanIn int, rows, cols int) *tensor.Dense {
		data := make([]float64, rows*cols)
		dist := distuv.Uniform{
			Min: -math.Sqrt(6.0 / float64(fanIn)),
			Max: math.Sqrt(6.0 / float64(fanIn)),
		}
		for i := range data {
			data[i] = dist.Rand()
		}
		return tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(data))
	}

	initBias := func(size int) *tensor.Dense {
		return tensor.New(tensor.WithShape(size), tensor.WithBacking(make([]float64, size)))
	}

	j.wQ = initProjection(j.attnDToken, j.attnDToken, dModel)
	j.wK = initProjection(j.attnDToken, j.attnDToken, dModel)
	j.wV = initProjection(j.attnDToken, j.attnDToken, dModel)
	j.wO = initProjection(dModel, dModel, dModel)

	j.bQ = initBias(dModel)
	j.bK = initBias(dModel)
	j.bV = initBias(dModel)
	j.bO = initBias(dModel)

	// Rebuild hidden[0] to accept numTokens * dModel inputs instead of 126
	newInputDim := numTokens * dModel
	j.inputDim = newInputDim
	h0Rows := j.hiddenDim[0]
	h0Data := make([]float64, h0Rows*newInputDim)
	fillRandom(h0Data, float64(newInputDim))
	j.hidden[0] = tensor.New(tensor.WithShape(h0Rows, newInputDim), tensor.WithBacking(h0Data))

	// Re-init bias for hidden[0]
	bh0Data := make([]float64, h0Rows)
	fillRandom(bh0Data, float64(h0Rows))
	j.bHidden[0] = tensor.New(tensor.WithShape(h0Rows), tensor.WithBacking(bh0Data))
}

// attentionForward runs the self-attention layer.
// Input: flat 126-dim vector. Output: flattened [numTokens * dModel] vector + cache.
func (j *JSDNN) attentionForward(flat []float64) ([]float64, *attentionCache) {
	dModel := j.attnDModel
	dToken := j.attnDToken
	n := j.attnNumTokens

	// Tokenize
	tokens := tokenize(flat, dToken)

	// Get weight data
	wqData := j.wQ.Data().([]float64)
	wkData := j.wK.Data().([]float64)
	wvData := j.wV.Data().([]float64)
	woData := j.wO.Data().([]float64)
	bqData := j.bQ.Data().([]float64)
	bkData := j.bK.Data().([]float64)
	bvData := j.bV.Data().([]float64)
	boData := j.bO.Data().([]float64)

	// Q = tokens @ Wq + bQ  → [n, dModel]
	Q := matMul2D(tokens, n, dToken, wqData, dModel)
	addBias(Q, bqData, n, dModel)

	// K = tokens @ Wk + bK  → [n, dModel]
	K := matMul2D(tokens, n, dToken, wkData, dModel)
	addBias(K, bkData, n, dModel)

	// V = tokens @ Wv + bV  → [n, dModel]
	V := matMul2D(tokens, n, dToken, wvData, dModel)
	addBias(V, bvData, n, dModel)

	// scores = Q @ K^T / sqrt(dModel)  → [n, n]
	scores := matMulTransB(Q, n, dModel, K, n)
	scale := 1.0 / math.Sqrt(float64(dModel))
	for i := range scores {
		scores[i] *= scale
	}

	// attnWeights = rowSoftmax(scores)  → [n, n]
	attnWeights := rowSoftmax(scores, n, n)

	// attnOut = attnWeights @ V  → [n, dModel]
	attnOut := matMul2D(attnWeights, n, n, V, dModel)

	// projected = attnOut @ Wo + bO  → [n, dModel]
	projected := matMul2D(attnOut, n, dModel, woData, dModel)
	addBias(projected, boData, n, dModel)

	cache := &attentionCache{
		tokens:      tokens,
		Q:           Q,
		K:           K,
		V:           V,
		attnWeights: attnWeights,
		attnOut:     attnOut,
	}

	return projected, cache
}

// attentionBackward computes gradients through the attention layer and updates weights.
// dFlat is the gradient of the loss w.r.t. the flattened attention output [numTokens * dModel].
func (j *JSDNN) attentionBackward(dFlat []float64, cache *attentionCache, lr float64) {
	dModel := j.attnDModel
	dToken := j.attnDToken
	n := j.attnNumTokens

	woData := j.wO.Data().([]float64)

	// dFlat is already [n * dModel], treat as dOut [n, dModel]
	dOut := dFlat

	// Output projection backward
	// dWo = attnOut^T @ dOut  [dModel, dModel]
	dWo := matMulTransA(cache.attnOut, n, dModel, dOut, dModel)
	// dbO = colSum(dOut)  [dModel]
	dbO := colSum(dOut, n, dModel)
	// dAttnOut = dOut @ Wo^T  [n, dModel]
	dAttnOut := matMulTransB(dOut, n, dModel, woData, dModel)

	// Weighted sum backward
	// dAttnWeights = dAttnOut @ V^T  [n, n]
	dAttnWeights := matMulTransB(dAttnOut, n, dModel, cache.V, n)
	// dV = attnWeights^T @ dAttnOut  [n, dModel]
	dV := matMulTransA(cache.attnWeights, n, n, dAttnOut, dModel)

	// Softmax backward
	dScores := softmaxBackward(dAttnWeights, cache.attnWeights, n, n)

	// Scale backward
	scale := 1.0 / math.Sqrt(float64(dModel))
	for i := range dScores {
		dScores[i] *= scale
	}

	// QK^T backward
	// dQ = dScores @ K  [n, dModel]
	dQ := matMul2D(dScores, n, n, cache.K, dModel)
	// dK = dScores^T @ Q  [n, dModel]
	dK := matMulTransA(dScores, n, n, cache.Q, dModel)

	// Projection weight gradients
	// dWq = tokens^T @ dQ  [dToken, dModel]
	dWq := matMulTransA(cache.tokens, n, dToken, dQ, dModel)
	dbQ := colSum(dQ, n, dModel)

	// dWk = tokens^T @ dK  [dToken, dModel]
	dWk := matMulTransA(cache.tokens, n, dToken, dK, dModel)
	dbK := colSum(dK, n, dModel)

	// dWv = tokens^T @ dV  [dToken, dModel]
	dWv := matMulTransA(cache.tokens, n, dToken, dV, dModel)
	dbV := colSum(dV, n, dModel)

	// Apply updates: weight = weight * decay + lr * gradient
	updateWeights := func(w *tensor.Dense, dw []float64) {
		wData := w.Data().([]float64)
		for i := range wData {
			grad := clipGrad(dw[i])
			wData[i] = wData[i]*weightDecay + lr*grad
		}
	}

	updateBias := func(b *tensor.Dense, db []float64) {
		bData := b.Data().([]float64)
		for i := range bData {
			grad := clipGrad(db[i])
			bData[i] += lr * grad
		}
	}

	updateWeights(j.wQ, dWq)
	updateWeights(j.wK, dWk)
	updateWeights(j.wV, dWv)
	updateWeights(j.wO, dWo)

	updateBias(j.bQ, dbQ)
	updateBias(j.bK, dbK)
	updateBias(j.bV, dbV)
	updateBias(j.bO, dbO)
}
