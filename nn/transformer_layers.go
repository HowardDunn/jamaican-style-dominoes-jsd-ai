package nn

import (
	"math"
)

// transformerLayer holds weights for one transformer layer (multi-head attention + FFN).
type transformerLayer struct {
	nHeads, dModel, dHead, dFF int

	// Multi-head attention weights
	wQ, wK, wV, wO *denseMatrix // [dModel x dModel]
	bQ, bK, bV, bO []float64   // [dModel]

	// Layer norm 1 (post-attention)
	ln1Gamma, ln1Beta []float64 // [dModel]

	// Feed-forward network
	ff1W []float64 // [dFF x dModel]
	ff1B []float64 // [dFF]
	ff2W []float64 // [dModel x dFF]
	ff2B []float64 // [dModel]

	// Layer norm 2 (post-FFN)
	ln2Gamma, ln2Beta []float64 // [dModel]
}

// denseMatrix is a simple row-major [rows x cols] matrix.
type denseMatrix struct {
	data     []float64
	rows, cols int
}

func newDenseMatrix(rows, cols int) *denseMatrix {
	return &denseMatrix{data: make([]float64, rows*cols), rows: rows, cols: cols}
}

// layerNormForward applies layer normalization to each row of x [seqLen x dModel].
// Returns normalized output and cache needed for backward pass.
func layerNormForward(x []float64, gamma, beta []float64, seqLen, dModel int) ([]float64, *layerNormCache) {
	out := make([]float64, len(x))
	means := make([]float64, seqLen)
	invStds := make([]float64, seqLen)
	xNorm := make([]float64, len(x))
	eps := 1e-5

	for i := 0; i < seqLen; i++ {
		base := i * dModel
		// Compute mean
		mean := 0.0
		for j := 0; j < dModel; j++ {
			mean += x[base+j]
		}
		mean /= float64(dModel)
		means[i] = mean

		// Compute variance
		variance := 0.0
		for j := 0; j < dModel; j++ {
			diff := x[base+j] - mean
			variance += diff * diff
		}
		variance /= float64(dModel)
		invStd := 1.0 / math.Sqrt(variance+eps)
		invStds[i] = invStd

		// Normalize and scale
		for j := 0; j < dModel; j++ {
			xNorm[base+j] = (x[base+j] - mean) * invStd
			out[base+j] = xNorm[base+j]*gamma[j] + beta[j]
		}
	}

	return out, &layerNormCache{
		xNorm:   xNorm,
		means:   means,
		invStds: invStds,
	}
}

type layerNormCache struct {
	xNorm   []float64 // [seqLen x dModel] normalized input
	means   []float64 // [seqLen]
	invStds []float64 // [seqLen]
}

// layerNormBackward computes gradients through layer normalization.
// dOut is [seqLen x dModel], returns dx [seqLen x dModel], dGamma [dModel], dBeta [dModel].
func layerNormBackward(dOut []float64, cache *layerNormCache, gamma []float64, seqLen, dModel int) ([]float64, []float64, []float64) {
	dx := make([]float64, len(dOut))
	dGamma := make([]float64, dModel)
	dBeta := make([]float64, dModel)
	N := float64(dModel)

	for i := 0; i < seqLen; i++ {
		base := i * dModel
		invStd := cache.invStds[i]

		// dBeta and dGamma
		for j := 0; j < dModel; j++ {
			dBeta[j] += dOut[base+j]
			dGamma[j] += dOut[base+j] * cache.xNorm[base+j]
		}

		// dx through layer norm
		// dxNorm = dOut * gamma
		// dx = invStd * (dxNorm - mean(dxNorm) - xNorm * mean(dxNorm * xNorm)) / 1
		dxNormMean := 0.0
		dxNormXNormMean := 0.0
		for j := 0; j < dModel; j++ {
			dxNorm := dOut[base+j] * gamma[j]
			dxNormMean += dxNorm
			dxNormXNormMean += dxNorm * cache.xNorm[base+j]
		}
		dxNormMean /= N
		dxNormXNormMean /= N

		for j := 0; j < dModel; j++ {
			dxNorm := dOut[base+j] * gamma[j]
			dx[base+j] = invStd * (dxNorm - dxNormMean - cache.xNorm[base+j]*dxNormXNormMean)
		}
	}

	return dx, dGamma, dBeta
}

// buildCausalMask creates a causal mask [seqLen x seqLen].
// mask[i][j] = 0 if j <= i (allowed), -inf if j > i (blocked).
func buildCausalMask(seqLen int) []float64 {
	mask := make([]float64, seqLen*seqLen)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j > i {
				mask[i*seqLen+j] = math.Inf(-1)
			}
		}
	}
	return mask
}

// multiHeadAttentionCache holds intermediate values for backward pass.
type multiHeadAttentionCache struct {
	input       []float64 // [seqLen x dModel]
	Q, K, V     []float64 // [seqLen x dModel]
	attnWeights []float64 // [nHeads x seqLen x seqLen] per-head softmax outputs
	attnOut     []float64 // [seqLen x dModel] concatenated head outputs before output proj
}

// multiHeadAttentionForward computes multi-head scaled dot-product attention.
// x is [seqLen x dModel], mask is [seqLen x seqLen] causal mask.
// Returns output [seqLen x dModel] and cache for backward.
func multiHeadAttentionForward(x []float64, layer *transformerLayer, mask []float64, seqLen int) ([]float64, *multiHeadAttentionCache) {
	dModel := layer.dModel
	nHeads := layer.nHeads
	dHead := layer.dHead

	// Q, K, V projections: x @ W^T + b
	Q := matMul2D(x, seqLen, dModel, layer.wQ.data, dModel)
	addBias(Q, layer.bQ, seqLen, dModel)

	K := matMul2D(x, seqLen, dModel, layer.wK.data, dModel)
	addBias(K, layer.bK, seqLen, dModel)

	V := matMul2D(x, seqLen, dModel, layer.wV.data, dModel)
	addBias(V, layer.bV, seqLen, dModel)

	// Per-head attention
	scale := 1.0 / math.Sqrt(float64(dHead))
	allAttnWeights := make([]float64, nHeads*seqLen*seqLen)
	headOuts := make([]float64, seqLen*dModel) // concatenated

	for h := 0; h < nHeads; h++ {
		hOff := h * dHead

		// Extract head slices for Q, K, V
		qH := extractHead(Q, seqLen, dModel, hOff, dHead)
		kH := extractHead(K, seqLen, dModel, hOff, dHead)
		vH := extractHead(V, seqLen, dModel, hOff, dHead)

		// scores = qH @ kH^T * scale → [seqLen x seqLen]
		scores := matMulTransB(qH, seqLen, dHead, kH, seqLen)
		for i := range scores {
			scores[i] = scores[i]*scale + mask[i]
		}

		// softmax per row
		attnW := rowSoftmax(scores, seqLen, seqLen)
		copy(allAttnWeights[h*seqLen*seqLen:], attnW)

		// attnOut = attnW @ vH → [seqLen x dHead]
		headOut := matMul2D(attnW, seqLen, seqLen, vH, dHead)

		// Write back into concatenated output
		for i := 0; i < seqLen; i++ {
			for j := 0; j < dHead; j++ {
				headOuts[i*dModel+hOff+j] = headOut[i*dHead+j]
			}
		}
	}

	// Output projection: headOuts @ wO + bO
	out := matMul2D(headOuts, seqLen, dModel, layer.wO.data, dModel)
	addBias(out, layer.bO, seqLen, dModel)

	cache := &multiHeadAttentionCache{
		input:       x,
		Q:           Q,
		K:           K,
		V:           V,
		attnWeights: allAttnWeights,
		attnOut:     headOuts,
	}

	return out, cache
}

// extractHead extracts head columns from a [seqLen x dModel] matrix → [seqLen x dHead].
func extractHead(x []float64, seqLen, dModel, offset, dHead int) []float64 {
	out := make([]float64, seqLen*dHead)
	for i := 0; i < seqLen; i++ {
		copy(out[i*dHead:(i+1)*dHead], x[i*dModel+offset:i*dModel+offset+dHead])
	}
	return out
}

// scatterHead writes head values back into a [seqLen x dModel] buffer.
func scatterHead(dst []float64, src []float64, seqLen, dModel, offset, dHead int) {
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dHead; j++ {
			dst[i*dModel+offset+j] += src[i*dHead+j]
		}
	}
}

// multiHeadAttentionBackward computes gradients through the multi-head attention.
// dOut is [seqLen x dModel]. Returns dx [seqLen x dModel] and weight/bias gradients.
func multiHeadAttentionBackward(dOut []float64, cache *multiHeadAttentionCache, layer *transformerLayer, mask []float64, seqLen int) (
	dx []float64, dwQ, dwK, dwV, dwO []float64, dbQ, dbK, dbV, dbO []float64) {

	dModel := layer.dModel
	nHeads := layer.nHeads
	dHead := layer.dHead

	// Output projection backward
	// dHeadOuts = dOut @ wO^T
	dHeadOuts := matMulTransB(dOut, seqLen, dModel, layer.wO.data, dModel)
	// dwO = headOuts^T @ dOut
	dwO = matMulTransA(cache.attnOut, seqLen, dModel, dOut, dModel)
	dbO = colSum(dOut, seqLen, dModel)

	// Per-head backward
	dQ := make([]float64, seqLen*dModel)
	dK := make([]float64, seqLen*dModel)
	dV := make([]float64, seqLen*dModel)
	scale := 1.0 / math.Sqrt(float64(dHead))

	for h := 0; h < nHeads; h++ {
		hOff := h * dHead

		// Extract per-head dOut
		dHeadOut := extractHead(dHeadOuts, seqLen, dModel, hOff, dHead)
		attnW := cache.attnWeights[h*seqLen*seqLen : (h+1)*seqLen*seqLen]
		qH := extractHead(cache.Q, seqLen, dModel, hOff, dHead)
		kH := extractHead(cache.K, seqLen, dModel, hOff, dHead)
		vH := extractHead(cache.V, seqLen, dModel, hOff, dHead)

		// dAttnW = dHeadOut @ vH^T → [seqLen x seqLen]
		dAttnW := matMulTransB(dHeadOut, seqLen, dHead, vH, seqLen)
		// dVH = attnW^T @ dHeadOut → [seqLen x dHead]
		dVH := matMulTransA(attnW, seqLen, seqLen, dHeadOut, dHead)

		// Softmax backward
		dScores := softmaxBackward(dAttnW, attnW, seqLen, seqLen)

		// Scale backward
		for i := range dScores {
			dScores[i] *= scale
		}

		// QK^T backward
		// dQH = dScores @ kH → [seqLen x dHead]
		dQH := matMul2D(dScores, seqLen, seqLen, kH, dHead)
		// dKH = dScores^T @ qH → [seqLen x dHead]
		dKH := matMulTransA(dScores, seqLen, seqLen, qH, dHead)

		// Scatter back into full dQ, dK, dV
		scatterHead(dQ, dQH, seqLen, dModel, hOff, dHead)
		scatterHead(dK, dKH, seqLen, dModel, hOff, dHead)
		scatterHead(dV, dVH, seqLen, dModel, hOff, dHead)
	}

	// Projection backward: Q = x @ wQ + bQ
	// dwQ = x^T @ dQ
	dwQ = matMulTransA(cache.input, seqLen, dModel, dQ, dModel)
	dbQ = colSum(dQ, seqLen, dModel)
	// dwK = x^T @ dK
	dwK = matMulTransA(cache.input, seqLen, dModel, dK, dModel)
	dbK = colSum(dK, seqLen, dModel)
	// dwV = x^T @ dV
	dwV = matMulTransA(cache.input, seqLen, dModel, dV, dModel)
	dbV = colSum(dV, seqLen, dModel)

	// dx = dQ @ wQ^T + dK @ wK^T + dV @ wV^T
	dx = matMulTransB(dQ, seqLen, dModel, layer.wQ.data, dModel)
	dxK := matMulTransB(dK, seqLen, dModel, layer.wK.data, dModel)
	dxV := matMulTransB(dV, seqLen, dModel, layer.wV.data, dModel)
	for i := range dx {
		dx[i] += dxK[i] + dxV[i]
	}

	return
}

// ffnForward computes the feed-forward network: ReLU(x @ ff1W^T + ff1B) @ ff2W^T + ff2B.
// x is [seqLen x dModel]. Returns output [seqLen x dModel] and cache.
func ffnForward(x []float64, layer *transformerLayer, seqLen int) ([]float64, *ffnCache) {
	dModel := layer.dModel
	dFF := layer.dFF

	// Hidden: x @ ff1W^T + ff1B → [seqLen x dFF]
	// ff1W is [dFF x dModel], so x @ ff1W^T = matMulTransB(x, seqLen, dModel, ff1W, dFF)
	hidden := matMulTransB(x, seqLen, dModel, layer.ff1W, dFF)
	addBias(hidden, layer.ff1B, seqLen, dFF)

	// ReLU
	preReLU := make([]float64, len(hidden))
	copy(preReLU, hidden)
	for i := range hidden {
		hidden[i] = relu(hidden[i])
	}

	// Output: hidden @ ff2W^T + ff2B → [seqLen x dModel]
	// ff2W is [dModel x dFF], so hidden @ ff2W^T = matMulTransB(hidden, seqLen, dFF, ff2W, dModel)
	out := matMulTransB(hidden, seqLen, dFF, layer.ff2W, dModel)
	addBias(out, layer.ff2B, seqLen, dModel)

	cache := &ffnCache{
		input:   x,
		preReLU: preReLU,
		hidden:  hidden,
	}

	return out, cache
}

type ffnCache struct {
	input   []float64 // [seqLen x dModel]
	preReLU []float64 // [seqLen x dFF] before ReLU
	hidden  []float64 // [seqLen x dFF] after ReLU
}

// ffnBackward computes gradients through the FFN.
// dOut is [seqLen x dModel]. Returns dx and weight/bias gradients.
func ffnBackward(dOut []float64, cache *ffnCache, layer *transformerLayer, seqLen int) (
	dx []float64, dff1W, dff1B, dff2W, dff2B []float64) {

	dModel := layer.dModel
	dFF := layer.dFF

	// ff2 backward: out = hidden @ ff2W^T + ff2B
	// dHidden = dOut @ ff2W → [seqLen x dFF]
	// ff2W is [dModel x dFF]
	dHidden := matMul2D(dOut, seqLen, dModel, layer.ff2W, dFF)
	// dff2W = dOut^T @ hidden → [dModel x dFF]
	dff2W = matMulTransA(dOut, seqLen, dModel, cache.hidden, dFF)
	dff2B = colSum(dOut, seqLen, dModel)

	// ReLU backward
	for i := range dHidden {
		if cache.preReLU[i] <= 0 {
			dHidden[i] = 0
		}
	}

	// ff1 backward: hidden = x @ ff1W^T + ff1B
	// dx = dHidden @ ff1W → [seqLen x dModel]
	// ff1W is [dFF x dModel]
	dx = matMul2D(dHidden, seqLen, dFF, layer.ff1W, dModel)
	// dff1W = dHidden^T @ x → [dFF x dModel]
	dff1W = matMulTransA(dHidden, seqLen, dFF, cache.input, dModel)
	dff1B = colSum(dHidden, seqLen, dFF)

	return
}

// transformerLayerCache holds all caches for one layer's forward pass.
type transformerLayerCache struct {
	// Pre-attention residual input
	residualAttn []float64
	attnCache    *multiHeadAttentionCache
	lnCache1     *layerNormCache
	// Pre-FFN residual input
	residualFFN []float64
	ffnCache    *ffnCache
	lnCache2    *layerNormCache
}

// transformerLayerForward runs one transformer layer.
// x is [seqLen x dModel], mask is [seqLen x seqLen].
// Post-norm: x = LN1(x + MHA(x)); x = LN2(x + FFN(x))
func transformerLayerForward(x []float64, layer *transformerLayer, mask []float64, seqLen int) ([]float64, *transformerLayerCache) {
	dModel := layer.dModel

	// Multi-head attention + residual
	attnOut, attnCache := multiHeadAttentionForward(x, layer, mask, seqLen)
	residualAttn := make([]float64, len(x))
	for i := range residualAttn {
		residualAttn[i] = x[i] + attnOut[i]
	}
	ln1Out, lnCache1 := layerNormForward(residualAttn, layer.ln1Gamma, layer.ln1Beta, seqLen, dModel)

	// FFN + residual
	ffnOut, ffnCache := ffnForward(ln1Out, layer, seqLen)
	residualFFN := make([]float64, len(ln1Out))
	for i := range residualFFN {
		residualFFN[i] = ln1Out[i] + ffnOut[i]
	}
	ln2Out, lnCache2 := layerNormForward(residualFFN, layer.ln2Gamma, layer.ln2Beta, seqLen, dModel)

	cache := &transformerLayerCache{
		residualAttn: residualAttn,
		attnCache:    attnCache,
		lnCache1:     lnCache1,
		residualFFN:  residualFFN,
		ffnCache:     ffnCache,
		lnCache2:     lnCache2,
	}

	return ln2Out, cache
}

// transformerLayerGrads holds all gradients for one layer.
type transformerLayerGrads struct {
	dwQ, dwK, dwV, dwO []float64
	dbQ, dbK, dbV, dbO []float64
	dln1Gamma, dln1Beta []float64
	dff1W, dff1B        []float64
	dff2W, dff2B        []float64
	dln2Gamma, dln2Beta []float64
}

// transformerLayerBackward computes gradients through one transformer layer.
// dOut is [seqLen x dModel]. Returns dx and all weight gradients.
func transformerLayerBackward(dOut []float64, cache *transformerLayerCache, layer *transformerLayer, mask []float64, seqLen int) ([]float64, *transformerLayerGrads) {
	dModel := layer.dModel

	// LN2 backward
	dResidualFFN, dln2Gamma, dln2Beta := layerNormBackward(dOut, cache.lnCache2, layer.ln2Gamma, seqLen, dModel)

	// FFN backward (input to FFN was LN1 output, which is the input stored in ffnCache)
	dFFN, dff1W, dff1B, dff2W, dff2B := ffnBackward(dResidualFFN, cache.ffnCache, layer, seqLen)

	// Residual connection: dLN1Out = dFFN + dResidualFFN
	dLN1Out := make([]float64, len(dFFN))
	for i := range dLN1Out {
		dLN1Out[i] = dFFN[i] + dResidualFFN[i]
	}

	// LN1 backward
	dResidualAttn, dln1Gamma, dln1Beta := layerNormBackward(dLN1Out, cache.lnCache1, layer.ln1Gamma, seqLen, dModel)

	// MHA backward
	dAttn, dwQ, dwK, dwV, dwO, dbQ, dbK, dbV, dbO := multiHeadAttentionBackward(dResidualAttn, cache.attnCache, layer, mask, seqLen)

	// Residual connection: dx = dAttn + dResidualAttn
	dx := make([]float64, len(dAttn))
	for i := range dx {
		dx[i] = dAttn[i] + dResidualAttn[i]
	}

	grads := &transformerLayerGrads{
		dwQ: dwQ, dwK: dwK, dwV: dwV, dwO: dwO,
		dbQ: dbQ, dbK: dbK, dbV: dbV, dbO: dbO,
		dln1Gamma: dln1Gamma, dln1Beta: dln1Beta,
		dff1W: dff1W, dff1B: dff1B,
		dff2W: dff2W, dff2B: dff2B,
		dln2Gamma: dln2Gamma, dln2Beta: dln2Beta,
	}

	return dx, grads
}
