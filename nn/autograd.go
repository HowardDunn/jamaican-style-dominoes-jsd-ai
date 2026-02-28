package nn

import (
	"fmt"
	"math"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// trainAutograd uses Gorgonia's automatic differentiation to compute gradients
// and update weights. It builds a computation graph mirroring the forward pass
// (attention + MLP), computes MSE loss, and applies gradients to the actual weights.
func (j *JSDNN) trainAutograd(x, y tensor.Tensor, learnRates []float64, mask tensor.Tensor) (float64, error) {
	g := gorgonia.NewGraph()

	// Prepare input — if attention is enabled, run attention forward (outside graph)
	var inputData []float64
	var inputDim int

	if j.useAttention {
		flat := x.Data().([]float64)
		projected, _ := j.attentionForward(flat)
		inputData = projected
		inputDim = len(projected)
	} else {
		inputData = make([]float64, len(x.Data().([]float64)))
		copy(inputData, x.Data().([]float64))
		inputDim = x.Shape()[0]
	}

	// Input node (not trainable)
	xNode := gorgonia.NewVector(g, gorgonia.Float64,
		gorgonia.WithShape(inputDim),
		gorgonia.WithValue(tensor.New(tensor.WithShape(inputDim), tensor.WithBacking(inputData))),
		gorgonia.WithName("x"))

	// Create weight/bias nodes for each hidden layer
	hiddenW := make([]*gorgonia.Node, len(j.hidden))
	hiddenB := make([]*gorgonia.Node, len(j.hidden))

	for i := range j.hidden {
		wData := make([]float64, len(j.hidden[i].Data().([]float64)))
		copy(wData, j.hidden[i].Data().([]float64))
		wShape := j.hidden[i].Shape()

		bData := make([]float64, len(j.bHidden[i].Data().([]float64)))
		copy(bData, j.bHidden[i].Data().([]float64))
		bShape := j.bHidden[i].Shape()

		hiddenW[i] = gorgonia.NewMatrix(g, gorgonia.Float64,
			gorgonia.WithShape(wShape...),
			gorgonia.WithValue(tensor.New(tensor.WithShape(wShape...), tensor.WithBacking(wData))),
			gorgonia.WithName(fmt.Sprintf("hidden_w_%d", i)))
		hiddenB[i] = gorgonia.NewVector(g, gorgonia.Float64,
			gorgonia.WithShape(bShape...),
			gorgonia.WithValue(tensor.New(tensor.WithShape(bShape...), tensor.WithBacking(bData))),
			gorgonia.WithName(fmt.Sprintf("hidden_b_%d", i)))
	}

	// Final layer weights
	fwData := make([]float64, len(j.final.Data().([]float64)))
	copy(fwData, j.final.Data().([]float64))
	fwShape := j.final.Shape()

	fbData := make([]float64, len(j.bFinal.Data().([]float64)))
	copy(fbData, j.bFinal.Data().([]float64))
	fbShape := j.bFinal.Shape()

	finalW := gorgonia.NewMatrix(g, gorgonia.Float64,
		gorgonia.WithShape(fwShape...),
		gorgonia.WithValue(tensor.New(tensor.WithShape(fwShape...), tensor.WithBacking(fwData))),
		gorgonia.WithName("final_w"))
	finalB := gorgonia.NewVector(g, gorgonia.Float64,
		gorgonia.WithShape(fbShape...),
		gorgonia.WithValue(tensor.New(tensor.WithShape(fbShape...), tensor.WithBacking(fbData))),
		gorgonia.WithName("final_b"))

	// Build forward pass: h = relu(W0 @ x + b0)
	var h *gorgonia.Node
	var err error

	h, err = gorgonia.Mul(hiddenW[0], xNode)
	if err != nil {
		return 0, fmt.Errorf("autograd hidden[0] mul: %w", err)
	}
	h, err = gorgonia.Add(h, hiddenB[0])
	if err != nil {
		return 0, fmt.Errorf("autograd hidden[0] add: %w", err)
	}
	h, err = gorgonia.Rectify(h)
	if err != nil {
		return 0, fmt.Errorf("autograd hidden[0] relu: %w", err)
	}

	for i := 1; i < len(j.hidden); i++ {
		h, err = gorgonia.Mul(hiddenW[i], h)
		if err != nil {
			return 0, fmt.Errorf("autograd hidden[%d] mul: %w", i, err)
		}
		h, err = gorgonia.Add(h, hiddenB[i])
		if err != nil {
			return 0, fmt.Errorf("autograd hidden[%d] add: %w", i, err)
		}
		h, err = gorgonia.Rectify(h)
		if err != nil {
			return 0, fmt.Errorf("autograd hidden[%d] relu: %w", i, err)
		}
	}

	// Final: out = W_final @ h + b_final
	out, err := gorgonia.Mul(finalW, h)
	if err != nil {
		return 0, fmt.Errorf("autograd final mul: %w", err)
	}
	out, err = gorgonia.Add(out, finalB)
	if err != nil {
		return 0, fmt.Errorf("autograd final add: %w", err)
	}

	if j.OutputActivation != "linear" {
		out, err = gorgonia.Rectify(out)
		if err != nil {
			return 0, fmt.Errorf("autograd output relu: %w", err)
		}
	}

	// Target node
	yData := make([]float64, len(y.Data().([]float64)))
	copy(yData, y.Data().([]float64))
	yNode := gorgonia.NewVector(g, gorgonia.Float64,
		gorgonia.WithShape(y.Shape()[0]),
		gorgonia.WithValue(tensor.New(tensor.WithShape(y.Shape()[0]), tensor.WithBacking(yData))),
		gorgonia.WithName("y"))

	// diff = target - prediction
	diff, err := gorgonia.Sub(yNode, out)
	if err != nil {
		return 0, fmt.Errorf("autograd sub: %w", err)
	}

	// Apply mask if provided
	if mask != nil {
		maskData := make([]float64, len(mask.Data().([]float64)))
		copy(maskData, mask.Data().([]float64))
		maskNode := gorgonia.NewVector(g, gorgonia.Float64,
			gorgonia.WithShape(mask.Shape()[0]),
			gorgonia.WithValue(tensor.New(tensor.WithShape(mask.Shape()[0]), tensor.WithBacking(maskData))),
			gorgonia.WithName("mask"))
		diff, err = gorgonia.HadamardProd(diff, maskNode)
		if err != nil {
			return 0, fmt.Errorf("autograd mask: %w", err)
		}
	}

	// Loss = sum(diff^2)
	sq, err := gorgonia.Square(diff)
	if err != nil {
		return 0, fmt.Errorf("autograd square: %w", err)
	}
	loss, err := gorgonia.Sum(sq)
	if err != nil {
		return 0, fmt.Errorf("autograd sum: %w", err)
	}

	// Request gradients for all trainable parameters
	trainableNodes := make(gorgonia.Nodes, 0, 2*len(j.hidden)+2)
	for i := range hiddenW {
		trainableNodes = append(trainableNodes, hiddenW[i], hiddenB[i])
	}
	trainableNodes = append(trainableNodes, finalW, finalB)

	_, err = gorgonia.Grad(loss, trainableNodes...)
	if err != nil {
		return 0, fmt.Errorf("autograd grad: %w", err)
	}

	// Run the computation graph
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()
	if err := machine.RunAll(); err != nil {
		return 0, fmt.Errorf("autograd run: %w", err)
	}

	// Read cost
	costVal := loss.Value().Data().(float64)

	// Apply gradients to actual weights
	// Gorgonia computes dLoss/dW. Since loss = sum((y - pred)^2),
	// gradient descent: w = w * decay - lr * dLoss/dW
	applyGradWeight := func(actual *tensor.Dense, node *gorgonia.Node, lr float64) error {
		gradVal, err := node.Grad()
		if err != nil {
			return err
		}
		gradData := gradVal.Data().([]float64)
		actualData := actual.Data().([]float64)
		for i := range actualData {
			gv := clipGrad(gradData[i])
			actualData[i] = actualData[i]*weightDecay - lr*gv
		}
		return nil
	}

	applyGradBiasFunc := func(actual *tensor.Dense, node *gorgonia.Node, lr float64) error {
		gradVal, err := node.Grad()
		if err != nil {
			return err
		}
		gradData := gradVal.Data().([]float64)
		actualData := actual.Data().([]float64)
		for i := range actualData {
			gv := clipGrad(gradData[i])
			actualData[i] -= lr * gv
		}
		return nil
	}

	// Apply hidden layer gradients
	for i := range j.hidden {
		lr := j.getLearnRate(learnRates, i)
		if err := applyGradWeight(j.hidden[i], hiddenW[i], lr); err != nil {
			return 0, fmt.Errorf("autograd apply hidden[%d] weight grad: %w", i, err)
		}
		if err := applyGradBiasFunc(j.bHidden[i], hiddenB[i], lr); err != nil {
			return 0, fmt.Errorf("autograd apply hidden[%d] bias grad: %w", i, err)
		}
	}

	// Apply final layer gradients
	finalLR := j.getLearnRate(learnRates, len(j.hidden))
	if err := applyGradWeight(j.final, finalW, finalLR); err != nil {
		return 0, fmt.Errorf("autograd apply final weight grad: %w", err)
	}
	if err := applyGradBiasFunc(j.bFinal, finalB, finalLR); err != nil {
		return 0, fmt.Errorf("autograd apply final bias grad: %w", err)
	}

	return math.Sqrt(costVal), nil
}
