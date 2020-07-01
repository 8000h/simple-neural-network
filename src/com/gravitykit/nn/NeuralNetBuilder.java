package com.gravitykit.nn;

import java.util.ArrayList;

public class NeuralNetBuilder {

    private ArrayList<Integer> layerNodeCounts = new ArrayList<>();
    private ITransferFunction transferFunction = null;
    private double rate                        = 1.0;

    public NeuralNetBuilder addLayer(int nodeCount) {
        this.layerNodeCounts.add(nodeCount);
        return this;
    }

    public NeuralNetBuilder setTransferFunction(ITransferFunction func) {
        this.transferFunction = func;
        return this;
    }

    public NeuralNetBuilder setRate(double rate) {
        this.rate = rate;
        return this;
    }

    public NeuralNet build() {
        NeuralNet nn = new NeuralNet(this.transferFunction, this.rate);

        // Each weight matrix depends on the current layer and next layer.
        // More precisely, the matrix will be NxM where
        // N = next layer node count
        // M = current layer node count
        for (int layerIdx = 0; layerIdx < layerNodeCounts.size() - 1; layerIdx++) {
            int N = layerNodeCounts.get(layerIdx + 1);
            int M = layerNodeCounts.get(layerIdx);
            nn.addLayerWeights(new Matrix(N, M, () -> Math.random()));
            nn.addBias(new Vector(N, () -> Math.random()));
        }

        return nn;
    }

}
