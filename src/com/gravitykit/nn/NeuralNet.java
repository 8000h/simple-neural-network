package com.gravitykit.nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Stack;

public class NeuralNet {

    private ArrayList<Matrix> weights      = new ArrayList<>();
    private ArrayList<Vector> layerInputs  = new ArrayList<>();
    private ArrayList<Vector> layerOutputs = new ArrayList<>();
    private ArrayList<Vector> layerBias    = new ArrayList<>();

    private ITransferFunction transferFunction;

    private double RATE = 0.2;

    public NeuralNet(ITransferFunction transferFunction, double rate) {
        this.transferFunction = transferFunction;
        this.RATE             = rate;
    }

    public void addLayerWeights(Matrix weights) {
        this.weights.add(weights);
    }
    public void addBias(Vector bias) { this.layerBias.add(bias); }
    public ArrayList<Matrix> getWeights() {
        return this.weights;
    }
    public ArrayList<Vector> getBias() {
        return this.layerBias;
    }

    private Matrix calcPartials(Vector deltas, Vector outputs) {
        Matrix partials = new Matrix(deltas.size(), outputs.size(), () -> 0.0);
        for (int outputIdx = 0; outputIdx < outputs.size(); outputIdx++)
            for (int deltaIdx = 0; deltaIdx < deltas.size(); deltaIdx++)
                partials.set(deltaIdx, outputIdx, -RATE * deltas.get(deltaIdx) * outputs.get(outputIdx));

        return partials;
    }

    private ArrayList<Matrix> backPropagate(IErrorFunction errorFunc, Vector desiredValues) {
        ArrayList<Matrix> gradient = new ArrayList<>();

        var outputLayerInput = layerInputs.get(layerInputs.size() - 1);
        var lastLayerOutput  = layerOutputs.get(layerOutputs.size() - 2);

        // First, calculate the output layer deltas.
        var deltas = layerOutputs.get(layerOutputs.size() - 1)
                .copy()
                .map((actual, desired) -> errorFunc.derivative(desired, actual), desiredValues)
                .mul(outputLayerInput
                        .copy()
                        .map(e -> transferFunction.derivative(e)));

        layerBias.get(layerBias.size() - 1).sub(deltas.copy().map(e -> e * RATE));
        gradient.add(calcPartials(deltas, lastLayerOutput));

        // Hidden layers
        var previousDeltas = deltas.copy();
        for (int layer = weights.size() - 1; layer > 0; layer--) {
            deltas = new Vector();
            var weightMatrix    = weights.get(layer);
            var layerInput      = layerInputs.get(layer - 1);
            var prevLayerOutput = layerOutputs.get(layer - 1);

            // Calculate the deltas for this layer.
            // The deltas depend on the next layer's deltas.
            for (int nodeIdx = 0; nodeIdx < layerInput.size(); nodeIdx++) {
                double sum = 0;
                for (int deltaIdx = 0; deltaIdx < previousDeltas.size(); deltaIdx++)
                    sum += weightMatrix.get(deltaIdx, nodeIdx) * previousDeltas.get(deltaIdx);

                double delta = sum * transferFunction.derivative(layerInput.get(nodeIdx));
                deltas.addDouble(delta);
            }

            gradient.add(0, calcPartials(deltas, prevLayerOutput));
            layerBias.get(layer - 1).sub(deltas.copy().map(e -> RATE * e));
            previousDeltas = deltas.copy();
        }

        return gradient;
    }

    public void train(IErrorFunction errorFunc, Vector desiredValues) {

        // This method first calculates the gradient, then
        // applies the values to the weights.
        // Mathematically the gradient is a vector but since
        // the weights are stored in a matrix for each layer,
        // the gradient will also be an array of matrices
        // for easy computation.
        var gradient = this.backPropagate(errorFunc, desiredValues);

        for (int matrixIdx = 0; matrixIdx < gradient.size(); matrixIdx++)
            weights.get(matrixIdx).add(gradient.get(matrixIdx));
    }

    public Vector simulate(Vector input) {
        // The input and output of each layer needs to be saved for training.
        layerInputs.clear();
        layerOutputs.clear();

        layerOutputs.add(input.copy());

        Vector output = null;

        for (int layerIdx = 0; layerIdx < weights.size(); layerIdx++) {
            var weightMatrix = weights.get(layerIdx);
            var biasVector   = layerBias.get(layerIdx);
            input = weightMatrix.multiply(input.copy()).add(biasVector);
            layerInputs.add(input.copy());
            output = input
                    .copy()
                    .map(e -> this.transferFunction.calculate(e));
            layerOutputs.add(output);
            input = output.copy();
        }

        return output.copy();
    }

}
