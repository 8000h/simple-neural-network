package com.gravitykit.nn;

import java.lang.reflect.Array;
import java.util.ArrayList;

public class NNTests {

    private static void train(IErrorFunction error, int epochs, NeuralNet nn, ArrayList<Sample> samples) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            Sample sample = samples.get(epoch % samples.size());
            nn.simulate(sample.getInput());
            nn.train(error, sample.getDesired());
        }
    }

    private static void printResult(NeuralNet nn, Vector input) {
        System.out.print(input);
        System.out.print(" -> nn -> ");
        System.out.println(nn.simulate(input));
    }

    // ====================================================================================
    // == testXOR =========================================================================
    // ====================================================================================

    public static void testXOR() {

        // XOR works well with a small learning rate, tanh(...) as the activation function,
        // and over 1000 epochs. If SigmoidTransfer is used then increase the rate and epochs.

        System.out.println("testXOR");

        NeuralNetBuilder builder = new NeuralNetBuilder();
        builder
                .setRate(0.2)
                .setTransferFunction(new TanHTransfer())
                .addLayer(2)
                .addLayer(2)
                .addLayer(1);

        NeuralNet nn = builder.build();

        ArrayList<Sample> samples = new ArrayList<>();
        samples.add(new Sample(
                new Double[] {0.0, 0.0},
                new Double[] {0.0})
        );

        samples.add(new Sample(
                new Double[] {1.0, 0.0},
                new Double[] {1.0})
        );

        samples.add(new Sample(
                new Double[] {0.0, 1.0},
                new Double[] {1.0})
        );

        samples.add(new Sample(
                new Double[] {1.0, 1.0},
                new Double[] {0.0})
        );

        train(new DiffSquareError(), 5000, nn, samples);

        for (var sample : samples)
            printResult(nn, sample.getInput());

        System.out.println();
    }

    // ====================================================================================
    // == testAND =========================================================================
    // ====================================================================================

    public static void testAND() {
        System.out.println("testAND");

        NeuralNetBuilder builder = new NeuralNetBuilder();
        builder
                .setRate(1)
                .setTransferFunction(new SigmoidTransfer())
                .addLayer(2)
                .addLayer(1);

        NeuralNet nn = builder.build();

        ArrayList<Sample> samples = new ArrayList<>();
        samples.add(new Sample(
                new Double[] {0.0, 0.0},
                new Double[] {0.0})
        );

        samples.add(new Sample(
                new Double[] {1.0, 0.0},
                new Double[] {0.0})
        );

        samples.add(new Sample(
                new Double[] {0.0, 1.0},
                new Double[] {0.0})
        );

        samples.add(new Sample(
                new Double[] {1.0, 1.0},
                new Double[] {1.0})
        );

        train(new DiffSquareError(), 10_000, nn, samples);

        for (var sample : samples)
            printResult(nn, sample.getInput());

        System.out.println();
    }

    // ====================================================================================
    // == testOR ==========================================================================
    // ====================================================================================

    public static void testOR() {
        System.out.println("testOR");

        NeuralNetBuilder builder = new NeuralNetBuilder();
        builder
                .setRate(0.2)
                .setTransferFunction(new TanHTransfer())
                .addLayer(2)
                .addLayer(1);

        NeuralNet nn = builder.build();

        ArrayList<Sample> samples = new ArrayList<>();
        samples.add(new Sample(
                new Double[] {0.0, 0.0},
                new Double[] {0.0})
        );

        samples.add(new Sample(
                new Double[] {1.0, 0.0},
                new Double[] {1.0})
        );

        samples.add(new Sample(
                new Double[] {0.0, 1.0},
                new Double[] {1.0})
        );

        samples.add(new Sample(
                new Double[] {1.0, 1.0},
                new Double[] {1.0})
        );

        train(new DiffSquareError(), 5000, nn, samples);

        for (var sample : samples)
            printResult(nn, sample.getInput());

        System.out.println();
    }

}
