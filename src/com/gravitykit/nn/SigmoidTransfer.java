package com.gravitykit.nn;

public class SigmoidTransfer implements ITransferFunction {

    @Override
    public double calculate(double input) {
        return 1.0 / (1.0 + Math.exp(-input));
    }

    @Override
    public double derivative(double atValue) {
        return calculate(atValue) * (1 - calculate(atValue));
    }
}
