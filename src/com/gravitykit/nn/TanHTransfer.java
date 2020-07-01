package com.gravitykit.nn;

public class TanHTransfer implements ITransferFunction {

    @Override
    public double calculate(double input) {
        return Math.tanh(input);
        // return (Math.exp(input) - Math.exp(-input)) / (Math.exp(input) + Math.exp(-input));
    }

    @Override
    public double derivative(double atValue) {
        return 1 - Math.pow(Math.tanh(atValue), 2);
    }
}
