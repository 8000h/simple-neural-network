package com.gravitykit.nn;

public class DiffSquareError implements IErrorFunction {

    @Override
    public double findError(double desired, double actual) {
        return Math.pow((desired - actual), 2) / 2.0;
    }

    @Override
    public double derivative(double desired, double actual) {
        return actual - desired;
    }
}
