package com.gravitykit.nn;

public interface ITransferFunction {

    public double calculate(double input);
    public double derivative(double atValue);

}
