package com.gravitykit.nn;

public interface IErrorFunction {

    public double findError(double desired, double actual);
    public double derivative(double desired, double actual);

}
