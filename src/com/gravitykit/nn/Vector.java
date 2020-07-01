package com.gravitykit.nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.function.*;

/*

    Vector is a glorified wrapper for ArrayList<Double>.
    The wrapping provides a clean way to perform arithmetic
    on each element.

    Example:

        Vector vec1 = new Vector(new Double[] {
            1.0, 2.0, 3.0
        });

        Vector vec2 = new Vector(new Double[] {
            4.0, 5.0, 6.0
        });

        vec1.dot(vec2); // 32.0
        vec1.add(vec2).map(e -> e * e); // vec1 = {25.0, 49.0, 81.0}

    By default, the operations on Vector change its contents.
    To prevent this, copy() must be called before the operation is
    performed.

    Vector added = vec1.copy().add(vec2); // vec1 doesn't change

 */

public class Vector {

    private static final String ERR_SIZE_MISMATCH = "Vector size mismatch";

    public static double dot(double[] left, Vector vecRight) {
        var right  = vecRight.vector;
        double sum = 0;

        if (left.length != right.size())
            throw new ArithmeticException(ERR_SIZE_MISMATCH);

        for (int idx = 0; idx < left.length; idx++)
            sum += left[idx] * right.get(idx);

        return sum;
    }

    public static double dot(Vector vecLeft, Vector vecRight) {
        var left   = vecLeft.vector;
        var right  = vecRight.vector;
        double sum = 0;

        if (left.size() != right.size())
            throw new ArithmeticException(ERR_SIZE_MISMATCH);

        for (int idx = 0; idx < left.size(); idx++)
            sum += left.get(idx) * right.get(idx);

        return sum;
    }

    public static Vector apply(UnaryOperator<Double> operator, Vector vec) {
        return vec.copy().map(operator);
    }

    public int size() {
        return vector.size();
    }

    public Vector sub(Vector other) {
        return this.map((a, b) -> a - b, other);
    }

    public Vector mul(Vector other) {
        return this.map((a, b) -> a * b, other);
    }

    public Vector() {}

    public Vector(Double[] values) {
        Collections.addAll(vector, values);
    }

    public Vector(int size, Supplier<Double> supplier) {
        for (int idx = 0; idx < size; idx++)
            vector.add(supplier.get());
    }

    public static Vector applyTransfer(ITransferFunction func, Vector vec) {
        var newVec = new Vector();
        for (var element : vec.vector)
            newVec.addDouble(func.calculate(element));
        return newVec;
    }

    public double get(int index) {
        return vector.get(index);
    }

    public Vector map(UnaryOperator<Double> operator) {
        this.vector.replaceAll(operator);
        return this;
    }

    public Vector map(BinaryOperator<Double> operator, Vector other) {
        if (this.vector.size() != other.size())
            throw new ArithmeticException(ERR_SIZE_MISMATCH);

        for (int idx = 0; idx < vector.size(); idx++)
            vector.set(idx, operator.apply(vector.get(idx), other.get(idx)));
        return this;
    }

    public Vector add(Vector other) {
        if (this.vector.size() != other.size())
            throw new ArithmeticException(ERR_SIZE_MISMATCH);
        return this.map((a, b) -> a + b, other);
    }

    private ArrayList<Double> vector = new ArrayList<>();

    public void addDouble(double d) {
        this.vector.add(d);
    }

    public ArrayList<Double> getVector() {
        return vector;
    }

    @Override
    public String toString() {
        return vector.toString();
    }

    public Vector copy() {
        Vector cloned = new Vector();
        for (Double d : this.vector)
            cloned.addDouble(d);
        return cloned;
    }
}
