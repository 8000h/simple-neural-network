package com.gravitykit.nn;

import java.util.ArrayList;
import java.util.function.Supplier;

public class Matrix {

    public static Matrix createScaleMatrix(ArrayList<Double> diagonals) {
        Matrix newMatrix = new Matrix(diagonals.size(), diagonals.size(), () -> 0.0);

        for (int idx = 0; idx < diagonals.size(); idx++)
            newMatrix.set(idx, idx, diagonals.get(idx));

        return newMatrix;
    }

    private int rows;
    private int columns;
    private double[][] matrix;

    public Matrix(int rows, int columns, Supplier<Double> supplier) {
        this.rows  = rows;
        this.columns = columns;
        this.matrix = new double[rows][columns];

        for (int rowIdx = 0; rowIdx < rows; rowIdx++)
            for (int colIdx = 0; colIdx < columns; colIdx++)
                this.matrix[rowIdx][colIdx] = supplier.get();
    }

    public Matrix(Double[][] values) {
        this.rows    = values.length;
        this.columns = values[0].length;

        this.matrix = new double[rows][columns];

        for (int rowIdx = 0; rowIdx < rows; rowIdx++)
            for (int colIdx = 0; colIdx < columns; colIdx++)
                this.matrix[rowIdx][colIdx] = values[rowIdx][colIdx];
    }

    public double[][] getMatrix() {
        return this.matrix;
    }

    public double[] getRow(int rowIdx) {
        return matrix[rowIdx];
    }

    public double get(int rowIdx, int colIdx) {
        return this.matrix[rowIdx][colIdx];
    }

    public int getRows() {
        return this.rows;
    }

    public int getCols() {
        return this.columns;
    }

    public void add(Matrix other) {
        if (this.rows != other.rows || this.columns != other.columns)
            throw new ArithmeticException();

        for (int rowIdx = 0; rowIdx < rows; rowIdx++)
            for (int colIdx = 0; colIdx < columns; colIdx++)
                this.matrix[rowIdx][colIdx] += other.get(rowIdx, colIdx);
    }

    public void sub(Matrix other) {
        if (this.rows != other.rows || this.columns != other.columns)
            throw new ArithmeticException();

        for (int rowIdx = 0; rowIdx < rows; rowIdx++)
            for (int colIdx = 0; colIdx < columns; colIdx++)
                this.matrix[rowIdx][colIdx] -= other.get(rowIdx, colIdx);
    }

    public void set(int rowIdx, int colIdx, double value) {
        this.matrix[rowIdx][colIdx] = value;
    }

    public Matrix multiply(Matrix other) {
        Matrix newMatrix = new Matrix(this.rows, other.columns, () -> 0.0);

        for (int rowIdx = 0; rowIdx < this.rows; rowIdx++)
            for (int otherColIdx = 0; otherColIdx < other.columns; otherColIdx++) {
                double sum = 0;
                for (int elementIdx = 0; elementIdx < this.columns; elementIdx++)
                    sum += this.get(rowIdx, elementIdx) * other.get(elementIdx, otherColIdx);

                newMatrix.set(rowIdx, otherColIdx, sum);
            }

        return newMatrix;
    }

    public Vector multiply(Vector rightVec) {
        if (this.columns != rightVec.size())
            throw new ArithmeticException();

        Vector newVec = new Vector();

        for (int row = 0; row < this.rows; row++)
            newVec.addDouble(Vector.dot(matrix[row], rightVec));

        return newVec;
    }

    @Override
    public String toString() {
        String matrixString = new String();
        matrixString += "\n";
        for (int rowIdx = 0; rowIdx < rows; rowIdx++) {
            matrixString += "[";
            for (int colIdx = 0; colIdx < columns; colIdx++)
                matrixString += " " + matrix[rowIdx][colIdx];
            matrixString += "]\n";
        }

        return matrixString;
    }
}
