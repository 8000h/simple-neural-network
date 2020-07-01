package com.gravitykit.nn;

/**
 * Sample implements a pair which combines sample input to the
 * network along with its solution.
 */

public class Sample {

    private Vector input;
    private Vector desired;

    public Sample(Vector input, Vector desired) {
        this.input   = input;
        this.desired = desired;
    }

    public Sample(Double[] input, Double[] desired) {
        this.input   = new Vector(input);
        this.desired = new Vector(desired);
    }

    public Vector getInput() {
        return this.input;
    }

    public Vector getDesired() {
        return this.desired;
    }

}
