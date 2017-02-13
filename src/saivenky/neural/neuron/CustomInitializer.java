package saivenky.neural.neuron;

import java.util.ArrayDeque;
import java.util.Queue;

/**
 * Created by saivenky on 2/13/17.
 */
public class CustomInitializer extends NeuronInitializer {
    private final Queue<Double> biases;
    private final Queue<Double> weights;

    public CustomInitializer() {
        super();
        biases = new ArrayDeque<>();
        weights = new ArrayDeque<>();
        biasInitializer = new Function() {
            @Override
            public double f() {
                return getNextBias();
            }
        };
        weightInitializer = new Function() {
            @Override
            public double f() {
                return getNextWeight();
            }
        };
    }

    public void addBiases(double ... biasesToAdd) {
        for(double bias : biasesToAdd) {
            biases.add(bias);
        }
    }

    public void addWeights(double ... weightsToAdd) {
        for(double weight : weightsToAdd) {
            weights.add(weight);
        }
    }

    private double getNextBias() {
        if (biases.isEmpty()) return 0;
        return biases.poll();
    }

    private double getNextWeight() {
        if (weights.isEmpty()) return 0;
        return weights.poll();
    }


}
