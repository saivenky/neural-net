package saivenky.neural;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import saivenky.neural.neuron.ZeroInitializer;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by saivenky on 2/6/17.
 */
public class NeuronPropertiesTests {
    private static final double PRECISION = 1e-7;

    private NeuronSet inputNeurons;

    @BeforeEach
    void setUp() {
        INeuron[] neuronArray = {
                new FakeNeuron(1),
                new FakeNeuron(2),
                new FakeNeuron(3),
                new FakeNeuron(4)
        };

        inputNeurons = new NeuronSet(neuronArray);
    }

    @Test
    void constructor_hasZeroCostGradients() {
        NeuronProperties properties = new NeuronProperties(ZeroInitializer.getInstance(), inputNeurons.size());
        assertEquals(inputNeurons.size(), properties.weights.length);
        assertEquals(inputNeurons.size(), properties.weightCostGradient.length);
        assertEquals(0, properties.biasCostGradient);
        for(double weightCostGradient : properties.weightCostGradient) {
            assertEquals(0, weightCostGradient);
        }
    }

    @Test
    void affine_evenWhenSelectedDoesAll() {
        NeuronProperties properties = new NeuronProperties(ZeroInitializer.getInstance(), inputNeurons.size());
        properties.weights = Vector.ize(0.1, 0.2, 0.3, 0.4);
        properties.bias = 100;
        int[] selected = {2};
        inputNeurons.select(selected);
        double result = 100 + 0.1 + 0.4 + 0.9 + 1.6;
        assertEquals(result, properties.affine(inputNeurons));
    }

    @Test
    void affineForSelected_onlyForSelected() {
        NeuronProperties properties = new NeuronProperties(ZeroInitializer.getInstance(), inputNeurons.size());
        properties.weights = Vector.ize(0.1, 0.2, 0.3, 0.4);
        properties.bias = 100;
        int[] selected = {2};
        inputNeurons.select(selected);
        double result = 100 + 0.9;
        assertEquals(result, properties.affineForSelected(inputNeurons));
    }

    @Test
    void update_noChangesWhenCostGradientsZero() {
        NeuronProperties properties = new NeuronProperties(ZeroInitializer.getInstance(), inputNeurons.size());
        properties.weights = Vector.ize(0.1, 0.2, 0.3, 0.4);
        properties.bias = 100;
        properties.update(0.1);
        assertEquals(100, properties.bias);
        Assertions.assertArrayEquals(Vector.ize(0.1, 0.2, 0.3, 0.4), properties.weights);
    }

    @Test
    void update_subtractsAtRateAndZerosCostsAfterPropertiesChanged() {
        NeuronProperties properties = new NeuronProperties(ZeroInitializer.getInstance(), inputNeurons.size());
        properties.weights = Vector.ize(0.1, 0.2, 0.3, 0.4);
        properties.bias = 100;
        properties.weightCostGradient = Vector.ize(-1, -1, 2, 2);
        properties.biasCostGradient = 3;

        properties.update(0.1);
        assertEquals(99.7, properties.bias);
        Assertions.assertArrayEquals(Vector.ize(0.2, 0.3, 0.1, 0.2), properties.weights, PRECISION);

        assertEquals(0, properties.biasCostGradient);
        for(double weightCostGradient : properties.weightCostGradient) {
            assertEquals(0, weightCostGradient);
        }
    }

    //TODO get mocking framework
    private static class FakeNeuron implements INeuron {

        private double activation;

        public FakeNeuron(double activation) {

            this.activation = activation;
        }

        @Override
        public double getActivation() {
            return activation;
        }

        @Override
        public void activate() {
            throw new NotImplementedException();
        }

        @Override
        public void activateScaled(double scale) {
            throw new NotImplementedException();
        }

        @Override
        public void gradientDescent(double rate) {
            throw new NotImplementedException();
        }

        @Override
        public void addToSignalCostGradient(double weight, double cost) {
            throw new NotImplementedException();
        }

        @Override
        public void backpropagate(boolean backpropagateToInputNeurons) {
            throw new NotImplementedException();
        }
    }
}
