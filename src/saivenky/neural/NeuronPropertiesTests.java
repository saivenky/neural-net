package saivenky.neural;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import saivenky.neural.neuron.CustomInitializer;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by saivenky on 2/6/17.
 */
public class NeuronPropertiesTests {
    private static final double PRECISION = 1e-7;

    private DropoutNeuronSet inputNeurons;
    private CustomInitializer customInitializer;

    @BeforeEach
    void setUp() {
        INeuron[] neuronArray = {
                new FakeNeuron(1),
                new FakeNeuron(2),
                new FakeNeuron(3),
                new FakeNeuron(4)
        };

        inputNeurons = new DropoutNeuronSet(neuronArray);
        customInitializer = new CustomInitializer();
        customInitializer.addWeights(0.1, 0.2, 0.3, 0.4);
        customInitializer.addBiases(100);
    }

    @Test
    void affine_evenWhenSelectedDoesAll() {
        NeuronProperties properties = new NeuronProperties(customInitializer, inputNeurons.size());
        int[] selected = {2};
        inputNeurons.select(selected);
        double result = 100 + 0.1 + 0.4 + 0.9 + 1.6;
        assertEquals(result, properties.affine(inputNeurons));
    }

    @Test
    void affineForSelected_onlyForSelected() {
        NeuronProperties properties = new NeuronProperties(customInitializer, inputNeurons.size());
        int[] selected = {2};
        inputNeurons.select(selected);
        double result = 100 + 0.9;
        assertEquals(result, properties.affineForSelected(inputNeurons));
    }

    @Test
    void update_noChangesWhenCostGradientsZero() {
        NeuronProperties properties = new NeuronProperties(customInitializer, inputNeurons.size());
        properties.update(0.1);
        assertEquals(100, properties.getBias());
        Assertions.assertArrayEquals(Vector.ize(0.1, 0.2, 0.3, 0.4), properties.getWeights());
    }

    @Test
    void update_subtractsAtRateAndZerosCostsAfterPropertiesChanged() {
        NeuronProperties properties = new NeuronProperties(customInitializer, inputNeurons.size());
        properties.addError(inputNeurons, 2);

        properties.update(0.5);
        assertEquals(99, properties.getBias());
        Assertions.assertArrayEquals(Vector.ize(-0.9, -1.8, -2.7, -3.6), properties.getWeights(), PRECISION);

        properties.update(0.5);
        assertEquals(99, properties.getBias());
        Assertions.assertArrayEquals(Vector.ize(-0.9, -1.8, -2.7, -3.6), properties.getWeights(), PRECISION);

        properties.addError(inputNeurons, 2);
        properties.update(0.5);
        assertEquals(98, properties.getBias());
        Assertions.assertArrayEquals(Vector.ize(-1.9, -3.8, -5.7, -7.6), properties.getWeights(), PRECISION);
    }

    //TODO get mocking framework
    private static class FakeNeuron implements INeuron {

        private double activation;

        FakeNeuron(double activation) {

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
        public void setSignalCostGradient(double cost) {
            throw new NotImplementedException();
        }

        @Override
        public void addToSignalCostGradient(double cost) {
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
