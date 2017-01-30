package saivenky.neural;

import org.junit.jupiter.api.Test;
import saivenky.neural.activation.Sigmoid;
import saivenky.neural.cost.Square;
import saivenky.neural.neuron.ZeroInitializer;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by saivenky on 1/29/17.
 */
public class NeuralNetworkTest {
    private static final double PRECISION = 1e-4;
    private static final double VERY_PRECISION = 1e-6;

    private static void assertSimilar(double expected, double actual) {
        assertEquals(expected, actual, PRECISION);
    }
    private static void assertVerySimilar(double expected, double actual) {
        assertEquals(expected, actual, VERY_PRECISION);
    }

    @Test
    void SingleNeuron() {
        int[] layers = {1, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        nn.layers[0].neurons[0].bias = 0.6;
        nn.layers[0].neurons[0].weights[0] = 0.9;

        Data.Example e = new Data.Example(Vector.ize(1), Vector.ize(0));
        nn.train(e.input, e.output);

        assertSimilar(0.8175, nn.predicted[0]);
        assertEquals(0.6, nn.layers[0].neurons[0].bias);
        assertEquals(0.9, nn.layers[0].neurons[0].weights[0]);

        assertSimilar(0.1219, nn.layers[0].error[0]);
        assertSimilar(0.1219, nn.layers[0].neurons[0].biasError);
        assertSimilar(0.1219, nn.layers[0].neurons[0].weightError[0]);

        nn.update(0.5);
        assertSimilar(0.5390, nn.layers[0].neurons[0].bias);
        assertSimilar(0.8390, nn.layers[0].neurons[0].weights[0]);
    }

    @Test
    void SingleNeuron_NegativeWeightAndBias() {
        int[] layers = {1, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        nn.layers[0].neurons[0].bias = -0.9;
        nn.layers[0].neurons[0].weights[0] = -0.6;

        Data.Example e = new Data.Example(Vector.ize(0.1), Vector.ize(1));
        nn.train(e.input, e.output);

        assertSimilar(0.2769, nn.predicted[0]);
        assertEquals(-0.9, nn.layers[0].neurons[0].bias);
        assertEquals(-0.6, nn.layers[0].neurons[0].weights[0]);

        assertSimilar(-0.1448, nn.layers[0].error[0]);
        assertSimilar(-0.1448, nn.layers[0].neurons[0].biasError);
        assertSimilar(-0.01448, nn.layers[0].neurons[0].weightError[0]);

        nn.update(0.5);
        assertSimilar(-0.8276, nn.layers[0].neurons[0].bias);
        assertSimilar(-0.5928, nn.layers[0].neurons[0].weights[0]);
    }

    @Test
    void SingleNeuron_TwoExamples() {
        int[] layers = {1, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        nn.layers[0].neurons[0].bias = 0.6;
        nn.layers[0].neurons[0].weights[0] = 0.9;

        Data.Example e = new Data.Example(Vector.ize(1), Vector.ize(0));
        nn.train(e.input, e.output);
        nn.train(e.input, e.output);

        assertSimilar(0.8175, nn.predicted[0]);
        assertEquals(0.6, nn.layers[0].neurons[0].bias);
        assertEquals(0.9, nn.layers[0].neurons[0].weights[0]);

        assertSimilar(0.1219, nn.layers[0].error[0]);
        assertSimilar(0.2438, nn.layers[0].neurons[0].biasError);
        assertSimilar(0.2438, nn.layers[0].neurons[0].weightError[0]);

        nn.update(0.5);
        assertSimilar(0.5390, nn.layers[0].neurons[0].bias);
        assertSimilar(0.8390, nn.layers[0].neurons[0].weights[0]);
    }

    @Test
    void SingleNeuron_MultipleUpdates() {
        int[] layers = {1, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        Data.Example e = new Data.Example(Vector.ize(1), Vector.ize(0));

        for(int i = 0; i < 100; i++) {
            nn.layers[0].neurons[0].bias = 0.6;
            nn.layers[0].neurons[0].weights[0] = 0.9;

            nn.train(e.input, e.output);
            nn.train(e.input, e.output);

            assertSimilar(0.8175, nn.predicted[0]);
            assertEquals(0.6, nn.layers[0].neurons[0].bias);
            assertEquals(0.9, nn.layers[0].neurons[0].weights[0]);

            assertSimilar(0.1219, nn.layers[0].error[0]);
            assertSimilar(0.2438, nn.layers[0].neurons[0].biasError);
            assertSimilar(0.2438, nn.layers[0].neurons[0].weightError[0]);

            nn.update(0.5);
            assertSimilar(0.5390, nn.layers[0].neurons[0].bias);
            assertSimilar(0.8390, nn.layers[0].neurons[0].weights[0]);
        }
    }

    @Test
    void TwoNeurons() {
        int[] layers = {1, 1, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        nn.layers[0].neurons[0].bias = 0.6;
        nn.layers[0].neurons[0].weights[0] = 0.9;
        nn.layers[1].neurons[0].bias = 0.7;
        nn.layers[1].neurons[0].weights[0] = 0.8;

        Data.Example e = new Data.Example(Vector.ize(1), Vector.ize(0));
        nn.train(e.input, e.output);

        assertSimilar(0.8175, nn.layers[0].activation[0]);
        assertSimilar(0.7948, nn.predicted[0]);
        assertEquals(0.6, nn.layers[0].neurons[0].bias);
        assertEquals(0.9, nn.layers[0].neurons[0].weights[0]);
        assertEquals(0.7, nn.layers[1].neurons[0].bias);
        assertEquals(0.8, nn.layers[1].neurons[0].weights[0]);

        assertSimilar(0.1296, nn.layers[1].error[0]);
        assertSimilar(0.1296, nn.layers[1].neurons[0].biasError);
        assertSimilar(0.1060, nn.layers[1].neurons[0].weightError[0]);

        assertSimilar(0.01546, nn.layers[0].error[0]);
        assertSimilar(0.01546, nn.layers[0].neurons[0].biasError);
        assertSimilar(0.01546, nn.layers[0].neurons[0].weightError[0]);

        nn.update(0.5);
        assertSimilar(0.6352, nn.layers[1].neurons[0].bias);
        assertSimilar(0.7470, nn.layers[1].neurons[0].weights[0]);
        assertSimilar(0.5923, nn.layers[0].neurons[0].bias);
        assertSimilar(0.8923, nn.layers[0].neurons[0].weights[0]);
    }

    @Test
    void TwoNeurons_MultipleUpdates() {
        int[] layers = {1, 1, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        Data.Example e = new Data.Example(Vector.ize(1), Vector.ize(0));

        for(int i = 0; i < 100; i++) {
            nn.layers[0].neurons[0].bias = 0.6;
            nn.layers[0].neurons[0].weights[0] = 0.9;
            nn.layers[1].neurons[0].bias = 0.7;
            nn.layers[1].neurons[0].weights[0] = 0.8;

            nn.train(e.input, e.output);

            assertSimilar(0.8175, nn.layers[0].activation[0]);
            assertSimilar(0.7948, nn.predicted[0]);
            assertEquals(0.6, nn.layers[0].neurons[0].bias);
            assertEquals(0.9, nn.layers[0].neurons[0].weights[0]);
            assertEquals(0.7, nn.layers[1].neurons[0].bias);
            assertEquals(0.8, nn.layers[1].neurons[0].weights[0]);

            assertSimilar(0.1296, nn.layers[1].error[0]);
            assertSimilar(0.1296, nn.layers[1].neurons[0].biasError);
            assertSimilar(0.1060, nn.layers[1].neurons[0].weightError[0]);

            assertSimilar(0.01546, nn.layers[0].error[0]);
            assertSimilar(0.01546, nn.layers[0].neurons[0].biasError);
            assertSimilar(0.01546, nn.layers[0].neurons[0].weightError[0]);

            nn.update(0.5);
            assertSimilar(0.6352, nn.layers[1].neurons[0].bias);
            assertSimilar(0.7470, nn.layers[1].neurons[0].weights[0]);
            assertSimilar(0.5923, nn.layers[0].neurons[0].bias);
            assertSimilar(0.8923, nn.layers[0].neurons[0].weights[0]);
        }
    }

    @Test
    void MultipleLayerNeurons() {
        int[] layers = {2, 2, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        Data.Example e = new Data.Example(Vector.ize(1, 0.5), Vector.ize(0));

        for(int i = 0; i < 100; i++) {
            nn.layers[0].neurons[0].bias = 0.9;
            nn.layers[0].neurons[1].bias = 0.3;
            nn.layers[0].neurons[0].weights = Vector.ize(0.8, 0.2);
            nn.layers[0].neurons[1].weights = Vector.ize(0.9, -0.4);

            nn.layers[1].neurons[0].bias = -0.2;
            nn.layers[1].neurons[0].weights = Vector.ize(0.5, -0.7);

            nn.train(e.input, e.output);

            assertEquals(0.9, nn.layers[0].neurons[0].bias);
            assertEquals(0.8, nn.layers[0].neurons[0].weights[0]);
            assertEquals(0.2, nn.layers[0].neurons[0].weights[1]);
            assertEquals(0.3, nn.layers[0].neurons[1].bias);
            assertEquals(0.9, nn.layers[0].neurons[1].weights[0]);
            assertEquals(-0.4, nn.layers[0].neurons[1].weights[1]);

            assertEquals(-0.2, nn.layers[1].neurons[0].bias);
            assertEquals(0.5, nn.layers[1].neurons[0].weights[0]);
            assertEquals(-0.7, nn.layers[1].neurons[0].weights[1]);

            assertSimilar(0.8581, nn.layers[0].activation[0]);
            assertSimilar(0.73106, nn.layers[0].activation[1]);
            assertSimilar(0.4298, nn.layers[1].activation[0]);

            assertSimilar(0.1053, nn.layers[1].error[0]);
            assertSimilar(0.006411, nn.layers[0].error[0]);
            assertSimilar(-0.014497, nn.layers[0].error[1]);

            nn.update(0.5);

            assertSimilar(0.8968, nn.layers[0].neurons[0].bias);
            assertSimilar(0.7968, nn.layers[0].neurons[0].weights[0]);
            assertSimilar(0.1984, nn.layers[0].neurons[0].weights[1]);
            assertSimilar(0.3072, nn.layers[0].neurons[1].bias);
            assertSimilar(0.9072, nn.layers[0].neurons[1].weights[0]);
            assertSimilar(-0.3964, nn.layers[0].neurons[1].weights[1]);

            assertSimilar(-0.2527, nn.layers[1].neurons[0].bias);
            assertSimilar(0.4548, nn.layers[1].neurons[0].weights[0]);
            assertSimilar(-0.7385, nn.layers[1].neurons[0].weights[1]);
        }
    }

    @Test
    void MultipleHiddenNeuronLayers() {
        int[] layers = {2, 3, 2, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        Data.Example e = new Data.Example(Vector.ize(0, 1), Vector.ize(0.8));

        //initialize
        nn.layers[0].neurons[0].bias = -0.6;
        nn.layers[0].neurons[1].bias = 0.3;
        nn.layers[0].neurons[2].bias = 0.1;

        nn.layers[1].neurons[0].bias = 0.9;
        nn.layers[1].neurons[1].bias = -0.4;

        nn.layers[2].neurons[0].bias = 0.5;

        nn.layers[0].neurons[0].weights[0] = 0.8;
        nn.layers[0].neurons[0].weights[1] = 0.7;
        nn.layers[0].neurons[1].weights[0] = -0.8;
        nn.layers[0].neurons[1].weights[1] = 0.4;
        nn.layers[0].neurons[2].weights[0] = -0.9;
        nn.layers[0].neurons[2].weights[1] = -0.3;

        nn.layers[1].neurons[0].weights[0] = -0.4;
        nn.layers[1].neurons[0].weights[1] = 0.7;
        nn.layers[1].neurons[0].weights[2] = 0.7;
        nn.layers[1].neurons[1].weights[0] = 0.6;
        nn.layers[1].neurons[1].weights[1] = -0.3;
        nn.layers[1].neurons[1].weights[2] = -0.2;

        nn.layers[2].neurons[0].weights[0] = -0.3;
        nn.layers[2].neurons[0].weights[1] = 0.5;

        //train and check no changes
        nn.train(e.input, e.output);

        assertEquals(-0.6, nn.layers[0].neurons[0].bias);
        assertEquals(0.3, nn.layers[0].neurons[1].bias);
        assertEquals(0.1, nn.layers[0].neurons[2].bias);

        assertEquals(0.9, nn.layers[1].neurons[0].bias);
        assertEquals(-0.4, nn.layers[1].neurons[1].bias);

        assertEquals(0.5, nn.layers[2].neurons[0].bias);

        assertEquals(0.8, nn.layers[0].neurons[0].weights[0]);
        assertEquals(0.7, nn.layers[0].neurons[0].weights[1]);
        assertEquals(-0.8, nn.layers[0].neurons[1].weights[0]);
        assertEquals(0.4, nn.layers[0].neurons[1].weights[1]);
        assertEquals(-0.9, nn.layers[0].neurons[2].weights[0]);
        assertEquals(-0.3, nn.layers[0].neurons[2].weights[1]);

        assertEquals(-0.4, nn.layers[1].neurons[0].weights[0]);
        assertEquals(0.7, nn.layers[1].neurons[0].weights[1]);
        assertEquals(0.7, nn.layers[1].neurons[0].weights[2]);
        assertEquals(0.6, nn.layers[1].neurons[1].weights[0]);
        assertEquals(-0.3, nn.layers[1].neurons[1].weights[1]);
        assertEquals(-0.2, nn.layers[1].neurons[1].weights[2]);

        assertEquals(-0.3, nn.layers[2].neurons[0].weights[0]);
        assertEquals(0.5, nn.layers[2].neurons[0].weights[1]);

        //feedforward
        assertVerySimilar(0.6129095, nn.predicted[0]);

        //update and check
        nn.update(0.7);
        assertVerySimilar(-0.59929781, nn.layers[0].neurons[0].bias);
        assertVerySimilar(0.29953107, nn.layers[0].neurons[1].bias);
        assertVerySimilar(0.09956932, nn.layers[0].neurons[2].bias);

        assertVerySimilar(0.89858573, nn.layers[1].neurons[0].bias);
        assertVerySimilar(-0.39624985, nn.layers[1].neurons[1].bias);

        assertVerySimilar(0.53107124, nn.layers[2].neurons[0].bias);

        assertVerySimilar(0.8, nn.layers[0].neurons[0].weights[0]);
        assertVerySimilar(0.70070219, nn.layers[0].neurons[0].weights[1]);
        assertVerySimilar(-0.8, nn.layers[0].neurons[1].weights[0]);
        assertVerySimilar(0.39953107, nn.layers[0].neurons[1].weights[1]);
        assertVerySimilar(-0.9, nn.layers[0].neurons[2].weights[0]);
        assertVerySimilar(-0.30043068, nn.layers[0].neurons[2].weights[1]);

        assertVerySimilar(-0.40074246, nn.layers[1].neurons[0].weights[0]);
        assertVerySimilar(0.699055, nn.layers[1].neurons[0].weights[1]);
        assertVerySimilar(0.69936334, nn.layers[1].neurons[0].weights[2]);
        assertVerySimilar(0.60196875, nn.layers[1].neurons[1].weights[0]);
        assertVerySimilar(-0.2974942, nn.layers[1].neurons[1].weights[1]);
        assertVerySimilar(-0.19831181, nn.layers[1].neurons[1].weights[2]);

        assertVerySimilar(-0.27472382, nn.layers[2].neurons[0].weights[0]);
        assertVerySimilar(0.51265259, nn.layers[2].neurons[0].weights[1]);
    }



}