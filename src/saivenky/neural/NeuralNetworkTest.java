package saivenky.neural;

import org.junit.jupiter.api.Test;
import saivenky.neural.activation.Linear;
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

    @Test
    void SingleNeuron() {
        int[] layers = {1, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        ((Neuron)nn.layers[0].neurons.get(0)).properties.bias = 0.6;
        ((Neuron)nn.layers[0].neurons.get(0)).properties.weights[0] = 0.9;

        Data.Example e = new Data.Example(Vector.ize(1), Vector.ize(0));
        nn.train(e.input, e.output);

        assertSimilar(0.8175, nn.predicted[0]);
        assertEquals(0.6, ((Neuron)nn.layers[0].neurons.get(0)).properties.bias);
        assertEquals(0.9, ((Neuron)nn.layers[0].neurons.get(0)).properties.weights[0]);

        assertSimilar(0.1219, ((Neuron)nn.layers[0].neurons.get(0)).properties.biasCostGradient);
        assertSimilar(0.1219, ((Neuron)nn.layers[0].neurons.get(0)).properties.weightCostGradient[0]);

        nn.update(0.5);
        assertSimilar(0.5390, ((Neuron)nn.layers[0].neurons.get(0)).properties.bias);
        assertSimilar(0.8390, ((Neuron)nn.layers[0].neurons.get(0)).properties.weights[0]);
    }

    @Test
    void SingleNeuron_NegativeWeights() {
        int[] layers = {1, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        ((Neuron)nn.layers[0].neurons.get(0)).properties.bias = -0.9;
        ((Neuron)nn.layers[0].neurons.get(0)).properties.weights[0] = -0.6;

        Data.Example e = new Data.Example(Vector.ize(0.1), Vector.ize(1));
        nn.train(e.input, e.output);

        assertSimilar(0.2769, nn.predicted[0]);
        assertEquals(-0.9, ((Neuron)nn.layers[0].neurons.get(0)).properties.bias);
        assertEquals(-0.6, ((Neuron)nn.layers[0].neurons.get(0)).properties.weights[0]);

        assertSimilar(-0.1448, ((Neuron)nn.layers[0].neurons.get(0)).properties.biasCostGradient);
        assertSimilar(-0.01448, ((Neuron)nn.layers[0].neurons.get(0)).properties.weightCostGradient[0]);

        nn.update(0.5);
        assertSimilar(-0.8276, ((Neuron)nn.layers[0].neurons.get(0)).properties.bias);
        assertSimilar(-0.5928, ((Neuron)nn.layers[0].neurons.get(0)).properties.weights[0]);
    }

    @Test
    void SingleNeuron_TwoExamples() {
        int[] layers = {1, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        ((Neuron)nn.layers[0].neurons.get(0)).properties.bias = 0.6;
        ((Neuron)nn.layers[0].neurons.get(0)).properties.weights[0] = 0.9;

        Data.Example e = new Data.Example(Vector.ize(1), Vector.ize(0));
        nn.train(e.input, e.output);
        nn.train(e.input, e.output);

        assertSimilar(0.8175, nn.predicted[0]);
        assertEquals(0.6, ((Neuron)nn.layers[0].neurons.get(0)).properties.bias);
        assertEquals(0.9, ((Neuron)nn.layers[0].neurons.get(0)).properties.weights[0]);

        assertSimilar(0.2438, ((Neuron)nn.layers[0].neurons.get(0)).properties.biasCostGradient);
        assertSimilar(0.2438, ((Neuron)nn.layers[0].neurons.get(0)).properties.weightCostGradient[0]);

        nn.update(0.5);
        assertSimilar(0.5390, ((Neuron)nn.layers[0].neurons.get(0)).properties.bias);
        assertSimilar(0.8390, ((Neuron)nn.layers[0].neurons.get(0)).properties.weights[0]);
    }

    @Test
    void SingleNeuron_MultipleUpdates() {
        int[] layers = {1, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        Data.Example e = new Data.Example(Vector.ize(1), Vector.ize(0));

        for(int i = 0; i < 100; i++) {
            ((Neuron)nn.layers[0].neurons.get(0)).properties.bias = 0.6;
            ((Neuron)nn.layers[0].neurons.get(0)).properties.weights[0] = 0.9;

            nn.train(e.input, e.output);
            nn.train(e.input, e.output);

            assertSimilar(0.8175, nn.predicted[0]);
            assertEquals(0.6, ((Neuron)nn.layers[0].neurons.get(0)).properties.bias);
            assertEquals(0.9, ((Neuron)nn.layers[0].neurons.get(0)).properties.weights[0]);

            assertSimilar(0.2438, ((Neuron)nn.layers[0].neurons.get(0)).properties.biasCostGradient);
            assertSimilar(0.2438, ((Neuron)nn.layers[0].neurons.get(0)).properties.weightCostGradient[0]);

            nn.update(0.5);
            assertSimilar(0.5390, ((Neuron)nn.layers[0].neurons.get(0)).properties.bias);
            assertSimilar(0.8390, ((Neuron)nn.layers[0].neurons.get(0)).properties.weights[0]);
        }
    }

    @Test
    void TwoNeurons_MultipleUpdates() {
        int[] layers = {1, 1, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        Data.Example e = new Data.Example(Vector.ize(1), Vector.ize(0));

        double[][] biases = {{0.6}, {0.7}};
        double[][][] weights = {{{0.9}}, {{0.8}}};

        for(int i = 0; i < 100; i++) {
            setProperties(nn, biases, weights);

            nn.train(e.input, e.output);

            assertSimilar(0.7948, nn.predicted[0]);

            checkProperties(nn, biases, weights, new ValueChecker() {
                @Override
                void areEqual(double expected, double actual, String message) {
                    assertEquals(expected, actual, message);
                }
            });

            assertSimilar(0.1296, ((Neuron)nn.layers[1].neurons.get(0)).properties.biasCostGradient);
            assertSimilar(0.1060, ((Neuron)nn.layers[1].neurons.get(0)).properties.weightCostGradient[0]);

            assertSimilar(0.01546, ((Neuron)nn.layers[0].neurons.get(0)).properties.biasCostGradient);
            assertSimilar(0.01546, ((Neuron)nn.layers[0].neurons.get(0)).properties.weightCostGradient[0]);

            nn.update(0.5);
            double[][] expectedBiases = {{0.5923}, {0.6352}};
            double[][][] expectedWeights = {{{0.8923}}, {{0.7470}}};
            checkProperties(nn, expectedBiases, expectedWeights, new ValueChecker() {
                @Override
                void areEqual(double expected, double actual, String message) {
                    assertEquals(expected, actual, PRECISION, message);
                }
            });
        }
    }

    @Test
    void MultipleLayerNeurons() {
        int[] layers = {2, 2, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        Data.Example e = new Data.Example(Vector.ize(1, 0.5), Vector.ize(0));

        double[][] biases = {
                {0.9, 0.3},
                {-0.2}
        };

        double[][][] weights = {
                {{0.8, 0.2}, {0.9, -0.4}},
                {{0.5, -0.7}}
        };

        for(int i = 0; i < 100; i++) {
            setProperties(nn, biases, weights);

            nn.train(e.input, e.output);
            checkProperties(nn, biases, weights, new ValueChecker() {
                @Override
                void areEqual(double expected, double actual, String message) {
                    assertEquals(expected, actual, message);
                }
            });

            assertSimilar(0.4298, nn.predicted[0]);

            //assertSimilar(0.1053, nn.layers[1].neurons.get(0).signalCostGradient);
            //assertSimilar(0.006411, nn.layers[0].neurons.get(0).signalCostGradient);
            //assertSimilar(-0.014497, nn.layers[0].neurons.get(1).signalCostGradient);

            nn.update(0.5);

            double[][] expectedBiases = {
                    {0.8968, 0.3072},
                    {-0.2527}
            };

            double[][][] expectedWeights = {
                    {{0.7968, 0.1984}, {0.9072, -0.3964}},
                    {{0.4548, -0.7385}}
            };

            checkProperties(nn, expectedBiases, expectedWeights, new ValueChecker() {
                @Override
                void areEqual(double expected, double actual, String message) {
                    assertEquals(expected, actual, PRECISION, message);
                }
            });
        }
    }

    @Test
    void MultipleHiddenNeuronLayers() {
        int[] layers = {2, 3, 2, 1};
        NeuralNetwork nn = new NeuralNetwork(
                layers, Sigmoid.getInstance(), Square.getInstance(), new ZeroInitializer());
        Data.Example e = new Data.Example(Vector.ize(0, 1), Vector.ize(0.8));

        double[][] biases = {
                {-0.6, 0.3, 0.1},
                {0.9, -0.4},
                {0.5}
        };

        double[][][] weights = {
                {{0.8, 0.7}, {-0.8, 0.4}, {-0.9, -0.3}},
                {{-0.4, 0.7, 0.7}, {0.6, -0.3, -0.2}},
                {{-0.3, 0.5}}
        };

        //initialize
        setProperties(nn, biases, weights);

        //train and check no changes
        nn.train(e.input, e.output);
        checkProperties(nn, biases, weights, new ValueChecker() {
            @Override
            void areEqual(double expected, double actual, String message) {
                assertEquals(expected, actual, message);
            }
        });

        //feedforward
        assertEquals(0.6129095, nn.predicted[0], VERY_PRECISION);

        //gradientDescent and check
        nn.update(0.7);
        double[][] expectedBiases = {
                {-0.59929781, 0.29953107, 0.09956932},
                {0.89858573, -0.39624985},
                {0.53107124}
        };

        double[][][] expectedWeights = {
                {{0.8, 0.70070219}, {-0.8, 0.39953107}, {-0.9, -0.30043068}},
                {{-0.40074246, 0.699055, 0.69936334}, {0.60196875, -0.2974942, -0.19831181}},
                {{-0.27472382, 0.51265259}}
        };

        checkProperties(nn, expectedBiases, expectedWeights, new ValueChecker() {
            @Override
            void areEqual(double expected, double actual, String message) {
                assertEquals(expected, actual, VERY_PRECISION, message);

            }
        });
    }

    @Test
    void ConvolutionPoolLayer() {
        NeuronSet input = new NeuronSet(new INeuron[2 * 2]);
        input.setShape(2, 2, 1);
        InputLayer inputLayer = new InputLayer(input);
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(Linear.getInstance(), 1, 1, 1, inputLayer.getNeurons());
        MaxPoolingLayer poolingLayer = new MaxPoolingLayer(2, 2, convolutionLayer.getNeurons());
        StandardLayer outputLayer = new StandardLayer(1, poolingLayer, Linear.getInstance(), ZeroInitializer.getInstance(), 0);
        ((Neuron)convolutionLayer.neurons.get(0)).properties.weights = Vector.ize(1);
        ((Neuron)convolutionLayer.neurons.get(0)).properties.bias = 0;
        ((Neuron)outputLayer.neurons.get(0)).properties.weights = Vector.ize(1);
        NeuralNetwork nn = new NeuralNetwork(inputLayer, Square.getInstance(), convolutionLayer, poolingLayer, outputLayer);
        nn.train(Vector.ize(0, 1, 0.1, 0), Vector.ize(0));
        nn.update(0.5);

        assertEquals(0.5, ((Neuron)convolutionLayer.neurons.get(0)).properties.weights[0]);
        assertEquals(-0.5, ((Neuron)convolutionLayer.neurons.get(0)).properties.bias);
    }

    private void setProperties(NeuralNetwork nn, double[][] biases, double[][][] weights) {
        for(int i = nn.layers.length - 1; i >= 0 ; i--) {
            Layer layer = nn.layers[i];
            double[] layerBiases = biases[i];
            double[][] layerWeights = weights[i];
            for(int j = 0; j < layer.neurons.size(); j++) {
                NeuronProperties properties = ((Neuron)layer.neurons.get(j)).properties;
                properties.bias = layerBiases[j];

                double[] neuronWeights = layerWeights[j];
                System.arraycopy(neuronWeights, 0, properties.weights, 0, properties.weights.length);
            }
        }
    }

    private void checkProperties(NeuralNetwork nn, double[][] biases, double[][][] weights, ValueChecker valueChecker) {
        for(int i = nn.layers.length - 1; i >= 0 ; i--) {
            Layer layer = nn.layers[i];
            double[] expectedLayerBiases = biases[i];
            double[][] expectedLayerWeights = weights[i];
            for(int j = 0; j < layer.neurons.size(); j++) {
                String message = String.format("layer %d, neuron %d, ", i, j);
                NeuronProperties properties = ((Neuron)layer.neurons.get(j)).properties;
                String biasMessage = message + "bias";
                valueChecker.areEqual(expectedLayerBiases[j], properties.bias, biasMessage);

                double[] expectedNeuronWeights = expectedLayerWeights[j];
                for (int k = 0; k < properties.weights.length; k++) {
                    String weightMessage = message + String.format("weights[%d]", k);
                    valueChecker.areEqual(expectedNeuronWeights[k], properties.weights[k], weightMessage);
                }
            }
        }
    }

    private static abstract class ValueChecker {
        abstract void areEqual(double expected, double actual, String message);
    }
}