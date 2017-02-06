package saivenky.neural;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import saivenky.neural.activation.Linear;
import saivenky.neural.neuron.ZeroInitializer;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by saivenky on 2/6/17.
 */
public class NeuronTests {
    private NeuronProperties properties;
    private NeuronSet inputNeurons;

    @BeforeEach
    public void setUp() {
        properties = new NeuronProperties(ZeroInitializer.getInstance(), 4);
        properties.weights = Vector.ize(0.1, 0.2, 0.3, 0.4);
        properties.bias = 100;

        INeuron[] neuronArray = {
                new InputNeuron(),
                new InputNeuron(),
                new InputNeuron(),
                new InputNeuron()
        };

        inputNeurons = new NeuronSet(neuronArray);
    }

    private void setActivations(double[] activations) {
        for(int i = 0; i < inputNeurons.size(); i++) {
            ((InputNeuron)inputNeurons.get(i)).setActivation(activations[i]);
        }
    }

    @Test
    void getActivation_correctAfterActivated() {
        Neuron neuron = new Neuron(properties, inputNeurons, Linear.getInstance());
        setActivations(Vector.ize(1, 2, 3, 4));
        int[] selected = {1, 2};
        inputNeurons.select(selected);
        double affineResult = properties.affineForSelected(inputNeurons);
        neuron.activate();

        assertEquals(affineResult, neuron.getActivation());
    }

    @Test
    void getActivation_correctAfterActivatedWithScale() {
        Neuron neuron = new Neuron(properties, inputNeurons, Linear.getInstance());
        setActivations(Vector.ize(1, 2, 3, 4));
        int[] selected = {1, 2};
        inputNeurons.select(selected);
        double affineResult = properties.affine(inputNeurons);
        neuron.activateScaled(0.5);

        assertEquals(affineResult * 0.5, neuron.getActivation());
        assertEquals(0, neuron.activation1);
    }

    @Test
    void addToSignalCostGradient_addsWithoutActivation1() {
        Neuron neuron = new Neuron(properties, inputNeurons, Linear.getInstance());
        setActivations(Vector.ize(1, 2, 3, 4));
        neuron.activation1 = 0.01;
        neuron.addToSignalCostGradient(100, 3);
        assertEquals(300, neuron.signalCostGradient);
    }

    @Test
    void backpropagate_usesActivation1AndZerosSignalCostGradientAfter() {
        Neuron neuron = new Neuron(properties, inputNeurons, Linear.getInstance());
        setActivations(Vector.ize(1, 2, 3, 4));
        neuron.activation1 = 0.01;
        neuron.addToSignalCostGradient(100, 3);
        assertEquals(300, neuron.signalCostGradient);

        neuron.backpropagate(false);
        assertEquals(3, properties.biasCostGradient);
        Assertions.assertArrayEquals(Vector.ize(3, 6, 9, 12), properties.weightCostGradient);
        assertEquals(0, neuron.signalCostGradient);
    }
}
