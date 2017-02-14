package saivenky.neural;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import saivenky.neural.activation.Sigmoid;
import saivenky.neural.neuron.CustomInitializer;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by saivenky on 2/6/17.
 */
public class NeuronTests {
    INeuron neuron;
    InputNeuron inputNeuron;

    @BeforeEach
    public void setUp() {
        CustomInitializer customInitializer = new CustomInitializer();
        customInitializer.addWeights(0.9);
        customInitializer.addBiases(0.6);
        inputNeuron = new InputNeuron();
        INeuron[] inputs = {inputNeuron};
        NeuronSet inputNeurons = new NeuronSet(inputs);
        neuron = new Neuron(customInitializer, inputNeurons, Sigmoid.getInstance());
    }

    @Test
    void getActivation_notActivatedReturnsZero() {
        inputNeuron.setActivation(1);
        assertEquals(0, neuron.getActivation());
    }

    @Test
    void getActivation_multipleUpdates() {
        inputNeuron.setActivation(1);
        neuron.activate();
        assertEquals(0.8175744761936437, neuron.getActivation());

        neuron.addToSignalCostGradient(0.25, neuron.getActivation());
        neuron.backpropagate(false);
        neuron.gradientDescent(2);
        neuron.activate();
        assertEquals(0.7986795151464976, neuron.getActivation());
    }

    @Test
    void getActivation_whenScaled() {
        inputNeuron.setActivation(1);
        neuron.activateScaled(0.5);
        assertEquals(0.4087872380968218, neuron.getActivation());

    }
}
