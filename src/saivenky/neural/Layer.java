package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;

/**
 * Created by saivenky on 1/26/17.
 */
class Layer {
    private ActivationFunction activationFunction;
    NeuronSet neurons;
    double dropoutRate;
    int[] nonDropout;

    Layer(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    final void setDropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
        int nonDropoutLength = neurons.size() - (int)(neurons.size() * dropoutRate);
        nonDropout = new int[nonDropoutLength];
        Vector.select(nonDropout, neurons.size());
        neurons.select(nonDropout);
    }

    final void reselectDropout() {
        Vector.select(nonDropout, neurons.size());
    }

    final void run() {
        neurons.activate(activationFunction);
    }

    private static double signalScaleFromDropout(double dropoutRate) {
        return (1 - dropoutRate);
    }

    final void runScaled(double inputDropoutRate) {
        double scale = signalScaleFromDropout(inputDropoutRate);
        neurons.activateScaled(activationFunction, scale);
    }

    final void backpropagate() {
        neurons.forSelectedWithCost(new NeuronSet.NeuronWithCostAction() {
            @Override
            void f(Neuron neuron, double cost, int i) {
                neuron.inputNeurons.addSignalCostGradient(neuron.properties.weights, cost);
            }
        });

        neurons.forSelected(new NeuronSet.NeuronAction() {
            @Override
            void f(Neuron neuron, int i) {
                neuron.inputNeurons.completeSignalCostGradient();
            }
        });

        neurons.forSelectedWithCost(new NeuronSet.NeuronWithCostAction() {
            @Override
            void f(Neuron neuron, double cost, int i) {
                neuron.properties.biasCostGradient += cost;
                Vector.multiplyAndAddSelected(neuron.inputNeurons.activation, cost, neuron.properties.weightCostGradient, neuron.inputNeurons.selected);
            }
        });
    }

    final void update(double rate) {
        neurons.forSelected(new NeuronSet.NeuronAction() {
            @Override
            void f(Neuron neuron, int i) {
                neuron.update(rate);
            }
        });
        neurons.forSelected(new NeuronSet.NeuronAction() {
            @Override
            void f(Neuron neuron, int i) {
                neuron.inputNeurons.clearSignalCostGradient();
            }
        });
    }
}
