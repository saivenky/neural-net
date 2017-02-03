package saivenky.neural;

import saivenky.neural.activation.pooling.PoolingActivationFunction;
import saivenky.neural.neuron.ZeroInitializer;

/**
 * Created by saivenky on 2/1/17.
 */
public class PoolingNeuron extends Neuron {

    private PoolingActivationFunction poolingActivationFunction;

    PoolingNeuron(PoolingActivationFunction poolingActivationFunction, NeuronSet inputNeurons) {
        super(new NeuronProperties(ZeroInitializer.getInstance(), inputNeurons.size()), inputNeurons, null);
        this.poolingActivationFunction = poolingActivationFunction;
        activation1 = 1;
    }

    @Override
    void signalForSelected() {
        activation = poolingActivationFunction.pool(inputNeurons.neurons, properties.weights);
    }

    @Override
    void signalScaled(double scale) {
        signalForSelected();
    }

    @Override
    void update(double rate) {}

    @Override
    void addToSignalCostGradient(double weight, double cost) {
        double poolCost = weight * cost;
        inputNeurons.forAll(new NeuronSet.NeuronAction() {
            @Override
            void f(Neuron neuron, int i) {
                neuron.addToSignalCostGradient(properties.weights[i], poolCost);
            }
        });
    }

    @Override
    void propagateToInputNeurons() {}

    @Override
    void propagateToProperties() {}
}
