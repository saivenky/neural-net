package saivenky.neural;

/**
 * Created by saivenky on 1/26/17.
 */
public abstract class Layer implements ILayer, IOutputLayer {
    NeuronSet neurons;

    Layer(NeuronSet neurons) {
        this.neurons = neurons;
    }

    public NeuronSet getNeurons() {
        return neurons;
    }

    public void run() {
        neurons.activate();
    }

    public void feedforward() {
        neurons.activate();
    }

    public void backpropagate(boolean backpropagateToPreviousLayer) {
        neurons.backpropagate(backpropagateToPreviousLayer);
    }

    public void gradientDescent(double rate) {
        neurons.gradientDescent(rate);
    }

    public void setExpected(double[][] expected) {
    }

    public void setSignalCostGradient(double[][] cost) {
        for (int b = 0; b < cost.length; b++) {
            for (int i = 0; i < neurons.size(); i++) {
                neurons.get(i).setSignalCostGradient(cost[b][i]);
            }
        }
    }

    public void getPredicted(double[][] predicted) {
        for (int i = 0; i < predicted.length; i++) {
            predicted[0][i] = neurons.get(i).getActivation();
        }
    }

    public int size() {
        return neurons.size();
    }
}
