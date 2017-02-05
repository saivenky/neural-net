package saivenky.neural;

/**
 * Created by saivenky on 1/26/17.
 */
public abstract class Layer implements ILayer {
    public NeuronSet neurons;

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

    public void backpropagate() {
        neurons.backpropagate();
    }

    public void updateGradient() {
        neurons.updateGradient();
    }

    public void gradientDescent(double rate) {
        neurons.update(rate);
    }

    public void setSignalCostGradient(double[] cost) {
        for (int i = 0; i < neurons.size(); i++) {
            neurons.get(i).addToSignalCostGradient(1, cost[i]);
        }
    }

    public void multiplySignalCostGradientByActivation1() {
        for(int i : neurons.selected) {
            neurons.get(i).multiplyByActivation1();
        }
    }
}
