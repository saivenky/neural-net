package saivenky.neural;

/**
 * Created by saivenky on 2/1/17.
 */
public class InputLayer implements IInputLayer {
    NeuronSet neurons;

    public InputLayer(int size) {
        this.neurons = new NeuronSet(new INeuron[size]);
        initializeNeurons();
    }

    public InputLayer(NeuronSet neurons) {
        this.neurons = neurons;
        initializeNeurons();
    }

    private void initializeNeurons() {
        for(int i = 0; i < neurons.size(); i++) {
            neurons.set(i, new InputNeuron());
        }
    }

    public void setInput(double[] input) {
        for(int i = 0; i < neurons.size(); i++) {
            ((InputNeuron)neurons.get(i)).setActivation(input[i]);
        }
    }

    @Override
    public NeuronSet getNeurons() {
        return neurons;
    }

    @Override
    public void run() {

    }

    @Override
    public void feedforward() {

    }

    @Override
    public void backpropagate(boolean backpropagateToPreviousLayer) {

    }

    @Override
    public void gradientDescent(double rate) {

    }

    @Override
    public void setSignalCostGradient(double[] cost) {

    }
}
