package saivenky.neural;

/**
 * Created by saivenky on 2/1/17.
 */
public class InputLayer {
    public InputNeuron[] neurons;
    public InputLayer(int size) {
        neurons = new InputNeuron[size];
        initializeNeurons();
    }

    private void initializeNeurons() {
        for(int i = 0; i < neurons.length; i++) {
            neurons[i] = new InputNeuron();
        }
    }

    public void setInput(double[] input) {
        for(int i = 0; i < neurons.length; i++) {
            neurons[i].setActivation(input[i]);
        }
    }
}
