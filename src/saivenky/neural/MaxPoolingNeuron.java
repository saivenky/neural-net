package saivenky.neural;

/**
 * Created by saivenky on 2/1/17.
 */
public class MaxPoolingNeuron implements INeuron {
    private NeuronSet inputNeurons;
    private double activation;
    private int indexOfMaxInputNeuron;

    MaxPoolingNeuron(NeuronSet inputNeurons) {
        this.inputNeurons = inputNeurons;
        activation = 0;
        indexOfMaxInputNeuron = -1;
    }

    public double getActivation() {
        return activation;
    }

    public void activate() {
        activation = 0;
        indexOfMaxInputNeuron = -1;
        for (int i = 0; i < inputNeurons.size(); i++) {
            double inputNeuronActivation = inputNeurons.get(i).getActivation();
            if(inputNeuronActivation > activation) {
                indexOfMaxInputNeuron = i;
                activation = inputNeuronActivation;
            }
        }
    }

    public void activateScaled(double scale) {
        activate();
        activation *= scale;
    }

    public void update(double rate) {}

    public void addToSignalCostGradient(double weight, double cost) {
        double poolCost = weight * cost;
        inputNeurons.get(indexOfMaxInputNeuron).addToSignalCostGradient(1, poolCost);
    }

    public void multiplyByActivation1() {
    }

    public void propagateToInputNeurons() {}

    public void propagateToProperties() {}
}
