package saivenky.neural;

/**
 * Created by saivenky on 1/30/17.
 */
public class NeuronSet implements Spatial3DStructure {
    INeuron[] neurons;

    private int width;
    private int height;
    private int depth;
    private int widthHeight;

    public NeuronSet(INeuron[] neurons) {
        this.neurons = neurons;
    }

    public INeuron get(int index) {
        return neurons[index];
    }

    void set(int index, INeuron neuron) {
        neurons[index] = neuron;
    }

    public int size() {
        return neurons.length;
    }

    public void activate() {
        for(INeuron neuron : neurons) {
            neuron.activate();
        }
    }

    public void backpropagate(boolean backpropagateToInputNeurons) {
        for(INeuron neuron : neurons) {
            neuron.backpropagate(backpropagateToInputNeurons);
        }
    }

    public void gradientDescent(double rate) {
        for(INeuron neuron : neurons) {
            neuron.gradientDescent(rate);
        }
    }

    public void addSignalCostGradient(double[] weight, double cost) {
        for(int i = 0; i < neurons.length; i++) {
            neurons[i].addToSignalCostGradient(weight[i], cost);
        }
    }

    public double affine(double[] weight, double bias) {
        double sum = 0;
        for(int i = 0; i < neurons.length; i++) {
            sum += weight[i] * neurons[i].getActivation();
        }

        return sum + bias;
    }

    public void addToWeightError(double[] weightError, double cost) {
        for(int i = 0; i < neurons.length; i++) {
            weightError[i] += neurons[i].getActivation() * cost;
        }
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append('[');
        for(int i = 0; i < neurons.length - 1; i++) {
            builder.append(neurons[i].getActivation() + ", ");
        }
        builder.append(neurons[neurons.length - 1].getActivation() + "]");
        return builder.toString();
    }

    @Override
    public int getWidth() {
        return width;
    }

    @Override
    public int getHeight() {
        return height;
    }

    @Override
    public int getDepth() {
        return depth;
    }

    @Override
    public void setShape(int width, int height, int depth) {
        this.width = width;
        this.height = height;
        this.depth = depth;
        widthHeight = width * height;
    }

    @Override
    public INeuron get(int x, int y, int z) {
        return neurons[z * widthHeight + y * width + x];
    }

    @Override
    public void set(int x, int y, int z, INeuron neuron) {
        neurons[z * widthHeight + y * width + x] = neuron;
    }

    @Override
    public INeuron[] getSegment(int startX, int startY, int startZ, int segmentWidth, int segmentHeight, int segmentDepth) {
        INeuron[] neurons = new INeuron[segmentWidth * segmentHeight * segmentDepth];
        int i = 0;
        int endX = startX + segmentWidth;
        int endY = startY + segmentHeight;
        int endZ = startZ + segmentDepth;

        for (int x = startX; x < endX; x++) {
            for (int y = startY; y < endY; y++) {
                for (int z = startZ; z < endZ; z++) {
                    neurons[i++] = get(x, y, z);
                }
            }
        }

        return neurons;
    }
}
