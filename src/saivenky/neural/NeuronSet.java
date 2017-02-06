package saivenky.neural;

/**
 * Created by saivenky on 1/30/17.
 */
public class NeuronSet implements Spatial2DStructure, Spatial3DStructure {
    INeuron[] neurons;
    public int[] selected;

    private int width;
    private int height;
    private int depth;
    private int widthHeight;

    public NeuronSet(INeuron[] neurons) {
        this.neurons = neurons;
        selected = new int[neurons.length];
        Vector.select(selected, neurons.length);
    }

    void select(int[] selected) {
        this.selected = selected;
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
        for(int i : selected) {
            neurons[i].activate();
        }
    }

    public void activateScaled(double scale) {
        for(int i = 0; i < neurons.length; i++) {
            neurons[i].activateScaled(scale);
        }
    }

    public void backpropagate(boolean backpropagateToInputNeurons) {
        for(int i : selected) {
            neurons[i].backpropagate(backpropagateToInputNeurons);
        }
    }

    public void gradientDescent(double rate) {
        for(int i : selected) {
            neurons[i].gradientDescent(rate);
        }
    }

    public void addSignalCostGradient(double[] weight, double cost) {
        for(int i : selected) {
            neurons[i].addToSignalCostGradient(weight[i], cost);
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
    public INeuron[] getSegmentSlice(int startX, int endX, int startY, int endY, int z) {
        INeuron[] neurons = new INeuron[(endX - startX + 1) * (endY - startY + 1)];
        int i = 0;
        for(int x = startX; x <= endX; x++) {
            for(int y = startY; y <= endY; y++) {
                neurons[i++] = get(x, y, z);
            }
        }

        return neurons;
    }

    @Override
    public void setShape(int width, int height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public INeuron get(int x, int y) {
        return neurons[y * width + x];
    }

    @Override
    public INeuron[] getSegment(int startX, int endX, int startY, int endY) {
        INeuron[] neurons = new INeuron[(endX - startX + 1) * (endY - startY + 1)];
        int i = 0;
        for(int x = startX; x <= endX; x++) {
            for(int y = startY; y <= endY; y++) {
                neurons[i++] = get(x, y);
            }
        }

        return neurons;
    }
}
