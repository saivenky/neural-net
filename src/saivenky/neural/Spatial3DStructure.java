package saivenky.neural;

/**
 * Created by saivenky on 2/1/17.
 */
public class Spatial3DStructure extends NeuronSet {
    final int width;
    final int height;
    final int depth;
    private final int widthHeight;

    public Spatial3DStructure(INeuron[] neurons, int width, int height, int depth) {
        super(neurons);
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.widthHeight = width * height;
    }

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

    public INeuron get(int x, int y, int z) {
        return neurons[z * widthHeight + y * width + x];
    }

    public void set(int x, int y, int z, INeuron neuron) {
        neurons[z * widthHeight + y * width + x] = neuron;
    }
}
