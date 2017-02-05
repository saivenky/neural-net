package saivenky.neural;

/**
 * Created by saivenky on 1/31/17.
 */
public class Spatial2DStructure extends NeuronSet {
    final int width;
    final int height;

    public Spatial2DStructure(INeuron[] neurons, int width, int height) {
        super(neurons);
        this.width = width;
        this.height = height;
    }

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

    public INeuron get(int x, int y) {
        return neurons[y * width + x];
    }
}
