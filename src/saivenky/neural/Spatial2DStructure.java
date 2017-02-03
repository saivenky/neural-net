package saivenky.neural;

/**
 * Created by saivenky on 1/31/17.
 */
public class Spatial2DStructure {
    private Neuron[] neurons;
    final int width;
    final int height;

    public Spatial2DStructure(Neuron[] neurons, int width, int height) {
        //0 1 2
        //3 4 5
        //6 7 8
        this.neurons = neurons;
        this.width = width;
        this.height = height;
    }

    public Neuron[] getSegment(int startX, int endX, int startY, int endY) {
        Neuron[] neurons = new Neuron[(endX - startX + 1) * (endY - startY + 1)];
        int i = 0;
        for(int x = startX; x <= endX; x++) {
            for(int y = startY; y <= endY; y++) {
                neurons[i++] = get(x, y);
            }
        }

        return neurons;
    }

    public Neuron get(int x, int y) {
        return neurons[y * width + x];
    }
}
