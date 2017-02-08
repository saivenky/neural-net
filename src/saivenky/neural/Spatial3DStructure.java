package saivenky.neural;

/**
 * Created by saivenky on 2/6/17.
 */
public interface Spatial3DStructure {
    int getWidth();
    int getHeight();
    int getDepth();
    void setShape(int width, int height, int depth);
    INeuron get(int x, int y, int z);
    void set(int x, int y, int z, INeuron neuron);
    INeuron[] getSegment(int startX, int startY, int startZ, int width, int height, int depth);
}
