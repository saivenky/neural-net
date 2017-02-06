package saivenky.neural;

/**
 * Created by saivenky on 2/6/17.
 */
public interface Spatial2DStructure {
    int getWidth();
    int getHeight();
    void setShape(int width, int height);
    INeuron get(int x, int y);
    INeuron[] getSegment(int startX, int endX, int startY, int endY);
}
