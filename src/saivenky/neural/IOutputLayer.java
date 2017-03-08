package saivenky.neural;

/**
 * Created by saivenky on 2/22/17.
 */
public interface IOutputLayer extends ILayer {
    void setExpected(float[][] expected);
    void setSignalCostGradient(float[][] cost);
    void getPredicted(float[][] predicted);
    int size();
}
