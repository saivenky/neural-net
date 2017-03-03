package saivenky.neural;

/**
 * Created by saivenky on 2/22/17.
 */
public interface IOutputLayer extends ILayer {
    void setExpected(double[][] expected);
    void setSignalCostGradient(double[][] cost);
    void getPredicted(double[][] predicted);
    int size();
}
