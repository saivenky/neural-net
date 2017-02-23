package saivenky.neural;

/**
 * Created by saivenky on 2/22/17.
 */
public interface IOutputLayer extends ILayer {
    void getPredicted(double[] predicted);
    int size();
}
