package saivenky.neural;

/**
 * Created by saivenky on 2/5/17.
 */
public interface ILayer {
    void feedforward();
    void backpropagate();
    void gradientDescent(double rate);
}

