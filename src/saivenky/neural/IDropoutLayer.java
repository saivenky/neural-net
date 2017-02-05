package saivenky.neural;

public interface IDropoutLayer extends ILayer {
    void runWithoutDropout();
    void reselectDropout();
}
