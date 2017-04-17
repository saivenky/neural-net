# neural-net - Self-Exercise of Implementing a CNN in C

Personal project to create an efficient Convolutional Neural Network in Java.
This transformed into an implementation in C with an interface in Java provided
through JNI.

## Supported Layers
- Convolution
- Max-pooling
- ReLU
- Sigmoid
- SoftMax + Cross Entropy
- Fully-connected

## Training
Only stochastic gradient descent is implemented. See [Future Plans](#future-plans).

## Example Usage in Java
Each C-implemented layer has a wrapper class in Java.

See snippet from saivenky.neural.mnist.MnistTester for sample initialization:

```java
InputLayer inputLayer = new InputLayer(IMAGE_WIDTH * IMAGE_HEIGHT, trainer.batchSize);
inputLayer.setShape(IMAGE_WIDTH, IMAGE_HEIGHT, 1);

int[] kernelShape = {5, 5, 1};
int[] poolShape = {2, 2, 1};
ConvolutionLayer convolutionLayer = new ConvolutionLayer(inputLayer, kernelShape, 20, 0);
MaxPoolingLayer poolingLayer = new MaxPoolingLayer(convolutionLayer, poolShape, 2);
ReluLayer reluLayer1 = new ReluLayer(poolingLayer);

FullyConnectedLayer fcLayer1 = new FullyConnectedLayer(reluLayer1, 100);
ReluLayer reluLayer2 = new ReluLayer(fcLayer1);

FullyConnectedLayer fcLayer2 = new FullyConnectedLayer(reluLayer2, 10);
SoftmaxCrossEntropyLayer outputLayer = new SoftmaxCrossEntropyLayer(fcLayer2, trainer.batchSize);

saivenky.neural.c.NeuralNetwork nn = new saivenky.neural.c.NeuralNetwork(
        trainer.batchSize,
        inputLayer,
        convolutionLayer,
        poolingLayer,
        reluLayer1,
        fcLayer1,
        reluLayer2,
        fcLayer2,
        outputLayer);
```

## Future Plans
- FFT for convolution
- ADAM optimization
- Efficient dropout (or batch normalization)
