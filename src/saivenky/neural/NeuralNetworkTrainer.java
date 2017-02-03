package saivenky.neural;

/**
 * Created by saivenky on 1/28/17.
 */
public class NeuralNetworkTrainer {
    private NeuralNetwork neuralNetwork;
    private Data.Example[] trainData;

    public NeuralNetworkTrainer(NeuralNetwork neuralNetwork, Data.Example[] trainData) {
        this.neuralNetwork = neuralNetwork;
        this.trainData = trainData;
    }

    public void train(double learningRate, int batchSize, Evaluator evaluator) {
        Data.shuffle(trainData);
        int batchEnd = batchSize - 1;
        int currentBatch = 0;
        int totalBatches = trainData.length / batchSize;
        for(int j = 0; j < trainData.length; j++) {
            Data.Example e = trainData[j];
            neuralNetwork.train(e.input, e.output);
            if (j == batchEnd) {
                neuralNetwork.update(learningRate);
                neuralNetwork.reselectDropouts();
                batchEnd += batchSize;
                if (batchEnd >= trainData.length) batchEnd = trainData.length - 1;
                evaluator.f(currentBatch++);
            }
        }
    }

    public void train(int epochs, double learningRate, int batchSize, Evaluator evaluator) {
        for (int i = 0; i < epochs; i++) {
            train(learningRate, batchSize, evaluator);
        }
    }

    public static abstract class Evaluator {
        public abstract void f(int batchNumber);
    }

    public static final Evaluator NullEvaluator = new Evaluator() {
        @Override
        public void f(int batchNumber) {
        }
    };
}
