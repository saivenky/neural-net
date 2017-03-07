package saivenky.neural;

/**
 * Created by saivenky on 1/28/17.
 */
public class NeuralNetworkTrainer {
    private INeuralNetwork neuralNetwork;
    private Data.Example[] trainData;
    public int batchSize;
    private double learningRate;
    private int epochs;

    public NeuralNetworkTrainer(Data.Example[] trainData) {
        this.trainData = trainData;
        this.neuralNetwork = null;
        batchSize = 0;
        learningRate = 0;
        epochs = 0;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public void setNeuralNetwork(INeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    private void validateTrainingParameters() {
        if (neuralNetwork == null) throw new IllegalArgumentException("neuralNetwork");
        if (batchSize < 1) throw new IllegalArgumentException("batchSize");
        System.out.println("Batch size: " + batchSize);
        if (learningRate < 0) throw new IllegalArgumentException("learningRate");
        System.out.println("Learning rate: " + learningRate);
        if (epochs < 1) throw new IllegalArgumentException("epochs");
        System.out.println("Epochs: " + epochs);
    }

    private void trainSingleEpoch(Evaluator batchEvaluator) {
        Data.shuffle(trainData);
        int batchEnd = batchSize - 1;
        int currentBatch = 0;
        long batchStartTime = System.currentTimeMillis();
        double[][] input = new double[batchSize][];
        double[][] output = new double[batchSize][];
        for(int j = 0; j < trainData.length; j++) {
            Data.Example e = trainData[j];
            input[j % batchSize] = e.input;
            output[j % batchSize] = e.output;
            if (j == batchEnd) {
                neuralNetwork.train(input, output);
                neuralNetwork.update(learningRate);
                batchEnd += batchSize;
                if (batchEnd >= trainData.length) batchEnd = trainData.length - 1;
                long batchEndTime = System.currentTimeMillis();
                batchEvaluator.f(currentBatch++, batchEndTime - batchStartTime);
                batchStartTime = batchEndTime;
            }
        }
    }

    public void train(Evaluator epochEvaluator, Evaluator batchEvaluator) {
        validateTrainingParameters();
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < epochs; i++) {
            trainSingleEpoch(batchEvaluator);
            long endTime = System.currentTimeMillis();
            epochEvaluator.f(i, endTime - startTime);
            startTime = endTime;
        }
    }

    public static abstract class Evaluator {
        public abstract void f(int iteration, long timeTaken);
    }

    public static final Evaluator NullEvaluator = new Evaluator() {
        @Override
        public void f(int iteration, long timeTaken) {
        }
    };
}
