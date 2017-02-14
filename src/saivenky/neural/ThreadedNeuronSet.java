package saivenky.neural;

import saivenky.neural.threading.SplitRunnable;
import saivenky.neural.threading.ThreadPool;

/**
 * Created by saivenky on 2/6/17.
 */
public class ThreadedNeuronSet extends NeuronSet {
    private static int N_THREADS = 2;

    private static final ThreadPool pool;

    static {
        Runtime runtime = Runtime.getRuntime();
        N_THREADS = runtime.availableProcessors();
        pool = new ThreadPool(N_THREADS);
    }
    private SplitActivateExecutor[] activateExecutors;
    private SplitAddCostExecutor[] addCostExecutors;
    private SplitDescentExecutor[] descentExecutors;

    ThreadedNeuronSet(INeuron[] neurons) {
        super(neurons);

        activateExecutors = new SplitActivateExecutor[N_THREADS];
        addCostExecutors = new SplitAddCostExecutor[N_THREADS];
        descentExecutors = new SplitDescentExecutor[N_THREADS];
        for(int i = 0; i < N_THREADS; i++) {
            activateExecutors[i] = new SplitActivateExecutor(N_THREADS, i);
            addCostExecutors[i] = new SplitAddCostExecutor(N_THREADS, i);
            descentExecutors[i] = new SplitDescentExecutor(N_THREADS, i);
        }
    }

    public void activate() {
        pool.runAllAndWait(activateExecutors);
    }

    public void backpropagate(boolean backpropagateToInputNeurons) {
        for(SplitAddCostExecutor addCostExecutor : addCostExecutors) {
            addCostExecutor.setInput(backpropagateToInputNeurons);
        }

        pool.runAllAndWait(addCostExecutors);
    }

    public void gradientDescent(double rate) {
        for(SplitDescentExecutor descentExecutor : descentExecutors) {
            descentExecutor.setInput(rate);
        }

        pool.runAllAndWait(descentExecutors);
    }

    public class SplitActivateExecutor extends SplitRunnable {
        public SplitActivateExecutor(int mod, int splitId) {
            super(neurons.length, mod, splitId);
        }

        @Override
        public void singleTask(int i) {
            neurons[i].activate();
        }
    }

    public class SplitAddCostExecutor extends SplitRunnable {
        private boolean backpropagateToInputNeurons;

        public SplitAddCostExecutor(int mod, int splitId) {
            super(neurons.length, mod, splitId);
        }

        public void setInput(boolean backpropagateToInputNeurons) {
            this.backpropagateToInputNeurons = backpropagateToInputNeurons;
        }

        @Override
        public void singleTask(int i) {
            neurons[i].backpropagate(backpropagateToInputNeurons);
        }
    }

    public class SplitDescentExecutor extends SplitRunnable {
        private double rate;

        public SplitDescentExecutor(int mod, int splitId) {
            super(neurons.length, mod, splitId);
        }

        public void setInput(double rate) {
            this.rate = rate;
        }

        @Override
        public void singleTask(int i) {
            neurons[i].gradientDescent(rate);
        }
    }
}
