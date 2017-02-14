package saivenky.neural;

import saivenky.neural.threading.SplitRunnable;
import saivenky.neural.threading.ThreadPool;

/**
 * Created by saivenky on 2/6/17.
 */
public class ThreadedDropoutNeuronSet extends DropoutNeuronSet {
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

    ThreadedDropoutNeuronSet(INeuron[] neurons) {
        super(neurons);
        activateExecutors = new SplitActivateExecutor[N_THREADS];
        addCostExecutors = new SplitAddCostExecutor[N_THREADS];
        descentExecutors = new SplitDescentExecutor[N_THREADS];
        createRunnables();
    }

    private void createRunnables() {
        for(int i = 0; i < N_THREADS; i++) {
            activateExecutors[i] = new SplitActivateExecutor(N_THREADS, i);
            addCostExecutors[i] = new SplitAddCostExecutor(N_THREADS, i);
            descentExecutors[i] = new SplitDescentExecutor(N_THREADS, i);
        }
    }

    void select(int[] selected) {
        boolean sizeDifferent = this.selected.length != selected.length;
        this.selected = selected;
        if (sizeDifferent) {
            createRunnables();
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
            super(selected.length, mod, splitId);
        }

        @Override
        public void singleTask(int i) {
            try {
                neurons[selected[i]].activate();
            }
            catch(ArrayIndexOutOfBoundsException e) {
                throw e;
            }
        }
    }

    public class SplitAddCostExecutor extends SplitRunnable {
        private boolean backpropagateToInputNeurons;

        public SplitAddCostExecutor(int mod, int splitId) {
            super(selected.length, mod, splitId);
        }

        public void setInput(boolean backpropagateToInputNeurons) {
            this.backpropagateToInputNeurons = backpropagateToInputNeurons;
        }

        @Override
        public void singleTask(int i) {
            neurons[selected[i]].backpropagate(backpropagateToInputNeurons);
        }
    }

    public class SplitDescentExecutor extends SplitRunnable {
        private double rate;

        public SplitDescentExecutor(int mod, int splitId) {
            super(selected.length, mod, splitId);
        }

        public void setInput(double rate) {
            this.rate = rate;
        }

        @Override
        public void singleTask(int i) {
            neurons[selected[i]].gradientDescent(rate);
        }
    }
}
