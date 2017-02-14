package saivenky.neural;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * Created by saivenky on 2/6/17.
 */
public class ThreadedNeuronSet extends NeuronSet {
    private static int N_THREADS = 2;

    static {
        Runtime runtime = Runtime.getRuntime();
        N_THREADS = runtime.availableProcessors();
    }

    private final ExecutorService pool;
    private List<SplitActivateExecutor> activateExecutors;
    private List<SplitAddCostExecutor> addCostExecutors;
    private List<SplitDescentExecutor> descentExecutors;

    ThreadedNeuronSet(INeuron[] neurons) {
        super(neurons);
        pool = Executors.newFixedThreadPool(N_THREADS);
        activateExecutors = new ArrayList<>();
        addCostExecutors = new ArrayList<>();
        descentExecutors = new ArrayList<>();
        for(int i = 0; i < N_THREADS; i++) {
            activateExecutors.add(new SplitActivateExecutor(N_THREADS, i));
            addCostExecutors.add(new SplitAddCostExecutor(N_THREADS, i));
            descentExecutors.add(new SplitDescentExecutor(N_THREADS, i));
        }
    }

    public void activate() {
        executeOnThreads(pool, activateExecutors);
    }

    public void backpropagate(boolean backpropagateToInputNeurons) {
        for(SplitAddCostExecutor addCostExecutor : addCostExecutors) {
            addCostExecutor.setInput(backpropagateToInputNeurons);
        }

        executeOnThreads(pool, addCostExecutors);
    }

    public void gradientDescent(double rate) {
        for(SplitDescentExecutor descentExecutor : descentExecutors) {
            descentExecutor.setInput(rate);
        }

        executeOnThreads(pool, descentExecutors);
    }

    private static void executeOnThreads(ExecutorService pool, List<? extends Callable<Integer>> callables) {
        List<Future<Integer>> futures = null;
        try {
            futures = pool.invokeAll(callables);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        if(futures == null) {
            throw new RuntimeException("thread pool did not successfully invokeAll");
        }

        for(Future<Integer> future : futures) {
            try {
                future.get();
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        }
    }

    public class SplitActivateExecutor implements Callable<Integer> {
        private int mod;
        private int splitId;

        public SplitActivateExecutor(int mod, int splitId) {
            this.mod = mod;
            this.splitId = splitId;
        }

        @Override
        public Integer call() throws Exception {
            for(int i = 0; i < neurons.length; i++) {
                if (i % mod == splitId) neurons[i].activate();
            }

            return 0;
        }
    }

    public class SplitAddCostExecutor implements Callable<Integer> {
        private int mod;
        private int splitId;
        private boolean backpropagateToInputNeurons;

        public SplitAddCostExecutor(int mod, int splitId) {
            this.mod = mod;
            this.splitId = splitId;
        }

        public void setInput(boolean backpropagateToInputNeurons) {
            this.backpropagateToInputNeurons = backpropagateToInputNeurons;
        }

        @Override
        public Integer call() throws Exception {
            for(int i = 0; i < neurons.length; i++) {
                if (i % mod == splitId) {
                    neurons[i].backpropagate(backpropagateToInputNeurons);
                }
            }

            return 0;
        }
    }

    public class SplitDescentExecutor implements Callable<Integer> {
        private int mod;
        private int splitId;
        private double rate;

        public SplitDescentExecutor(int mod, int splitId) {
            this.mod = mod;
            this.splitId = splitId;
        }

        public void setInput(double rate) {
            this.rate = rate;
        }

        @Override
        public Integer call() throws Exception {
            for(int i = 0; i < neurons.length; i++) {
                if (i % mod == splitId) {
                    neurons[i].gradientDescent(rate);
                }
            }

            return 0;
        }
    }
}
