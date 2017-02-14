package saivenky.neural.threading;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Created by saivenky on 2/14/17.
 */
public class ThreadPool {
    private final ExecutorService pool;
    private final Future[] futures;

    public ThreadPool(int size) {
        pool = Executors.newFixedThreadPool(size);
        futures = new Future[size];
    }

    public void runAllAndWait(Runnable[] runnables) {
        for(int i = 0; i < runnables.length; i++) {
            futures[i] = pool.submit(runnables[i]);
        }

        waitAll();
    }

    private void waitAll() {
        for(Future future : futures) {
            try {
                future.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            } catch (ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
