package saivenky.neural.threading;

/**
 * Created by saivenky on 2/14/17.
 */
public abstract class SplitRunnable implements Runnable {
    private final int start;
    private final int end;

    private static int[] getRange(int elements, int total, int splitNumber) {
        //i.e. total = 4, split = {0, 1, 2, 3}
        int size = elements / total;
        int start = size * splitNumber;
        int end = elements;

        if (splitNumber + 1 != total) {
            end = size * (splitNumber + 1);
        }

        return new int[]{start, end};
    }

    public SplitRunnable(int elements, int totalSplits, int splitNumber) {
        int[] range = getRange(elements, totalSplits, splitNumber);
        start = range[0];
        end = range[1];
        System.out.printf("total %d, split %d, range: [%d, %d)\n", totalSplits, splitNumber, start, end);
    }

    public void run() {
        for(int i = start; i < end; i++) {
            singleTask(i);
        }
    }

    public abstract void singleTask(int i);
}
