package saivenky.neural.c;

import saivenky.neural.ILayer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Created by saivenky on 2/19/17.
 */
public abstract class Layer implements ILayer {
    static final int SIZEOF_DOUBLE = 8;

    private static final ByteOrder NATIVE_ORDER;

    static {
        System.loadLibrary("neural");
        System.out.println("Loaded 'neural' library");
        NATIVE_ORDER = ByteOrder.nativeOrder();
    }

    void adjustByteOrderOnBuffers() {
        if (outputSignals != null) {
            for (ByteBuffer outputSignal : outputSignals) {
                outputSignal.order(NATIVE_ORDER);
            }
        }

        if (outputErrors != null) {
            for (ByteBuffer outputError : outputErrors) {
                outputError.order(NATIVE_ORDER);
            }
        }
    }

    protected ByteBuffer[] outputSignals;
    protected ByteBuffer[] outputErrors;
    long nativeLayerPtr;
    protected int[] shape;
}
