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
        if (inputActivation != null) inputActivation.order(NATIVE_ORDER);
        if (inputError != null) inputError.order(NATIVE_ORDER);
        if (outputSignal != null) outputSignal.order(NATIVE_ORDER);
        if (outputError != null) outputError.order(NATIVE_ORDER);
    }

    protected ByteBuffer inputActivation;
    protected ByteBuffer inputError;
    protected ByteBuffer outputSignal;
    protected ByteBuffer outputError;
    long nativeLayerPtr;
    protected int[] shape;
}
