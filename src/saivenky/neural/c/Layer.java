package saivenky.neural.c;

import saivenky.neural.ILayer;

import java.nio.ByteBuffer;

/**
 * Created by saivenky on 2/19/17.
 */
public abstract class Layer implements ILayer {
    static final int SIZEOF_DOUBLE = 8;

    protected ByteBuffer inputActivation;
    protected ByteBuffer inputError;
    protected ByteBuffer outputSignal;
    protected ByteBuffer outputError;
    protected long nativeLayerPtr;
}
