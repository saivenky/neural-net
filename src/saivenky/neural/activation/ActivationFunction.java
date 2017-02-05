package saivenky.neural.activation;

/**
 * Created by saivenky on 1/26/17.
 */
public interface ActivationFunction {
    double f(double z);
    double f1(double z);

    int SIGMOID = 0x001;
    int LINEAR = 0x002;
    int TANH = 0x003;
    int RELU = 0x004;

    static ActivationFunction get(int activationFunctionId) {
        switch (activationFunctionId) {
            case SIGMOID: return Sigmoid.getInstance();
            case LINEAR: return Linear.getInstance();
            case TANH: return Tanh.getInstance();
            case RELU: return RectifiedLinear.getInstance();
            default: return Sigmoid.getInstance();
        }
    }

    static int id(ActivationFunction activationFunction) {
        if(activationFunction instanceof Sigmoid) return SIGMOID;
        if(activationFunction instanceof Linear) return LINEAR;
        if(activationFunction instanceof Tanh) return TANH;
        if(activationFunction instanceof RectifiedLinear) return RELU;
        throw new RuntimeException("Unknown activation function");
    }
}
