package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;

/**
 * Created by saivenky on 1/30/17.
 */
public class NeuronSet {
    Neuron[] neurons;
    double[] signalCostGradient;
    boolean newSignalCostAdded; //optimization to only multiply by activation1 once
    boolean isSignalCostZeroed; //optimization to zero only once

    int[] selected;
    double[] activation;
    private double[] activation1;

    static abstract class NeuronAction {
        abstract void f(Neuron neuron, int i);
    }

    public static abstract class NeuronWithCostAction {
        abstract void f(Neuron neuron, double cost, int i);
    }

    NeuronSet(Neuron[] neurons) {
        this.neurons = neurons;
        selected = null;
        activation = new double[neurons.length];
        activation1 = new double[neurons.length];
        signalCostGradient = new double[neurons.length];
        newSignalCostAdded = false;
        isSignalCostZeroed = true;

    }

    NeuronSet() {
    }

    void addSignalCostGradient(double[] cost, double scalar) {
        newSignalCostAdded = true;
        isSignalCostZeroed = false;
        Vector.multiplyAndAddSelected(cost, scalar, signalCostGradient, selected);
    }

    void completeSignalCostGradient() {
        if(newSignalCostAdded) {
            Vector.multiplySelected(signalCostGradient, activation1, signalCostGradient, selected);
            newSignalCostAdded = false;
        }
    }

    void clearSignalCostGradient() {
        if (!isSignalCostZeroed) {
            isSignalCostZeroed = true;
            Vector.zero(signalCostGradient);
        }
    }

    void select(int[] selected) {
        this.selected = selected;
    }

    Neuron get(int index) {
        return neurons[index];
    }

    void set(int index, Neuron neuron) {
        neurons[index] = neuron;
    }

    int size() {
        return neurons.length;
    }

    void forSelected(NeuronAction action) {
        if(selected == null) {
            forAll(action);
            return;
        }

        for(int i : selected) {
            action.f(neurons[i], i);
        }
    }

    void forSelectedWithCost(NeuronWithCostAction action) {
        if(selected == null) {
            forAll(action);
            return;
        }

        for(int i : selected) {
            action.f(neurons[i], signalCostGradient[i], i);
        }
    }

    private void forAll(NeuronAction action) {
        for (int i = 0; i < neurons.length; i++) {
            action.f(neurons[i], i);
        }
    }

    private void forAll(NeuronWithCostAction action) {
        for (int i = 0; i < neurons.length; i++) {
            action.f(neurons[i], signalCostGradient[i], i);
        }
    }

    public void activate(ActivationFunction activationFunction) {
        forSelected(new NeuronAction() {
            @Override
            void f(Neuron neuron, int i) {
                double signal = neuron.signalForSelected();
                activation[i] = activationFunction.f(signal);
                activation1[i] = activationFunction.f1(signal);
            }
        });
    }

    public void activateScaled(ActivationFunction activationFunction, double scale) {
        forAll(new NeuronAction() {
            @Override
            void f(Neuron neuron, int i) {
                double signal = neuron.signalScaled(scale);
                activation[i] = activationFunction.f(signal);
                activation1[i] = activationFunction.f1(signal);
            }
        });
    }
}
