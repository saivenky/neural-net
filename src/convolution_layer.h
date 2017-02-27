#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H
#include "kernel_dim.h"
#include "neuron_props.h"

struct convolution_layer {
  struct shape inputShape;
  struct shape outputShape;
  struct shape kernelShape;
  struct dim inputDim;
  struct dim outputDim;
  struct dim kernelDim;
  int padding;
  double *inputActivation;
  double *inputError;
  double *outputSignal;
  double *outputError;
  struct properties **props;
};

struct convolution_layer *create_convolution_layer(int *inputShape, int *kernelShape, int frames, int stride, int padding, double *inputActivation, double *inputError);
int destroy_convolution_layer(struct convolution_layer *l);
void feedforward_convolution_layer(struct convolution_layer *l);
void backpropogate_convolution_layer(struct convolution_layer *l);
void backpropogate_to_input_convolution_layer(struct convolution_layer *l);
void backpropogate_to_props_convolution_layer(struct convolution_layer *l);
void update_convolution_layer(struct convolution_layer *l, double rate);
#endif
