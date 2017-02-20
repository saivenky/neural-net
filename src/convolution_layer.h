#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include "neuron_props.h"

#define SHAPE_LEN 3
#define SHAPE_SIZE sizeof(int) * 3
#define LAST_DIM 2

struct shape {
  int width;
  int height;
  int depth;
};

struct dim {
  int dim0;
  int dim1;
  int dim2;
};

struct convolution_layer {
  struct shape inputShape;
  struct shape outputShape;
  struct shape kernelShape;
  struct dim inputDim;
  struct dim outputDim;
  struct dim kernelDim;
  int frames;
  double *inputActivation;
  double *inputError;
  double *outputSignal;
  double *outputError;
  struct properties **props;
};

struct shape calcoutsize(struct shape inputShape, struct shape kernelShape, int stride);
struct dim calcdim(struct shape);

struct convolution_layer *create_convolution_layer(int *inputShape, int *kernelShape, int frames, int stride, double *inputActivation, double *inputError);
int destroy_convolution_layer(struct convolution_layer *l);
void feedforward_convolution_layer(struct convolution_layer *l);
void backpropogate_convolution_layer(struct convolution_layer *l);
void backpropogate_to_input_convolution_layer(struct convolution_layer *l);
void backpropogate_to_props_convolution_layer(struct convolution_layer *l);
void update_convolution_layer(struct convolution_layer *l, double rate);
#endif
