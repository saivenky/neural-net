#ifndef NEURON_H
#define NEURON_H

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

struct layer {
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

struct layer;
struct layer *create_layer(int *inputShape, int *kernelShape, int frames, int stride);
int destroy_layer(struct layer *l);
void apply_kernel(struct layer *l);
void backpropogate_to_props(struct layer *l);
#endif
