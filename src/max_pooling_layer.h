#ifndef MAX_POOLING_LAYER_H
#define MAX_POOLING_LAYER_H
#include "kernel_dim.h"

struct max_pooling_layer {
  struct shape inputShape;
  struct shape outputShape;
  struct shape poolShape;
  struct dim inputDim;
  struct dim outputDim;
  struct dim poolDim;
  struct dim inputStrideDim;
  double *inputActivation;
  double *inputError;
  double *outputSignal;
  double *outputError;
  int *outputArgmaxIndex;
};

struct max_pooling_layer *create_max_pooling_layer(int *inputShape, int *poolShape, int stride, double *inputActivation, double *inputError);
int destroy_max_pooling_layer(struct max_pooling_layer *l);
void feedforward_max_pooling_layer(struct max_pooling_layer *l);
void backpropogate_max_pooling_layer(struct max_pooling_layer *l);
#endif
