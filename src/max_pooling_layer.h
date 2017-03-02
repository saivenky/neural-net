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
};

struct max_pooling_layer *create_max_pooling_layer(int *inputShape, int *poolShape, int stride);
struct activation create_activation_max_pooling_layer(struct max_pooling_layer *l, double *inputActivation);
struct gradient create_gradient_max_pooling_layer(struct max_pooling_layer *l, double *inputError);
int destroy_max_pooling_layer(struct max_pooling_layer *l);

void feedforward_max_pooling_layer(struct max_pooling_layer *l, struct activation);
void backpropogate_max_pooling_layer(struct max_pooling_layer *l, struct activation, struct gradient);
#endif
