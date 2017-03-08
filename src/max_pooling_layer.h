#ifndef MAX_POOLING_LAYER_H
#define MAX_POOLING_LAYER_H
#include "activation.h"
#include "gradient.h"
#include "kernel_dim.h"

struct max_pooling_layer {
  struct shape inputShape;
  struct shape outputShape;
  struct shape poolShape;
  struct stride inputStride;
};

struct max_pooling_layer *create_max_pooling_layer(int *inputShape, int *poolShape, int stride);
struct activation create_activation_max_pooling_layer(struct max_pooling_layer *l, float_t *inputActivation);
struct gradient create_gradient_max_pooling_layer(struct max_pooling_layer *l, float_t *inputError);
int destroy_max_pooling_layer(struct max_pooling_layer *l);

void feedforward_max_pooling_layer(struct max_pooling_layer *l, struct activation);
void backpropogate_max_pooling_layer(struct max_pooling_layer *l, struct activation, struct gradient);
#endif
