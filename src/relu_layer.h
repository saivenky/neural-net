#ifndef RELU_LAYER_H
#define RELU_LAYER_H
#include "activation.h"
#include "gradient.h"

struct relu_layer {
  int size;
};

struct relu_layer *create_relu_layer(int size);
struct activation create_activation_relu_layer(struct relu_layer *, float_t *inputActivation);
struct gradient create_gradient_relu_layer(struct relu_layer *, float_t *inputError);
void destroy_relu_layer(struct relu_layer *l);
void feedforward_relu_layer(struct relu_layer *l, struct activation a);
void backpropogate_relu_layer(struct relu_layer *l, struct activation, struct gradient);
#endif
