#ifndef SIGMOID_LAYER_H
#define SIGMOID_LAYER_H
#include "activation.h"
#include "gradient.h"

struct sigmoid_layer {
  int size;
};

struct sigmoid_layer *create_sigmoid_layer(int size);
struct activation create_activation_sigmoid_layer(struct sigmoid_layer *, double *inputActivation);
struct gradient create_gradient_sigmoid_layer(struct sigmoid_layer *, double *inputError);
void destroy_sigmoid_layer(struct sigmoid_layer *l);
void feedforward_sigmoid_layer(struct sigmoid_layer *l, struct activation);
void backpropogate_sigmoid_layer(struct sigmoid_layer *l, struct activation, struct gradient);
#endif
