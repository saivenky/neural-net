#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H
#include "kernel_dim.h"
#include "activation.h"
#include "gradient.h"

struct convolution_layer {
  struct shape inputShape;
  struct shape outputShape;
  struct shape kernelShape;
  int padding;
  float_t *weights;
  float_t *biases;
};

struct convolution_layer_gradient {
  float_t *weightErrors;
  float_t *biasErrors;
};

struct convolution_layer *create_convolution_layer(int *inputShape, int *kernelShape, int frames, int stride, int padding);
struct activation create_activation_convolution_layer(struct convolution_layer *l, float_t *inputActivation);
struct gradient create_gradient_convolution_layer(struct convolution_layer *l, float_t *inputError);
int destroy_convolution_layer(struct convolution_layer *l);

void feedforward_convolution_layer(struct convolution_layer *l, struct activation a);
void backpropogate_convolution_layer(struct convolution_layer *l, struct activation a, struct gradient g);
void backpropogate_to_input_convolution_layer(struct convolution_layer *l, struct gradient g);
void backpropogate_to_props_convolution_layer(struct convolution_layer *l, struct activation a, struct gradient g);
void update_convolution_layer(struct convolution_layer *l, float_t rate, struct gradient g);
#endif
