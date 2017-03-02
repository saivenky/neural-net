#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H
#include "kernel_dim.h"
#include "neuron_props.h"
#include "activation.h"
#include "gradient.h"

struct convolution_layer {
  struct shape inputShape;
  struct shape outputShape;
  struct shape kernelShape;
  int padding;
  struct properties **props;
};

struct convolution_layer *create_convolution_layer(int *inputShape, int *kernelShape, int frames, int stride, int padding);
struct activation create_activation_convolution_layer(struct convolution_layer *l, double *inputActivation);
struct gradient create_gradient_convolution_layer(struct convolution_layer *l, double *inputError);
int destroy_convolution_layer(struct convolution_layer *l);

void feedforward_convolution_layer(struct convolution_layer *l, struct activation a);
void backpropogate_convolution_layer(struct convolution_layer *l, struct activation a, struct gradient g);
void backpropogate_to_input_convolution_layer(struct convolution_layer *l, struct gradient g);
void backpropogate_to_props_convolution_layer(struct convolution_layer *l, struct activation a, struct gradient g);
void update_convolution_layer(struct convolution_layer *l, double rate, struct gradient g);
#endif
