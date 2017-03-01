#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H
#include "neuron_props.h"
#include "activation.h"
#include "gradient.h"

struct fully_connected_layer {
  int inputSize;
  int outputSize;
  double *weights;
  double *biases;
  double *weightErrors;
  double *biasErrors;
};

struct fully_connected_layer *create_fully_connected_layer(int inputSize, int outputSize);
struct activation create_activation_fully_connected_layer(struct fully_connected_layer *l, double *inputActivation);
struct gradient create_gradient_fully_connected_layer(struct fully_connected_layer *l, double *inputError);
int destroy_fully_connected_layer(struct fully_connected_layer *l);

void feedforward_fully_connected_layer(struct fully_connected_layer *l, struct activation a);
void backpropogate_fully_connected_layer(struct fully_connected_layer *l, struct activation a, struct gradient g);
void backpropogate_to_input_fully_connected_layer(struct fully_connected_layer *l, struct gradient g);
void backpropogate_to_props_fully_connected_layer(struct fully_connected_layer *l, struct activation a, struct gradient g);
void update_fully_connected_layer(struct fully_connected_layer *l, double rate);
#endif
