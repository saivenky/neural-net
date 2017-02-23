#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H
#include "neuron_props.h"

struct fully_connected_layer {
  long inputSize;
  long outputSize;
  double *inputActivation;
  double *inputError;
  double *outputSignal;
  double *outputError;
  double *weights;
  double *biases;
  double *weightErrors;
  double *biasErrors;
};

struct fully_connected_layer *create_fully_connected_layer(long inputSize, long outputSize, double *inputActivation, double *inputError);
int destroy_fully_connected_layer(struct fully_connected_layer *l);
void feedforward_fully_connected_layer(struct fully_connected_layer *l);
void backpropogate_fully_connected_layer(struct fully_connected_layer *l);
void backpropogate_to_input_fully_connected_layer(struct fully_connected_layer *l);
void backpropogate_to_props_fully_connected_layer(struct fully_connected_layer *l);
void update_fully_connected_layer(struct fully_connected_layer *l, double rate);
#endif
