#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "fully_connected_layer.h"
#include "rand.h"
#include "activation.h"
#include "gradient.h"

struct fully_connected_layer *create_fully_connected_layer(int inputSize, int outputSize) {
  struct fully_connected_layer *l = malloc(sizeof(struct fully_connected_layer));
  l->inputSize = inputSize;
  l->outputSize = outputSize;

  int weightsLen = inputSize * outputSize;
  l->weights = malloc(weightsLen * sizeof(double));
  l->biases = malloc(outputSize * sizeof(double));
  l->weightErrors = calloc(weightsLen, sizeof(double));
  l->biasErrors = calloc(outputSize, sizeof(double));
  init_rand_truncated_norm(l->weights, weightsLen, 0.1);
  init_const(l->biases, outputSize, 0.1);
  return l;
}

struct activation create_activation_fully_connected_layer(struct fully_connected_layer *l, double *inputActivation) {
  return create_activation(inputActivation, l->outputSize);
}

struct gradient create_gradient_fully_connected_layer(struct fully_connected_layer *l, double *inputError) {
  return create_gradient(inputError, l->outputSize);
}

int destroy_fully_connected_layer(struct fully_connected_layer *l) {
  free(l->weights);
  free(l->biases);
  free(l->weightErrors);
  free(l->biasErrors);
  free(l);
  return 0;
}

void feedforward_fully_connected_layer(struct fully_connected_layer *l, struct activation a) {
  for (int i = 0; i < l->outputSize; i++) {
    double sum = l->biases[i];
    double *weights = l->weights + i * l->inputSize;
    for (int j = 0; j < l->inputSize; j++) {
      sum += a.inputActivation[j] * weights[j];
    }
    a.outputSignal[i] = sum;
  }
}

void backpropogate_fully_connected_layer(struct fully_connected_layer *l, struct activation a, struct gradient g) {
  backpropogate_to_props_fully_connected_layer(l, a, g);
  if (g.inputError != NULL) {
    backpropogate_to_input_fully_connected_layer(l, g);
  }
  memset(g.outputError, 0, l->outputSize * sizeof(double));
}

void backpropogate_to_input_fully_connected_layer(struct fully_connected_layer *l, struct gradient g) {
  for (int i = 0; i < l->outputSize; i++) {
    double error = g.outputError[i];
    double *weights = l->weights + i * l->inputSize;
    for (int j = 0; j < l->inputSize; j++) {
      g.inputError[j] += error * weights[j];
    }
  }
}

void backpropogate_to_props_fully_connected_layer(struct fully_connected_layer *l, struct activation a, struct gradient g) {
  for (int i = 0; i < l->outputSize; i++) {
    double error = g.outputError[i];
    l->biasErrors[i] += error;
    double *weightErrors = l->weightErrors + i * l->inputSize;
    for(int j = 0; j < l->inputSize; j++) {
      weightErrors[j] += error * a.inputActivation[j];
    }
  }
}

void update_fully_connected_layer(struct fully_connected_layer *l, double rate) {
  for (int i = 0; i < l->outputSize; i++) {
    l->biases[i] -= rate * l->biasErrors[i];
  }

  int weightsLen = l->inputSize * l->outputSize;
  for (int i = 0; i < weightsLen; i++) {
    l->weights[i] -= rate * l->weightErrors[i];
  }
  memset(l->biasErrors, 0, l->outputSize * sizeof(double));
  memset(l->weightErrors, 0, weightsLen * sizeof(double));
}
