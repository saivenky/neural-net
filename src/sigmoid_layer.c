#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sigmoid_layer.h"

double sigmoid(double z) {
  return 1.0 / (1.0 + exp(-z));
}

double sigmoid1(double z) {
  double s = sigmoid(z);
  return s * (1.0 - s);
}

struct sigmoid_layer *create_sigmoid_layer(long size, double *inputActivation, double *inputError) {
  struct sigmoid_layer *l = malloc(sizeof(struct sigmoid_layer));
  l->size = size;
  l->inputActivation = inputActivation;
  l->inputError = inputError;

  l->outputSignal = malloc(size * sizeof(double));
  l->outputError = calloc(size, sizeof(double));

  return l;
}

void destroy_sigmoid_layer(struct sigmoid_layer *l) {
  free(l->outputSignal);
  free(l->outputError);
  free(l);
}

void feedforward_sigmoid_layer(struct sigmoid_layer *layer) {
  for (int i = 0; i < layer->size; i++) {
    layer->outputSignal[i] = sigmoid(layer->inputActivation[i]);
  }
}

void backpropogate_sigmoid_layer(struct sigmoid_layer *l) {
  if (l->inputError != NULL) {
    for (int i = 0; i < l->size; i++) {
      l->inputError[i] += sigmoid1(l->inputActivation[i]) * l->outputError[i];
    }
  }
  memset(l->outputError, 0, l->size * sizeof(double));
}

void update_sigmoid_layer(struct sigmoid_layer *l) {
}
