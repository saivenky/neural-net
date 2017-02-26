#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "relu_layer.h"

double relu(double z) {
  if (z < 0.0) {
    return 0.0;
  }
  return z;
}

double relu1(double z) {
  if (z < 0.0) {
    return 0.0;
  }
  return 1.0;
}

struct relu_layer *create_relu_layer(long size, double *inputActivation, double *inputError) {
  struct relu_layer *l = malloc(sizeof(struct relu_layer));
  l->size = size;
  l->inputActivation = inputActivation;
  l->inputError = inputError;

  l->outputSignal = malloc(size * sizeof(double));
  l->outputError = calloc(size, sizeof(double));

  return l;
}

void destroy_relu_layer(struct relu_layer *l) {
  free(l->outputSignal);
  free(l->outputError);
  free(l);
}

void feedforward_relu_layer(struct relu_layer *layer) {
  for (int i = 0; i < layer->size; i++) {
    layer->outputSignal[i] = relu(layer->inputActivation[i]);
  }
}

void backpropogate_relu_layer(struct relu_layer *l) {
  if (l->inputError != NULL) {
    for (int i = 0; i < l->size; i++) {
      l->inputError[i] += relu1(l->inputActivation[i]) * l->outputError[i];
    }
  }
  memset(l->outputError, 0, l->size * sizeof(double));
}
