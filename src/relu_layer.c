#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "activation.h"
#include "gradient.h"
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

struct relu_layer *create_relu_layer(int size) {
  struct relu_layer *l = malloc(sizeof(struct relu_layer));
  l->size = size;
  return l;
}

struct activation create_activation_relu_layer(struct relu_layer *l, double *inputActivation) {
  return create_activation(inputActivation, l->size);
}

struct gradient create_gradient_relu_layer(struct relu_layer *l, double *inputError) {
  return create_gradient(inputError, l->size);
}

void destroy_relu_layer(struct relu_layer *l) {
  free(l);
}

void feedforward_relu_layer(struct relu_layer *l, struct activation a) {
  for (int i = 0; i < l->size; i++) {
    a.outputSignal[i] = relu(a.inputActivation[i]);
  }
}

void backpropogate_relu_layer(struct relu_layer *l, struct activation a, struct gradient g) {
  if (g.inputError != NULL) {
    for (int i = 0; i < l->size; i++) {
      g.inputError[i] += relu1(a.inputActivation[i]) * g.outputError[i];
    }
  }
  memset(g.outputError, 0, l->size * sizeof(double));
}
