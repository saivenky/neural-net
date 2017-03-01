#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "activation.h"
#include "gradient.h"
#include "sigmoid_layer.h"

double sigmoid(double z) {
  return 1.0 / (1.0 + exp(-z));
}

double sigmoid1(double z) {
  double s = sigmoid(z);
  return s * (1.0 - s);
}

struct sigmoid_layer *create_sigmoid_layer(int size) {
  struct sigmoid_layer *l = malloc(sizeof(struct sigmoid_layer));
  l->size = size;
  return l;
}

struct activation create_activation_sigmoid_layer(struct sigmoid_layer *l, double *inputActivation) {
  return create_activation(inputActivation, l->size);
}

struct gradient create_gradient_sigmoid_layer(struct sigmoid_layer *l, double *inputError) {
  return create_gradient(inputError, l->size);
}

void destroy_sigmoid_layer(struct sigmoid_layer *l) {
  free(l);
}

void feedforward_sigmoid_layer(struct sigmoid_layer *l, struct activation a) {
  for (int i = 0; i < l->size; i++) {
    a.outputSignal[i] = sigmoid(a.inputActivation[i]);
  }
}

void backpropogate_sigmoid_layer(struct sigmoid_layer *l, struct activation a, struct gradient g) {
  if (g.inputError != NULL) {
    for (int i = 0; i < l->size; i++) {
      g.inputError[i] += sigmoid1(a.inputActivation[i]) * g.outputError[i];
    }
  }
  memset(g.outputError, 0, l->size * sizeof(double));
}
