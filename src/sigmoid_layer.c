#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "activation.h"
#include "gradient.h"
#include "sigmoid_layer.h"

float_t sigmoid(float_t z) {
  return 1.0f / (1.0f + expf(-z));
}

float_t sigmoid1(float_t z) {
  float_t s = sigmoid(z);
  return s * (1.0f - s);
}

struct sigmoid_layer *create_sigmoid_layer(int size) {
  struct sigmoid_layer *l = malloc(sizeof(struct sigmoid_layer));
  l->size = size;
  return l;
}

struct activation create_activation_sigmoid_layer(struct sigmoid_layer *l, float_t *inputActivation) {
  return create_activation(inputActivation, l->size);
}

struct gradient create_gradient_sigmoid_layer(struct sigmoid_layer *l, float_t *inputError) {
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
  memset(g.outputError, 0, l->size * sizeof(float_t));
}
