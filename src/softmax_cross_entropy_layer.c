#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "activation.h"
#include "gradient.h"
#include "softmax_cross_entropy_layer.h"

struct softmax_cross_entropy_layer *create_softmax_cross_entropy_layer(int size) {
  struct softmax_cross_entropy_layer *l = malloc(sizeof(struct softmax_cross_entropy_layer));
  l->size = size;
  return l;
}

struct activation create_activation_softmax_cross_entropy_layer(
    struct softmax_cross_entropy_layer *l, float_t *inputActivation) {
  return create_activation(inputActivation, l->size);
}

struct gradient create_gradient_softmax_cross_entropy_layer(
    struct softmax_cross_entropy_layer *l, float_t *inputError) {
  struct gradient g = create_gradient_passthru(inputError);
  g.extra = (float_t *)malloc(l->size * sizeof(float_t));
  return g;
}

void destroy_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l) {
  free(l);
}

void feedforward_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l, struct activation a) {
  float_t max_val = -FLOAT_MAX;
  for (int i = 0; i < l->size; i++) {
    if (a.inputActivation[i] > max_val) {
      max_val = a.inputActivation[i];
    }
  }

  float_t sum = 0.0;
  for (int i = 0; i < l->size; i++) {
    float_t soft_out = expf(a.inputActivation[i]-max_val);
    if (isnan(soft_out) || isinf(soft_out)) {
      printf("ERROR: softmax failed, number too large\n");
      exit(-1);
    }
    a.outputSignal[i] = soft_out;
    sum += soft_out;
  }

  for (int i = 0; i < l->size; i++) {
    a.outputSignal[i] /= sum;
  }
}

void backpropogate_softmax_cross_entropy_layer(
    struct softmax_cross_entropy_layer *l, struct activation a, struct gradient g) {
  float_t *expected = (float_t *)g.extra;
  for (int i = 0; i < l->size; i++) {
    g.inputError[i] =  a.outputSignal[i] - expected[i];
  }
}

void set_expected_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l, struct gradient g, float_t *expected) {
  memcpy(g.extra, expected, l->size * sizeof(float_t));
}
