#include <stdlib.h>
#include <stdio.h>
#include <float.h>
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
    struct softmax_cross_entropy_layer *l, double *inputActivation) {
  return create_activation(inputActivation, l->size);
}

struct gradient create_gradient_softmax_cross_entropy_layer(
    struct softmax_cross_entropy_layer *l, double *inputError) {
  return create_gradient_passthru(inputError);
}

void destroy_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l) {
  free(l);
}

void feedforward_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l, struct activation a) {
  double max_val = -DBL_MAX;
  for (int i = 0; i < l->size; i++) {
    if (a.inputActivation[i] > max_val) {
      max_val = a.inputActivation[i];
    }
  }

  double sum = 0.0;
  for (int i = 0; i < l->size; i++) {
    double soft_out = exp(a.inputActivation[i]-max_val);
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

void set_expected_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l, struct activation a, struct gradient g, double *expected) {
  for (int i = 0; i < l->size; i++) {
    g.inputError[i] =  a.outputSignal[i] - expected[i];
  }
}
