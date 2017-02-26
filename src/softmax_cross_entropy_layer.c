#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include "softmax_cross_entropy_layer.h"

struct softmax_cross_entropy_layer *create_softmax_cross_entropy_layer(long size, double *inputActivation, double *inputError) {
  struct softmax_cross_entropy_layer *l = malloc(sizeof(struct softmax_cross_entropy_layer));
  l->size = size;
  l->inputActivation = inputActivation;
  l->inputError = inputError;
  l->outputSignal = malloc(size * sizeof(double));
  return l;
}

void destroy_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l) {
  free(l->outputSignal);
  free(l);
}

void feedforward_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l) {
  double max_val = -DBL_MAX;
  for (int i = 0; i < l->size; i++) {
    if (l->inputActivation[i] > max_val) {
      max_val = l->inputActivation[i];
    }
  }

  double sum = 0.0;
  for (int i = 0; i < l->size; i++) {
    double soft_out = exp(l->inputActivation[i]-max_val);
    if (isnan(soft_out) || isinf(soft_out)) {
      printf("ERROR: softmax failed, number too large\n");
      exit(-1);
    }
    l->outputSignal[i] = soft_out;
    sum += soft_out;
  }

  for (int i = 0; i < l->size; i++) {
    l->outputSignal[i] /= sum;
  }
}

void set_expected_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l, double *expected) {
  for (int i = 0; i < l->size; i++) {
    l->inputError[i] =  l->outputSignal[i] - expected[i];
  }
}
