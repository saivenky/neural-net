#include <stdlib.h>
#include "neuron_props.h"
#include "rand.h"

struct properties *create_properties(int inputSize) {
  struct properties *p = malloc(sizeof(struct properties));
  p->inputSize = inputSize;
  p->weights = malloc(sizeof(double) * inputSize);
  init_rand_norm(p->weights, inputSize, 1.0);
  p->weightErrors = calloc(inputSize, sizeof(double));
  p->bias = rand_norm(1.0);
  p->biasError = 0;
  return p;
}

int destroy_properties(struct properties *p) {
  free(p->weights);
  free(p->weightErrors);
  free(p);
  return 0;
}

void update_properties(struct properties *p, double rate) {
  for(int i = 0; i < p->inputSize; i++) {
    p->weights[i] -= rate * p->weightErrors[i];
    p->weightErrors[i] = 0;
  }
  p->bias -= rate * p->biasError;
  p->biasError = 0;
}
