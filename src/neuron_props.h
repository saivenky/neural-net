#ifndef NEURON_PROPS_H
#define NEURON_PROPS_H

struct properties {
  int inputSize;
  double *weights;
  double bias;
  double *weightErrors;
  double biasError;
};

struct properties *create_properties(int inputSize);
int destroy_properties(struct properties *);
void update_properties(struct properties *, double rate);
#endif
