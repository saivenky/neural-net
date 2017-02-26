#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "fully_connected_layer.h"
#include "neuron_props.h"
#include "rand.h"

struct fully_connected_layer *create_fully_connected_layer(long inputSize, long outputSize, double *inputActivation, double *inputError) {
  struct fully_connected_layer *l = malloc(sizeof(struct fully_connected_layer));
  l->inputSize = inputSize;
  l->outputSize = outputSize;

  if (inputActivation == NULL) {
    printf("ERROR: inputActivation is NULL\n");
    fflush(stdout);
  }

  l->inputActivation = inputActivation;

  if (inputError == NULL) {
    printf("NATIVE: inputError is NULL\n");
    fflush(stdout);
  }

  l->inputError = inputError;

  l->outputSignal = malloc(outputSize * sizeof(double));
  l->outputError = calloc(outputSize, sizeof(double));

  int weightsLen = inputSize * outputSize;
  l->weights = malloc(weightsLen * sizeof(double));
  l->biases = malloc(outputSize * sizeof(double));
  l->weightErrors = calloc(weightsLen, sizeof(double));
  l->biasErrors = calloc(outputSize, sizeof(double));
  init_rand_norm(l->weights, weightsLen, 1.0);
  init_rand_norm(l->biases, outputSize, 1.0);
  return l;
}

int destroy_fully_connected_layer(struct fully_connected_layer *l) {
  free(l->outputSignal);
  free(l->outputError);
  free(l->weights);
  free(l->biases);
  free(l->weightErrors);
  free(l->biasErrors);
  free(l);
  return 0;
}

void feedforward_fully_connected_layer(struct fully_connected_layer *l) {
  for (int i = 0; i < l->outputSize; i++) {
    double sum = l->biases[i];
    double *weights = l->weights + i * l->inputSize;
    for (int j = 0; j < l->inputSize; j++) {
      sum += l->inputActivation[j] * weights[j];
    }
    l->outputSignal[i] = sum;
  }
}

void backpropogate_fully_connected_layer(struct fully_connected_layer *l) {
  backpropogate_to_props_fully_connected_layer(l);
  if (l->inputError != NULL) {
    backpropogate_to_input_fully_connected_layer(l);
  }
  memset(l->outputError, 0, l->outputSize * sizeof(double));
}

void backpropogate_to_input_fully_connected_layer(struct fully_connected_layer *l) {
  for (int i = 0; i < l->outputSize; i++) {
    double error = l->outputError[i];
    double *weights = l->weights + i * l->inputSize;
    for (int j = 0; j < l->inputSize; j++) {
      l->inputError[j] += error * weights[j];
    }
  }
}

void backpropogate_to_props_fully_connected_layer(struct fully_connected_layer *l) {
  for (int i = 0; i < l->outputSize; i++) {
    double error = l->outputError[i];
    l->biasErrors[i] += error;
    double *weightErrors = l->weightErrors + i * l->inputSize;
    for(int j = 0; j < l->inputSize; j++) {
      weightErrors[j] += error * l->inputActivation[j];
    }
  }
}

void update_fully_connected_layer(struct fully_connected_layer *l, double rate) {
  for (int i = 0; i < l->outputSize; i++) {
    l->biases[i] -= rate * l->biasErrors[i];
  }

  int weightsLen = l->inputSize * l->outputSize;
  for (int i = 0; i < weightsLen; i++) {
    l->weights[i] -= rate * l->weightErrors[i];
  }
  memset(l->biasErrors, 0, l->outputSize * sizeof(double));
  memset(l->weightErrors, 0, weightsLen * sizeof(double));
}
