#include <stdlib.h>
#include <stdio.h>
#include "activation.h"

struct activation create_activation_passthru(float_t *inputActivation) {
  struct activation a;
  a.inputActivation = inputActivation;
  a.outputSignal = inputActivation;
  return a;
}

struct activation create_activation(float_t *inputActivation, int size) {
  struct activation a;
  if (inputActivation == NULL) {
    printf("ERROR: inputActivation is NULL\n");
    fflush(stdout);
  }
  a.inputActivation = inputActivation;
  a.outputSignal = malloc(size * sizeof(float_t));
  return a;
}

void destroy_activation(struct activation a) {
  if (a.outputSignal != a.inputActivation && a.outputSignal != NULL) {
    free(a.outputSignal);
  }
}
