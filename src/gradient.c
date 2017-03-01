#include <stdlib.h>
#include <stdio.h>
#include "gradient.h"

struct gradient create_gradient_passthru(double *inputError) {
  struct gradient g;
  g.outputError = inputError;
  g.inputError = inputError;
  return g;
}

struct gradient create_gradient(double *inputError, int size) {
  struct gradient g;
  if (inputError == NULL) {
    printf("NATIVE: inputError is NULL\n");
    fflush(stdout);
  }
  g.inputError = inputError;
  g.outputError = calloc(size, sizeof(double));
  return g;
}

void destroy_gradient(struct gradient g) {
  if (g.inputError != g.outputError && g.outputError != NULL) {
    free(g.outputError);
  }
}
