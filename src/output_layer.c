#include <stdlib.h>
#include "output_layer.h"

struct output_layer *create_output_layer(long size, double *inputActivation, double *inputError) {
  struct output_layer *l = malloc(sizeof(struct output_layer));
  l->size = size;
  l->inputActivation = inputActivation;
  l->inputError = inputError;
  return l;
}

void destroy_output_layer(struct output_layer *l) {
  free(l);
}
