#include <stdlib.h>
#include "output_layer.h"
#include "activation.h"
#include "gradient.h"

struct output_layer *create_output_layer(int size) {
  struct output_layer *l = malloc(sizeof(struct output_layer));
  l->size = size;
  return l;
}

struct activation create_activation_output_layer(double *inputActivation) {
  return create_activation_passthru(inputActivation);
}

struct gradient create_gradient_output_layer(double *inputError) {
  return create_gradient_passthru(inputError);
}

void destroy_output_layer(struct output_layer *l) {
  free(l);
}
