#include <stdlib.h>
#include "input_layer.h"

struct input_layer *create_input_layer(long size) {
  struct input_layer *l = malloc(sizeof(struct input_layer));
  l->size = size;
  l->outputSignal = malloc(size * sizeof(double));
  return l;
}

void destroy_input_layer(struct input_layer *l) {
  free(l->outputSignal);
  free(l);
}
