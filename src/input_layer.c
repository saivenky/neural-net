#include <stdlib.h>
#include "input_layer.h"
#include "activation.h"

struct input_layer *create_input_layer(int size) {
  struct input_layer *l = malloc(sizeof(struct input_layer));
  l->size = size;
  return l;
}

struct activation create_activation_input_layer(struct input_layer *l) {
  return create_activation(NULL, l->size);
}

struct gradient create_gradient_input_layer(struct input_layer *l) {
  return create_gradient_passthru(NULL);
}

void destroy_input_layer(struct input_layer *l) {
  free(l);
}
