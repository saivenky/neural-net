#include <stdlib.h>
#include "activation.h"
#include "gradient.h"
#include "network_layer.h"

struct network_layer *create_network_layer(void *layer) {
  struct network_layer *l = malloc(sizeof(struct network_layer));
  l->layer = layer;
  return l;
}

void destroy_network_layer(struct network_layer *l) {
  free(l);
}
