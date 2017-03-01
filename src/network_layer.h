#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H
#include "activation.h"
#include "gradient.h"

struct network_layer {
  void *layer;
  struct activation activation;
  struct gradient gradient;
};

struct network_layer *create_network_layer(void *layer);
void destroy_network_layer(struct network_layer *l);
#endif
