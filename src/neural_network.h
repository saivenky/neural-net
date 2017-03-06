#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "network_layer.h"

struct neural_network {
  int size;
  struct network_layer **layers;
};

struct neural_network *create_neural_network(struct network_layer **layers, int size);
void destroy_neural_network(struct neural_network *);
void feedforward_neural_network(struct neural_network *);
void backpropogate_neural_network(struct neural_network *);
void update_neural_network(struct neural_network *, double rate);
#endif
