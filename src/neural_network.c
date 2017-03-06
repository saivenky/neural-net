#include <stdlib.h>
#include <string.h>
#include "neural_network.h"
#include "network_layer.h"

struct neural_network *create_neural_network(struct network_layer **layers, int size) {
  struct neural_network *nn = malloc(sizeof(struct neural_network));
  nn->size = size;
  nn->layers = malloc(nn->size * sizeof(struct network_layer *));
  nn->layers = memcpy(nn->layers, layers, nn->size * sizeof(struct network_layer *));
  return nn;
}

void feedforward_neural_network(struct neural_network *nn) {
  for (int i = 0; i < nn->size; i++) {
    feedforward_network_layer((void *)(nn->layers[i]));
  }
}

void backpropogate_neural_network(struct neural_network *nn) {
  for (int i = nn->size - 1; i >= 0; i--) {
    backpropogate_network_layer((void *)(nn->layers[i]));
  }
}

void update_neural_network(struct neural_network *nn, double rate) {
  for (int i = 0; i < nn->size; i++) {
    update_network_layer((void *)(nn->layers[i]), rate);
  }
}
