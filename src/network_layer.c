#include <stdlib.h>
#include "activation.h"
#include "gradient.h"
#include "network_layer.h"

struct network_layer *create_network_layer(
    void *layer,
    struct network_layer *previousLayer,
    create_activation_type create_activation,
    create_gradient_type create_gradient,
    feedforward_type feedforward,
    backpropogate_type backpropogate,
    update_type update) {
  struct network_layer *l = malloc(sizeof(struct network_layer));
  l->layer = layer;
  l->miniBatchSize = previousLayer->miniBatchSize;
  l->activations = malloc(previousLayer->miniBatchSize * sizeof(struct activation));
  l->gradients = malloc(previousLayer->miniBatchSize * sizeof(struct gradient));

  for (int i = 0; i < l->miniBatchSize; i++) {
    double *outputSignal = previousLayer->activations ? previousLayer->activations[i].outputSignal : NULL;
    double *outputError = previousLayer->gradients ? previousLayer->gradients[i].outputError : NULL;
    l->activations[i] = create_activation(layer, outputSignal);
    l->gradients[i] = create_gradient(layer, outputError);
  }

  l->create_activation = create_activation;
  l->create_gradient = create_gradient;
  l->feedforward = feedforward;
  l->backpropogate = backpropogate;
  l->update = update;

  return l;
}

void feedforward_network_layer(void *nativeLayerPtr, int batchIndex) {
  struct network_layer *network_layer = (struct network_layer *)nativeLayerPtr;
  if (network_layer->feedforward == NULL) return;
  network_layer->feedforward(
      network_layer->layer, network_layer->activations[batchIndex]);
}

void backpropogate_network_layer(void *nativeLayerPtr, int batchIndex) {
  struct network_layer *network_layer = (struct network_layer *)nativeLayerPtr;
  if (network_layer->backpropogate == NULL) return;
  network_layer->backpropogate(
      network_layer->layer,
      network_layer->activations[batchIndex],
      network_layer->gradients[batchIndex]);
}

void update_network_layer(void *nativeLayerPtr, double rate, int batchIndex) {
  struct network_layer *network_layer = (struct network_layer *)nativeLayerPtr;
  if (network_layer->update == NULL) return;
  network_layer->update(
      network_layer->layer, rate, network_layer->gradients[batchIndex]);

}

void destroy_network_layer(struct network_layer *l) {
  free(l->activations);
  free(l->gradients);
  free(l);
}
