#include <stdlib.h>
#include "activation.h"
#include "gradient.h"
#include "network_layer.h"

struct network_layer *create_network_layer(
    void *layer,
    struct network_layer *previousLayer,
    struct activation (*create_activation)(void *,double *),
    struct gradient (*create_gradient)(void *,double *)) {
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

  return l;
}

void feedforward_network_layer(
    void *nativeLayerPtr,
    void (*feedforward_layer)(void *,struct activation)) {
  struct network_layer *network_layer = (struct network_layer *)nativeLayerPtr;
  for (int i = 0; i < network_layer->miniBatchSize; i++) {
    feedforward_layer(
        network_layer->layer, network_layer->activations[i]);
  }
}

void backpropogate_network_layer(
    void *nativeLayerPtr,
    void (*backpropogate_layer)(void *,struct activation,struct gradient)) {
  struct network_layer *network_layer = (struct network_layer *)nativeLayerPtr;
  for (int i = 0; i < network_layer->miniBatchSize; i++) {
    backpropogate_layer(
        network_layer->layer, network_layer->activations[i], network_layer->gradients[i]);
  }
}

void destroy_network_layer(struct network_layer *l) {
  free(l->activations);
  free(l->gradients);
  free(l);
}
