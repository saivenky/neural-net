#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H
#include "activation.h"
#include "gradient.h"

struct network_layer {
  void *layer;
  int miniBatchSize;
  struct activation *activations;
  struct gradient *gradients;
};

typedef struct activation(*create_activation_type)(void *, double *);
typedef struct gradient(*create_gradient_type)(void *, double *);
typedef void (*feedforward_type)(void *, struct activation);
typedef void (*backpropogate_type)(void *, struct activation, struct gradient);

struct network_layer *create_network_layer(
    void *layer,
    struct network_layer *previousLayer,
    create_activation_type,
    create_gradient_type);
void feedforward_network_layer(
    void *nativeLayerPtr,
    feedforward_type);
void backpropogate_network_layer(
    void *nativeLayerPtr,
    backpropogate_type);
void destroy_network_layer(struct network_layer *l);
#endif
