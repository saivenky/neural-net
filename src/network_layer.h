#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H
#include "activation.h"
#include "gradient.h"

typedef struct activation(*create_activation_type)(void *, float_t *);
typedef struct gradient(*create_gradient_type)(void *, float_t *);
typedef void (*feedforward_type)(void *, struct activation);
typedef void (*backpropogate_type)(void *, struct activation, struct gradient);
typedef void (*update_type)(void *, float_t, struct gradient);

struct network_layer {
  void *layer;
  int miniBatchSize;
  struct activation *activations;
  struct gradient *gradients;
  create_activation_type create_activation;
  create_gradient_type create_gradient;
  feedforward_type feedforward;
  backpropogate_type backpropogate;
  update_type update;
};

struct network_layer *create_network_layer(
    void *layer,
    struct network_layer *previousLayer,
    create_activation_type,
    create_gradient_type,
    feedforward_type,
    backpropogate_type,
    update_type);
void feedforward_network_layer(void *nativeLayerPtr, int batchIndex);
void backpropogate_network_layer(void *nativeLayerPtr, int batchIndex);
void update_network_layer(void *nativeLayerPtr, float_t rate, int batchIndex);
void destroy_network_layer(struct network_layer *l);
#endif
