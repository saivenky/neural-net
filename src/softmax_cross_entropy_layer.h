#ifndef SOFTMAX_CROSS_ENTROPY_LAYER_H
#define SOFTMAX_CROSS_ENTROPY_LAYER_H
#include "activation.h"
#include "gradient.h"

struct softmax_cross_entropy_layer {
  int size;
};

struct softmax_cross_entropy_layer *create_softmax_cross_entropy_layer(int size);
struct activation create_activation_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *, float_t *inputActivation);
struct gradient create_gradient_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *, float_t *inputError);

void destroy_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l);
void feedforward_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l, struct activation);
void backpropogate_softmax_cross_entropy_layer(
    struct softmax_cross_entropy_layer *l, struct activation a, struct gradient g);
void set_expected_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l, struct gradient g, float_t *expected);
#endif
