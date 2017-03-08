#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H
#include "activation.h"
#include "gradient.h"

struct output_layer {
  int size;
};

struct output_layer *create_output_layer(int size);
struct activation create_activation_output_layer(struct output_layer *, float_t *inputActivation);
struct gradient create_gradient_output_layer(struct output_layer *, float_t *inputError);
void destroy_output_layer(struct output_layer *l);
#endif
