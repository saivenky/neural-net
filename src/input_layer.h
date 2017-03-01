#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H
#include "activation.h"
#include "gradient.h"

struct input_layer {
  int size;
};

struct input_layer *create_input_layer(int size);
struct activation create_activation_input_layer(struct input_layer *l);
struct gradient create_gradient_input_layer(struct input_layer *l);
void destroy_input_layer(struct input_layer *l);
#endif
