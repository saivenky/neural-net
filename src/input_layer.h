#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H
struct input_layer {
  long size;
  double *outputSignal;
};

struct input_layer *create_input_layer(long size);
void destroy_input_layer(struct input_layer *l);
#endif
