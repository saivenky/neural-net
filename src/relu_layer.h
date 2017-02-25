#ifndef RELU_LAYER_H
#define RELU_LAYER_H
struct relu_layer {
  long size;
  double *inputActivation;
  double *inputError;
  double *outputSignal;
  double *outputError;
};

struct relu_layer *create_relu_layer(long size, double *inputActivation, double *inputError);
void destroy_relu_layer(struct relu_layer *l);
void feedforward_relu_layer(struct relu_layer *l);
void backpropogate_relu_layer(struct relu_layer *l);
void update_relu_layer(struct relu_layer *l);
#endif
