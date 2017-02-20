#ifndef SIGMOID_LAYER_H
#define SIGMOID_LAYER_H
struct sigmoid_layer {
  long size;
  double *inputActivation;
  double *inputError;
  double *outputSignal;
  double *outputError;
};

struct sigmoid_layer *create_sigmoid_layer(long size, double *inputActivation, double *inputError);
void destroy_sigmoid_layer(struct sigmoid_layer *l);
void feedforward_sigmoid_layer(struct sigmoid_layer *l);
void backpropogate_sigmoid_layer(struct sigmoid_layer *l);
void update_sigmoid_layer(struct sigmoid_layer *l);
#endif
