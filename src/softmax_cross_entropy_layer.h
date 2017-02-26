#ifndef SOFTMAX_CROSS_ENTROPY_LAYER_H
#define SOFTMAX_CROSS_ENTROPY_LAYER_H
struct softmax_cross_entropy_layer {
  long size;
  double *inputActivation;
  double *inputError;
  double *outputSignal;
};

struct softmax_cross_entropy_layer *create_softmax_cross_entropy_layer(long size, double *inputActivation, double *inputError);
void destroy_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l);
void feedforward_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l);
void set_expected_softmax_cross_entropy_layer(struct softmax_cross_entropy_layer *l, double *expected);
#endif
