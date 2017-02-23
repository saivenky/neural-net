#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H
struct output_layer {
  long size;
  double *inputActivation;
  double *inputError;
};

struct output_layer *create_output_layer(long size, double *inputActivation, double *inputError);
void destroy_output_layer(struct output_layer *l);
#endif
