#ifndef ACTIVATION_H
#define ACTIVATION_H
struct activation {
  double *inputActivation;
  double *outputSignal;
  void *extra;
};

struct activation create_activation_passthru(double *inputActivation);
struct activation create_activation(double *inputActivation, int size);
void destroy_activation(struct activation a);
#endif
