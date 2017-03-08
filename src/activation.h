#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "types.h"
struct activation {
  float_t *inputActivation;
  float_t *outputSignal;
  void *extra;
};

struct activation create_activation_passthru(float_t *inputActivation);
struct activation create_activation(float_t *inputActivation, int size);
void destroy_activation(struct activation a);
#endif
