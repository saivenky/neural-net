#ifndef GRADIENT_H
#define GRADIENT_H
#include "types.h"
struct gradient {
  float_t *inputError;
  float_t *outputError;
  void *extra;
};

struct gradient create_gradient_passthru(float_t *inputError);
struct gradient create_gradient(float_t *inputError, int size);
void destroy_gradient(struct gradient g);
#endif
