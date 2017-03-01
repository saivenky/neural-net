#ifndef GRADIENT_H
#define GRADIENT_H
struct gradient {
  double *inputError;
  double *outputError;
};

struct gradient create_gradient_passthru(double *inputError);
struct gradient create_gradient(double *inputError, int size);
void destroy_gradient(struct gradient g);
#endif
