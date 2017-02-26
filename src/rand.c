#include <float.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "rand.h"

#define TWO_PI 6.28318530717958647692

inline double rand_uniform() {
  double result = (double)rand() / RAND_MAX;
  return result;
}

inline double rand_norm(double stdev) {
  static double z0, z1;
  static bool generate;
  generate = !generate;
  if (!generate) {
    return z1 * stdev;
  }

  double u1, u2;
  do {
    u1 = rand_uniform();
    u2 = rand_uniform();
  } while (u1 <= DBL_MIN);

  z0 = sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
  z1 = sqrt(-2.0 * log(u1)) * sin(TWO_PI * u2);
  return z0 * stdev;
}

inline double rand_truncated_norm(double stdev) {
  return fabs(rand_norm(stdev));
}

inline void init_rand_norm(double *array, int len, double stdev) {
  for(int i = 0; i < len; i++) {
    array[i] = rand_norm(stdev);
  }
}

inline void init_rand_truncated_norm(double *array, int len, double stdev) {
  for(int i = 0; i < len; i++) {
    array[i] = rand_truncated_norm(stdev);
  }
}

inline void init_const(double *array, int len, double value) {
  for(int i = 0; i < len; i++) {
    array[i] = value;
  }
}
