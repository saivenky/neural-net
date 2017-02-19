#include <float.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include "rand.h"

#define TWO_PI 6.28318530717958647692

inline double randf() {
  double result = (double)rand() / RAND_MAX;
  return result;
}

inline double randnormf() {
  static double z0, z1;
  static bool generate;
  generate = !generate;
  if (!generate) return z1;

  double u1, u2;
  do {
    u1 = randf();
    u2 = randf();
  } while (u1 <= DBL_MIN);

  z0 = sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
  z1 = sqrt(-2.0 * log(u1)) * sin(TWO_PI * u2);
  return z0;
}

inline void init_rand(double *array, int len) {
  for(int i = 0; i < len; i++) {
    array[i] = randnormf();
  }
}
