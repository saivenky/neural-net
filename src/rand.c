#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "types.h"
#include "rand.h"

#define TWO_PI 6.28318530717958647692F

inline float_t rand_uniform() {
  float_t result = (float_t)rand() / (float_t)RAND_MAX;
  return result;
}

inline float_t rand_norm(float_t stdev) {
  static float_t z0, z1;
  static bool generate;
  generate = !generate;
  if (!generate) {
    return z1 * stdev;
  }

  float_t u1, u2;
  do {
    u1 = rand_uniform();
    u2 = rand_uniform();
  } while (u1 <= FLOAT_MIN);

  z0 = sqrtf(-2.0f * logf(u1)) * cosf(TWO_PI * u2);
  z1 = sqrtf(-2.0f * logf(u1)) * sinf(TWO_PI * u2);
  return z0 * stdev;
}

inline float_t rand_truncated_norm(float_t stdev) {
  float_t n;
  do {
    n = rand_norm(stdev);
  } while (fabs(n / stdev) >= 2.0f);
  return n;
}

inline void init_rand_norm(float_t *array, int len, float_t stdev) {
  for(int i = 0; i < len; i++) {
    array[i] = rand_norm(stdev);
  }
}

inline void init_rand_truncated_norm(float_t *array, int len, float_t stdev) {
  for(int i = 0; i < len; i++) {
    array[i] = rand_truncated_norm(stdev);
  }
}

inline void init_const(float_t *array, int len, float_t value) {
  for(int i = 0; i < len; i++) {
    array[i] = value;
  }
}
