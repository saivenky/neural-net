#ifndef RAND_H
#define RAND_H
float_t rand_uniform();
float_t rand_norm(float_t);
float_t rand_truncated_norm(float_t);
void init_rand_norm(float_t *, int, float_t);
void init_rand_truncated_norm(float_t *, int, float_t);
void init_const(float_t *, int, float_t);
#endif
