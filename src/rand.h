#ifndef RAND_H
#define RAND_H
double rand_uniform();
double rand_norm();
double rand_truncated_norm();
void init_rand_norm(double *, int );
void init_rand_truncated_norm(double *, int );
void init_const(double *, int, double);
#endif
