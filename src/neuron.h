#ifndef NEURON_H
#define NEURON_H

#define SHAPE_LEN 3
#define SHAPE_SIZE sizeof(int) * 3
#define LAST_DIM 2

double randf();
double randnormf();
void init_rand(double *, int );

struct properties {
  int inputSize;
  double *weights;
  double bias;
  double *weightErrors;
  double biasError;
};

struct layer {
  int *inputShape;
  int *outputShape;
  int *kernelShape;
  int *inputDim;
  int *outputDim;
  int *kernelDim;
  int frames;
  double *inputActivation;
  double *inputError;
  double *outputSignal;
  double *outputError;
  struct properties **props;
};

struct properties *create_properties(int );
int destroy_properties(struct properties *);
void update_properties(struct properties *, double );

int *calcoutsize(int *, int *, int );
int *calcdim(int *shape);

struct layer;
struct layer *create_layer(int *inputShape, int *kernelShape, int frames, int stride);
int destroy_layer(struct layer *l);
void apply_kernel(struct layer *l);
void backpropogate_to_props(struct layer *l);
#endif
