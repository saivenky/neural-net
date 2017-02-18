#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "neuron.h"
#include "float.h"

#define TWO_PI 6.28318530717958647692

double randf() {
  double result = (double)rand() / RAND_MAX;
  return result;
}

double randnormf() {
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

void init_rand(double *array, int len) {
  for(int i = 0; i < len; i++) {
    array[i] = randnormf();
  }
}

struct properties *create_properties(int inputSize) {
  struct properties *p = malloc(sizeof(struct properties));
  p->inputSize = inputSize;
  p->weights = malloc(sizeof(double) * inputSize);
  init_rand(p->weights, inputSize);
  p->weightErrors = calloc(inputSize, sizeof(double));
  p->bias = randnormf();
  p->biasError = 0;
  return p;
}

int destroy_properties(struct properties *p) {
  free(p->weights);
  free(p->weightErrors);
  free(p);
  return 0;
}

void update_properties(struct properties *p, double rate) {
  for(int i = 0; i < p->inputSize; i++) {
    p->weights[i] -= rate * p->weightErrors[i];
    p->weightErrors[i] = 0;
  }
  p->bias -= rate * p->biasError;
  p->biasError = 0;
}

int *calcoutsize(int *inputShape, int *kernelShape, int stride) {
  int *outputShape = malloc(SHAPE_SIZE);
  for(int i = 0; i < SHAPE_LEN; i++) {
    int temp = inputShape[i] - kernelShape[i];
    if (temp % stride != 0) {
      puts("ERROR: kernel and stride size do not create round output size.");
    }
    outputShape[i] = temp / stride + 1;
  }

  return outputShape;
}

int *calcdim(int *shape) {
  int *dim = malloc(SHAPE_SIZE);
  dim[0] = shape[0];
  for(int i = 1; i < SHAPE_LEN; i++) {
    dim[i] = dim[i-1] * shape[i];
  }

  return dim;
}

struct layer *create_layer(int *inputShape, int *kernelShape, int frames, int stride) {
  struct layer *l = malloc(sizeof(struct layer));
  int *inputShapeCopy = malloc(SHAPE_SIZE);
  int *kernelShapeCopy = malloc(SHAPE_SIZE);
  l->inputShape = memcpy(inputShapeCopy, inputShape, SHAPE_SIZE);
  l->kernelShape = memcpy(kernelShapeCopy, kernelShape, SHAPE_SIZE);
  l->outputShape = calcoutsize(inputShape, kernelShape, stride);
  l->inputDim = calcdim(inputShape);
  l->outputDim = calcdim(l->outputShape);
  l->kernelDim = calcdim(kernelShape);
  l->frames = frames;

  long inputSize = l->inputDim[LAST_DIM] * sizeof(double);
  l->inputActivation = malloc(inputSize);
  l->inputError = malloc(inputSize);

  long outputSize = frames * l->outputDim[LAST_DIM] * sizeof(double);
  l->outputSignal = malloc(outputSize);
  l->outputError = malloc(outputSize);

  l->props = malloc(sizeof(struct properties *) * frames);
  for(int i = 0; i < frames; i++) {
    l->props[i] = create_properties(l->kernelDim[LAST_DIM]);
  }

  printf("NATIVE: input(%ld), output(%ld), kernel(%ld)\n", inputSize, outputSize, l->kernelDim[LAST_DIM] * sizeof(double));
  fflush(stdout);
  return l;
}

int destroy_layer(struct layer *l) {
  free(l->inputShape);
  free(l->kernelShape);
  free(l->outputShape);
  free(l->inputDim);
  free(l->kernelDim);
  free(l->outputDim);

  free(l->inputActivation);
  free(l->inputError);
  free(l->outputSignal);
  free(l->outputError);

  for(int i = 0; i < l->frames; i++) {
    destroy_properties(l->props[i]);
  }

  free(l->props);
  free(l);
  return 0;
}

struct apply_kernel_args {
  struct layer *l;
  int frame;
};

void *apply_kernel_single_frame (void *args) {
  struct apply_kernel_args *fargs = (struct apply_kernel_args *)args;
  struct layer layer = *(fargs->l);
  int frame = fargs->frame;
  int frameStart = frame * layer.outputDim[LAST_DIM];
  double *weights = layer.props[frame]->weights;
  double bias = layer.props[frame]->bias;

  for (int outZ = 0, inZInit = 0; outZ < layer.outputDim[2]; outZ += layer.outputDim[1], inZInit += layer.inputDim[1]) {
    for (int outY = 0, inYInit = 0; outY < layer.outputDim[1]; outY += layer.outputDim[0], inYInit += layer.inputDim[0]) {
      for (int outX = 0; outX < layer.outputDim[0]; outX++) {

        double sum = 0;
        for (int kernZ = 0, inZ = inZInit; kernZ < layer.kernelDim[2]; kernZ += layer.kernelDim[1], inZ += layer.inputDim[1]) {
          for (int kernY = 0, inYZ = inZ + inYInit; kernY < layer.kernelDim[1]; kernY += layer.kernelDim[0], inYZ += layer.inputDim[0]) {
            for (int kernX = 0, inXYZ = inYZ + outX; kernX < layer.kernelDim[0]; kernX++, inXYZ++) {
              double activation = layer.inputActivation[inXYZ];
              double weight = weights[kernZ + kernY + kernX];
              sum += activation * weight;
            }
          }
        }

        int outIndex = frameStart + outZ + outY + outX;
        layer.outputSignal[outIndex] = sum + bias;
      }
    }
  }
  return NULL;
}
void apply_kernel(struct layer *l) {
  struct apply_kernel_args args[l->frames];
  for (int frame = 0; frame < l->frames; frame++) {
    args[frame].l = l;
    args[frame].frame = frame;
    apply_kernel_single_frame(&args[frame]);
  }
}

struct backpropogate_to_props_args {
  struct layer *l;
  int frame;
};

void *backpropogate_to_props_single_frame(void *args) {
  int frame = ((struct backpropogate_to_props_args *)args)->frame;
  struct layer layer = *(((struct backpropogate_to_props_args *)args)->l);
  int frameStart = frame * layer.outputDim[LAST_DIM];
  struct properties *prop = layer.props[frame];

  for (int outZ = 0, inZInit = 0; outZ < layer.outputDim[2]; outZ += layer.outputDim[1], inZInit += layer.inputDim[1]) {
    for (int outY = 0, inYInit = 0; outY < layer.outputDim[1]; outY += layer.outputDim[0], inYInit += layer.inputDim[0]) {
      for (int outX = 0; outX < layer.outputDim[0]; outX++) {

        int outIndex = frameStart + outZ + outY + outX;
        double error = layer.outputError[outIndex];
        prop->biasError += error;

        for (int kernZ = 0, inZ = inZInit; kernZ < layer.kernelDim[2]; kernZ += layer.kernelDim[1], inZ += layer.inputDim[1]) {
          for (int kernY = 0, inYZ = inZ + inYInit; kernY < layer.kernelDim[1]; kernY += layer.kernelDim[0], inYZ += layer.inputDim[0]) {
            for (int kernX = 0, inXYZ = inYZ + outX; kernX < layer.kernelDim[0]; kernX++, inXYZ++) {
              double activation = layer.inputActivation[inXYZ];
              prop->weightErrors[kernZ + kernY + kernX] += activation * error;
            }
          }
        }
      }
    }
  }
  return NULL;
}

void backpropogate_to_props(struct layer *l) {
  struct backpropogate_to_props_args args[l->frames];
  for (int frame = 0; frame < l->frames; frame++) {
    args[frame].l = l;
    args[frame].frame = frame;
    backpropogate_to_props_single_frame(&args[frame]);
  }
}

void bye() {
  printf("Goodbye\n");
}

int main() {
  atexit(bye);
  printf("Hello World\n");
  int inputShape[] = {4, 5, 6};
  int kernelShape[] = {2, 3, 6};
  struct layer *l = create_layer(inputShape, kernelShape, 20, 1);
  destroy_layer(l);
  for(int i = 0; i < 20; i++) {
    printf("%f %d\n", randf(), RAND_MAX);
  }
}
