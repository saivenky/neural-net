#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "neuron.h"

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

struct frame_args {
  struct layer *l;
  int frame;
};

void *apply_kernel_single_frame (void *args) {
  struct frame_args *fargs = (struct frame_args *)args;
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
  struct frame_args args[l->frames];
  for (int frame = 0; frame < l->frames; frame++) {
    args[frame].l = l;
    args[frame].frame = frame;
    apply_kernel_single_frame(&args[frame]);
  }
}

void *backpropogate_to_props_single_frame(void *args) {
  struct frame_args *fargs = (struct frame_args *)args;
  struct layer layer = *(fargs->l);
  int frame = fargs->frame;
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
  struct frame_args args[l->frames];
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
}
