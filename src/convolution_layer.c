#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "neuron_props.h"
#include "kernel_dim.h"
#include "convolution_layer.h"

struct convolution_layer *create_convolution_layer(int *inputShape, int *kernelShape, int frames, int stride, double *inputActivation, double *inputError) {
  struct convolution_layer *l = malloc(sizeof(struct convolution_layer));
  l->inputShape = create_shape(inputShape);
  l->kernelShape = create_shape(kernelShape);
  l->outputShape = calcoutsize(l->inputShape, l->kernelShape, stride, frames);
  l->inputDim = calcdim(l->inputShape);
  l->outputDim = calcdim(l->outputShape);
  l->kernelDim = calcdim(l->kernelShape);

  if (inputActivation == NULL) {
    printf("ERROR: inputActivation is NULL\n");
    fflush(stdout);
  }

  l->inputActivation = inputActivation;

  if (inputError == NULL) {
    printf("NATIVE: inputError is NULL\n");
    fflush(stdout);
  }

  l->inputError = inputError;

  l->outputSignal = malloc(frames * l->outputDim.dim2 * sizeof(double));
  l->outputError = calloc(frames * l->outputDim.dim2, sizeof(double));

  l->props = malloc(sizeof(struct properties *) * frames);
  for(int i = 0; i < frames; i++) {
    l->props[i] = create_properties(l->kernelDim.dim2);
  }

  return l;
}

int destroy_convolution_layer(struct convolution_layer *l) {
  free(l->inputActivation);
  free(l->inputError);
  free(l->outputSignal);
  free(l->outputError);

  for(int i = 0; i < l->outputShape.depth; i++) {
    destroy_properties(l->props[i]);
  }

  free(l->props);
  free(l);
  return 0;
}

struct frame_args {
  struct convolution_layer *l;
  int frame;
};

void feedforward_convolution_layer(struct convolution_layer *l) {
  for (int outZ = 0, frame = 0; outZ < l->outputDim.dim2; outZ += l->outputDim.dim1, frame++) {
    double *weights = l->props[frame]->weights;
    double bias = l->props[frame]->bias;
    for (int outY = 0, inYInit = 0; outY < l->outputDim.dim1; outY += l->outputDim.dim0, inYInit += l->inputDim.dim0) {
      for (int outX = 0; outX < l->outputDim.dim0; outX++) {

        double sum = 0;
        for (int kernZ = 0, inZ = 0; kernZ < l->kernelDim.dim2; kernZ += l->kernelDim.dim1, inZ += l->inputDim.dim1) {
          for (int kernY = 0, inYZ = inZ + inYInit; kernY < l->kernelDim.dim1; kernY += l->kernelDim.dim0, inYZ += l->inputDim.dim0) {
            for (int kernX = 0, inXYZ = inYZ + outX; kernX < l->kernelDim.dim0; kernX++, inXYZ++) {
              double activation = l->inputActivation[inXYZ];
              double weight = weights[kernZ + kernY + kernX];
              sum += activation * weight;
            }
          }
        }

        int outIndex = outZ + outY + outX;
        l->outputSignal[outIndex] = sum + bias;
      }
    }
  }
}

void backpropogate_to_input_convolution_layer(struct convolution_layer *l) {
  for (int outZ = 0, frame = 0; outZ < l->outputDim.dim2; outZ += l->outputDim.dim1, frame++) {
    double *weights = l->props[frame]->weights;
    for (int outY = 0, inYInit = 0; outY < l->outputDim.dim1; outY += l->outputDim.dim0, inYInit += l->inputDim.dim0) {
      for (int outX = 0; outX < l->outputDim.dim0; outX++) {
        double error = l->outputError[outX + outY + outZ];

        for (int kernZ = 0, inZ = 0; kernZ < l->kernelDim.dim2; inZ += l->inputDim.dim1, kernZ += l->kernelDim.dim1) {
          for (int kernY = 0, inYZ = inZ + inYInit; kernY < l->kernelDim.dim1; inYZ += l->inputDim.dim0, kernY += l->kernelDim.dim0) {
            for (int kernX = 0, inXYZ = inYZ + outX; kernX < l->kernelDim.dim0; kernX++, inXYZ++) {
              double weight = weights[kernZ + kernY + kernX];
              l->inputError[inXYZ] += weight * error;
            }
          }
        }
      }
    }
  }
}

void backpropogate_to_props_convolution_layer(struct convolution_layer *l) {
  for (int outZ = 0, frame = 0; outZ < l->outputDim.dim2; outZ += l->outputDim.dim1, frame++) {
    struct properties *prop = l->props[frame];
    for (int outY = 0, inYInit = 0; outY < l->outputDim.dim1; outY += l->outputDim.dim0, inYInit += l->inputDim.dim0) {
      for (int outX = 0; outX < l->outputDim.dim0; outX++) {

        int outIndex = outZ + outY + outX;
        double error = l->outputError[outIndex];
        prop->biasError += error;

        for (int kernZ = 0, inZ = 0; kernZ < l->kernelDim.dim2; kernZ += l->kernelDim.dim1, inZ += l->inputDim.dim1) {
          for (int kernY = 0, inYZ = inZ + inYInit; kernY < l->kernelDim.dim1; kernY += l->kernelDim.dim0, inYZ += l->inputDim.dim0) {
            int kernYZ = kernY + kernZ;
            for (int kernX = 0, inXYZ = inYZ + outX; kernX < l->kernelDim.dim0; kernX++, inXYZ++) {
              double activation = l->inputActivation[inXYZ];
              prop->weightErrors[kernYZ + kernX] += activation * error;
            }
          }
        }
      }
    }
  }
}

void backpropogate_convolution_layer(struct convolution_layer *l) {
  backpropogate_to_props_convolution_layer(l);
  if (l->inputError != NULL) {
    backpropogate_to_input_convolution_layer(l);
  }
  memset(l->outputError, 0, l->outputDim.dim2 * sizeof(double));
}

void update_convolution_layer(struct convolution_layer *l, double rate) {
  for(int i = 0; i < l->outputShape.depth; i++) {
    update_properties(l->props[i], rate);
  }
}
