#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include "neuron_props.h"
#include "kernel_dim.h"
#include "max_pooling_layer.h"
#define MIN_DBL -DBL_MAX

struct dim dim2stridedim(struct dim dim, int stride) {
  struct dim stridedim;
  stridedim.dim0 = stride;
  stridedim.dim1 = dim.dim0 * stride;
  stridedim.dim2 = dim.dim1 * stride;
  return stridedim;
}

struct max_pooling_layer *create_max_pooling_layer(int *inputShape, int *poolShape, int stride, double *inputActivation, double *inputError) {
  struct max_pooling_layer *l = malloc(sizeof(struct max_pooling_layer));
  l->inputShape = create_shape(inputShape);
  l->poolShape = create_shape(poolShape);
  l->outputShape = calcoutsize(l->inputShape, l->poolShape, stride, 1);

  l->inputDim = calcdim(l->inputShape);
  l->outputDim = calcdim(l->outputShape);
  l->poolDim = calcdim(l->poolShape);

  l->inputStrideDim = dim2stridedim(l->inputDim, stride);

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

  l->outputSignal = malloc(l->outputDim.dim2 * sizeof(double));
  l->outputError = calloc(l->outputDim.dim2, sizeof(double));
  l->outputArgmaxIndex = malloc(l->outputDim.dim2 * sizeof(int));
  return l;
}

int destroy_max_pooling_layer(struct max_pooling_layer *l) {
  free(l->outputSignal);
  free(l->outputError);
  free(l->outputArgmaxIndex);
  free(l);
  return 0;
}

void feedforward_max_pooling_layer(struct max_pooling_layer *l) {
  for (int outZ = 0, inZInit = 0; outZ < l->outputDim.dim2; outZ += l->outputDim.dim1, inZInit += l->inputStrideDim.dim2) {
    for (int outY = 0, inYInit = 0; outY < l->outputDim.dim1; outY += l->outputDim.dim0, inYInit += l->inputStrideDim.dim1) {
      for (int outX = 0, inXInit = 0; outX < l->outputDim.dim0; outX++, inXInit += l->inputStrideDim.dim0) {

        double maxActivation = MIN_DBL;
        int argMaxIndex = -1;
        for (int poolZ = 0, inZ = inZInit; poolZ < l->poolShape.depth; poolZ++, inZ += l->inputDim.dim1) {
          for (int poolY = 0, inYZ = inZ + inYInit; poolY < l->poolShape.height; poolY++, inYZ += l->inputDim.dim0) {
            for (int poolX = 0, inXYZ = inYZ + inXInit; poolX < l->poolShape.width; poolX++, inXYZ++) {
              if (maxActivation < l->inputActivation[inXYZ]) {
                maxActivation = l->inputActivation[inXYZ];
                argMaxIndex = inXYZ;
              }
            }
          }
        }

        if (argMaxIndex == -1) {
          printf("ERROR: argmax is -1\n");
        }
        int outIndex = outZ + outY + outX;
        l->outputSignal[outIndex] = maxActivation;
        l->outputArgmaxIndex[outIndex] = argMaxIndex;
      }
    }
  }
}

void backpropogate_max_pooling_layer(struct max_pooling_layer *l) {
  for (int i = 0; i < l->outputDim.dim2; i++) {
    int argMaxIndex = l->outputArgmaxIndex[i];
    l->inputError[argMaxIndex] += l->outputError[i];
  }
  memset(l->outputError, 0, l->outputDim.dim2 * sizeof(double));
}
