#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include "neuron_props.h"
#include "kernel_dim.h"
#include "activation.h"
#include "gradient.h"
#include "max_pooling_layer.h"

struct dim dim2stridedim(struct dim dim, int stride) {
  struct dim stridedim;
  stridedim.dim0 = stride;
  stridedim.dim1 = dim.dim0 * stride;
  stridedim.dim2 = dim.dim1;
  return stridedim;
}

struct max_pooling_layer *create_max_pooling_layer(int *inputShape, int *poolShape, int stride, double *inputActivation, double *inputError) {
  struct max_pooling_layer *l = malloc(sizeof(struct max_pooling_layer));
  l->inputShape = create_shape(inputShape);
  l->poolShape = create_shape(poolShape);
  l->outputShape = calcoutsize(l->inputShape, l->poolShape, 0, stride, 1);

  l->inputDim = calcdim(l->inputShape);
  l->outputDim = calcdim(l->outputShape);
  l->poolDim = calcdim(l->poolShape);

  l->inputStrideDim = dim2stridedim(l->inputDim, stride);

  return l;
}

struct activation create_activation_max_pooling_layer(struct max_pooling_layer *l, double *inputActivation) {
  struct activation a = create_activation(inputActivation, l->outputDim.dim2);
  a.extra = malloc(l->outputDim.dim2 * sizeof(int));
  return a;
}

struct gradient create_gradient_max_pooling_layer(struct max_pooling_layer *l, double *inputError) {
  return create_gradient(inputError, l->outputDim.dim2);
}

int destroy_max_pooling_layer(struct max_pooling_layer *l) {
  free(l);
  return 0;
}

void feedforward_max_pooling_layer(struct max_pooling_layer *l, struct activation a) {
  int *outputArgmaxIndex = (int *)(a.extra);
  for (int outZ = 0, inZInit = 0; outZ < l->outputDim.dim2; outZ += l->outputDim.dim1, inZInit += l->inputStrideDim.dim2) {
    for (int outY = 0, inYInit = 0; outY < l->outputDim.dim1; outY += l->outputDim.dim0, inYInit += l->inputStrideDim.dim1) {
      for (int outX = 0, inXInit = 0; outX < l->outputDim.dim0; outX++, inXInit += l->inputStrideDim.dim0) {

        double maxActivation = -DBL_MAX;
        int argMaxIndex = -1;
        for (int poolZ = 0, inZ = inZInit; poolZ < l->poolShape.depth; poolZ++, inZ += l->inputDim.dim1) {
          for (int poolY = 0, inYZ = inZ + inYInit; poolY < l->poolShape.height; poolY++, inYZ += l->inputDim.dim0) {
            for (int poolX = 0, inXYZ = inYZ + inXInit; poolX < l->poolShape.width; poolX++, inXYZ++) {
              if (maxActivation < a.inputActivation[inXYZ]) {
                maxActivation = a.inputActivation[inXYZ];
                argMaxIndex = inXYZ;
              }
            }
          }
        }

        if (argMaxIndex < 0 || argMaxIndex >= 11520) {
          printf("ERROR: max_pooling_layer - argmax is %d\n", argMaxIndex);
          exit(-1);
        }
        int outIndex = outZ + outY + outX;
        a.outputSignal[outIndex] = maxActivation;
        outputArgmaxIndex[outIndex] = argMaxIndex;
      }
    }
  }
}

void backpropogate_max_pooling_layer(struct max_pooling_layer *l, struct activation a, struct gradient g) {
  int *outputArgmaxIndex = (int *)(a.extra);
  for (int i = 0; i < l->outputDim.dim2; i++) {
    int argMaxIndex = outputArgmaxIndex[i];
    g.inputError[argMaxIndex] += g.outputError[i];
  }
  memset(g.outputError, 0, l->outputDim.dim2 * sizeof(double));
}
