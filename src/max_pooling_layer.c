#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include "neuron_props.h"
#include "kernel_dim.h"
#include "activation.h"
#include "gradient.h"
#include "max_pooling_layer.h"

struct max_pooling_layer *create_max_pooling_layer(int *inputShape, int *poolShape, int stride) {
  struct max_pooling_layer *l = malloc(sizeof(struct max_pooling_layer));
  l->inputShape = create_shape(inputShape);
  l->poolShape = create_shape(poolShape);
  l->outputShape = calcoutsize(l->inputShape, l->poolShape, 0, stride, 1);
  l->inputStride = shape2stride(l->inputShape, stride);

  return l;
}

struct activation create_activation_max_pooling_layer(struct max_pooling_layer *l, double *inputActivation) {
  struct activation a = create_activation(inputActivation, l->outputShape.dim2);
  a.extra = malloc(l->outputShape.dim2 * sizeof(int));
  return a;
}

struct gradient create_gradient_max_pooling_layer(struct max_pooling_layer *l, double *inputError) {
  return create_gradient(inputError, l->outputShape.dim2);
}

int destroy_max_pooling_layer(struct max_pooling_layer *l) {
  free(l);
  return 0;
}

void feedforward_max_pooling_layer(struct max_pooling_layer *l, struct activation a) {
  int *outputArgmaxIndex = (int *)(a.extra);
  for (int outZ = 0, inZInit = 0; outZ < l->outputShape.dim2; outZ += l->outputShape.dim1, inZInit += l->inputStride.dim2) {
    for (int outY = 0, inYInit = 0; outY < l->outputShape.dim1; outY += l->outputShape.dim0, inYInit += l->inputStride.dim1) {
      for (int outX = 0, inXInit = 0; outX < l->outputShape.dim0; outX++, inXInit += l->inputStride.dim0) {

        double maxActivation = -DBL_MAX;
        int argMaxIndex = -1;
        for (int poolZ = 0, inZ = inZInit; poolZ < l->poolShape.depth; poolZ++, inZ += l->inputShape.dim1) {
          for (int poolY = 0, inYZ = inZ + inYInit; poolY < l->poolShape.height; poolY++, inYZ += l->inputShape.dim0) {
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
  for (int i = 0; i < l->outputShape.dim2; i++) {
    int argMaxIndex = outputArgmaxIndex[i];
    g.inputError[argMaxIndex] += g.outputError[i];
  }
  memset(g.outputError, 0, l->outputShape.dim2 * sizeof(double));
}
