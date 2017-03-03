#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "neuron_props.h"
#include "kernel_dim.h"
#include "activation.h"
#include "gradient.h"
#include "convolution_layer.h"

struct convolution_layer *create_convolution_layer(
    int *inputShape, int *kernelShape, int frames, int stride, int padding) {
  struct convolution_layer *l = malloc(sizeof(struct convolution_layer));
  l->padding = padding;
  l->inputShape = create_shape(inputShape);
  l->kernelShape = create_shape(kernelShape);
  l->outputShape = calcoutsize(l->inputShape, l->kernelShape, l->padding, stride, frames);

  l->props = malloc(sizeof(struct properties *) * frames);
  for(int i = 0; i < frames; i++) {
    l->props[i] = create_properties(l->kernelShape.dim2);
  }

  return l;
}

struct activation create_activation_convolution_layer(
    struct convolution_layer *l,
    double *inputActivation) {
  return create_activation(inputActivation, l->outputShape.dim2);
}

struct gradient create_gradient_convolution_layer(
    struct convolution_layer *l,
    double *inputError) {
  return create_gradient(inputError, l->outputShape.dim2);
}

int destroy_convolution_layer(struct convolution_layer *l) {
  for(int i = 0; i < l->outputShape.depth; i++) {
    destroy_properties(l->props[i]);
  }

  free(l->props);
  free(l);
  return 0;
}

void feedforward_convolution_layer(struct convolution_layer *l, struct activation a) {
  int yStart = -(l->padding * l->inputShape.dim0);
  int xStart = -(l->padding);
  for (int outZ = 0, frame = 0; outZ < l->outputShape.dim2; outZ += l->outputShape.dim1, frame++) {
    double *weights = l->props[frame]->weights;
    double bias = l->props[frame]->bias;
    for (int outY = 0, inYInit = yStart; outY < l->outputShape.dim1; outY += l->outputShape.dim0, inYInit += l->inputShape.dim0) {
      for (int outX = 0, inXInit = xStart; outX < l->outputShape.dim0; outX++, inXInit++) {

        double sum = 0;
        for (int kernZ = 0, inZ = 0; kernZ < l->kernelShape.dim2; kernZ += l->kernelShape.dim1, inZ += l->inputShape.dim1) {
          for (int kernY = 0, inY = inYInit; kernY < l->kernelShape.dim1 && inY < l->inputShape.dim1; kernY += l->kernelShape.dim0, inY += l->inputShape.dim0) {
            if (inY < 0) continue;
            for (int kernX = 0, inX = inXInit; kernX < l->kernelShape.dim0 && inX < l->inputShape.dim0; kernX++, inX++) {
              if (inX < 0) continue;
              double activation = a.inputActivation[inX + inY + inZ];
              double weight = weights[kernZ + kernY + kernX];
              sum += activation * weight;
            }
          }
        }

        int outIndex = outZ + outY + outX;
        a.outputSignal[outIndex] = sum + bias;
      }
    }
  }
}

void backpropogate_to_input_convolution_layer(struct convolution_layer *l, struct gradient g) {
  int yStart = -(l->padding * l->inputShape.dim0);
  int xStart = -(l->padding);

  for (int outZ = 0, frame = 0; outZ < l->outputShape.dim2; outZ += l->outputShape.dim1, frame++) {
    double *weights = l->props[frame]->weights;
    for (int outY = 0, inYInit = yStart; outY < l->outputShape.dim1; outY += l->outputShape.dim0, inYInit += l->inputShape.dim0) {
      for (int outX = 0, inXInit = xStart; outX < l->outputShape.dim0; outX++, inXInit++) {
        double error = g.outputError[outX + outY + outZ];

        for (int kernZ = 0, inZ = 0; kernZ < l->kernelShape.dim2; inZ += l->inputShape.dim1, kernZ += l->kernelShape.dim1) {
          for (int kernY = 0, inY = inYInit; kernY < l->kernelShape.dim1 && inY < l->inputShape.dim1; inY += l->inputShape.dim0, kernY += l->kernelShape.dim0) {
            if (inY < 0) continue;
            for (int kernX = 0, inX = inXInit; kernX < l->kernelShape.dim0 && inX < l->inputShape.dim0; kernX++, inX++) {
              if (inX < 0) continue;
              double weight = weights[kernZ + kernY + kernX];
              g.inputError[inZ + inY + inX] += weight * error;
            }
          }
        }
      }
    }
  }
}

void backpropogate_to_props_convolution_layer(struct convolution_layer *l, struct activation a, struct gradient g) {
  int yStart = -(l->padding * l->inputShape.dim0);
  int xStart = -(l->padding);

  for (int outZ = 0, frame = 0; outZ < l->outputShape.dim2; outZ += l->outputShape.dim1, frame++) {
    struct properties *prop = l->props[frame];
    for (int outY = 0, inYInit = yStart; outY < l->outputShape.dim1; outY += l->outputShape.dim0, inYInit += l->inputShape.dim0) {
      for (int outX = 0, inXInit = xStart; outX < l->outputShape.dim0; outX++, inXInit++) {

        int outIndex = outZ + outY + outX;
        double error = g.outputError[outIndex];
        prop->biasError += error;
        for (int kernZ = 0, inZ = 0; kernZ < l->kernelShape.dim2; kernZ += l->kernelShape.dim1, inZ += l->inputShape.dim1) {
          for (int kernY = 0, inY = inYInit; kernY < l->kernelShape.dim1 && inY < l->inputShape.dim1; kernY += l->kernelShape.dim0, inY += l->inputShape.dim0) {
            if (inY < 0) continue;
            for (int kernX = 0, inX = inXInit; kernX < l->kernelShape.dim0 && inX < l->inputShape.dim0; kernX++, inX++) {
              if (inX < 0) continue;
              double activation = a.inputActivation[inZ + inY + inX];
              prop->weightErrors[kernZ + kernY + kernX] += activation * error;
            }
          }
        }
      }
    }
  }
}

void backpropogate_convolution_layer(struct convolution_layer *l, struct activation a, struct gradient g) {
  backpropogate_to_props_convolution_layer(l, a, g);
  if (g.inputError != NULL) {
    backpropogate_to_input_convolution_layer(l, g);
  }
  memset(g.outputError, 0, l->outputShape.dim2 * sizeof(double));
}

void update_convolution_layer(struct convolution_layer *l, double rate, struct gradient g) {
  for(int i = 0; i < l->outputShape.depth; i++) {
    update_properties(l->props[i], rate);
  }
}
