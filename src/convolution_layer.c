#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "kernel_dim.h"
#include "activation.h"
#include "gradient.h"
#include "rand.h"
#include "convolution_layer.h"

struct convolution_layer *create_convolution_layer(
    int *inputShape, int *kernelShape, int frames, int stride, int padding) {
  struct convolution_layer *l = malloc(sizeof(struct convolution_layer));
  l->padding = padding;
  l->inputShape = create_shape(inputShape);
  l->kernelShape = create_shape(kernelShape);
  l->outputShape = calcoutsize(l->inputShape, l->kernelShape, l->padding, stride, frames);

  l->weights = malloc(frames * l->kernelShape.dim2 * sizeof(float_t));
  l->biases = malloc(frames * sizeof(float_t));
  init_rand_truncated_norm(l->weights, frames * l->kernelShape.dim2, 0.1f);
  init_const(l->biases, frames, 0.1f);

  return l;
}

struct activation create_activation_convolution_layer(
    struct convolution_layer *l,
    float_t *inputActivation) {
  return create_activation(inputActivation, l->outputShape.dim2);
}

struct gradient create_gradient_convolution_layer(
    struct convolution_layer *l,
    float_t *inputError) {
  struct gradient g = create_gradient(inputError, l->outputShape.dim2);
  struct convolution_layer_gradient *local = malloc(sizeof(struct convolution_layer_gradient));
  local->weightErrors = calloc(l->outputShape.depth * l->kernelShape.dim2, sizeof(float_t));
  local->biasErrors = calloc(l->outputShape.depth, sizeof(float_t));
  g.extra = local;
  return g;
}

int destroy_convolution_layer(struct convolution_layer *l) {
  free(l->weights);
  free(l->biases);
  free(l);
  return 0;
}

void feedforward_convolution_layer(struct convolution_layer *l, struct activation a) {
  int yStart = -(l->padding * l->inputShape.dim0);
  int xStart = -(l->padding);
  for (int outZ = 0, frame = 0; outZ < l->outputShape.dim2; outZ += l->outputShape.dim1, frame++) {
    float_t *weights = l->weights + frame * l->kernelShape.dim2;
    float_t bias = l->biases[frame];
    for (int outY = 0, inYInit = yStart; outY < l->outputShape.dim1; outY += l->outputShape.dim0, inYInit += l->inputShape.dim0) {
      for (int outX = 0, inXInit = xStart; outX < l->outputShape.dim0; outX++, inXInit++) {

        float_t sum = 0;
        for (int kernZ = 0, inZ = 0; kernZ < l->kernelShape.dim2; kernZ += l->kernelShape.dim1, inZ += l->inputShape.dim1) {
          for (int kernY = 0, inY = inYInit; kernY < l->kernelShape.dim1 && inY < l->inputShape.dim1; kernY += l->kernelShape.dim0, inY += l->inputShape.dim0) {
            if (inY < 0) continue;
            for (int kernX = 0, inX = inXInit; kernX < l->kernelShape.dim0 && inX < l->inputShape.dim0; kernX++, inX++) {
              if (inX < 0) continue;
              float_t activation = a.inputActivation[inX + inY + inZ];
              float_t weight = weights[kernZ + kernY + kernX];
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
    float_t *weights = l->weights + frame * l->kernelShape.dim2;
    for (int outY = 0, inYInit = yStart; outY < l->outputShape.dim1; outY += l->outputShape.dim0, inYInit += l->inputShape.dim0) {
      for (int outX = 0, inXInit = xStart; outX < l->outputShape.dim0; outX++, inXInit++) {
        float_t error = g.outputError[outX + outY + outZ];

        for (int kernZ = 0, inZ = 0; kernZ < l->kernelShape.dim2; inZ += l->inputShape.dim1, kernZ += l->kernelShape.dim1) {
          for (int kernY = 0, inY = inYInit; kernY < l->kernelShape.dim1 && inY < l->inputShape.dim1; inY += l->inputShape.dim0, kernY += l->kernelShape.dim0) {
            if (inY < 0) continue;
            for (int kernX = 0, inX = inXInit; kernX < l->kernelShape.dim0 && inX < l->inputShape.dim0; kernX++, inX++) {
              if (inX < 0) continue;
              float_t weight = weights[kernZ + kernY + kernX];
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
  struct convolution_layer_gradient *local_g = (struct convolution_layer_gradient *)g.extra;

  for (int outZ = 0, frame = 0; outZ < l->outputShape.dim2; outZ += l->outputShape.dim1, frame++) {
    float_t *weightErrors = local_g->weightErrors + frame * l->kernelShape.dim2;
    float_t *biasError = local_g->biasErrors + frame;
    for (int outY = 0, inYInit = yStart; outY < l->outputShape.dim1; outY += l->outputShape.dim0, inYInit += l->inputShape.dim0) {
      for (int outX = 0, inXInit = xStart; outX < l->outputShape.dim0; outX++, inXInit++) {

        int outIndex = outZ + outY + outX;
        float_t error = g.outputError[outIndex];
        *biasError += error;
        for (int kernZ = 0, inZ = 0; kernZ < l->kernelShape.dim2; kernZ += l->kernelShape.dim1, inZ += l->inputShape.dim1) {
          for (int kernY = 0, inY = inYInit; kernY < l->kernelShape.dim1 && inY < l->inputShape.dim1; kernY += l->kernelShape.dim0, inY += l->inputShape.dim0) {
            if (inY < 0) continue;
            for (int kernX = 0, inX = inXInit; kernX < l->kernelShape.dim0 && inX < l->inputShape.dim0; kernX++, inX++) {
              if (inX < 0) continue;
              float_t activation = a.inputActivation[inZ + inY + inX];
              weightErrors[kernZ + kernY + kernX] += activation * error;
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
  memset(g.outputError, 0, l->outputShape.dim2 * sizeof(float_t));
}

void update_convolution_layer(struct convolution_layer *l, float_t rate, struct gradient g) {
  struct convolution_layer_gradient *local_g = (struct convolution_layer_gradient *)g.extra;
  for(int i = 0; i < l->outputShape.depth; i++) {
    float_t *weights = l->weights + i * l->kernelShape.dim2;
    float_t *weightErrors = local_g->weightErrors + i * l->kernelShape.dim2;
    l->biases[i] -= rate * local_g->biasErrors[i];
    for(int j = 0; j < l->kernelShape.dim2; j++) {
      weights[j] -= rate * weightErrors[j];
    }
  }
  memset(local_g->weightErrors, 0, l->outputShape.depth * l->kernelShape.dim2 * sizeof(float_t));
  memset(local_g->biasErrors, 0, l->outputShape.depth * sizeof(float_t));
}
