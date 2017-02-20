#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "neuron_props.h"
#include "convolution_layer.h"

inline int calcoutsize_single(int inputSize, int kernelSize, int stride) {
  int temp = inputSize - kernelSize;
  if (temp % stride != 0) {
    puts("ERROR: kernel and stride size do not create round output size.");
  }
  return temp / stride + 1;
}

inline struct shape calcoutsize(struct shape inputShape, struct shape kernelShape, int stride) {
  struct shape outputShape;
  outputShape.width = calcoutsize_single(inputShape.width, kernelShape.width, stride);
  outputShape.height = calcoutsize_single(inputShape.height, kernelShape.height, stride);
  outputShape.depth = calcoutsize_single(inputShape.depth, kernelShape.depth, stride);
  return outputShape;
}

inline struct dim calcdim(struct shape shape) {
  struct dim dim;
  dim.dim0 = shape.width;
  dim.dim1 = dim.dim0 * shape.height;
  dim.dim2 = dim.dim1 * shape.depth;
  return dim;
}

inline struct shape create_shape(int *shapeArray) {
  struct shape shape;
  shape.width = shapeArray[0];
  shape.height = shapeArray[1];
  shape.depth = shapeArray[2];
  return shape;
}

struct convolution_layer *create_convolution_layer(int *inputShape, int *kernelShape, int frames, int stride, double *inputActivation, double *inputError) {
  struct convolution_layer *l = malloc(sizeof(struct convolution_layer));
  l->inputShape = create_shape(inputShape);
  l->kernelShape = create_shape(kernelShape);
  l->outputShape = calcoutsize(l->inputShape, l->kernelShape, stride);
  l->inputDim = calcdim(l->inputShape);
  l->outputDim = calcdim(l->outputShape);
  l->kernelDim = calcdim(l->kernelShape);
  l->frames = frames;

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

  for(int i = 0; i < l->frames; i++) {
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

void *apply_kernel_single_frame (void *args) {
  struct frame_args *fargs = (struct frame_args *)args;
  struct convolution_layer *layer = fargs->l;
  int frame = fargs->frame;
  int frameStart = frame * layer->outputDim.dim2;
  double *weights = layer->props[frame]->weights;
  double bias = layer->props[frame]->bias;

  for (int outZ = 0, inZInit = 0; outZ < layer->outputDim.dim2; outZ += layer->outputDim.dim1, inZInit += layer->inputDim.dim1) {
    for (int outY = 0, inYInit = 0; outY < layer->outputDim.dim1; outY += layer->outputDim.dim0, inYInit += layer->inputDim.dim0) {
      for (int outX = 0; outX < layer->outputDim.dim0; outX++) {

        double sum = 0;
        for (int kernZ = 0, inZ = inZInit; kernZ < layer->kernelDim.dim2; kernZ += layer->kernelDim.dim1, inZ += layer->inputDim.dim1) {
          for (int kernY = 0, inYZ = inZ + inYInit; kernY < layer->kernelDim.dim1; kernY += layer->kernelDim.dim0, inYZ += layer->inputDim.dim0) {
            for (int kernX = 0, inXYZ = inYZ + outX; kernX < layer->kernelDim.dim0; kernX++, inXYZ++) {
              double activation = layer->inputActivation[inXYZ];
              double weight = weights[kernZ + kernY + kernX];
              sum += activation * weight;
            }
          }
        }

        int outIndex = frameStart + outZ + outY + outX;
        layer->outputSignal[outIndex] = sum + bias;
      }
    }
  }
  return NULL;
}

void feedforward_convolution_layer(struct convolution_layer *l) {
  struct frame_args args[l->frames];
  for (int frame = 0; frame < l->frames; frame++) {
    args[frame].l = l;
    args[frame].frame = frame;
    apply_kernel_single_frame(&args[frame]);
  }
}

void backpropogate_to_input_convolution_layer(struct convolution_layer *l) {
  for (int frame = 0; frame < l->frames; frame++) {
    int frameStart = frame * l->outputDim.dim2;
    double *weights = l->props[frame]->weights;

    for (int outZ = 0, inZInit = 0; outZ < l->outputDim.dim2; outZ += l->outputDim.dim1, inZInit += l->inputDim.dim1) {
      for (int outY = 0, inYInit = 0; outY < l->outputDim.dim1; outY += l->outputDim.dim0, inYInit += l->inputDim.dim0) {
        for (int outX = 0; outX < l->outputDim.dim0; outX++) {
          double error = l->outputError[frameStart + outX + outY + outZ];

          for (int kernZ = 0, inZ = inZInit; kernZ < l->kernelDim.dim2; inZ += l->inputDim.dim1, kernZ += l->kernelDim.dim1) {
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
}

void *backpropogate_to_props_single_frame(void *args) {
  struct frame_args *fargs = (struct frame_args *)args;
  struct convolution_layer *layer = fargs->l;
  int frame = fargs->frame;
  int frameStart = frame * layer->outputDim.dim2;
  struct properties *prop = layer->props[frame];

  for (int outZ = 0, inZInit = 0; outZ < layer->outputDim.dim2; outZ += layer->outputDim.dim1, inZInit += layer->inputDim.dim1) {
    for (int outY = 0, inYInit = 0; outY < layer->outputDim.dim1; outY += layer->outputDim.dim0, inYInit += layer->inputDim.dim0) {
      for (int outX = 0; outX < layer->outputDim.dim0; outX++) {

        int outIndex = frameStart + outZ + outY + outX;
        double error = layer->outputError[outIndex];
        prop->biasError += error;

        for (int kernZ = 0, inZ = inZInit; kernZ < layer->kernelDim.dim2; kernZ += layer->kernelDim.dim1, inZ += layer->inputDim.dim1) {
          for (int kernY = 0, inYZ = inZ + inYInit; kernY < layer->kernelDim.dim1; kernY += layer->kernelDim.dim0, inYZ += layer->inputDim.dim0) {
            int kernYZ = kernY + kernZ;
            for (int kernX = 0, inXYZ = inYZ + outX; kernX < layer->kernelDim.dim0; kernX++, inXYZ++) {
              double activation = layer->inputActivation[inXYZ];
              prop->weightErrors[kernYZ + kernX] += activation * error;
            }
          }
        }
      }
    }
  }
  return NULL;
}

void backpropogate_to_props_convolution_layer(struct convolution_layer *l) {
  struct frame_args args[l->frames];
  for (int frame = 0; frame < l->frames; frame++) {
    args[frame].l = l;
    args[frame].frame = frame;
    backpropogate_to_props_single_frame(&args[frame]);
  }
}

void backpropogate_convolution_layer(struct convolution_layer *l) {
  backpropogate_to_props_convolution_layer(l);
  if (l->inputError != NULL) {
    backpropogate_to_input_convolution_layer(l);
  }
  memset(l->outputError, 0, l->outputDim.dim2 * l->frames * sizeof(double));
}

void update_convolution_layer(struct convolution_layer *l, double rate) {
  for(int i = 0; i < l->frames; i++) {
    update_properties(l->props[i], rate);
  }
}