#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "kernel_dim.h"

inline int calcoutsize_single(int inputSize, int kernelSize, int padding, int stride) {
  int temp = inputSize - kernelSize + 2*padding;
  if (temp % stride != 0) {
    puts("ERROR: kernel and stride size do not create round output size.");
  }
  return temp / stride + 1;
}

inline struct shape calcoutsize(struct shape inputShape, struct shape kernelShape, int padding, int stride, int frames) {
  struct shape outputShape;
  outputShape.width = calcoutsize_single(inputShape.width, kernelShape.width, padding, stride);
  outputShape.height = calcoutsize_single(inputShape.height, kernelShape.height, padding, stride);
  outputShape.depth = frames * (inputShape.depth - kernelShape.depth + 1);
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
