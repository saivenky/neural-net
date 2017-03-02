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

inline void fill_dim_for_shape(struct shape *shape) {
  shape->dim0 = shape->width;
  shape->dim1 = shape->dim0 * shape->height;
  shape->dim2 = shape->dim1 * shape->depth;
}
inline struct shape calcoutsize(struct shape inputShape, struct shape kernelShape, int padding, int stride, int frames) {
  struct shape outputShape;
  outputShape.width = calcoutsize_single(inputShape.width, kernelShape.width, padding, stride);
  outputShape.height = calcoutsize_single(inputShape.height, kernelShape.height, padding, stride);
  outputShape.depth = frames * (inputShape.depth - kernelShape.depth + 1);
  fill_dim_for_shape(&outputShape);
  return outputShape;
}

inline struct shape create_shape(int *shapeArray) {
  struct shape shape;
  shape.width = shapeArray[0];
  shape.height = shapeArray[1];
  shape.depth = shapeArray[2];
  fill_dim_for_shape(&shape);
  return shape;
}

struct stride shape2stride(struct shape shape, int stride) {
  struct stride strideResult;
  strideResult.dim0 = stride;
  strideResult.dim1 = shape.dim0 * stride;
  strideResult.dim2 = shape.dim1;
  return strideResult;
}

