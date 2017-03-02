#ifndef KERNEL_DIM_H
#define KERNEL_DIM_H

struct shape {
  int width;
  int height;
  int depth;
  int dim0;
  int dim1;
  int dim2;
};

struct stride {
  int dim0;
  int dim1;
  int dim2;
};

struct shape create_shape(int *shape);
struct shape calcoutsize(struct shape inputShape, struct shape kernelShape, int padding, int stride, int frames);
struct stride shape2stride(struct shape shape, int stride);
#endif
