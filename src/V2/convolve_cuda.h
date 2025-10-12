#ifndef _CONVOLVE_CUDA_H_
#define _CONVOLVE_CUDA_H_

#include "klt.h"
#include "klt_util.h"
#include <cuda.h>

static void checkCuda(cudaError_t err, const char *msg);

__global__ void convolveHorizKernel(
  const float *imgin, 
  float *imgout,
  int ncols, 
  int nrows,
  const float *kernel, 
  int kwidth
);

__global__ void convolveVertKernel(
  const float *imgin, 
  float *imgout,
  int ncols, 
  int nrows,
  const float *kernel, 
  int kwidth
);

__host__ void _convolveImageHorizCUDA(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout
);

__host__ void _convolveImageVertCUDA(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout
);

__host__ void convolveSeperateCUDA(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout
);

__host__ void computeSmoothedImageCUDA(
	_KLT_FloatImage d_img,
	float sigma,
	_KLT_FloatImage d_smooth
);

__host__ void computePyramidCUDA(
	_KLT_FloatImage d_img,
	_KLT_Pyramid d_pyramid,
	float sigma
);

__host__ void computeGradientsCUDA(
  _KLT_FloatImage d_img,
  _KLT_FloatImage d_gradx,
  _KLT_FloatImage d_grady);

#endif
