#ifndef _CONVOLVE_CUDA_H_
#define _CONVOLVE_CUDA_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "base.h"
#include "error.h"
#include "klt.h"
#include "klt_util.h"
#include "pyramid.h"
#include "convolve.h"

#include <cuda_runtime.h>

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

__host__ void convolveImageHorizCUDAold(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout,
  int ncols,
  int nrows,
  int kwidth
);

__host__ void convolveImageVertCUDAold(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout,
  int ncols,
  int nrows,
  int kwidth
);

__host__ void convolveSeperateCUDAold(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout
);

__host__ void computeSmoothedImageCUDAold(
	_KLT_FloatImage d_img,
	float sigma,
	_KLT_FloatImage d_smooth
);

__host__ void computePyramidCUDAold(
	_KLT_FloatImage d_img,
	_KLT_Pyramid d_pyramid,
	float sigma
);

__host__ void computeGradientsCUDAold(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady);


// ------------------------------------------- v3.5 --------------------------------------

__host__ void convolveImageHorizCUDA(
  float *d_imgin,
  ConvolutionKernel h_kernel,
  float *d_imgout,
  int ncols,
  int nrows,
  int kwidth,
  cudaStream_t stream
);

__host__ void convolveImageVertCUDA(
  float *d_imgin,
  ConvolutionKernel h_kernel,
  float *d_imgout,
  int ncols,
  int nrows,
  int kwidth,
  cudaStream_t stream
);

__host__ void convolveSeparateCUDA(
  float *d_imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  float *d_imgout,
  int ncols,
  int nrows,
  cudaStream_t stream
);

__host__ void computeSmoothedImageCUDA(
  float *d_img,
  float sigma,
  float *d_smooth_img,
  int ncols,
  int nrows,
  cudaStream_t stream
);

__global__ void subsampleKernel(
    const float *d_src, float *d_dst,
    int src_ncols, int src_nrows,
    int dst_ncols, int dst_nrows,
    int subsampling, int subhalf
);

__host__ void computePyramidCUDA(
  float *d_img,
  float *d_pyramid,
  float sigma_fact,
  int ncols,
  int nrows,
  int subsampling,
  int nLevels,
  cudaStream_t stream
);

__host__ void computeGradientsCUDA(
  float *d_img,
  float sigma,
  float *d_gradx,
  float *d_grady,
  int ncols,
  int nrows,
  cudaStream_t stream
);

  /* CPU */
void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg);

static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv);

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady);

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width);

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth);

#ifdef __cplusplus
}
#endif

#endif