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

#define MAX_KERNEL_WIDTH 	71

__constant__ float c_gauss_kernel_07[MAX_KERNEL_WIDTH];
__constant__ float c_gaussderiv_kernel_07[MAX_KERNEL_WIDTH];
__constant__ float c_gauss_kernel_36[MAX_KERNEL_WIDTH]; 
__constant__ float c_gaussderiv_kernel_36[MAX_KERNEL_WIDTH];
__constant__ float c_gauss_kernel_10[MAX_KERNEL_WIDTH];
__constant__ float c_gaussderiv_kernel_10[MAX_KERNEL_WIDTH];

static int gauss_width;
static int gaussderiv_width;

typedef enum {
    SIGMA_07 = 0,
    SIGMA_36 = 1, 
    SIGMA_10 = 2,
    SIGMA_OTHER = 3  // Fallback for other sigma values
} SigmaType;

typedef enum {
    KERNEL_GAUSSIAN = 0,
    KERNEL_GAUSSIAN_DERIV = 1
} KernelType;


__global__ void convolveHorizKernel(
    const float *imgin, float *imgout,
    int ncols, int nrows,
    int kwidth,
    KernelType kernel_type,
    float sigma
);

__global__ void convolveVertKernel(
    const float *imgin, float *imgout,
    int ncols, int nrows,
    int kwidth,
    KernelType kernel_type,
    float sigma
);

__host__ __device__ SigmaType getSigmaType(float sigma);

__device__ const float* getKernelPtr(KernelType kernel_type, SigmaType sigma_type);

__host__ void convolveImageHorizCUDA(
  float *d_imgin,
  float *d_imgout,
  int ncols,
  int nrows,
  int kwidth,
  KernelType kernel_type,
  float sigma,
  cudaStream_t stream
);

__host__ void convolveImageVertCUDA(
  float *d_imgin,
  float *d_imgout,
  int ncols,
  int nrows,
  int kwidth,
  KernelType kernel_type,
  float sigma,
  cudaStream_t stream
);

__host__ void convolveSeparateCUDA(
  float *d_imgin,
  int horiz_kwidth,
  KernelType horiz_kernel_type,
  int vert_kwidth,
  KernelType vert_kernel_type,
  float sigma,
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

__host__ void computeKernelsConstant(
  float sigma
);

__host__ void initializePrecomputedKernels();


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