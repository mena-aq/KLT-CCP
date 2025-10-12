/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

/* CUDA Implementation */
#include "convolve_cuda.h"


/* Small helper for CUDA errors */
static void checkCuda(cudaError_t err, const char *msg)
{
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
    exit(1);
  }
}

__global__ void convolveHorizKernel(const float *imgin, float *imgout,
                                    int ncols, int nrows,
                                    const float *kernel, int kwidth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= ncols || y >= nrows) return;

  int radius = kwidth / 2;

  /* Zero border */
  if (x < radius || x >= ncols - radius) {
    imgout[y * ncols + x] = 0.0f;
    return;
  }

  float sum = 0.0f;
  int start = x - radius;
  for (int k = 0; k < kwidth; k++) {
    sum += imgin[y * ncols + (start + k)] * kernel[kwidth - 1 - k];
  }
  imgout[y * ncols + x] = sum;
}

__global__ void convolveVertKernel(const float *imgin, float *imgout,
                                    int ncols, int nrows,
                                    const float *kernel, int kwidth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= ncols || y >= nrows) return;

  int radius = kwidth / 2;

  /* Zero border */
  if (y < radius || y >= nrows - radius) {
    imgout[y * ncols + x] = 0.0f;
    return;
  }

  float sum = 0.0f;
  int start = y - radius;
  for (int k = 0; k < kwidth; k++) {
    sum += imgin[(start + k) * ncols + x] * kernel[kwidth - 1 - k];
  }
  imgout[y * ncols + x] = sum;
}


__host__ void convolveImageHorizCUDA(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  /* Basic checks to match original behavior */
  assert(kernel.width % 2 == 1);
  assert(imgin != imgout);
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  size_t npix = (size_t)ncols * nrows;
  size_t img_bytes = npix * sizeof(float);
  size_t kernel_bytes = kwidth * sizeof(float);

  float *d_imgin = NULL, *d_imgout = NULL, *d_kernel = NULL;
  checkCuda(cudaMalloc((void**)&d_imgin, img_bytes), "malloc imgin");
  checkCuda(cudaMalloc((void**)&d_imgout, img_bytes), "malloc imgout");
  checkCuda(cudaMalloc((void**)&d_kernel, kernel_bytes), "malloc kernel");

  checkCuda(cudaMemcpy(d_imgin, h_imgin, img_bytes, cudaMemcpyHostToDevice), "copy imgin");
  checkCuda(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice), "copy kernel");

  dim3 blockSize(16, 16);
  dim3 gridSize((ncols + blockSize.x - 1)/blockSize.x, (nrows + blockSize.y - 1)/blockSize.y);

  convolveHorizKernel<<<gridSize, blockSize>>>(d_imgin, d_imgout, ncols, nrows, d_kernel, kwidth);
  checkCuda(cudaGetLastError(), "kernel launch horiz");
  checkCuda(cudaDeviceSynchronize(), "synchronize horiz");

  checkCuda(cudaMemcpy(h_imgout, d_imgout, img_bytes, cudaMemcpyDeviceToHost), "copy out horiz");

  cudaFree(d_imgin);
  cudaFree(d_imgout);
  cudaFree(d_kernel);

}


__host__ void convolveImageVertCUDA(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  /* Basic checks to match original behavior */
  assert(kernel.width % 2 == 1);
  assert(imgin != imgout);
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);
  size_t npix = (size_t)ncols * nrows;
  size_t img_bytes = npix * sizeof(float);
  size_t kernel_bytes = kwidth * sizeof(float);

  float *d_imgin = NULL, *d_imgout = NULL, *d_kernel = NULL;
  checkCuda(cudaMalloc((void**)&d_imgin, img_bytes), "malloc imgin");
  checkCuda(cudaMalloc((void**)&d_imgout, img_bytes), "malloc imgout");
  checkCuda(cudaMalloc((void**)&d_kernel, kernel_bytes), "malloc kernel");

  checkCuda(cudaMemcpy(d_imgin, h_imgin, img_bytes, cudaMemcpyHostToDevice), "copy imgin");
  checkCuda(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice), "copy kernel");

  dim3 blockSize(16, 16);
  dim3 gridSize((ncols + blockSize.x - 1)/blockSize.x, (nrows + blockSize.y - 1)/blockSize.y);

  convolveVertKernel<<<gridSize, blockSize>>>(d_imgin, d_imgout, ncols, nrows, d_kernel, kwidth);
  checkCuda(cudaGetLastError(), "kernel launch vert");
  checkCuda(cudaDeviceSynchronize(), "synchronize vert");

  checkCuda(cudaMemcpy(h_imgout, d_imgout, img_bytes, cudaMemcpyDeviceToHost), "copy out vert");

  cudaFree(d_imgin);
  cudaFree(d_imgout);
  cudaFree(d_kernel);
}


__host__ void convolveSeparateCUDA(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  /* Create temporary image */
  _KLT_FloatImage tmpimg;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);

  convolveImageHorizCUDA(imgin, horiz_kernel, tmpimg);
  convolveImageVertCUDA(tmpimg, vert_kernel, imgout);

  /* Free memory */
  _KLTFreeFloatImage(tmpimg);
}

__host__ void computeSmoothedImageCUDA(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabsf(sigma - sigma_last) > 0.05f)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  convolveSeparateCUDA(img, gauss_kernel, gauss_kernel, smooth);
}

__host__ void computePyramidCUDA(
  _KLT_FloatImage img, 
  _KLT_Pyramid pyramid,
  float sigma_fact)
{
  _KLT_FloatImage currimg, tmpimg;
  int ncols = img->ncols, nrows = img->nrows;
  int subsampling = pyramid->subsampling;
  int subhalf = subsampling / 2;
  float sigma = subsampling * sigma_fact;  /* empirically determined */
  int oldncols;
  int i, x, y;
	
  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTComputePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  assert(pyramid->ncols[0] == img->ncols);
  assert(pyramid->nrows[0] == img->nrows);

  /* Copy original image to level 0 of pyramid */
  memcpy(pyramid->img[0]->data, img->data, ncols*nrows*sizeof(float));

  currimg = img;
  for (i = 1 ; i < pyramid->nLevels ; i++)  {
    tmpimg = _KLTCreateFloatImage(ncols, nrows);

    computeSmoothedImageCUDA(currimg, sigma, tmpimg);

    /* Subsample */
    // downsample with the smoothed img
    oldncols = ncols;
    ncols /= subsampling;  nrows /= subsampling;
    for (y = 0 ; y < nrows ; y++)
      for (x = 0 ; x < ncols ; x++)
        pyramid->img[i]->data[y*ncols+x] = 
          tmpimg->data[(subsampling*y+subhalf)*oldncols +
                      (subsampling*x+subhalf)];

    /* Reassign current image */
    currimg = pyramid->img[i]; //curr image for next level is the current level img
				
    _KLTFreeFloatImage(tmpimg);
  }
}

__host__ void computeGradientsCUDA(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabsf(sigma - sigma_last) > 0.05f)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  convolveSeparateCUDA(img, gaussderiv_kernel, gauss_kernel, gradx);
  convolveSeparateCUDA(img, gauss_kernel, gaussderiv_kernel, grady);
}





