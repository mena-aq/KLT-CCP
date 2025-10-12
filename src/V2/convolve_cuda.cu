/*********************************************************************
 * convolve_cuda.cu
 *********************************************************************/

/*********************************************************************
 * Naive CUDA implementation of horizontal/vertical separable
 * convolution. This file intentionally keeps the kernels simple:
 * one thread per pixel, no shared-memory or other optimizations.
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* CUDA runtime */
#include <cuda_runtime.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve_cuda.h"
#include "klt_util.h"

/* Kernels state (kept on host as before) */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0f;

/* Small helper for CUDA errors */
static void checkCuda(cudaError_t err, const char *msg)
{
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
    exit(1);
  }
}

/*********************************************************************
 * CPU helpers kept from original implementation
 */

static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;   /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0f);

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));

    /* Compute gauss and deriv */
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ; 
          i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; 
          i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
                "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;
    
    den = 0.0f;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0f;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}

/*********************************************************************
 * CUDA device kernels (naive)
 * - convolveHorizKernel: computes horizontal convolution per-pixel
 * - convolveVertKernel: computes vertical convolution per-pixel
 *
 * Both kernels expect kernel data to be available in device memory.
 */

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


/* Host wrappers that allocate device memory, copy inputs, launch kernel,
    and copy back result. These are intentionally simple and naive. */

static void launchConvolveHoriz(const float *h_imgin, float *h_imgout,
                                int ncols, int nrows,
                                const float *h_kernel, int kwidth)
{
  size_t npix = (size_t)ncols * nrows;
  size_t img_bytes = npix * sizeof(float);
  size_t kernel_bytes = kwidth * sizeof(float);

  float *d_imgin = NULL, *d_imgout = NULL, *d_kernel = NULL;
  checkCuda(cudaMalloc((void**)&d_imgin, img_bytes), "malloc imgin");
  checkCuda(cudaMalloc((void**)&d_imgout, img_bytes), "malloc imgout");
  checkCuda(cudaMalloc((void**)&d_kernel, kernel_bytes), "malloc kernel");

  checkCuda(cudaMemcpy(d_imgin, h_imgin, img_bytes, cudaMemcpyHostToDevice), "copy imgin");
  checkCuda(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice), "copy kernel");

  dim3 block(16, 16);
  dim3 grid((ncols + block.x - 1)/block.x, (nrows + block.y - 1)/block.y);

  convolveHorizKernel<<<grid, block>>>(d_imgin, d_imgout, ncols, nrows, d_kernel, kwidth);
  checkCuda(cudaGetLastError(), "kernel launch horiz");
  checkCuda(cudaDeviceSynchronize(), "synchronize horiz");

  checkCuda(cudaMemcpy(h_imgout, d_imgout, img_bytes, cudaMemcpyDeviceToHost), "copy out horiz");

  cudaFree(d_imgin);
  cudaFree(d_imgout);
  cudaFree(d_kernel);
}

static void launchConvolveVert(const float *h_imgin, float *h_imgout,
                                int ncols, int nrows,
                                const float *h_kernel, int kwidth)
{
  size_t npix = (size_t)ncols * nrows;
  size_t img_bytes = npix * sizeof(float);
  size_t kernel_bytes = kwidth * sizeof(float);

  float *d_imgin = NULL, *d_imgout = NULL, *d_kernel = NULL;
  checkCuda(cudaMalloc((void**)&d_imgin, img_bytes), "malloc imgin");
  checkCuda(cudaMalloc((void**)&d_imgout, img_bytes), "malloc imgout");
  checkCuda(cudaMalloc((void**)&d_kernel, kernel_bytes), "malloc kernel");

  checkCuda(cudaMemcpy(d_imgin, h_imgin, img_bytes, cudaMemcpyHostToDevice), "copy imgin");
  checkCuda(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice), "copy kernel");

  dim3 block(16, 16);
  dim3 grid((ncols + block.x - 1)/block.x, (nrows + block.y - 1)/block.y);

  convolveVertKernel<<<grid, block>>>(d_imgin, d_imgout, ncols, nrows, d_kernel, kwidth);
  checkCuda(cudaGetLastError(), "kernel launch vert");
  checkCuda(cudaDeviceSynchronize(), "synchronize vert");

  checkCuda(cudaMemcpy(h_imgout, d_imgout, img_bytes, cudaMemcpyDeviceToHost), "copy out vert");

  cudaFree(d_imgin);
  cudaFree(d_imgout);
  cudaFree(d_kernel);
}


/* Public functions that replace the old CPU implementations but keep the
    same function signatures so other code can remain unchanged. */

void _convolveImageHoriz(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  /* Basic checks to match original behavior */
  assert(kernel.width % 2 == 1);
  assert(imgin != imgout);
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  launchConvolveHoriz(imgin->data, imgout->data, imgin->ncols, imgin->nrows,
                      kernel.data, kernel.width);
}

void _convolveImageVert(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  /* Basic checks to match original behavior */
  assert(kernel.width % 2 == 1);
  assert(imgin != imgout);
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  launchConvolveVert(imgin->data, imgout->data, imgin->ncols, imgin->nrows,
                      kernel.data, kernel.width);
}


/* The rest of the original API (convolution orchestration) stays the same
    and reuses the new GPU-enabled functions. */

static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  /* Create temporary image */
  _KLT_FloatImage tmpimg;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);

  /* Do convolution (now on GPU) */
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
  _convolveImageVert(tmpimg, vert_kernel, imgout);

  /* Free memory */
  _KLTFreeFloatImage(tmpimg);
}

void _KLTComputeGradients(
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

  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
}


void _KLTComputeSmoothedImage(
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

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}



