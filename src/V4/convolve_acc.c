/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "convolve_acc.h"
#include "klt_util.h"


/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

/*********************************************************************
 * _KLTToFloatImage - OpenACC optimized
 */

void _KLTToFloatImageACC(
  KLT_PixelType *img,
  int ncols, int nrows,
  float *floatimg)  
{

  #pragma acc parallel loop copyin(img[0:ncols*nrows]) \
                           copyout(floatimg[0:ncols*nrows])
  for (int i = 0; i < ncols * nrows; i++) {
    floatimg[i] = (float)img[i];
  }
}


/*********************************************************************
 * _convolveImageHoriz - Optimized for flat buffers
 */

static void _convolveImageHoriz(
  float *imgin, int in_ncols, int in_nrows,
  ConvolutionKernel kernel,
  float *imgout, int out_ncols, int out_nrows)
{
  int radius = kernel.width / 2;
  int kernel_width = kernel.width;

  #pragma acc parallel loop gang vector collapse(2) \
              copyin(imgin[0:in_ncols*in_nrows], kernel) \
              copyout(imgout[0:out_ncols*out_nrows])
  for (int j = 0; j < in_nrows; j++) {
    for (int i = 0; i < in_ncols; i++) {
      if (i < radius || i >= in_ncols - radius) {
        imgout[j * out_ncols + i] = 0.0f;
      } else {
        float sum = 0.0f;
        #pragma acc loop seq
        for (int k = 0; k < kernel_width; k++) {
          sum += imgin[j * in_ncols + i - radius + k] * 
                 kernel.data[kernel_width - 1 - k];
        }
        imgout[j * out_ncols + i] = sum;
      }
    }
  }
}

/*********************************************************************
 * _convolveImageVert - Optimized for flat buffers
 */

static void _convolveImageVert(
  float *imgin, int in_ncols, int in_nrows,
  ConvolutionKernel kernel,
  float *imgout, int out_ncols, int out_nrows)
{
  int radius = kernel.width / 2;
  int kernel_width = kernel.width;

  #pragma acc parallel loop gang vector collapse(2) \
              copyin(imgin[0:in_ncols*in_nrows], kernel) \
              copyout(imgout[0:out_ncols*out_nrows])
  for (int i = 0; i < in_ncols; i++) {
    for (int j = 0; j < in_nrows; j++) {
      if (j < radius || j >= in_nrows - radius) {
        imgout[j * out_ncols + i] = 0.0f;
      } else {
        float sum = 0.0f;
        #pragma acc loop seq
        for (int k = 0; k < kernel_width; k++) {
          sum += imgin[(j - radius + k) * in_ncols + i] * 
                 kernel.data[kernel_width - 1 - k];
        }
        imgout[j * out_ncols + i] = sum;
      }
    }
  }
}

/*********************************************************************
 * _convolveSeparate - Using flat buffers
 */

static void _convolveSeparate(
  float *imgin, int in_ncols, int in_nrows,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  float *imgout, int out_ncols, int out_nrows)
{
  // Use device memory for temporary image
  float *tmpimg = (float*)malloc(in_ncols * in_nrows * sizeof(float));
  
  #pragma acc data create(tmpimg[0:in_ncols*in_nrows])
  {
    _convolveImageHoriz(imgin, in_ncols, in_nrows, horiz_kernel, 
                       tmpimg, in_ncols, in_nrows);
    _convolveImageVert(tmpimg, in_ncols, in_nrows, vert_kernel, 
                      imgout, out_ncols, out_nrows);
  }
  
  free(tmpimg);
}


/*********************************************************************
 * _KLTComputeGradients - Using flat buffers
 */

void _KLTComputeGradientsACC(
  float *img, int img_ncols, int img_nrows,
  float sigma,
  float *gradx, int gradx_ncols, int gradx_nrows,
  float *grady, int grady_ncols, int grady_nrows)
{
  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  
  #pragma acc data present(img, gradx, grady)
  {
    _convolveSeparate(img, img_ncols, img_nrows,
                     gaussderiv_kernel, gauss_kernel,
                     gradx, gradx_ncols, gradx_nrows);
    _convolveSeparate(img, img_ncols, img_nrows,
                     gauss_kernel, gaussderiv_kernel,
                     grady, grady_ncols, grady_nrows);
  }
}

/*********************************************************************
 * _KLTComputeSmoothedImage - Using flat buffers
 */

void _KLTComputeSmoothedImageACC(
  float *img, int img_ncols, int img_nrows,
  float sigma,
  float *smooth, int smooth_ncols, int smooth_nrows)
{
  /* Compute kernel, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  #pragma acc data present(img, smooth)
  {
    _convolveSeparate(img, img_ncols, img_nrows,
                     gauss_kernel, gauss_kernel,
                     smooth, smooth_ncols, smooth_nrows);
  }
}

/*********************************************************************
 * Pyramid computation with OpenACC
 */

void computePyramidOpenACC(
  float *img, int img_ncols, int img_nrows,
  float sigma_fact,
  float *pyramid, int subsampling, int nLevels)
{
  int subhalf = subsampling / 2;
  float sigma = subsampling * sigma_fact;

  // Copy base level
  #pragma acc parallel loop present(img, pyramid)
  for (int i = 0; i < img_ncols * img_nrows; i++) {
    pyramid[i] = img[i];
  }

  float *current_smooth = (float*)malloc(img_ncols * img_nrows * sizeof(float));
  int current_ncols = img_ncols;
  int current_nrows = img_nrows;
  size_t current_offset = 0;

  #pragma acc data create(current_smooth[0:img_ncols*img_nrows])
  {
    for (int i = 1; i < nLevels; i++) {
      size_t next_offset = current_offset + current_ncols * current_nrows;
      int next_ncols = current_ncols / subsampling;
      int next_nrows = current_nrows / subsampling;

      // Smooth current level
      _KLTComputeSmoothedImageACC(pyramid + current_offset, current_ncols, current_nrows,
                              sigma, current_smooth, current_ncols, current_nrows);

      // Subsample into next level
      #pragma acc parallel loop collapse(2) \
                  present(pyramid, current_smooth)
      for (int y = 0; y < next_nrows; y++) {
        for (int x = 0; x < next_ncols; x++) {
          int src_x = subsampling * x + subhalf;
          int src_y = subsampling * y + subhalf;
          pyramid[next_offset + y * next_ncols + x] = 
            current_smooth[src_y * current_ncols + src_x];
        }
      }

      current_offset = next_offset;
      current_ncols = next_ncols;
      current_nrows = next_nrows;
    }
  }

  free(current_smooth);
}
