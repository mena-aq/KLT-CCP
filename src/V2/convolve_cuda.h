/*********************************************************************
 * convolve_cuda.h
 *
 * Minimal header exposing naive CUDA convolution entry points.
 * This header intentionally contains only the horizontal/vertical
 * convolution API used by the CUDA implementation.
 *********************************************************************/

#ifndef _CONVOLVE_CUDA_H_
#define _CONVOLVE_CUDA_H_

#include "klt.h"
#include "klt_util.h"

#define MAX_KERNEL_WIDTH 71

typedef struct {
  int width;
  float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

/* Naive GPU-enabled convolution functions. They perform the convolution
   by copying input data and kernel to the device, launching a simple
   kernel (one thread per pixel), and copying the result back. */

void _convolveImageHoriz(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout);

void _convolveImageVert(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout);

#endif
