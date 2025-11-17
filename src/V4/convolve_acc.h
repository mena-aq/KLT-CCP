#ifndef __CONVOLVE_ACC_H__
#define __CONVOLVE_ACC_H__

/* Necessary includes */
#include "base.h"
#include "klt_util.h"


void _KLTToFloatImageACC(
  KLT_PixelType *img,
  int ncols, int nrows,
  float *floatimg);

void _KLTComputeGradientsACC(
  float *img, int img_ncols, int img_nrows,
  float sigma,
  float *gradx, int gradx_ncols, int gradx_nrows,
  float *grady, int grady_ncols, int grady_nrows);

void _KLTComputeSmoothedImageACC(
  float *img, int img_ncols, int img_nrows,
  float sigma,
  float *smooth, int smooth_ncols, int smooth_nrows);

void computePyramidOpenACC(
  float *img, int img_ncols, int img_nrows,
  float sigma_fact,
  float *pyramid, int subsampling, int nLevels);

#endif 
