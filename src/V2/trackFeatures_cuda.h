# include <cuda.h>

/* Standard includes */
#include <assert.h>
#include <math.h>		/* fabs() */
#include <stdlib.h>		/* malloc() */
#include <stdio.h>		/* fflush() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"	/* for computing pyramid */
#include "klt.h"
#include "klt_util.h"	/* _KLT_FloatImage */
#include "pyramid.h"	/* _KLT_Pyramid */


__global__ void convolveImageHorizontalKernel();
__global__ void convolveImageVerticalKernel();


__global__ void trackFeaturesKernel(
	KLT_TrackingContext *d_tc,
	_KLT_Pyramid d_pyramid1,
	_KLT_Pyramid d_pyramid1_gradx,
	_KLT_Pyramid d_pyramid1_grady,
	_KLT_Pyramid d_pyramid2,
	_KLT_Pyramid d_pyramid2_gradx,
	_KLT_Pyramid d_pyramid2_grady,
	KLT_FeatureList *d_fl,
	int ncols,
	int nrows
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

__host__ void kltTrackFeaturesCUDA(
  KLT_TrackingContext *h_tc,
  KLT_PixelType *h_img1,
  KLT_PixelType *h_img2,
  KLT_FeatureList *h_fl,
  KLT_TrackingContext *d_tc,
  KLT_PixelType *img1,
  KLT_PixelType *img2,
  int ncols,
  int nrows,
  KLT_FeatureList *d_fl);