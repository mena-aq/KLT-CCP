#ifndef _TRACKFEATURES_CUDA_H_
#define _TRACKFEATURES_CUDA_H_

/* Standard includes */
#include <assert.h>
#include <math.h>		/* fabs() */
#include <stdlib.h>		/* malloc() */
#include <stdio.h>		/* fflush() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve_cuda.h"	
#include "klt.h"
#include "klt_util.h"	/* _KLT_FloatImage */
#include "pyramid.h"	/* _KLT_Pyramid */

/*CUDA implementation*/
#include <cuda.h>


__host__device__ float sumAbsFloatWindowCUDA(
	float* fw,
	int width,
	int height
);

__host__device__ float interpolateCUDA(
	float x, 
	float y, 
	const float *buf, 
	int level
);

__global__ void trackFeatureKernel(
	KLT_TrackingContext *d_tc,
	const float *d_pyramid1,
	const float *d_pyramid1_gradx,
	const float *d_pyramid1_grady,
	const float *d_pyramid2,
	const float *d_pyramid2_gradx,
	const float *d_pyramid2_grady,
	const float *d_in_x,
	const float *d_in_y,
	const int *d_in_val,
	float *d_out_x,
	float *d_out_y,
	int *d_out_val,
	int window_width,
	int window_height,
	float step_factor,
	int max_iterations,
	float small,
	float th,
	float max_residue
);


__host__ void kltTrackFeaturesCUDA(
  KLT_TrackingContext *h_tc,
  KLT_PixelType *h_img1,
  KLT_PixelType *h_img2,
  KLT_FeatureList *h_fl,
  KLT_TrackingContext *d_tc,
  KLT_PixelType *img1,
  KLT_PixelType *img2,
  KLT_FeatureList *d_fl,
  int ncols,
  int nrows
);

#endif