#ifndef _TRACKFEATURES_CUDA_H_
#define _TRACKFEATURES_CUDA_H_

#ifdef __cplusplus
extern "C" {
#endif

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


// Shared pyramid metadata structure (host-side only)
typedef struct {
    int nLevels;
    float subsampling;
    int *nrows;    // array of size nLevels
    int *ncols;    // array of size nLevels
    size_t *offsets;   // array of size nLevels (offsets into pure image data buffer)
    size_t total_size; // total floats needed for all image data
} PyramidMetadata;


// Device-side pyramid metadata (passed as parameters to kernels)
typedef struct {
    int nLevels;
    float subsampling;
    // Device arrays for pyramid dimensions and offsets  
    const int *nrows;    // device pointer to nrows array
    const int *ncols;    // device pointer to ncols array
    const size_t *offsets; // device pointer to offsets array
} DevicePyramidMeta;


// CUDA version of sumAbsFloatWindow for GPU 
__host__ __device__ float sumAbsFloatWindowCUDA(
	float* fw,
	int width,
	int height
);


// device implementation of _interpolate (uses constant memory)
__device__ float interpolateCUDA(
    float x, float y, 
    const float *img_data, 
    int level
);

// CUDA kernel for tracking features
// Tracks a singular feature point across two image pyramids
// Each thread handles one pixel in the feature window
__global__ void trackFeatureKernel(
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


// Host function to track features using CUDA
// Calls the CUDA kernels to for convolution and feature tracking
__host__ void kltTrackFeaturesCUDA(
  KLT_TrackingContext h_tc,
  KLT_PixelType *h_img1,
  KLT_PixelType *h_img2,
  int ncols,
  int nrows,
  KLT_FeatureList h_fl
);


// Host function to allocate GPU resources needed for tracking
__host__ void allocateGPUResources(int numFeatures, KLT_TrackingContext h_tc, int ncols, int nrows);
__host__ void freeGPUResources();

#ifdef __cplusplus
}
#endif


#endif