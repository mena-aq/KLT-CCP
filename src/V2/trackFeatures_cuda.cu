#include "trackFeatures_cuda.h"
#include "convolve_cuda.h"

__host__device__ float sumAbsFloatWindowCUDA(
	float* fw,
	int width,
	int height
)
{
	float sum = 0.0f;
	for (int i = 0; i < width * height; ++i) {
		sum += fabsf(fw[i]);
	}
	return sum;
}

// Helper for single contiguous device pyramid buffer
typedef struct {
    float *d_buffer; // device buffer
    size_t total_floats;
} DevicePyramidAlloc;

// Device accessor for packed pyramid buffer
__device__ float getPyramidVal(const float *buf, int idx) { return buf[idx]; }
__device__ int getPyramidInt(const float *buf, int idx) { return (int)buf[idx]; }
__device__ float getPyramidSubsampling(const float *buf) { return buf[1]; }
__device__ int getPyramidNLevels(const float *buf) { return (int)buf[0]; }
__device__ int getPyramidNRows(const float *buf, int level) { int nLevels = getPyramidNLevels(buf); return (int)buf[2 + level]; }
__device__ int getPyramidNCols(const float *buf, int level) { int nLevels = getPyramidNLevels(buf); return (int)buf[2 + nLevels + level]; }
__device__ int getPyramidDataOffset(const float *buf, int level) { int nLevels = getPyramidNLevels(buf); return (int)buf[2 + 2*nLevels + level]; }


__device__ float interpolateCUDA(float x, float y, const float *buf, int level) {
	int nLevels = getPyramidNLevels(buf);
	int nc = getPyramidNCols(buf, level);
	int nr = getPyramidNRows(buf, level);
	int offset = getPyramidDataOffset(buf, level);
	int xt = (int)x;
	int yt = (int)y;
	float ax = x - xt;
	float ay = y - yt;
	const float *ptr = buf + offset + yt * nc + xt;
	return ( (1-ax) * (1-ay) * ptr[0] +
			 ax   * (1-ay) * ptr[1] +
			 (1-ax) *   ay   * ptr[nc] +
			 ax   *   ay   * ptr[nc+1] );
}

// same as interpolate in cpu but __host__device__
/*
__host__device__ float interpolateCUDA(
	float x,
	float y,
	_KLT_FloatImage img
)
{
  int xt = (int) x;  
  int yt = (int) y;
  float ax = x - xt;
  float ay = y - yt;
  float *ptr = img->data + (img->ncols*yt) + xt;

#ifndef _DNDEBUG
  if (xt<0 || yt<0 || xt>=img->ncols-1 || yt>=img->nrows-1) {
    fprintf(stderr, "(xt,yt)=(%d,%d)  imgsize=(%d,%d)\n"
            "(x,y)=(%f,%f)  (ax,ay)=(%f,%f)\n",
            xt, yt, img->ncols, img->nrows, x, y, ax, ay);
    fflush(stderr);
  }
#endif

  assert (xt >= 0 && yt >= 0 && xt <= img->ncols - 2 && yt <= img->nrows - 2);

  return ( (1-ax) * (1-ay) * *ptr + //top-left
           ax   * (1-ay) * *(ptr+1) + //top-right
           (1-ax) *   ay   * *(ptr+(img->ncols)) + //bottom-left
           ax   *   ay   * *(ptr+(img->ncols)+1) ); //bottom-right
}
*/


// Deep-copy a host pyramid to the device. Returns DevicePyramidAlloc; on error, d_pyr will be NULL.
// Packs: [nLevels, subsampling, nrows[n], ncols[n], img_ptrs[n], img_data...]
static DevicePyramidAlloc deepCopyPyramidToDevice(_KLT_Pyramid src) {
	DevicePyramidAlloc out;
	out.d_buffer = NULL; out.total_floats = 0;
	if (!src) return out;
	int nLevels = src->nLevels;
	size_t n_floats = 0;
	// Header: nLevels, subsampling
	n_floats += 2;
	// nrows, ncols, img_ptrs
	n_floats += nLevels * 3;
	// img_data
	size_t *level_offsets = (size_t*)malloc(sizeof(size_t) * nLevels);
	size_t data_offset = n_floats;
	for (int i = 0; i < nLevels; ++i) {
		level_offsets[i] = data_offset;
		_KLT_FloatImage h_img = src->img[i];
		data_offset += (size_t)h_img->ncols * (size_t)h_img->nrows;
	}
	n_floats = data_offset;
	out.total_floats = n_floats;
	float *h_buffer = (float*)malloc(sizeof(float) * n_floats);
	if (!h_buffer) { free(level_offsets); return out; }
	// Pack header
	h_buffer[0] = (float)nLevels;
	h_buffer[1] = (float)src->subsampling;
	// Pack nrows, ncols, img_ptrs (as float offsets)
	for (int i = 0; i < nLevels; ++i) {
		h_buffer[2 + i] = (float)src->nrows[i];
		h_buffer[2 + nLevels + i] = (float)src->ncols[i];
		h_buffer[2 + 2*nLevels + i] = (float)level_offsets[i];
	}
	// Pack image data
	for (int i = 0; i < nLevels; ++i) {
		_KLT_FloatImage h_img = src->img[i];
		int w = h_img->ncols;
		int h = h_img->nrows;
		size_t n_pix = (size_t)w * (size_t)h;
		float *dst = h_buffer + level_offsets[i];
		memcpy(dst, h_img->data, sizeof(float) * n_pix);
	}
	// Allocate device buffer and copy
	float *d_buffer = NULL;
	if (cudaMalloc((void**)&d_buffer, sizeof(float) * n_floats) != cudaSuccess) {
		free(h_buffer); free(level_offsets); return out;
	}
	cudaMemcpy(d_buffer, h_buffer, sizeof(float) * n_floats, cudaMemcpyHostToDevice);
	out.d_buffer = d_buffer;
	free(h_buffer); free(level_offsets);
	return out;
}


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
)
{

	int status;
	int iteration = 0;
	int hw = window_width / 2;
	int hh = window_height / 2;
  	float one_plus_eps = 1.001f;
	int val = KLT_TRACKED;

	int nPyramidLevels = getPyramidNLevels(d_pyramid1);
	float subsampling = getPyramidSubsampling(d_pyramid1);

	// which feature to track
	int featureIdx = blockIdx.x;
	// only track features that are not lost
	if (d_in_val[featureIdx] < 0) {
		// mark output as lost immediately
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			d_out_x[featureIdx] = -1.0f;
			d_out_y[featureIdx] = -1.0f;
			d_out_val[featureIdx] = d_in_val[featureIdx];
		}
		return;
	}

	// Prepare per-feature coordinates from input arrays
	float xloc = d_in_x[featureIdx];
	float yloc = d_in_y[featureIdx];
	float xlocout, ylocout;

	// Thread ID
	int tid = threadIdx.y * blockDim.x + threadIdx.x;

	// shared memory allocation
	extern __shared__ float shared[];
	float* imgdiff = shared; //imgdiff: first part of shared memory
	float* gradx = imgdiff + window_width * window_height; //gradx: second part of shared memory
	float* grady = gradx + window_width * window_height; //grady: third part of shared memory

	// Transform to coarsest
	for (int rr = nPyramidLevels - 1; rr >= 0; --rr) {
		xloc /= subsampling; yloc /= subsampling;
	}
	xlocout = xloc; ylocout = yloc;

	// shared vars for x1,y1 dx,dy and final x2,y2 of feature
	__shared__ float s_x1, s_y1;
	__shared__ float s_dx, s_dy;
	__shared__ float s_x2, s_y2;
	__shared__ int s_status; // use KLT_* codes
	__shared__ int s_continue;

	if (tid == 0) {
		s_x1 = xloc; s_y1 = yloc;
		s_x2 = xlocout; s_y2 = ylocout; 
		s_dx = 0.0f; s_dy = 0.0f; 
		s_status = KLT_TRACKED; s_continue = 1;
	}
	__syncthreads();

	for (int r = nPyramidLevels - 1; r >= 0; --r) {

		// If already lost, skip levels
		if (s_status != KLT_TRACKED) break;

		// Scale coordinates 
		if (tid == 0) {
			s_x1 *= subsampling; s_y1 *= subsampling;
			s_x2 *= subsampling; s_y2 *= subsampling;
		}
		__syncthreads();

		// Each thread computes its window pixel
		int i = threadIdx.x - hw;
		int j = threadIdx.y - hh;
		int idx = threadIdx.y * window_width + threadIdx.x;

		// iterative refinement
		iteration = 0;
		s_continue = 1;
		do {
            // compute per-thread sample coordinates (pixel)
            float samp_x1 = s_x1 + (float)i;
            float samp_y1 = s_y1 + (float)j;
            float samp_x2 = s_x2 + (float)i;
            float samp_y2 = s_y2 + (float)j;

			float i1 = interpolateCUDA(samp_x1, samp_y1, d_pyramid1, r);
			float i2 = interpolateCUDA(samp_x2, samp_y2, d_pyramid2, r);
			imgdiff[idx] = i1 - i2;

			float gx1 = interpolateCUDA(samp_x1, samp_y1, d_pyramid1_gradx, r);
			float gx2 = interpolateCUDA(samp_x2, samp_y2, d_pyramid2_gradx, r);
			gradx[idx] = gx1 + gx2;
			float gy1 = interpolateCUDA(samp_x1, samp_y1, d_pyramid1_grady, r);
			float gy2 = interpolateCUDA(samp_x2, samp_y2, d_pyramid2_grady, r);
			grady[idx] = gy1 + gy2;

			__syncthreads();

			if (tid == 0) {
				// gradient matrix & error vector
				float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f, ex = 0.0f, ey = 0.0f;
				int W = window_width * window_height;
				for (int k = 0; k < W; ++k) {
					float gx = gradx[k]; float gy = grady[k]; float diff = imgdiff[k];
					gxx += gx*gx; gxy += gx*gy; gyy += gy*gy;
					ex += diff * gx; ey += diff * gy;
				}
				ex *= step_factor; ey *= step_factor;
				float det = gxx*gyy - gxy*gxy;
				s_dx = 0.0f; s_dy = 0.0f;
				// solve equation
				if (det < small) {
					s_status = KLT_SMALL_DET;
					s_continue = 0; //if small det, stop
				} else {
					s_dx = (gyy*ex - gxy*ey)/det;
					s_dy = (gxx*ey - gxy*ex)/det;
					s_x2 += s_dx; s_y2 += s_dy;
					// stop if step is small enough
					s_continue = (fabsf(s_dx) >= th || fabsf(s_dy) >= th) ? 1 : 0;
				}
			}
			__syncthreads();

			iteration++;

		} while (s_continue && iteration < max_iterations && s_status == KLT_TRACKED);

		// Recompute intensity difference at final position for residue
		float samp_x1_f = s_x1 + (float)i;
		float samp_y1_f = s_y1 + (float)j;
		float samp_x2_f = s_x2 + (float)i;
		float samp_y2_f = s_y2 + (float)j;
		imgdiff[idx] = interpolateCUDA(samp_x1_f, samp_y1_f, d_pyramid1, r) - interpolateCUDA(samp_x2_f, samp_y2_f, d_pyramid2, r);
		__syncthreads();

		if (tid == 0) {
			int W = window_width * window_height;
			float residue = sumAbsFloatWindowCUDA(imgdiff, window_width, window_height) / (float)W;
			if (residue > max_residue) s_status = KLT_LARGE_RESIDUE;
		}
		__syncthreads();

		if (s_status != KLT_TRACKED) break;
	}

	// record feature into output arrays
	if (tid == 0) {
		float final_x = s_x2;
		float final_y = s_y2;
		int final_val = KLT_TRACKED;

		// out-of-bounds check using final coordinates
		if (final_x < d_tc->borderx || final_x >= window_width - d_tc->borderx ||
			final_y < d_tc->bordery || final_y >= window_height - d_tc->bordery) {
			final_val = KLT_OOB;
		}

		// Map shared status codes
		if (s_status == KLT_SMALL_DET) final_val = KLT_SMALL_DET;
		else if (s_status == KLT_LARGE_RESIDUE) final_val = KLT_LARGE_RESIDUE;
		else if (s_status == KLT_MAX_ITERATIONS) final_val = KLT_MAX_ITERATIONS;

		if (final_val != KLT_TRACKED) {
			d_out_x[featureIdx] = -1.0f;
			d_out_y[featureIdx] = -1.0f;
			d_out_val[featureIdx] = final_val;
		} else {
			d_out_x[featureIdx] = final_x;
			d_out_y[featureIdx] = final_y;
			d_out_val[featureIdx] = KLT_TRACKED;
		}
	}

	__syncthreads();
	
};


__host__ void kltTrackFeaturesCUDA(
  KLT_TrackingContext *h_tc,
  KLT_PixelType *h_img1,
  KLT_PixelType *h_img2,
  KLT_FeatureList *h_fl,
  KLT_TrackingContext *d_tc,
  KLT_PixelType *d_img1,
  KLT_PixelType *d_img2,
  KLT_FeatureList *d_fl,
  int ncols,
  int nrows)
{

	_KLT_FloatImage tmpimg, floatimg1, floatimg2;
	_KLT_Pyramid pyramid1, pyramid1_gradx, pyramid1_grady,
		pyramid2, pyramid2_gradx, pyramid2_grady;
	float subsampling = (float) h_tc->subsampling;
	float xloc, yloc, xlocout, ylocout;
	int val;
	int indx, r;
	KLT_BOOL floatimg1_created = FALSE;
	int i;

	if (KLT_verbose >= 1)  {
		fprintf(stderr,  "(KLT) Tracking %d features in a %d by %d image...  ",
			KLTCountRemainingFeatures(h_fl), ncols, nrows);
		fflush(stderr);
	}

	/* Check window size (and correct if necessary) */
	if (h_tc->window_width % 2 != 1) {
		h_tc->window_width = h_tc->window_width+1;
		KLTWarning("Tracking context's window width must be odd.  "
			"Changing to %d.\n", h_tc->window_width);
	}
	if (h_tc->window_height % 2 != 1) {
		h_tc->window_height = h_tc->window_height+1;
		KLTWarning("Tracking context's window height must be odd.  "
			"Changing to %d.\n", h_tc->window_height);
	}
	if (h_tc->window_width < 3) {
		h_tc->window_width = 3;
		KLTWarning("Tracking context's window width must be at least three.  \n"
			"Changing to %d.\n", h_tc->window_width);
	}
	if (h_tc->window_height < 3) {
		h_tc->window_height = 3;
		KLTWarning("Tracking context's window height must be at least three.  \n"
			"Changing to %d.\n", h_tc->window_height);
	}

	/* Create temporary image */
	tmpimg = _KLTCreateFloatImage(ncols, nrows);

	/* Process first image by converting to float, smoothing, computing */
	/* pyramid, and computing gradient pyramids */

  	// if sequential mode and previous pyramid exists, reuse it
  	// example3 uses sequential mode which means the first if block is executed for all frames except the first one
	if (h_tc->sequentialMode && h_tc->pyramid_last != NULL)  {
		pyramid1 = (_KLT_Pyramid) h_tc->pyramid_last;
		pyramid1_gradx = (_KLT_Pyramid) h_tc->pyramid_last_gradx;
		pyramid1_grady = (_KLT_Pyramid) h_tc->pyramid_last_grady;
		if (pyramid1->ncols[0] != ncols || pyramid1->nrows[0] != nrows)
			KLTError("(KLTTrackFeatures) Size of incoming image (%d by %d) "
			"is different from size of previous image (%d by %d)\n",
			ncols, nrows, pyramid1->ncols[0], pyramid1->nrows[0]);
		assert(pyramid1_gradx != NULL);
		assert(pyramid1_grady != NULL);

	} else  {

    	// ------ this block is only executed ONCE in example3 for the first frame ------
		floatimg1_created = TRUE;
		floatimg1 = _KLTCreateFloatImage(ncols, nrows);
		_KLTToFloatImage(h_img1, ncols, nrows, tmpimg);

    	// ------------- convolve_seperate with Gaussian kernel (blur) ---------------
		computeSmoothedImageCUDA(tmpimg, _KLTComputeSmoothSigma(h_tc), floatimg1);
    	//---------------------------------------------------------------------------

		pyramid1 = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);

		// ------------- calls ComputeSmoothedImage at each level which calls convolve_seperate ---------------
    	computePyramidCUDA(floatimg1, pyramid1, h_tc->pyramid_sigma_fact);
    	// --------------------------------------------------------------------------

		pyramid1_gradx = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);
		pyramid1_grady = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);

    	// ------ calls convolve_seperate for both gradx and grady for each level ---------
		for (i = 0 ; i < h_tc->nPyramidLevels ; i++)
			computeGradientsCUDA(pyramid1->img[i], h_tc->grad_sigma, 
			pyramid1_gradx->img[i],
			pyramid1_grady->img[i]);
    	// --------------------------------------------------------------------------
    
	}

	/* Do the same thing with second image */
	floatimg2 = _KLTCreateFloatImage(ncols, nrows);
	_KLTToFloatImage(h_img2, ncols, nrows, tmpimg);

  	// ------------- convolve_seperate with Gaussian kernel (blur) ---------------
	computeSmoothedImageCUDA(tmpimg, _KLTComputeSmoothSigma(h_tc), floatimg2);
	// --------------------------------------------------------------------------

  	pyramid2 = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);
	
  	// ------------- calls ComputeSmoothedImage at each level which calls convolve_seperate ---------------
  	computePyramidCUDA(floatimg2, pyramid2, h_tc->pyramid_sigma_fact);
	// --------------------------------------------------------------------------

  	pyramid2_gradx = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);
	pyramid2_grady = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);
	
  	// ------ calls convolve_seperate for both gradx and grady for each level ---------
  	for (i = 0 ; i < h_tc->nPyramidLevels ; i++)
		computeGradientsCUDA(pyramid2->img[i], h_tc->grad_sigma, 
		pyramid2_gradx->img[i],
		pyramid2_grady->img[i]);
  	// --------------------------------------------------------------------------

	/* Write internal images */
	if (h_tc->writeInternalImages)  {
		char fname[80];
		for (i = 0 ; i < h_tc->nPyramidLevels ; i++)  {
			sprintf(fname, "kltimg_tf_i%d.pgm", i);
			_KLTWriteFloatImageToPGM(pyramid1->img[i], fname);
			sprintf(fname, "kltimg_tf_i%d_gx.pgm", i);
			_KLTWriteFloatImageToPGM(pyramid1_gradx->img[i], fname);
			sprintf(fname, "kltimg_tf_i%d_gy.pgm", i);
			_KLTWriteFloatImageToPGM(pyramid1_grady->img[i], fname);
			sprintf(fname, "kltimg_tf_j%d.pgm", i);
			_KLTWriteFloatImageToPGM(pyramid2->img[i], fname);
			sprintf(fname, "kltimg_tf_j%d_gx.pgm", i);
			_KLTWriteFloatImageToPGM(pyramid2_gradx->img[i], fname);
			sprintf(fname, "kltimg_tf_j%d_gy.pgm", i);
			_KLTWriteFloatImageToPGM(pyramid2_grady->img[i], fname);
		}
	}

	// allocate device memory for pyramids
	// Deep-copy pyramids to device (contiguous buffer)
	DevicePyramidAlloc d_pyramid1 = deepCopyPyramidToDevice(pyramid1);
	DevicePyramidAlloc d_pyramid1_gradx = deepCopyPyramidToDevice(pyramid1_gradx);
	DevicePyramidAlloc d_pyramid1_grady = deepCopyPyramidToDevice(pyramid1_grady);
	DevicePyramidAlloc d_pyramid2 = deepCopyPyramidToDevice(pyramid2);
	DevicePyramidAlloc d_pyramid2_gradx = deepCopyPyramidToDevice(pyramid2_gradx);
	DevicePyramidAlloc d_pyramid2_grady = deepCopyPyramidToDevice(pyramid2_grady);

	// launch kernel to track each feature
	int window_width = h_tc->window_width;
	int window_height = h_tc->window_height;
	float step_factor = h_tc->step_factor;
	int max_iterations = h_tc->max_iterations;
	float small = h_tc->min_determinant;
	float th = h_tc->min_displacement;
	float max_residue = h_tc->max_residue;

	int numFeatures = KLTCountRemainingFeatures(h_fl);
	dim3 blockSize(window_width, window_height);
	dim3 gridSize(numFeatures);
	size_t sharedMemSize = window_width * window_height * sizeof(float) * 3; // for imgdiff, gradx, grady (3)

	// Prepare device input arrays for feature list (input & output)
	float *d_in_x = NULL, *d_in_y = NULL;
	int *d_in_val = NULL;
	float *d_out_x = NULL, *d_out_y = NULL;
	int *d_out_val = NULL;

	float *h_in_x = (float*)malloc(sizeof(float) * numFeatures);
	float *h_in_y = (float*)malloc(sizeof(float) * numFeatures);
	int *h_in_val = (int*)malloc(sizeof(int) * numFeatures);
	float *h_out_x = (float*)malloc(sizeof(float) * numFeatures);
	float *h_out_y = (float*)malloc(sizeof(float) * numFeatures);
	int *h_out_val = (int*)malloc(sizeof(int) * numFeatures);

	for (i = 0; i < numFeatures; ++i) {
		h_in_x[i] = h_fl->feature[i]->x;
		h_in_y[i] = h_fl->feature[i]->y;
		h_in_val[i] = h_fl->feature[i]->val;
	}

	cudaMalloc((void**)&d_in_x, sizeof(float) * numFeatures);
	cudaMalloc((void**)&d_in_y, sizeof(float) * numFeatures);
	cudaMalloc((void**)&d_in_val, sizeof(int) * numFeatures);
	cudaMalloc((void**)&d_out_x, sizeof(float) * numFeatures);
	cudaMalloc((void**)&d_out_y, sizeof(float) * numFeatures);
	cudaMalloc((void**)&d_out_val, sizeof(int) * numFeatures);

	cudaMemcpy(d_in_x, h_in_x, sizeof(float) * numFeatures, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_y, h_in_y, sizeof(float) * numFeatures, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in_val, h_in_val, sizeof(int) * numFeatures, cudaMemcpyHostToDevice);

	// Launch kernel with feature arrays and packed pyramids
	trackFeatureKernel<<<gridSize, blockSize, sharedMemSize>>>(
		d_tc,
		d_pyramid1.d_buffer,
		d_pyramid1_gradx.d_buffer,
		d_pyramid1_grady.d_buffer,
		d_pyramid2.d_buffer,
		d_pyramid2_gradx.d_buffer,
		d_pyramid2_grady.d_buffer,
		d_in_x,
		d_in_y,
		d_in_val,
		d_out_x,
		d_out_y,
		d_out_val,
		window_width,
		window_height,
		step_factor,
		max_iterations,
		small,
		th,
		max_residue
	);
	cudaDeviceSynchronize();

	// Copy back outputs
	cudaMemcpy(h_out_x, d_out_x, sizeof(float) * numFeatures, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out_y, d_out_y, sizeof(float) * numFeatures, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out_val, d_out_val, sizeof(int) * numFeatures, cudaMemcpyDeviceToHost);

	// Merge results into host feature list
	for (i = 0; i < numFeatures; ++i) {
		h_fl->feature[i]->x = h_out_x[i];
		h_fl->feature[i]->y = h_out_y[i];
		h_fl->feature[i]->val = h_out_val[i];
	}

	// free temp host/device arrays
	free(h_in_x); free(h_in_y); free(h_in_val);
	free(h_out_x); free(h_out_y); free(h_out_val);
	cudaFree(d_in_x); cudaFree(d_in_y); cudaFree(d_in_val);
	cudaFree(d_out_x); cudaFree(d_out_y); cudaFree(d_out_val);

	// free device memory for pyramids (single buffer)
	#define FREE_DEVICE_PYRAMID(dp) do { if ((dp).d_buffer) cudaFree((dp).d_buffer); } while(0)
	FREE_DEVICE_PYRAMID(d_pyramid1);
	FREE_DEVICE_PYRAMID(d_pyramid1_gradx);
	FREE_DEVICE_PYRAMID(d_pyramid1_grady);
	FREE_DEVICE_PYRAMID(d_pyramid2);
	FREE_DEVICE_PYRAMID(d_pyramid2_gradx);
	FREE_DEVICE_PYRAMID(d_pyramid2_grady);
	#undef FREE_DEVICE_PYRAMID

  	// to reuse pyramid of current image as previous image in next call
	if (h_tc->sequentialMode)  {
		h_tc->pyramid_last = pyramid2;
		h_tc->pyramid_last_gradx = pyramid2_gradx;
		h_tc->pyramid_last_grady = pyramid2_grady;
	} else  {
		_KLTFreePyramid(pyramid2);
		_KLTFreePyramid(pyramid2_gradx);
		_KLTFreePyramid(pyramid2_grady);
	}

	/* Free memory */
	_KLTFreeFloatImage(tmpimg);
	if (floatimg1_created)  _KLTFreeFloatImage(floatimg1);
	_KLTFreeFloatImage(floatimg2);
	_KLTFreePyramid(pyramid1);
	_KLTFreePyramid(pyramid1_gradx);
	_KLTFreePyramid(pyramid1_grady);

	if (KLT_verbose >= 1)  {
		fprintf(stderr,  "\n\t%d features successfully tracked.\n",
			KLTCountRemainingFeatures(featurelist));
		if (h_tc->writeInternalImages)
			fprintf(stderr,  "\tWrote images to 'kltimg_tf*.pgm'.\n");
		fflush(stderr);
	}


}
