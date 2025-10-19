#include "trackFeatures_cuda.h"
#include "convolve_cuda.h"

// cuda check macro for error handling
#define CUDA_CHECK(call)                                                    \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

// helper function to round up to nearest multiple of 32 for thread warps
static inline int roundUp32(int x) {
    return ((x + 31) / 32) * 32;
}


// Device accessor for packed pyramid buffer
__device__ float getPyramidVal(const float *buf, int idx) { return buf[idx]; }
__device__ int getPyramidInt(const float *buf, int idx) { return (int)buf[idx]; }
__device__ float getPyramidSubsampling(const float *buf) { return buf[1]; }
__device__ int getPyramidNLevels(const float *buf) { return (int)buf[0]; }
__device__ int getPyramidNRows(const float *buf, int level) { int nLevels = getPyramidNLevels(buf); return (int)buf[2 + level]; }
__device__ int getPyramidNCols(const float *buf, int level) { int nLevels = getPyramidNLevels(buf); return (int)buf[2 + nLevels + level]; }
__device__ int getPyramidDataOffset(const float *buf, int level) { int nLevels = getPyramidNLevels(buf); return (int)buf[2 + 2*nLevels + level]; }

extern int KLT_verbose;
typedef float *_FloatWindow;

// V3.2: to reuse pyr2 as pyr1 and d_out as d_in bw frames (double buffering)
static bool first_frame = true;

// reusable buffers to store feature list 
static float *d_in_x = NULL, *d_in_y = NULL;
static int *d_in_val = NULL;
static float *d_out_x = NULL, *d_out_y = NULL;  
static int *d_out_val = NULL;
static size_t feature_pool_size = 0;

// reusable pyramid device buffers
static float *d_pyramid1 = NULL;
static float *d_pyramid1_gradx = NULL;
static float *d_pyramid1_grady = NULL;
static float *d_pyramid2 = NULL;
static float *d_pyramid2_gradx = NULL;
static float *d_pyramid2_grady = NULL;

// V3.5 trying something w image so convolve doesnt need to copy and copy back
static float *d_img1 = NULL,*d_img2 =NULL;
static float *d_smooth_img1 = NULL, *d_smooth_img2 = NULL;


static void allocateFeatureList(int numFeatures) {
    if (feature_pool_size < numFeatures) {
        printf("Allocating feature list...\n");
        // Free existing if too small
        if (d_in_x) cudaFree(d_in_x);
        if (d_in_y) cudaFree(d_in_y);
        if (d_in_val) cudaFree(d_in_val);
        if (d_out_x) cudaFree(d_out_x);
        if (d_out_y) cudaFree(d_out_y);
        if (d_out_val) cudaFree(d_out_val);
        
        // Allocate new larger pool
        CUDA_CHECK(cudaMalloc((void**)&d_in_x, sizeof(float) * numFeatures));
        CUDA_CHECK(cudaMalloc((void**)&d_in_y, sizeof(float) * numFeatures));
        CUDA_CHECK(cudaMalloc((void**)&d_in_val, sizeof(int) * numFeatures));
        CUDA_CHECK(cudaMalloc((void**)&d_out_x, sizeof(float) * numFeatures));
        CUDA_CHECK(cudaMalloc((void**)&d_out_y, sizeof(float) * numFeatures));
        CUDA_CHECK(cudaMalloc((void**)&d_out_val, sizeof(int) * numFeatures));
        
        feature_pool_size = numFeatures;
        
        if (KLT_verbose) {
            printf("Allocated feature pool for %d features\n", numFeatures);
        }
    }
}

static size_t estimatePyramidSize(_KLT_Pyramid pyramid) {
    if (!pyramid) return 0;
    
    size_t total = 2; // nLevels + subsampling
    total += pyramid->nLevels * 3; // 1d nrows + ncols + offsets for nlevels

    // img pyramids 
    for (int i = 0; i < pyramid->nLevels; i++) {
        total += (size_t)pyramid->ncols[i] * (size_t)pyramid->nrows[i];
    }
    
    return total;
}

static void allocatePyramidBuffers(_KLT_Pyramid pyramid) {

    if (!d_pyramid1)
        printf("allocating pyramid buffers\n");
    // Calculate size once
    size_t total_floats = estimatePyramidSize(pyramid);
    size_t bytes_needed = total_floats * sizeof(float);
    
    // Allocate if null
    if (!d_pyramid1) CUDA_CHECK(cudaMalloc(&d_pyramid1, bytes_needed));
    if (!d_pyramid1_gradx) CUDA_CHECK(cudaMalloc(&d_pyramid1_gradx, bytes_needed));
    if (!d_pyramid1_grady) CUDA_CHECK(cudaMalloc(&d_pyramid1_grady, bytes_needed));
    if (!d_pyramid2) CUDA_CHECK(cudaMalloc(&d_pyramid2, bytes_needed));
    if (!d_pyramid2_gradx) CUDA_CHECK(cudaMalloc(&d_pyramid2_gradx, bytes_needed));
    if (!d_pyramid2_grady) CUDA_CHECK(cudaMalloc(&d_pyramid2_grady, bytes_needed));
}


static void copyPyramidToDevice(_KLT_Pyramid src, float* d_dest) {
    if (!src || !d_dest) return;
    
    int nLevels = src->nLevels;
    size_t n_floats = estimatePyramidSize(src);
    
    // Allocate temporary host buffer
    float *h_buffer = (float*)malloc(sizeof(float) * n_floats);
    
    // Pack data into host buffer (same as before)
    size_t *level_offsets = (size_t*)malloc(sizeof(size_t) * nLevels);
    size_t data_offset = 2 + nLevels * 3;  // header + arrays
    
    // Pack header
    h_buffer[0] = (float)nLevels;
    h_buffer[1] = (float)src->subsampling;
    
    // Pack nrows, ncols, offsets
    for (int i = 0; i < nLevels; i++) {
        h_buffer[2 + i] = (float)src->nrows[i];
        h_buffer[2 + nLevels + i] = (float)src->ncols[i];
        level_offsets[i] = data_offset;
        h_buffer[2 + 2*nLevels + i] = (float)level_offsets[i];
        data_offset += (size_t)src->ncols[i] * (size_t)src->nrows[i];
    }
    
    // Pack image data
    for (int i = 0; i < nLevels; i++) {
        _KLT_FloatImage h_img = src->img[i];
        int w = h_img->ncols;
        int h = h_img->nrows;
        size_t n_pix = (size_t)w * (size_t)h;
        float *dst = h_buffer + level_offsets[i];
        memcpy(dst, h_img->data, sizeof(float) * n_pix);
    }
    
    // Copy to pre-allocated device buffer
    CUDA_CHECK(cudaMemcpy(d_dest, h_buffer, sizeof(float) * n_floats, cudaMemcpyHostToDevice));
    
    free(h_buffer);
    free(level_offsets);
}

__host__ __device__ float sumAbsFloatWindowCUDA(
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
    int borderx,
    int bordery,
    int window_width,
    int window_height,
    float step_factor,
    int max_iterations,
    float small,
    float th,
    float max_residue
)
{
    int featureIdx = blockIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    
    // Only track features that are not lost
    if (d_in_val[featureIdx] < 0) {
        if (tid == 0) {
            d_out_x[featureIdx] = -1.0f;
            d_out_y[featureIdx] = -1.0f;
            d_out_val[featureIdx] = d_in_val[featureIdx];
        }
        return;
    }

    int hw = window_width / 2;
    int hh = window_height / 2;
	int window_elems = window_width * window_height;

    // Bounds check: skip threads outside the window
    if (tid >= window_elems) return;

    int nPyramidLevels = getPyramidNLevels(d_pyramid1);
    float subsampling = getPyramidSubsampling(d_pyramid1);
    
    // Shared variables for feature state 
    __shared__ float s_x1, s_y1, s_x2, s_y2;
    __shared__ float s_dx, s_dy;
    __shared__ int s_status, s_continue;
    __shared__ int s_iteration;
    
    // Shared reduction variables
    __shared__ float s_gxx, s_gxy, s_gyy, s_ex, s_ey;
    
    if (tid == 0) {
        // Initialize coordinates
        float xloc = d_in_x[featureIdx];
        float yloc = d_in_y[featureIdx];
        
        // Transform to coarsest resolution
        for (int r = nPyramidLevels - 1; r >= 0; r--) {
            xloc /= subsampling; yloc /= subsampling;
        }
        s_x1 = xloc; s_y1 = yloc;
        s_x2 = xloc; s_y2 = yloc;
        s_status = KLT_TRACKED;
    }
    __syncthreads();

    // Main pyramid levels loop
    for (int r = nPyramidLevels - 1; r >= 0; r--) {
        if (s_status != KLT_TRACKED) break;
        
        if (tid == 0) {
            s_x1 *= subsampling; s_y1 *= subsampling;
            s_x2 *= subsampling; s_y2 *= subsampling;
        }
        __syncthreads();

        // Iterative refinement for this pyramid level
        s_iteration = 0;
        s_continue = 1;
        
        do {
            // Each thread computes its portion of the window
            int i = threadIdx.x - hw;
            int j = threadIdx.y - hh;
            
            float samp_x1 = s_x1 + (float)i;
            float samp_y1 = s_y1 + (float)j;
            float samp_x2 = s_x2 + (float)i;
            float samp_y2 = s_y2 + (float)j;

            // Compute this thread's contribution
            float imgdiff = interpolateCUDA(samp_x1, samp_y1, d_pyramid1, r) - 
                           interpolateCUDA(samp_x2, samp_y2, d_pyramid2, r);
            
            float gx = interpolateCUDA(samp_x1, samp_y1, d_pyramid1_gradx, r) + 
                      interpolateCUDA(samp_x2, samp_y2, d_pyramid2_gradx, r);
            
            float gy = interpolateCUDA(samp_x1, samp_y1, d_pyramid1_grady, r) + 
                      interpolateCUDA(samp_x2, samp_y2, d_pyramid2_grady, r);

            // Each thread computes its partial sums
            float my_gxx = gx * gx;
            float my_gxy = gx * gy;
            float my_gyy = gy * gy;
            float my_ex = imgdiff * gx;
            float my_ey = imgdiff * gy;

            // Parallel reduction across threads
            if (tid == 0) {
                s_gxx = 0.0f; s_gxy = 0.0f; s_gyy = 0.0f; 
                s_ex = 0.0f; s_ey = 0.0f;
            }
            __syncthreads();

            // atomic add all thread contributions
            atomicAdd(&s_gxx, my_gxx);
            atomicAdd(&s_gxy, my_gxy);
            atomicAdd(&s_gyy, my_gyy);
            atomicAdd(&s_ex, my_ex);
            atomicAdd(&s_ey, my_ey);
            __syncthreads();

            // thread 0 computes the displacement update
            if (tid == 0) {
                s_ex *= step_factor;
                s_ey *= step_factor;
                
                float det = s_gxx * s_gyy - s_gxy * s_gxy;
                s_dx = 0.0f; s_dy = 0.0f;
                
                if (det < small) {
                    s_status = KLT_SMALL_DET;
                    s_continue = 0;
                } else {
                    s_dx = (s_gyy * s_ex - s_gxy * s_ey) / det;
                    s_dy = (s_gxx * s_ey - s_gxy * s_ex) / det;
                    s_x2 += s_dx;
                    s_y2 += s_dy;
                    s_continue = (fabsf(s_dx) >= th || fabsf(s_dy) >= th) ? 1 : 0;
                }
                s_iteration++;
            }
            __syncthreads();

        } while (s_continue && s_iteration < max_iterations && s_status == KLT_TRACKED);

        // check residue
        if (s_status == KLT_TRACKED) {
            __shared__ float s_residue_sum;
            if (tid == 0) s_residue_sum = 0.0f;
            __syncthreads();

            // Each thread computes its pixel's contribution to residue
            int i = threadIdx.x - hw;
            int j = threadIdx.y - hh;
            float samp_x1 = s_x1 + (float)i;
            float samp_y1 = s_y1 + (float)j;
            float samp_x2 = s_x2 + (float)i;
            float samp_y2 = s_y2 + (float)j;
            
            float residue_val = fabsf(interpolateCUDA(samp_x1, samp_y1, d_pyramid1, r) - 
                                     interpolateCUDA(samp_x2, samp_y2, d_pyramid2, r));
            
            atomicAdd(&s_residue_sum, residue_val);
            __syncthreads();

			// thread 0 checks average residue
            if (tid == 0) {
                float residue = s_residue_sum / (window_width * window_height);
                if (residue > max_residue) s_status = KLT_LARGE_RESIDUE;
            }
            __syncthreads();
        }
    }

    // record final value and status of the feature
    if (tid == 0) {
        float final_x = s_x2;
        float final_y = s_y2;
        int final_val = s_status;

        // bounds check
        int nc = getPyramidNCols(d_pyramid1, 0);
        int nr = getPyramidNRows(d_pyramid1, 0);
        
        if (final_x - hw < borderx || final_x + hw >= nc - borderx ||
            final_y - hh < bordery || final_y + hh >= nr - bordery) {
            final_val = KLT_OOB;
        }

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
}

__host__ void kltTrackFeaturesCUDA(
  KLT_TrackingContext h_tc,
  KLT_PixelType *h_img1,
  KLT_PixelType *h_img2,
  int ncols,
  int nrows,
  KLT_FeatureList h_fl
)
{
    printf("hello");
    _KLT_FloatImage tmpimg, floatimg1, floatimg2;
	_KLT_Pyramid pyramid1, pyramid1_gradx, pyramid1_grady,
		pyramid2, pyramid2_gradx, pyramid2_grady;
	float subsampling = (float) h_tc->subsampling;
	//float xloc, yloc, xlocout, ylocout;
	//int val;
	//int indx, r;
	KLT_BOOL floatimg1_created = FALSE;
	int i;
    
	int numFeatures = KLTCountRemainingFeatures(h_fl);

    // V3.1: allocate feature list once
    allocateFeatureList(numFeatures);

    // V3.1: allocate pyramids once
    _KLT_Pyramid temp_pyramid_for_size = NULL;
    if (h_tc->sequentialMode && h_tc->pyramid_last != NULL) {
        temp_pyramid_for_size = (_KLT_Pyramid) h_tc->pyramid_last;
    } else {
        // Create a temporary pyramid just for size estimation
        temp_pyramid_for_size = _KLTCreatePyramid(ncols, nrows, (int)subsampling, h_tc->nPyramidLevels);
    }
    
    allocatePyramidBuffers(temp_pyramid_for_size);
    
    // Free temporary pyramid if we created it
    if (temp_pyramid_for_size != h_tc->pyramid_last) {
        _KLTFreePyramid(temp_pyramid_for_size);
    }

    // v3.5
    if (first_frame){
        //CUDA_CHECK(cudaMalloc((void**)&d_img1, ncols * nrows * sizeof(KLT_PixelType)));
    	//CUDA_CHECK(cudaMalloc((void**)&d_img2, ncols * nrows * sizeof(KLT_PixelType)));
    }

	// allocate device memory for 2 sequential frames
    // v3.4 realising i dont need these at all
    /*
	KLT_PixelType *d_img1, *d_img2;
	CUDA_CHECK(cudaMalloc((void**)&d_img1, ncols * nrows * sizeof(KLT_PixelType)));
	CUDA_CHECK(cudaMalloc((void**)&d_img2, ncols * nrows * sizeof(KLT_PixelType)));
	// copy the 2 frames to track b/w to device
    CUDA_CHECK(cudaMemcpy(d_img1, h_img1, ncols * nrows * sizeof(KLT_PixelType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_img2, h_img2, ncols * nrows * sizeof(KLT_PixelType), cudaMemcpyHostToDevice));
    */

    
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
	if (h_tc->sequentialMode && h_tc->pyramid_last != NULL)  {
        
		pyramid1 = (_KLT_Pyramid) h_tc->pyramid_last;
		pyramid1_gradx = (_KLT_Pyramid) h_tc->pyramid_last_gradx;
		pyramid1_grady = (_KLT_Pyramid) h_tc->pyramid_last_grady;    

        //v3.2  Swap: pyramid2 becomes pyramid1 (reuse previous frames pyramid2, device to device)
        printf("swap pyramids\n");
        std::swap(d_pyramid1, d_pyramid2);
        std::swap(d_pyramid1_gradx, d_pyramid2_gradx);
        std::swap(d_pyramid1_grady, d_pyramid2_grady);

	} else  {

		floatimg1_created = TRUE;
		floatimg1 = _KLTCreateFloatImage(ncols, nrows);
		_KLTToFloatImage(h_img1, ncols, nrows, tmpimg);

		// smoothing using convolution kernels
        computeSmoothedImageCUDA(tmpimg, _KLTComputeSmoothSigma(h_tc), floatimg1);

		pyramid1 = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);

		// compute image pyramids for subsampling
    	computePyramidCUDA(floatimg1, pyramid1, h_tc->pyramid_sigma_fact);

		pyramid1_gradx = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);
		pyramid1_grady = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);

		for (i = 0 ; i < h_tc->nPyramidLevels ; i++)
			// compute img gradients in x and y
            computeGradientsCUDA(pyramid1->img[i], h_tc->grad_sigma, 
			pyramid1_gradx->img[i], pyramid1_grady->img[i]);

        // v3.2 Copy pyramids 1 to device
        // v3.5 no need to do this
        copyPyramidToDevice(pyramid1, d_pyramid1);
        copyPyramidToDevice(pyramid1_gradx, d_pyramid1_gradx);
        copyPyramidToDevice(pyramid1_grady, d_pyramid1_grady);

	}


	/* Do the same thing with second image */
	floatimg2 = _KLTCreateFloatImage(ncols, nrows);
	_KLTToFloatImage(h_img2, ncols, nrows, tmpimg);

    computeSmoothedImageCUDA(tmpimg, _KLTComputeSmoothSigma(h_tc), floatimg2);

  	pyramid2 = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);
	
    computePyramidCUDA(floatimg2, pyramid2, h_tc->pyramid_sigma_fact);

  	pyramid2_gradx = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);
	pyramid2_grady = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);
	
  	for (i = 0 ; i < h_tc->nPyramidLevels ; i++)
        computeGradientsCUDA(pyramid2->img[i], h_tc->grad_sigma, 
		pyramid2_gradx->img[i], pyramid2_grady->img[i]);


    // v3.2 Copy pyramids 2 to device
    // v3.5 no need 
    copyPyramidToDevice(pyramid2, d_pyramid2);
    copyPyramidToDevice(pyramid2_gradx, d_pyramid2_gradx);
    copyPyramidToDevice(pyramid2_grady, d_pyramid2_grady);

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
    /*
	DevicePyramidAlloc d_pyramid1 = deepCopyPyramidToDevice(pyramid1);
	DevicePyramidAlloc d_pyramid1_gradx = deepCopyPyramidToDevice(pyramid1_gradx);
	DevicePyramidAlloc d_pyramid1_grady = deepCopyPyramidToDevice(pyramid1_grady);
	DevicePyramidAlloc d_pyramid2 = deepCopyPyramidToDevice(pyramid2);
	DevicePyramidAlloc d_pyramid2_gradx = deepCopyPyramidToDevice(pyramid2_gradx);
	DevicePyramidAlloc d_pyramid2_grady = deepCopyPyramidToDevice(pyramid2_grady);
    */


    // copy to d pyramid
    /*
    copyPyramidToDevice(pyramid1, d_pyramid1);
    copyPyramidToDevice(pyramid1_gradx, d_pyramid1_gradx);
    copyPyramidToDevice(pyramid1_grady, d_pyramid1_grady);
    copyPyramidToDevice(pyramid2, d_pyramid2);
    copyPyramidToDevice(pyramid2_gradx, d_pyramid2_gradx);
    copyPyramidToDevice(pyramid2_grady, d_pyramid2_grady);
    */

	// prepare parameters for kernel launch
	int window_width = h_tc->window_width;
	int window_height = h_tc->window_height;
	float step_factor = h_tc->step_factor;
	int max_iterations = h_tc->max_iterations;
	float small = h_tc->min_determinant;
	float th = h_tc->min_displacement;
	float max_residue = h_tc->max_residue;
    int borderx = h_tc->borderx;
    int bordery = h_tc->bordery;

	// Prepare device input arrays for feature list (input & output)
    /*
	float *d_in_x = NULL, *d_in_y = NULL; //input feature coords
	int *d_in_val = NULL; //input feature vals (status)
	float *d_out_x = NULL, *d_out_y = NULL; //output feature coords
	int *d_out_val = NULL;	 //output feature vals (status)
    */

	// allocate temp host arrays for feature list
	float *h_in_x = (float*)malloc(sizeof(float) * numFeatures);
	float *h_in_y = (float*)malloc(sizeof(float) * numFeatures);
	int *h_in_val = (int*)malloc(sizeof(int) * numFeatures);
	float *h_out_x = (float*)malloc(sizeof(float) * numFeatures);
	float *h_out_y = (float*)malloc(sizeof(float) * numFeatures);
	int *h_out_val = (int*)malloc(sizeof(int) * numFeatures);

	// fill input feature list arrays
	for (i = 0; i < numFeatures; ++i) {
		h_in_x[i] = h_fl->feature[i]->x;
		h_in_y[i] = h_fl->feature[i]->y;
		h_in_val[i] = h_fl->feature[i]->val;
	}

	// allocate device arrays
    /*
	CUDA_CHECK(cudaMalloc((void**)&d_in_x, sizeof(float) * numFeatures));
	CUDA_CHECK(cudaMalloc((void**)&d_in_y, sizeof(float) * numFeatures));
	CUDA_CHECK(cudaMalloc((void**)&d_in_val, sizeof(int) * numFeatures));
	CUDA_CHECK(cudaMalloc((void**)&d_out_x, sizeof(float) * numFeatures));
	CUDA_CHECK(cudaMalloc((void**)&d_out_y, sizeof(float) * numFeatures));
	CUDA_CHECK(cudaMalloc((void**)&d_out_val, sizeof(int) * numFeatures));
    */

	// copy host inputs to device (feature list arrays)
    /*
	CUDA_CHECK(cudaMemcpy(d_in_x, h_in_x, sizeof(float) * numFeatures, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_in_y, h_in_y, sizeof(float) * numFeatures, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_in_val, h_in_val, sizeof(int) * numFeatures, cudaMemcpyHostToDevice));
    */

        
    // V3.2: double buffering of feature list
    if (first_frame) {
        // First frame: need to copy everything
        first_frame = false;
        printf("First frame\n");
        // Copy input features to device
        cudaMemcpy(d_in_x, h_in_x, sizeof(float) * numFeatures, cudaMemcpyHostToDevice);
        cudaMemcpy(d_in_y, h_in_y, sizeof(float) * numFeatures, cudaMemcpyHostToDevice);
        cudaMemcpy(d_in_val, h_in_val, sizeof(int) * numFeatures, cudaMemcpyHostToDevice);
    } else {
        // Subsequent frames: reuse outputs as inputs
        // out is now in (device-to-device)
        std::swap(d_in_x, d_out_x);
        std::swap(d_in_y, d_out_y); 
        std::swap(d_in_val, d_out_val);
    }


	// define block and grid sizes
    /*
	int padded_window_width = roundUp32(window_width);
    int padded_window_height = roundUp32(window_height);
	dim3 blockSize(padded_window_width, padded_window_height);
    dim3 gridSize(numFeatures);
    */

    // V3.3: update launch configuration occupancy (windowsize=blocksize)
	dim3 blockSize(window_width,window_height);
    dim3 gridSize(numFeatures);

	// launch kernel with feature arrays and packed pyramids
	// we use contiguous single buffer for pyramids to ease memory management
	// each block tracks one feature
	trackFeatureKernel<<<gridSize, blockSize>>>(
		d_pyramid1,
		d_pyramid1_gradx,
		d_pyramid1_grady,
		d_pyramid2,
		d_pyramid2_gradx,
		d_pyramid2_grady,
		d_in_x,
		d_in_y,
		d_in_val,
		d_out_x,
		d_out_y,
		d_out_val,
		borderx,
		bordery,
		window_width,
		window_height,
		step_factor,
		max_iterations,
		small,
		th,
		max_residue
	);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
	// synchronize
	cudaDeviceSynchronize();

	// Copy back outputs
	// we need only the output feature list arrays
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
	//cudaFree(d_in_x); cudaFree(d_in_y); cudaFree(d_in_val);
	//cudaFree(d_out_x); cudaFree(d_out_y); cudaFree(d_out_val);

	// free device memory for pyramids (single buffer)
    /*
	#define FREE_DEVICE_PYRAMID(dp) do { if ((dp).d_buffer) cudaFree((dp).d_buffer); } while(0)
	FREE_DEVICE_PYRAMID(d_pyramid1);
	FREE_DEVICE_PYRAMID(d_pyramid1_gradx);
	FREE_DEVICE_PYRAMID(d_pyramid1_grady);
	FREE_DEVICE_PYRAMID(d_pyramid2);
	FREE_DEVICE_PYRAMID(d_pyramid2_gradx);
	FREE_DEVICE_PYRAMID(d_pyramid2_grady);
	#undef FREE_DEVICE_PYRAMID
    */

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


}