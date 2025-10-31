#include "trackFeatures_cuda.h"
#include "convolve_cuda.h"

// for debugging
#include <float.h>

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


////////////// THIS IS JUST FOR DEBUGGING /////////////////////////////
// Add this function to verify pyramid contents
static void verifyPyramidContents(const char* name, float* d_pyramid, int nLevels, int subsampling, 
                                  int base_ncols, int base_nrows) {
    printf("=== Verifying %s Pyramid ===\n", name);
    
    // Calculate level dimensions
    int current_ncols = base_ncols;
    int current_nrows = base_nrows;
    size_t total_offset = 0;
    
    for (int level = 0; level < nLevels; level++) {
        size_t level_size = (size_t)current_ncols * (size_t)current_nrows;
        
        // Allocate host buffer for this level
        float *h_level_data = (float*)malloc(level_size * sizeof(float));
        
        // Copy level data from device
        CUDA_CHECK(cudaMemcpy(h_level_data, d_pyramid + total_offset, 
                             level_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Print level info and sample data
        printf("Level %d (%dx%d):\n", level, current_ncols, current_nrows);
        
        // Print first row
        printf("  First row: ");
        for (int x = 0; x < min(10, current_ncols); x++) {
            printf("%6.1f ", h_level_data[x]);
        }
        printf("\n");
        
        // Print center region (if image is large enough)
        if (current_nrows > 5 && current_ncols > 5) {
            int center_y = current_nrows / 2;
            int center_x = current_ncols / 2;
            printf("  Center [%d,%d]: ", center_y, center_x);
            for (int dx = -2; dx <= 2; dx++) {
                int x = center_x + dx;
                if (x >= 0 && x < current_ncols) {
                    float val = h_level_data[center_y * current_ncols + x];
                    printf("%6.1f ", val);
                }
            }
            printf("\n");
        }
        
        // Check for all zeros (might indicate a problem)
        int zero_count = 0;
        float min_val = FLT_MAX, max_val = -FLT_MAX;
        for (size_t i = 0; i < level_size; i++) {
            if (h_level_data[i] == 0.0f) zero_count++;
            if (h_level_data[i] < min_val) min_val = h_level_data[i];
            if (h_level_data[i] > max_val) max_val = h_level_data[i];
        }
        
        printf("  Stats: zeros=%zu/%zu (%.1f%%), min=%.2f, max=%.2f\n",
               zero_count, level_size, (100.0f * zero_count) / level_size, min_val, max_val);
        
        free(h_level_data);
        
        // Update for next level
        total_offset += level_size;
        if (level < nLevels - 1) {
            current_ncols /= subsampling;
            current_nrows /= subsampling;
        }
    }
    printf("=== End %s Pyramid Verification ===\n\n", name);
}


extern int KLT_verbose;
typedef float *_FloatWindow;

// Global shared metadata (since all pyramids have same structure)
static PyramidMetadata g_pyramid_meta = {0};
static bool g_pyramid_meta_initialized = false;

// Initialize shared pyramid metadata from a reference pyramid
static void initializePyramidMetadata(_KLT_Pyramid pyramid) {
    if (g_pyramid_meta_initialized) return;
    
    int nLevels = pyramid->nLevels;
    g_pyramid_meta.nLevels = nLevels;
    g_pyramid_meta.subsampling = (float)pyramid->subsampling;
    
    // Allocate arrays
    g_pyramid_meta.nrows = (int*)malloc(sizeof(int) * nLevels);
    g_pyramid_meta.ncols = (int*)malloc(sizeof(int) * nLevels);
    g_pyramid_meta.offsets = (size_t*)malloc(sizeof(size_t) * nLevels);
    
    // Calculate offsets and total size
    size_t offset = 0;
    for (int i = 0; i < nLevels; i++) {
        g_pyramid_meta.nrows[i] = pyramid->nrows[i];
        g_pyramid_meta.ncols[i] = pyramid->ncols[i];
        g_pyramid_meta.offsets[i] = offset;
        offset += (size_t)pyramid->ncols[i] * (size_t)pyramid->nrows[i];
    }
    g_pyramid_meta.total_size = offset;
    
    g_pyramid_meta_initialized = true;
    
    if (KLT_verbose) {
        /*printf("Initialized shared pyramid metadata: %d levels, total size: %zu floats\n", 
               nLevels, g_pyramid_meta.total_size);*/
    }
}

// Cleanup shared metadata
static void freePyramidMetadata() {
    if (!g_pyramid_meta_initialized) return;
    
    free(g_pyramid_meta.nrows);
    free(g_pyramid_meta.ncols);
    free(g_pyramid_meta.offsets);
    g_pyramid_meta_initialized = false;
}

// helper function to get device pointer to a specific pyramid level
static float* getDevicePyramidLevel(float *d_pyramid, int level) {
    if (!g_pyramid_meta_initialized) {
        fprintf(stderr, "Error: Pyramid metadata not initialized\n");
        return NULL;
    }
    if (level < 0 || level >= g_pyramid_meta.nLevels) {
        fprintf(stderr, "Error: Invalid pyramid level %d\n", level);
        return NULL;
    }
    return d_pyramid + g_pyramid_meta.offsets[level];
}



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
static float *d_pyramid3 = NULL;
static float *d_pyramid3_gradx = NULL;
static float *d_pyramid3_grady = NULL;

// Device metadata in constant memory (accessible from all kernels)
__constant__ int c_nLevels;
__constant__ float c_subsampling;
__constant__ int c_nrows[32];      // Max 32 pyramid levels should be enough
__constant__ int c_ncols[32];
__constant__ size_t c_offsets[32];

// V3.5 trying something w image so convolve doesnt need to copy and copy back
static float *d_img1 = NULL,*d_img2 =NULL;
static float *d_smooth_img1 = NULL, *d_smooth_img2 = NULL;
static float *d_img3 = NULL;
static float *d_smooth_img3 = NULL;
int frame_count = 0;

// v3.6 trying pinned memory 
// Add these global pinned buffers
static float *h_in_x_pinned = NULL, *h_in_y_pinned = NULL;
static int *h_in_val_pinned = NULL;
static float *h_out_x_pinned = NULL, *h_out_y_pinned = NULL;  
static int *h_out_val_pinned = NULL;
static size_t pinned_pool_size = 0;

// v3.7 streams
static cudaStream_t stream1, stream2;
static cudaEvent_t pyramid3_ready;
static cudaEvent_t tracking_done;

static void allocatePinnedBuffers(int numFeatures) {
    if (pinned_pool_size < numFeatures) {
        //printf("Allocating pinned memory for %d features...\n", numFeatures);
        
        // Free existing if too small
        if (h_in_x_pinned) cudaFreeHost(h_in_x_pinned);
        if (h_in_y_pinned) cudaFreeHost(h_in_y_pinned);
        if (h_in_val_pinned) cudaFreeHost(h_in_val_pinned);
        if (h_out_x_pinned) cudaFreeHost(h_out_x_pinned);
        if (h_out_y_pinned) cudaFreeHost(h_out_y_pinned);
        if (h_out_val_pinned) cudaFreeHost(h_out_val_pinned);
        
        // Allocate new pinned memory
        CUDA_CHECK(cudaHostAlloc((void**)&h_in_x_pinned, sizeof(float) * numFeatures, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc((void**)&h_in_y_pinned, sizeof(float) * numFeatures, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc((void**)&h_in_val_pinned, sizeof(int) * numFeatures, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc((void**)&h_out_x_pinned, sizeof(float) * numFeatures, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc((void**)&h_out_y_pinned, sizeof(float) * numFeatures, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc((void**)&h_out_val_pinned, sizeof(int) * numFeatures, cudaHostAllocDefault));
        
        pinned_pool_size = numFeatures;
    }
}

static void allocateFeatureList(int numFeatures) {
    if (feature_pool_size < numFeatures) {
        //printf("Allocating feature list...\n");
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
            //printf("Allocated feature pool for %d features\n", numFeatures);
        }
    }
}

static size_t estimatePyramidSize(_KLT_Pyramid pyramid) {
    if (!pyramid) return 0;
    
    size_t total = 0; // only image data, no header
    
    // Just the image pyramids 
    for (int i = 0; i < pyramid->nLevels; i++) {
        total += (size_t)pyramid->ncols[i] * (size_t)pyramid->nrows[i];
    }
    
    return total;
}

static void allocatePyramidBuffers(_KLT_Pyramid pyramid) {

    if (!d_pyramid1)
        //printf("allocating pyramid buffers\n");
    
    // Initialize shared metadata first
    initializePyramidMetadata(pyramid);
    
    // Calculate size for pure image data (no header)
    size_t bytes_needed = g_pyramid_meta.total_size * sizeof(float);
    
    // Allocate if null
    if (!d_pyramid1) CUDA_CHECK(cudaMalloc(&d_pyramid1, bytes_needed));
    if (!d_pyramid1_gradx) CUDA_CHECK(cudaMalloc(&d_pyramid1_gradx, bytes_needed));
    if (!d_pyramid1_grady) CUDA_CHECK(cudaMalloc(&d_pyramid1_grady, bytes_needed));
    if (!d_pyramid2) CUDA_CHECK(cudaMalloc(&d_pyramid2, bytes_needed));
    if (!d_pyramid2_gradx) CUDA_CHECK(cudaMalloc(&d_pyramid2_gradx, bytes_needed));
    if (!d_pyramid2_grady) CUDA_CHECK(cudaMalloc(&d_pyramid2_grady, bytes_needed));
    if (!d_pyramid3) CUDA_CHECK(cudaMalloc(&d_pyramid3, bytes_needed));
    if (!d_pyramid3_gradx) CUDA_CHECK(cudaMalloc(&d_pyramid3_gradx, bytes_needed));
    if (!d_pyramid3_grady) CUDA_CHECK(cudaMalloc(&d_pyramid3_grady, bytes_needed));
    
    // Copy metadata to constant memory (one-time setup)
    int nLevels = g_pyramid_meta.nLevels;
    static bool constant_memory_initialized = false;
    if (!constant_memory_initialized) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_nLevels, &g_pyramid_meta.nLevels, sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_subsampling, &g_pyramid_meta.subsampling, sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_nrows, g_pyramid_meta.nrows, nLevels * sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_ncols, g_pyramid_meta.ncols, nLevels * sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_offsets, g_pyramid_meta.offsets, nLevels * sizeof(size_t)));
        constant_memory_initialized = true;
        
        if (KLT_verbose) {
            //printf("Copied pyramid metadata to constant memory\n");
        }
    }
}

__host__ void allocateGPUResources(int numFeatures, KLT_TrackingContext h_tc, int ncols, int nrows) {

    // if first frame allocate image buffers
    //printf("allocate image buffers");
    CUDA_CHECK(cudaMalloc((void**)&d_img1, ncols * nrows * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_img2, ncols * nrows * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_img3, ncols * nrows * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_smooth_img1, ncols * nrows * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_smooth_img2, ncols * nrows * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_smooth_img3, ncols * nrows * sizeof(float)));

    // V3.1: allocate feature list once
    allocateFeatureList(numFeatures);

    // V3.1: allocate pyramids once
    _KLT_Pyramid temp_pyramid_for_size = NULL;
    if (h_tc->sequentialMode && h_tc->pyramid_last != NULL) {
        temp_pyramid_for_size = (_KLT_Pyramid) h_tc->pyramid_last;
    } else {
        // Create a temporary pyramid just for size estimation
        int subsampling = (int)h_tc->subsampling;
        temp_pyramid_for_size = _KLTCreatePyramid(ncols, nrows, subsampling, h_tc->nPyramidLevels);
    }
    allocatePyramidBuffers(temp_pyramid_for_size);
    // Free temporary pyramid if we created it
    if (temp_pyramid_for_size != h_tc->pyramid_last) {
        _KLTFreePyramid(temp_pyramid_for_size);
    }

    // allocate host pinned memory feature list buffers
    allocatePinnedBuffers(numFeatures);

    // create streams
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaEventCreate(&pyramid3_ready);
    cudaEventCreate(&tracking_done);

    //computeKernelsConstant(_KLTComputeSmoothSigma(h_tc));
    initializePrecomputedKernels();
}

__host__ void freeGPUResources(){
    // free feature list
    if (d_in_x) cudaFree(d_in_x);
    if (d_in_y) cudaFree(d_in_y);
    if (d_in_val) cudaFree(d_in_val);
    if (d_out_x) cudaFree(d_out_x);
    if (d_out_y) cudaFree(d_out_y);
    if (d_out_val) cudaFree(d_out_val);
    d_in_x = d_in_y = NULL;
    d_in_val = NULL;
    d_out_x = d_out_y = NULL;
    d_out_val = NULL;

    //free pyramids
    if (d_pyramid1) cudaFree(d_pyramid1);
    if (d_pyramid1_gradx) cudaFree(d_pyramid1_gradx);
    if (d_pyramid1_grady) cudaFree(d_pyramid1_grady);
    if (d_pyramid2) cudaFree(d_pyramid2);
    if (d_pyramid2_gradx) cudaFree(d_pyramid2_gradx);
    if (d_pyramid2_grady) cudaFree(d_pyramid2_grady);
    d_pyramid1 = d_pyramid1_gradx = d_pyramid1_grady = NULL;
    d_pyramid2 = d_pyramid2_gradx = d_pyramid2_grady = NULL;
    freePyramidMetadata();

    //free host pinned memory
    if (h_in_x_pinned) cudaFreeHost(h_in_x_pinned);
    if (h_in_y_pinned) cudaFreeHost(h_in_y_pinned);
    if (h_in_val_pinned) cudaFreeHost(h_in_val_pinned);
    if (h_out_x_pinned) cudaFreeHost(h_out_x_pinned);
    if (h_out_y_pinned) cudaFreeHost(h_out_y_pinned);
    if (h_out_val_pinned) cudaFreeHost(h_out_val_pinned);
    
    h_in_x_pinned = h_in_y_pinned = NULL;
    h_in_val_pinned = NULL;
    h_out_x_pinned = h_out_y_pinned = NULL;
    h_out_val_pinned = NULL;
    pinned_pool_size = 0;

    // destroy streams
    if (stream1) cudaStreamDestroy(stream1);
    if (stream2) cudaStreamDestroy(stream2);
    stream1 = stream2 = NULL;
}

/*
static void copyPyramidToDevice(_KLT_Pyramid src, float* d_dest) {
    if (!src || !d_dest) return;
    
    if (!g_pyramid_meta_initialized) {
        fprintf(stderr, "Error: Pyramid metadata not initialized\n");
        return;
    }
    
    int nLevels = src->nLevels;
    
    // Copy image data for each level directly to device buffer
    for (int i = 0; i < nLevels; i++) {
        _KLT_FloatImage h_img = src->img[i];
        int w = h_img->ncols;
        int h = h_img->nrows;
        size_t n_pix = (size_t)w * (size_t)h;
        size_t bytes = n_pix * sizeof(float);
        
        float *d_level_start = d_dest + g_pyramid_meta.offsets[i];
        CUDA_CHECK(cudaMemcpy(d_level_start, h_img->data, bytes, cudaMemcpyHostToDevice));
    }
}
*/

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

// Device accessor for interpolation (uses constant memory)
__device__ float interpolateCUDA(float x, float y, const float *img_data, int level) 
{
    int nc = c_ncols[level];
    int nr = c_nrows[level];
    size_t offset = c_offsets[level];
    int xt = (int)x;
    int yt = (int)y;
    float ax = x - xt;
    float ay = y - yt;
    const float *ptr = img_data + offset + yt * nc + xt;
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
    const float one_plus_eps = 1.001f;

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

    int nPyramidLevels = c_nLevels;
    
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
            xloc /= c_subsampling; yloc /= c_subsampling;
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
            s_x1 *= c_subsampling; s_y1 *= c_subsampling;
            s_x2 *= c_subsampling; s_y2 *= c_subsampling;
        }
        __syncthreads();

        // Iterative refinement for this pyramid level
        s_iteration = 0;
        s_continue = 1;
        
        do {

            // Add boundary check at start of each iteration (like CPU)
            if (tid == 0) {
                int nc = c_ncols[r];
                int nr = c_nrows[r];
                
                // Exact CPU boundary check logic
                if (s_x1 - hw < 0.0f || nc - (s_x1 + hw) < one_plus_eps ||
                    s_x2 - hw < 0.0f || nc - (s_x2 + hw) < one_plus_eps ||
                    s_y1 - hh < 0.0f || nr - (s_y1 + hh) < one_plus_eps ||
                    s_y2 - hh < 0.0f || nr - (s_y2 + hh) < one_plus_eps) {
                    s_status = KLT_OOB;
                    s_continue = 0;
                }
            }
            __syncthreads();
            
            if (s_status != KLT_TRACKED) break;


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

                /*
                // DEBUG: Always print residue for features that get lost
                if (residue > max_residue) {
                    printf("Feature %d LARGE_RESIDUE: pos=(%.1f,%.1f), residue=%.6f > threshold=%.6f\n", 
                           featureIdx, s_x2, s_y2, residue, max_residue);
                } else if (featureIdx < 5) { // Print a few successful ones for comparison
                    printf("Feature %d TRACKED: pos=(%.1f,%.1f), residue=%.6f\n", 
                           featureIdx, s_x2, s_y2, residue);
                }
                */
                                                                           
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
        int nc = c_ncols[0];
        int nr = c_nrows[0];

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

      // In trackFeatureKernel, before the final write:
        /*
        if (tid == 0) {
            if (final_val != KLT_TRACKED) {
                printf("Feature %d LOST: status=%d, pos=(%.1f,%.1f)->(%.1f,%.1f)\n", 
                       featureIdx, final_val, d_in_x[featureIdx], d_in_y[featureIdx], final_x, final_y);
            } else if (featureIdx < 5) {  // Print first few successful ones
                printf("Feature %d TRACKED: (%.1f,%.1f)->(%.1f,%.1f)\n",
                       featureIdx, d_in_x[featureIdx], d_in_y[featureIdx], final_x, final_y);
            }
        }
        */

    }
}


__global__ void trackFeatureKernelTest(
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
    const float one_plus_eps = 1.001f;

	int featureIdx = blockIdx.x;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int window_elems = window_width * window_height;
	
	// Only track features that are not lost
	if (d_in_val[featureIdx] < 0) {
		if (tid == 0) {
			d_out_x[featureIdx] = -1.0f;
			d_out_y[featureIdx] = -1.0f;
			d_out_val[featureIdx] = d_in_val[featureIdx];
		}
		return;
	}

    // Bounds check: skip threads outside the window
    if (tid >= window_elems) return;

	int hw = window_width / 2;
	int hh = window_height / 2;
	int nPyramidLevels = c_nLevels;

	// shared memory allocation
	extern __shared__ float shared[];
	float* imgdiff = shared; //imgdiff: first part of shared memory
	float* gradx = imgdiff + window_width * window_height; //gradx: second part of shared memory
	float* grady = gradx + window_width * window_height; //grady: third part of shared memory

	// Shared variables for feature state 
	__shared__ float s_x1, s_y1, s_x2, s_y2;
	__shared__ float s_dx, s_dy;
	__shared__ int s_status, s_continue;
	__shared__ int s_iteration;

	if (tid == 0) {
		// Initialize coordinates
		float xloc = d_in_x[featureIdx];
		float yloc = d_in_y[featureIdx];
		
		// Transform to coarsest resolution
		for (int r = nPyramidLevels - 1; r >= 0; r--) {
			xloc /= c_subsampling; yloc /= c_subsampling;
		}
		s_x1 = xloc; s_y1 = yloc;
		s_x2 = xloc; s_y2 = yloc;
		s_status = KLT_TRACKED;
	}
	__syncthreads();

	// Main pyramid levels loop
	for (int r = nPyramidLevels - 1; r >= 0; r--) {
		// If already lost, skip levels
		if (s_status != KLT_TRACKED) break;

		// Scale coordinates 
		if (tid == 0) {
			s_x1 *= c_subsampling; s_y1 *= c_subsampling;
			s_x2 *= c_subsampling; s_y2 *= c_subsampling;
		}
		__syncthreads();

		// Iterative refinement for this pyramid level
		s_iteration = 0;
		s_continue = 1;
		
		do {
			// Add boundary check at start of each iteration (like CPU)
			if (tid == 0) {
				int nc = c_ncols[r];
				int nr = c_nrows[r];
				
				// Exact CPU boundary check logic
				if (s_x1 - hw < 0.0f || nc - (s_x1 + hw) < one_plus_eps ||
					s_x2 - hw < 0.0f || nc - (s_x2 + hw) < one_plus_eps ||
					s_y1 - hh < 0.0f || nr - (s_y1 + hh) < one_plus_eps ||
					s_y2 - hh < 0.0f || nr - (s_y2 + hh) < one_plus_eps) {
					s_status = KLT_OOB;
					s_continue = 0;
				}
			}
			__syncthreads();
			
			if (s_status != KLT_TRACKED) break;

			// Each thread computes its window pixel
			int i = threadIdx.x - hw;
			int j = threadIdx.y - hh;
			int idx = threadIdx.y * window_width + threadIdx.x;

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
				s_iteration++;
			}
			__syncthreads();

		} while (s_continue && s_iteration < max_iterations && s_status == KLT_TRACKED);

		// check residue
		if (s_status == KLT_TRACKED) {
			// Each thread computes its window pixel for residue
			int i = threadIdx.x - hw;
			int j = threadIdx.y - hh;
			int idx = threadIdx.y * window_width + threadIdx.x;
			
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
		}
	}

	// record final value and status of the feature
	if (tid == 0) {
		float final_x = s_x2;
		float final_y = s_y2;
		int final_val = s_status;

		// bounds check
		int nc = c_ncols[0];
		int nr = c_nrows[0];

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
	
};


__host__ void kltTrackFeaturesCUDA(
  KLT_TrackingContext h_tc,
  KLT_PixelType *h_img1,
  KLT_PixelType *h_img2,
  KLT_PixelType *h_img3,
  int ncols,
  int nrows,
  KLT_FeatureList h_fl
)
{
    
    _KLT_FloatImage tmpimg, floatimg1, floatimg2;
	_KLT_Pyramid pyramid1, pyramid1_gradx, pyramid1_grady,
		pyramid2, pyramid2_gradx, pyramid2_grady;
	float subsampling = (float) h_tc->subsampling;
	KLT_BOOL floatimg1_created = FALSE;
	int i;
    
	int numFeatures = KLTCountRemainingFeatures(h_fl);

    
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


    if (frame_count==0){

        //process img1
        _KLT_FloatImage tmpimg1 = _KLTCreateFloatImage(ncols, nrows);
        _KLTToFloatImage(h_img1, ncols, nrows, tmpimg1);

        CUDA_CHECK(cudaMemcpyAsync(d_img1, tmpimg1->data, ncols * nrows * sizeof(float), cudaMemcpyHostToDevice,stream1));
        computeSmoothedImageCUDA(d_img1, _KLTComputeSmoothSigma(h_tc), d_smooth_img1, ncols, nrows, stream1);
        computePyramidCUDA(d_smooth_img1, d_pyramid1, h_tc->pyramid_sigma_fact, ncols, nrows, subsampling, h_tc->nPyramidLevels,stream1);

        for (int i = 0; i < h_tc->nPyramidLevels; i++) {
            float *d_img = getDevicePyramidLevel(d_pyramid1, i);
            float *d_gradx = getDevicePyramidLevel(d_pyramid1_gradx, i);
            float *d_grady = getDevicePyramidLevel(d_pyramid1_grady, i);
            int level_ncols = g_pyramid_meta.ncols[i];
            int level_nrows = g_pyramid_meta.nrows[i];
            computeGradientsCUDA(d_img, h_tc->grad_sigma, d_gradx, d_grady, level_ncols, level_nrows,stream1);
        }
        _KLTFreeFloatImage(tmpimg1);

        // In kltTrackFeaturesCUDA, after processing img1 in frame_count==0 block:
        //printf("=== VERIFYING FIRST FRAME PYRAMIDS ===\n");
        //verifyPyramidContents("PYRAMID1_FIRST_FRAME", d_pyramid1, h_tc->nPyramidLevels, (int)subsampling, ncols, nrows);
        

        //process img2
        _KLT_FloatImage tmpimg2 = _KLTCreateFloatImage(ncols, nrows);
        _KLTToFloatImage(h_img2, ncols, nrows, tmpimg2);

        CUDA_CHECK(cudaMemcpyAsync(d_img2, tmpimg2->data, ncols * nrows * sizeof(float), cudaMemcpyHostToDevice,stream1));
        computeSmoothedImageCUDA(d_img2, _KLTComputeSmoothSigma(h_tc), d_smooth_img2, ncols, nrows,stream1);
        computePyramidCUDA(d_smooth_img2, d_pyramid2, h_tc->pyramid_sigma_fact, ncols, nrows, subsampling, h_tc->nPyramidLevels,stream1);

        // Compute gradients for pyramid2
        for (int i = 0; i < h_tc->nPyramidLevels; i++) {
            float *d_img = getDevicePyramidLevel(d_pyramid2, i);
            float *d_gradx = getDevicePyramidLevel(d_pyramid2_gradx, i);
            float *d_grady = getDevicePyramidLevel(d_pyramid2_grady, i);
            int level_ncols = g_pyramid_meta.ncols[i];
            int level_nrows = g_pyramid_meta.nrows[i];
            computeGradientsCUDA(d_img, h_tc->grad_sigma, d_gradx, d_grady, level_ncols, level_nrows,stream1);
        }
        _KLTFreeFloatImage(tmpimg2);

        CUDA_CHECK(cudaStreamSynchronize(stream1));

        // After processing img2 in frame_count==0 block:  
        //verifyPyramidContents("PYRAMID2_FIRST_FRAME", d_pyramid2, h_tc->nPyramidLevels, (int)subsampling, ncols, nrows);
        
        // Also verify the gradients:
        //verifyPyramidContents("GRADX1_FIRST_FRAME", d_pyramid1_gradx, h_tc->nPyramidLevels, (int)subsampling, ncols, nrows);
        //verifyPyramidContents("GRADY1_FIRST_FRAME", d_pyramid1_grady, h_tc->nPyramidLevels, (int)subsampling, ncols, nrows);
    
    }
    else{
        CUDA_CHECK(cudaStreamSynchronize(stream1)); 
        
        // rotate buffers
        std::swap(d_pyramid1, d_pyramid2);
        std::swap(d_pyramid1_gradx, d_pyramid2_gradx);
        std::swap(d_pyramid1_grady, d_pyramid2_grady);

        // stream synchronise here
        CUDA_CHECK(cudaEventSynchronize(pyramid3_ready));
        
        std::swap(d_pyramid2, d_pyramid3);
        std::swap(d_pyramid2_gradx, d_pyramid3_gradx);
        std::swap(d_pyramid2_grady, d_pyramid3_grady);
        
        // Rotate image buffers
        std::swap(d_img1, d_img2);
        std::swap(d_img2, d_img3);
        std::swap(d_smooth_img1, d_smooth_img2);
        std::swap(d_smooth_img2, d_smooth_img3);

        //cuda memset new pyramid3 buffers to zero
        CUDA_CHECK(cudaMemsetAsync(d_pyramid3, 0, g_pyramid_meta.total_size * sizeof(float), stream2));
        CUDA_CHECK(cudaMemsetAsync(d_pyramid3_gradx, 0, g_pyramid_meta.total_size * sizeof(float), stream2));
        CUDA_CHECK(cudaMemsetAsync(d_pyramid3_grady, 0, g_pyramid_meta.total_size * sizeof(float), stream2));
    }

    // ------------ launch stream to start copying next img -----------
    _KLT_FloatImage tmpimg3 = _KLTCreateFloatImage(ncols, nrows);
    _KLTToFloatImage(h_img3, ncols, nrows, tmpimg3);

    //printf("Host image center: %.1f\n", tmpimg3->data[120 * ncols + 160]);

    // Start async copy and pyramid computation in stream2
    CUDA_CHECK(cudaMemcpyAsync(d_img3, tmpimg3->data, ncols * nrows * sizeof(float), cudaMemcpyHostToDevice, stream2));
    
    float device_sample;
    CUDA_CHECK(cudaMemcpyAsync(&device_sample, &d_img3[120 * ncols + 160], sizeof(float), cudaMemcpyDeviceToHost, stream2));
    CUDA_CHECK(cudaStreamSynchronize(stream2));  // Wait for copy to complete
    //printf("Device image center after copy: %.1f\n", device_sample);

    computeSmoothedImageCUDA(d_img3, _KLTComputeSmoothSigma(h_tc), d_smooth_img3, ncols, nrows, stream2);

    float smooth_sample;
    CUDA_CHECK(cudaMemcpyAsync(&smooth_sample, &d_smooth_img3[120 * ncols + 160], sizeof(float), cudaMemcpyDeviceToHost, stream2));
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    //printf("Smoothed image center: %.1f\n", smooth_sample);

    computePyramidCUDA(d_smooth_img3, d_pyramid3, h_tc->pyramid_sigma_fact, ncols, nrows, subsampling, h_tc->nPyramidLevels, stream2);
    
    // VERIFY PYRAMID3 IMMEDIATELY
    //printf("=== VERIFYING PYRAMID3 AFTER COMPUTATION ===\n");
    //verifyPyramidContents("PYRAMID3_RAW", d_pyramid3, h_tc->nPyramidLevels, (int)subsampling, ncols, nrows);

    // Compute gradients for pyramid3 in stream2
    for (int i = 0; i < h_tc->nPyramidLevels; i++) {
        float *d_img = getDevicePyramidLevel(d_pyramid3, i);
        float *d_gradx = getDevicePyramidLevel(d_pyramid3_gradx, i);
        float *d_grady = getDevicePyramidLevel(d_pyramid3_grady, i);
        int level_ncols = g_pyramid_meta.ncols[i];
        int level_nrows = g_pyramid_meta.nrows[i];
        computeGradientsCUDA(d_img, h_tc->grad_sigma, d_gradx, d_grady, level_ncols, level_nrows, stream2);
    }
    //verifyPyramidContents("GRADX3_RAW", d_pyramid3_gradx, h_tc->nPyramidLevels, (int)subsampling, ncols, nrows);

    _KLTFreeFloatImage(tmpimg3);
    CUDA_CHECK(cudaEventRecord(pyramid3_ready, stream2));


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

    // Copy data to pinned memory (fast CPU copy)
    for (i = 0; i < numFeatures; ++i) {
        h_in_x_pinned[i] = h_fl->feature[i]->x;
        h_in_y_pinned[i] = h_fl->feature[i]->y; 
        h_in_val_pinned[i] = h_fl->feature[i]->val;
    }

        
    // V3.2: double buffering of feature list
    if (first_frame) {
        first_frame = false;
        //printf("First frame - initializing device feature buffers\n");
         
        // Copy to device (fast pinned→device)
        CUDA_CHECK(cudaMemcpyAsync(d_in_x, h_in_x_pinned, sizeof(float) * numFeatures, cudaMemcpyHostToDevice,stream1));
        CUDA_CHECK(cudaMemcpyAsync(d_in_y, h_in_y_pinned, sizeof(float) * numFeatures, cudaMemcpyHostToDevice,stream1));
        CUDA_CHECK(cudaMemcpyAsync(d_in_val, h_in_val_pinned, sizeof(int) * numFeatures, cudaMemcpyHostToDevice,stream1));
        
    } else {
        //printf("Subsequent frame - copying outputs to inputs\n");

        CUDA_CHECK(cudaEventSynchronize(tracking_done));
        /*
        std::swap(d_in_x, d_out_x);
        std::swap(d_in_y, d_out_y);
        std::swap(d_in_val, d_out_val);
        */
        CUDA_CHECK(cudaMemcpyAsync(d_in_x, d_out_x, sizeof(float) * numFeatures, cudaMemcpyDeviceToDevice,stream1));
        CUDA_CHECK(cudaMemcpyAsync(d_in_y, d_out_y, sizeof(float) * numFeatures, cudaMemcpyDeviceToDevice,stream1));
        CUDA_CHECK(cudaMemcpyAsync(d_in_val, d_out_val, sizeof(int) * numFeatures, cudaMemcpyDeviceToDevice,stream1)); 
    }

   
    // V3.3: update launch configuration occupancy (windowsize=blocksize)
	dim3 blockSize(window_width,window_height);
    dim3 gridSize(numFeatures);
    size_t sharedMemSize = window_width * window_height * sizeof(float) * 3;

    /*
    printf("=== BEFORE KERNEL LAUNCH (Frame 2→3) ===\n");
    verifyPyramidContents("PYRAMID1_FRAME2", d_pyramid1, h_tc->nPyramidLevels, (int)subsampling, ncols, nrows);
    verifyPyramidContents("PYRAMID2_FRAME3", d_pyramid2, h_tc->nPyramidLevels, (int)subsampling, ncols, nrows);
    verifyPyramidContents("GRADX1_FRAME2", d_pyramid1_gradx, h_tc->nPyramidLevels, (int)subsampling, ncols, nrows);
    verifyPyramidContents("GRADX2_FRAME3", d_pyramid2_gradx, h_tc->nPyramidLevels, (int)subsampling, ncols, nrows);
    printf("======================================\n");
    */

	trackFeatureKernelTest<<<gridSize, blockSize,sharedMemSize,stream1>>>(
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
	//cudaDeviceSynchronize();
    CUDA_CHECK(cudaEventRecord(tracking_done, stream1));
    //CUDA_CHECK(cudaStreamSynchronize(stream1));

    CUDA_CHECK(cudaMemcpyAsync(h_out_x_pinned, d_out_x, sizeof(float) * numFeatures, cudaMemcpyDeviceToHost,stream1));
    CUDA_CHECK(cudaMemcpyAsync(h_out_y_pinned, d_out_y, sizeof(float) * numFeatures, cudaMemcpyDeviceToHost,stream1));
    CUDA_CHECK(cudaMemcpyAsync(h_out_val_pinned, d_out_val, sizeof(int) * numFeatures, cudaMemcpyDeviceToHost,stream1));

    // Copy to feature list (fast CPU copy)
    for (i = 0; i < numFeatures; ++i) {
        h_fl->feature[i]->x = h_out_x_pinned[i];
        h_fl->feature[i]->y = h_out_y_pinned[i];
        h_fl->feature[i]->val = h_out_val_pinned[i];
    }

    frame_count++;

    /*
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

	_KLTFreeFloatImage(tmpimg);
	if (floatimg1_created)  _KLTFreeFloatImage(floatimg1);
	_KLTFreeFloatImage(floatimg2);
	_KLTFreePyramid(pyramid1);
	_KLTFreePyramid(pyramid1_gradx);
	_KLTFreePyramid(pyramid1_grady);
    */


}
