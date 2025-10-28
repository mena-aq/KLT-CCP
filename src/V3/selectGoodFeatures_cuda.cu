#include "selectGoodFeatures_cuda.h"
#include "convolve_cuda.h"
#include "base.h"
#include "error.h"
#include "klt_util.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>

// External variables
extern int KLT_verbose;

// CUDA error checking macro
#define CUDA_CHECK(call)                                                    \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

// Device function to compute minimum eigenvalue
__device__ float minEigenvalueCUDA(float gxx, float gxy, float gyy) {
    float trace = gxx + gyy;
    float det = gxx * gyy - gxy * gxy;
    float discriminant = trace * trace - 4.0f * det;
    if (discriminant < 0.0f) return 0.0f;
    return 0.5f * (trace - sqrtf(discriminant));
}


// CUDA kernel to compute eigenvalues for feature selection NO SHARED MEMORY
__global__ void computeEigenvaluesKernel(
    const float *d_gradx,
    const float *d_grady, 
    int *d_pointlist,
    int ncols,
    int nrows,
    int window_hw,
    int window_hh,
    int borderx,
    int bordery,
    int nSkippedPixels,
    int *d_npoints
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Map to actual pixel coordinates with skipping
    int x = borderx + idx * (nSkippedPixels + 1);
    int y = bordery + idy * (nSkippedPixels + 1);
    
    if (x >= ncols - borderx || y >= nrows - bordery) return;
    
    // Compute gradients in window
    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
    for (int yy = y - window_hh; yy <= y + window_hh; yy++) {
        for (int xx = x - window_hw; xx <= x + window_hw; xx++) {
            float gx = d_gradx[yy * ncols + xx];
            float gy = d_grady[yy * ncols + xx];
            gxx += gx * gx;
            gxy += gx * gy;
            gyy += gy * gy;
        }
    }
    
    // Compute minimum eigenvalue
    float val = minEigenvalueCUDA(gxx, gxy, gyy);
    
    // Clamp to int limit
    const float int_limit = 2147483647.0f; // INT_MAX
    if (val > int_limit) val = int_limit;
    
    // Store result atomically
    int point_idx = atomicAdd(d_npoints, 1);
    if (point_idx < ncols * nrows) { // Safety check
        int base_idx = point_idx * 3;
        d_pointlist[base_idx] = x;
        d_pointlist[base_idx + 1] = y;
        d_pointlist[base_idx + 2] = (int)val;
    }
}

/*
__global__ void computeEigenvaluesKernel(
    const float *d_gradx,
    const float *d_grady, 
    int *d_pointlist,
    int ncols,
    int nrows,
    int window_hw,
    int window_hh,
    int borderx,
    int bordery,
    int nSkippedPixels,
    int *d_npoints
) 
{
    // Shared memory layout optimized for coalesced access
    extern __shared__ float2 shared_grads[];  // Using float2 for better coalescing
    
    const int tile_width = blockDim.x + 2 * window_hw;
    const int tile_height = blockDim.y + 2 * window_hh;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    
    // Calculate the actual output coordinates this block will process
    int block_output_start_x = borderx + blockIdx.x * (blockDim.x * (nSkippedPixels + 1));
    int block_output_start_y = bordery + blockIdx.y * (blockDim.y * (nSkippedPixels + 1));
    
    // Cooperative loading of the tile - we need to load a larger area to cover all output pixels
    for (int pos = tid; pos < tile_width * tile_height; pos += total_threads) {
        int load_x = pos % tile_width;
        int load_y = pos / tile_width;
        
        // Calculate image coordinates for loading (include halo for window operations)
        int img_x = block_output_start_x + load_x - window_hw;
        int img_y = block_output_start_y + load_y - window_hh;
        
        float2 grad_val;
        if (img_x >= 0 && img_x < ncols && img_y >= 0 && img_y < nrows) {
            grad_val.x = d_gradx[img_y * ncols + img_x];
            grad_val.y = d_grady[img_y * ncols + img_x];
        } else {
            grad_val.x = 0.0f;
            grad_val.y = 0.0f;
        }
        shared_grads[load_y * tile_width + load_x] = grad_val;
    }
    
    __syncthreads();
    
    // Each thread processes one output pixel with skipping pattern
    int output_x = block_output_start_x + threadIdx.x * (nSkippedPixels + 1);
    int output_y = block_output_start_y + threadIdx.y * (nSkippedPixels + 1);
    
    // Check bounds
    if (output_x >= ncols - borderx || output_y >= nrows - bordery) {
        return;
    }
    
    // Now compute the eigenvalue for this output pixel
    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
    
    for (int wy = -window_hh; wy <= window_hh; wy++) {
        for (int wx = -window_hw; wx <= window_hw; wx++) {
            // Calculate position in shared memory
            int shared_x = threadIdx.x * (nSkippedPixels + 1) + wx + window_hw;
            int shared_y = threadIdx.y * (nSkippedPixels + 1) + wy + window_hh;
            
            // Ensure we're within the loaded shared memory bounds
            if (shared_x >= 0 && shared_x < tile_width && shared_y >= 0 && shared_y < tile_height) {
                float2 grad_val = shared_grads[shared_y * tile_width + shared_x];
                gxx += grad_val.x * grad_val.x;
                gxy += grad_val.x * grad_val.y;
                gyy += grad_val.y * grad_val.y;
            }
        }
    }
    
    float val = minEigenvalueCUDA(gxx, gxy, gyy);
    const float int_limit = 2147483647.0f;
    if (val > int_limit) val = int_limit;
    
    if (val > 0.0f) {
        int point_idx = atomicAdd(d_npoints, 1);
        if (point_idx < ncols * nrows) {
            int base_idx = point_idx * 3;
            d_pointlist[base_idx] = output_x;
            d_pointlist[base_idx + 1] = output_y;
            d_pointlist[base_idx + 2] = (int)val;
        }
    }
}
*/

// Helper functions (implemented locally to avoid external dependencies)
static void _fillFeaturemap(int x, int y, unsigned char *featuremap, int mindist, int ncols, int nrows) {
    int ix, iy;
    for (iy = y - mindist; iy <= y + mindist; iy++)
        for (ix = x - mindist; ix <= x + mindist; ix++)
            if (ix >= 0 && ix < ncols && iy >= 0 && iy < nrows)
                featuremap[iy * ncols + ix] = 1;
}

static int _outOfBounds(float x, float y, int ncols, int nrows, int borderx, int bordery) {
    return (x < borderx || x > ncols - 1 - borderx ||
            y < bordery || y > nrows - 1 - bordery);
}

static int _comparePoints(const void *a, const void *b) {
    int *x = (int *) a;
    int *y = (int *) b;
    if (x[2] > y[2]) return(-1);
    else if (x[2] < y[2]) return(1);
    else return(0);
}

static void _sortPointList(int *pointlist, int npoints) {
    qsort(pointlist, npoints, 3 * sizeof(int), _comparePoints);
}



// Host function for GPU-accelerated feature selection
void _KLTSelectGoodFeaturesCUDA(
    KLT_TrackingContext tc,
    KLT_PixelType *img, 
    int ncols, 
    int nrows,
    KLT_FeatureList featurelist,
    selectionMode mode)
{
    _KLT_FloatImage floatimg, gradx, grady;
    int window_hw, window_hh;
    int *pointlist;
    int npoints = 0;
    KLT_BOOL overwriteAllFeatures = (mode == SELECTING_ALL) ? TRUE : FALSE;
    KLT_BOOL floatimages_created = FALSE;
    KLT_BOOL use_gpu_computation = TRUE;
    int nSkippedPixels = tc->nSkippedPixels;


    float *d_img, *d_smoothed;
    float *d_gradx, *d_grady;

    /* Check window size (and correct if necessary) */
    if (tc->window_width % 2 != 1) {
        tc->window_width = tc->window_width+1;
        KLTWarning("Tracking context's window width must be odd.  "
                   "Changing to %d.\n", tc->window_width);
    }
    if (tc->window_height % 2 != 1) {
        tc->window_height = tc->window_height+1;
        KLTWarning("Tracking context's window height must be odd.  "
                   "Changing to %d.\n", tc->window_height);
    }
    if (tc->window_width < 3) {
        tc->window_width = 3;
        KLTWarning("Tracking context's window width must be at least three.  \n"
                   "Changing to %d.\n", tc->window_width);
    }
    if (tc->window_height < 3) {
        tc->window_height = 3;
        KLTWarning("Tracking context's window height must be at least three.  \n"
                   "Changing to %d.\n", tc->window_height);
    }
    window_hw = tc->window_width/2; 
    window_hh = tc->window_height/2;

    /* Create pointlist for results */
    pointlist = (int *) malloc(ncols * nrows * 3 * sizeof(int));

    /* Create temporary images, etc. */
    if (mode == REPLACING_SOME && 
        tc->sequentialMode && tc->pyramid_last != NULL)  
    {
        floatimg = ((_KLT_Pyramid) tc->pyramid_last)->img[0];
        gradx = ((_KLT_Pyramid) tc->pyramid_last_gradx)->img[0];
        grady = ((_KLT_Pyramid) tc->pyramid_last_grady)->img[0];
        assert(gradx != NULL);
        assert(grady != NULL);
        use_gpu_computation = FALSE; // Use existing CPU gradients

    } else {
        floatimages_created = TRUE;
        //floatimg = _KLTCreateFloatImage(ncols, nrows);
        //gradx    = _KLTCreateFloatImage(ncols, nrows);
        //grady    = _KLTCreateFloatImage(ncols, nrows);

        _KLT_FloatImage tmpimg;
        tmpimg = _KLTCreateFloatImage(ncols, nrows);
        _KLTToFloatImage(img, ncols, nrows, tmpimg);

        CUDA_CHECK(cudaMalloc(&d_img, ncols * nrows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_smoothed, ncols * nrows * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_img, tmpimg->data, ncols * nrows * sizeof(float), cudaMemcpyHostToDevice));

        // Smoothing
        if (tc->smoothBeforeSelecting)  {
            computeSmoothedImageCUDA(d_img, _KLTComputeSmoothSigma(tc), d_smoothed, ncols, nrows, 0);
        } else {
            CUDA_CHECK(cudaMemcpy(d_smoothed, d_img, ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice)); // not actually smoothed
        }
        
        // Compute Gradients
        CUDA_CHECK(cudaMalloc(&d_gradx, ncols * nrows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grady, ncols * nrows * sizeof(float)));
        computeGradientsCUDA(d_smoothed, tc->grad_sigma, d_gradx, d_grady, ncols, nrows, 0);

    }

    /*
    // comment out bc its lowkey pointless
     // Write internal images 
    if (tc->writeInternalImages)  {
        _KLTWriteFloatImageToPGM(floatimg, "kltimg_sgfrlf.pgm");
        _KLTWriteFloatImageToPGM(gradx, "kltimg_sgfrlf_gx.pgm");
        _KLTWriteFloatImageToPGM(grady, "kltimg_sgfrlf_gy.pgm");
    }
    */

    /* Compute trackability using GPU */
    {
        int borderx = tc->borderx;
        int bordery = tc->bordery;
        
        if (borderx < window_hw)  borderx = window_hw;
        if (bordery < window_hh)  bordery = window_hh;

        // GPU buffers
        int *d_pointlist, *d_npoints;
        
        // Allocate GPU memory
        CUDA_CHECK(cudaMalloc(&d_pointlist, ncols * nrows * 3 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_npoints, sizeof(int)));
                
        // Initialize point counter
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_npoints, &zero, sizeof(int), cudaMemcpyHostToDevice));

/*
         // In your host code, replace the grid calculation:
        int pixels_x = (ncols - 2 * borderx + tc->nSkippedPixels) / (tc->nSkippedPixels + 1);
        int pixels_y = (nrows - 2 * bordery + tc->nSkippedPixels) / (tc->nSkippedPixels + 1);
        
        dim3 blockSize(16, 16);
        // Calculate grid size based on how many blocks we need to cover all output pixels
        dim3 gridSize(
            (pixels_x + blockSize.x - 1) / blockSize.x,
            (pixels_y + blockSize.y - 1) / blockSize.y
        );
        
        // Calculate shared memory requirements
        int tile_width = blockSize.x * (nSkippedPixels + 1) + 2 * window_hw;
        int tile_height = blockSize.y * (nSkippedPixels + 1) + 2 * window_hh;
        size_t shared_mem_size = tile_width * tile_height * sizeof(float2);

                
        if (KLT_verbose) {
            printf("GPU eigenvalue computation: grid(%d,%d), block(%d,%d), pixels(%d,%d)\n",
                   gridSize.x, gridSize.y, blockSize.x, blockSize.y, pixels_x, pixels_y);
        }
        
        // Launch eigenvalue computation kernel
        computeEigenvaluesKernel<<<gridSize, blockSize, shared_mem_size>>>(
            d_gradx, d_grady, d_pointlist, ncols, nrows,
            window_hw, window_hh, borderx, bordery, tc->nSkippedPixels, d_npoints);
*/

        // Calculate grid dimensions
        int pixels_x = (ncols - 2 * borderx + tc->nSkippedPixels) / (tc->nSkippedPixels + 1);
        int pixels_y = (nrows - 2 * bordery + tc->nSkippedPixels) / (tc->nSkippedPixels + 1);
        
        dim3 blockSize(16, 16);  // 256 threads per block
        dim3 gridSize(
            (pixels_x + blockSize.x - 1) / blockSize.x,
            (pixels_y + blockSize.y - 1) / blockSize.y
        );
        
        if (KLT_verbose) {
            printf("Launch config: grid(%d,%d) blocks, block(%d,%d) threads, processing %dx%d pixels\n",
                   gridSize.x, gridSize.y, blockSize.x, blockSize.y, pixels_x, pixels_y);
        }
        
        // Launch kernel WITHOUT shared memory
        computeEigenvaluesKernel<<<gridSize, blockSize>>>(
            d_gradx, d_grady, d_pointlist, ncols, nrows,
            window_hw, window_hh, borderx, bordery, tc->nSkippedPixels, d_npoints);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy results back
        CUDA_CHECK(cudaMemcpy(&npoints, d_npoints, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (npoints > 0) {
            CUDA_CHECK(cudaMemcpy(pointlist, d_pointlist, npoints * 3 * sizeof(int), cudaMemcpyDeviceToHost));
        }
        
        if (KLT_verbose) {
            printf("GPU found %d candidate points\n", npoints);
        }
        
        // Cleanup GPU memory
        cudaFree(d_pointlist);
        cudaFree(d_npoints);
    }
    cudaFree(d_img);
    cudaFree(d_smoothed);
    cudaFree(d_gradx);
    cudaFree(d_grady);


    /* Sort the features */
    if (npoints > 0) {
        _sortPointList(pointlist, npoints);
    }

    /* Check tc->mindist */
    if (tc->mindist < 0)  {
        KLTWarning("(KLTSelectGoodFeaturesCUDA) Tracking context field tc->mindist "
                   "is negative (%d); setting to zero", tc->mindist);
        tc->mindist = 0;
    }

    /* Enforce minimum distance between features */
    {
        KLT_Feature feature;
        int ncols_fe = featurelist->nFeatures, nrows_fe = 1;
        unsigned char *featuremap;
        int *ptr, *listend;
        int x, y, val;
        int mindistsq = tc->mindist * tc->mindist;
        int id;
        float window_halfwidth = 0.5 * tc->window_width;
        float window_halfheight = 0.5 * tc->window_height;
        
        /* Cannot add features with tracking context's window size */
        if (tc->window_width % 2 != 1 || tc->window_height % 2 != 1 ||
            tc->window_width < 3 || tc->window_height < 3)
            KLTError("(KLTSelectGoodFeaturesCUDA) Bad tracking context");

        /* Make array of features */
        featuremap = (unsigned char *) calloc(ncols * nrows, sizeof(unsigned char));
        
        /* Mark existing features if we're replacing some */
        // only if we are replacing not selecting
        if (!overwriteAllFeatures) {
            for (id = 0 ; id < featurelist->nFeatures ; id++)
                if (featurelist->feature[id]->val >= 0) {
                    feature = featurelist->feature[id];
                    x = (int) (feature->x + 0.5);
                    y = (int) (feature->y + 0.5);
                    _fillFeaturemap(x, y, featuremap, tc->mindist, ncols, nrows);
                }
        }

        /* Add features, making sure they don't interfere with each other
           (by checking the featuremap) and not too close to the border */
        ptr = pointlist;
        listend = pointlist + 3 * npoints;
        for (id = 0 ; id < featurelist->nFeatures && ptr < listend ; id++) {
            if (overwriteAllFeatures || featurelist->feature[id]->val < 0) {
                while (ptr < listend) {
                    x = *ptr++;
                    y = *ptr++;
                    val = *ptr++;
                    if (val < tc->min_eigenvalue) break;
                    if (x - window_halfwidth < 0 || 
                        ncols - (x + window_halfwidth) < 1 ||
                        y - window_halfheight < 0 || 
                        nrows - (y + window_halfheight) < 1) continue;
                    if (_outOfBounds(x, y, ncols, nrows, tc->borderx, tc->bordery)) continue;
                    if (featuremap[y*ncols+x] == 0) break;
                }
                if (ptr < listend) {
                    featurelist->feature[id]->x = (float) x;
                    featurelist->feature[id]->y = (float) y;
                    featurelist->feature[id]->val = val;
                    _fillFeaturemap(x, y, featuremap, tc->mindist, ncols, nrows);
                } else {
                    featurelist->feature[id]->val = KLT_NOT_FOUND;
                }
            }
        }

        /* Fill in remaining features with KLT_NOT_FOUND */
        for ( ; id < featurelist->nFeatures ; id++)
            if (overwriteAllFeatures || featurelist->feature[id]->val < 0)
                featurelist->feature[id]->val = KLT_NOT_FOUND;

        free(featuremap);
    }


    /* Free memory */
    /*
    if (floatimages_created) {
        _KLTFreeFloatImage(floatimg);
        _KLTFreeFloatImage(gradx);
        _KLTFreeFloatImage(grady);
    }
    */
    free(pointlist);
}

/*********************************************************************
 * KLTSelectGoodFeatures - Public wrapper function
 */
void KLTSelectGoodFeaturesCUDA(
    KLT_TrackingContext tc,
    KLT_PixelType *img, 
    int ncols, 
    int nrows,
    KLT_FeatureList featurelist)
{
    if (KLT_verbose >= 1) {
        fprintf(stderr,  "(KLT) Selecting the %d best features "
                "from a %d by %d image...  ", featurelist->nFeatures, ncols, nrows);
        fflush(stderr);
    }

    _KLTSelectGoodFeaturesCUDA(tc, img, ncols, nrows, featurelist, SELECTING_ALL);

    if (KLT_verbose >= 1) {
        fprintf(stderr,  "\n\t%d features found.\n", 
                KLTCountRemainingFeatures(featurelist));
        if (tc->writeInternalImages)
            fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
        fflush(stderr);
    }
}

/*********************************************************************
 * KLTReplaceLostFeatures - Public wrapper function
 */
void KLTReplaceLostFeaturesCUDA(
    KLT_TrackingContext tc,
    KLT_PixelType *img, 
    int ncols, 
    int nrows,
    KLT_FeatureList featurelist)
{
    int nLostFeatures = featurelist->nFeatures - KLTCountRemainingFeatures(featurelist);

    if (KLT_verbose >= 1) {
        fprintf(stderr,  "(KLT) Attempting to replace %d features "
                "in a %d by %d image...  ", nLostFeatures, ncols, nrows);
        fflush(stderr);
    }

    /* If there are any lost features, replace them */
    if (nLostFeatures > 0)
        _KLTSelectGoodFeaturesCUDA(tc, img, ncols, nrows, featurelist, REPLACING_SOME);

    if (KLT_verbose >= 1) {
        fprintf(stderr,  "\n\t%d features replaced.\n",
                nLostFeatures - featurelist->nFeatures + KLTCountRemainingFeatures(featurelist));
        if (tc->writeInternalImages)
            fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
        fflush(stderr);
    }
}