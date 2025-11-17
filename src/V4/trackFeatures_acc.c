/*********************************************************************
 * trackFeatures.c - OpenACC Optimized Version
 *********************************************************************/

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
#include "convolve_acc.h"  // Include the new header
#include "klt.h"
#include "klt_util.h"
#include "pyramid.h"
#include <float.h>

extern int KLT_verbose;

// Use flat buffer types
typedef float *FloatWindow;

// Flat pyramid structure for OpenACC
typedef struct {
    float *data;           // Flat buffer for all pyramid levels
    int nLevels;
    int *ncols;
    int *nrows;
    size_t *offsets;
    size_t total_size;
} FlatPyramid;

typedef struct {
    float gxx, gxy, gyy;
} GradientMatrix;

typedef struct {
    float ex, ey;
} ErrorVector;

typedef struct {
    float dx, dy;
    int status;
} SolutionResult;

// Global pyramid metadata
FlatPyramid pyramid_meta = {0};
bool pyramid_meta_initialized = false;

// Global reusable buffers
float *pyramid1_flat = NULL;
float *pyramid1_gradx_flat = NULL;
float *pyramid1_grady_flat = NULL;
float *pyramid2_flat = NULL;
float *pyramid2_gradx_flat = NULL;
float *pyramid2_grady_flat = NULL;

float *img1_flat = NULL;    
float *img2_flat = NULL;   
float *smooth_img1_flat = NULL, *smooth_img2_flat = NULL;
bool device_buffs_allocated = false;

bool first_frame = true;
int frame_count = 0;

static void verifyPyramidContents(const char* name, float* pyramid, int nLevels, int subsampling, 
                                  int base_ncols, int base_nrows) {
    printf("=== Verifying %s Pyramid ===\n", name);
    
    int current_ncols = base_ncols;
    int current_nrows = base_nrows;
    size_t total_offset = 0;
    
    for (int level = 0; level < nLevels; level++) {
        size_t level_size = (size_t)current_ncols * (size_t)current_nrows;
        float *level_data = pyramid + total_offset;
        
        printf("Level %d (%dx%d):\n", level, current_ncols, current_nrows);
        
        // Print first row
        printf("  First row: ");
        for (int x = 0; x < (current_ncols < 10 ? current_ncols : 10); x++) {
            printf("%6.1f ", level_data[x]);
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
                    float val = level_data[center_y * current_ncols + x];
                    printf("%6.1f ", val);
                }
            }
            printf("\n");
        }
        
        // Check for all zeros, min/max
        int zero_count = 0;
        float min_val = FLT_MAX, max_val = -FLT_MAX;
        for (size_t i = 0; i < level_size; i++) {
            if (level_data[i] == 0.0f) zero_count++;
            if (level_data[i] < min_val) min_val = level_data[i];
            if (level_data[i] > max_val) max_val = level_data[i];
        }
        
        printf("  Stats: zeros=%d/%zu (%.1f%%), min=%.2f, max=%.2f\n",
               zero_count, level_size, (100.0f * zero_count) / level_size, min_val, max_val);
        
        total_offset += level_size;
        if (level < nLevels - 1) {
            current_ncols /= subsampling;
            current_nrows /= subsampling;
        }
    }
    printf("=== End %s Pyramid Verification ===\n\n", name);
}


void initializePyramidMetadata(int base_ncols, int base_nrows, int subsampling, int nLevels) {
    if (pyramid_meta_initialized) return;
    
    pyramid_meta.nLevels = nLevels;
    pyramid_meta.ncols = (int*)malloc(nLevels * sizeof(int));
    pyramid_meta.nrows = (int*)malloc(nLevels * sizeof(int));
    pyramid_meta.offsets = (size_t*)malloc(nLevels * sizeof(size_t));
    
    // Calculate total size and level dimensions
    size_t total_size = 0;
    int current_ncols = base_ncols;
    int current_nrows = base_nrows;
    
    for (int i = 0; i < nLevels; i++) {
        pyramid_meta.ncols[i] = current_ncols;
        pyramid_meta.nrows[i] = current_nrows;
        pyramid_meta.offsets[i] = total_size;
        total_size += (size_t)current_ncols * current_nrows;
        
        if (i < nLevels - 1) {
            current_ncols /= subsampling;
            current_nrows /= subsampling;
        }
    }
    
    pyramid_meta.total_size = total_size;
    pyramid_meta.data = (float*)malloc(total_size * sizeof(float));
    pyramid_meta_initialized = true;
}

void freePyramidMetadata() {
    if (!pyramid_meta_initialized) return;
    
    free(pyramid_meta.data);
    free(pyramid_meta.ncols);
    free(pyramid_meta.nrows);
    free(pyramid_meta.offsets);
    pyramid_meta_initialized = false;
}

float* getPyramidLevel(int level) {
    if (!pyramid_meta_initialized || level < 0 || level >= pyramid_meta.nLevels)
        return NULL;
    return pyramid_meta.data + pyramid_meta.offsets[level];
}

size_t getPyramidTotalSize() {
    return pyramid_meta.total_size;
}

int getPyramidLevelCols(int level) {
    if (!pyramid_meta_initialized || level < 0 || level >= pyramid_meta.nLevels)
        return 0;
    return pyramid_meta.ncols[level];
}

int getPyramidLevelRows(int level) {
    if (!pyramid_meta_initialized || level < 0 || level >= pyramid_meta.nLevels)
        return 0;
    return pyramid_meta.nrows[level];
}

#pragma acc routine seq
static float _sumAbsFloatWindow(
  FloatWindow fw,
  int width,
  int height)
{
  float sum = 0.0;
  int w;

  for ( ; height > 0 ; height--)
    for (w=0 ; w < width ; w++)
      sum += (float) fabs(*fw++);

  return sum;
}

/*********************************************************************
 * _interpolate - OpenACC compatible
 */
#pragma acc routine seq
static float _interpolate(
  float x, float y, 
  const float *imdata, int imncols, int imnrows)
{
  int xt = (int)x;
  int yt = (int)y;
  float ax = x - xt;
  float ay = y - yt;
  const float *ptr = imdata + yt * imncols + xt;

  if (xt < 0 || yt < 0 || xt > imncols - 2 || yt > imnrows - 2) {
    printf("[_interpolate] Out of bounds: x=%.2f y=%.2f xt=%d yt=%d imncols=%d imnrows=%d\n",
           x, y, xt, yt, imncols, imnrows);
  }

  assert(xt >= 0 && yt >= 0 && xt <= imncols - 2 && yt <= imnrows - 2);

  return ((1-ax) * (1-ay) * ptr[0] +
          ax   * (1-ay) * ptr[1] +
          (1-ax) *   ay   * ptr[imncols] +
          ax   *   ay   * ptr[imncols+1]);
}

/*********************************************************************
 * _computeIntensityDifference - OpenACC optimized
 */
//#pragma acc routine seq  
static void _computeIntensityDifference(
  const float *img1, int img1_ncols, int img1_nrows,
  const float *img2, int img2_ncols, int img2_nrows,
  float x1, float y1,
  float x2, float y2,
  int width, int height,
  FloatWindow imgdiff)
{
  int hw = width/2, hh = height/2;

  #pragma acc loop collapse(2) 
  for (int j = -hh; j <= hh; j++) {
    for (int i = -hw; i <= hw; i++) {
      int idx = (j + hh) * width + (i + hw);
      float g1 = _interpolate(x1 + i, y1 + j, img1, img1_ncols, img1_nrows);
      float g2 = _interpolate(x2 + i, y2 + j, img2, img2_ncols, img2_nrows);
      imgdiff[idx] = g1 - g2;
    }
  }
}


/*********************************************************************
 * _computeGradientSum - OpenACC optimized
 */
//#pragma acc routine seq  
static void _computeGradientSum(
  const float *gradx1, int gx1_ncols, int gx1_nrows,
  const float *grady1, int gy1_ncols, int gy1_nrows,
  const float *gradx2, int gx2_ncols, int gx2_nrows,
  const float *grady2, int gy2_ncols, int gy2_nrows,
  float x1, float y1,
  float x2, float y2,
  int width, int height,
  FloatWindow gradx,
  FloatWindow grady)
{
  int hw = width/2, hh = height/2;

  #pragma acc loop collapse(2) 
  for (int j = -hh; j <= hh; j++) {
    for (int i = -hw; i <= hw; i++) {
      int idx = (j + hh) * width + (i + hw);
      float gx1 = _interpolate(x1 + i, y1 + j, gradx1, gx1_ncols, gx1_nrows);
      float gx2 = _interpolate(x2 + i, y2 + j, gradx2, gx2_ncols, gx2_nrows);
      float gy1 = _interpolate(x1 + i, y1 + j, grady1, gy1_ncols, gy1_nrows);
      float gy2 = _interpolate(x2 + i, y2 + j, grady2, gy2_ncols, gy2_nrows);
      gradx[idx] = gx1 + gx2;
      grady[idx] = gy1 + gy2;
    }
  }
}



/*********************************************************************
 * _compute2by2GradientMatrix - OpenACC optimized
 */
#pragma acc routine seq 
static void _compute2by2GradientMatrix(
  FloatWindow gradx,
  FloatWindow grady,
  int width, int height,
  float *gxx, float *gxy, float *gyy)
{
  float local_gxx = 0.0, local_gxy = 0.0, local_gyy = 0.0;

  #pragma acc loop reduction(+:local_gxx,local_gxy,local_gyy)
  for (int i = 0; i < width * height; i++) {
    float gx = gradx[i];
    float gy = grady[i];
    local_gxx += gx * gx;
    local_gxy += gx * gy;
    local_gyy += gy * gy;
  }
  *gxx = local_gxx;
  *gxy = local_gxy;
  *gyy = local_gyy;
}

#pragma acc routine seq
static GradientMatrix _compute2by2GradientMatrixACC(
    FloatWindow gradx, FloatWindow grady, int width, int height)
{
    float local_gxx = 0.0, local_gxy = 0.0, local_gyy = 0.0;
    GradientMatrix result = {0};

    #pragma acc loop reduction(+:local_gxx,local_gxy,local_gyy)
    for (int i = 0; i < width * height; i++) {
        float gx = gradx[i];
        float gy = grady[i];
        local_gxx += gx * gx;
        local_gxy += gx * gy;
        local_gyy += gy * gy;
    }
    result.gxx = local_gxx;
    result.gxy = local_gxy;
    result.gyy = local_gyy;
    return result;
}

/*********************************************************************
 * _compute2by1ErrorVector - OpenACC optimized
 */
#pragma acc routine seq 
static void _compute2by1ErrorVector(
  FloatWindow imgdiff,
  FloatWindow gradx,
  FloatWindow grady,
  int width, int height,
  float step_factor,
  float *ex, float *ey)
{
  float local_ex = 0.0, local_ey = 0.0;
  
  #pragma acc loop reduction(+:local_ex,local_ey) 
  for (int i = 0; i < width * height; i++) {
    float diff = imgdiff[i];
    local_ex += diff * gradx[i];
    local_ey += diff * grady[i];
  }

  *ex = local_ex * step_factor;
  *ey = local_ey * step_factor;
}

#pragma acc routine seq 
static ErrorVector _compute2by1ErrorVectorACC(
    FloatWindow imgdiff, FloatWindow gradx, FloatWindow grady,
    int width, int height, float step_factor)
{
    float local_ex = 0.0, local_ey = 0.0;
    ErrorVector result = {0};

    #pragma acc loop reduction(+:local_ex,local_ey) 
    for (int i = 0; i < width * height; i++) {
        float diff = imgdiff[i];
        local_ex += diff * gradx[i];
        local_ey += diff * grady[i];
    }
    result.ex = local_ex * step_factor;
    result.ey = local_ey * step_factor;
    return result;
}

/*********************************************************************
 * _solveEquation - OpenACC compatible
 */

#pragma acc routine seq 
static int _solveEquation(
  float gxx, float gxy, float gyy,
  float ex, float ey,
  float small,
  float *dx, float *dy)
{
  float det = gxx * gyy - gxy * gxy;

  if (det < small) return KLT_SMALL_DET;

  *dx = (gyy * ex - gxy * ey) / det;
  *dy = (gxx * ey - gxy * ex) / det;
  return KLT_TRACKED;
}

#pragma acc routine seq
static SolutionResult _solveEquationACC(
    float gxx, float gxy, float gyy, float ex, float ey, float small)
{
    SolutionResult result = {0};
    float det = gxx * gyy - gxy * gxy;
    if (det < small) {
        result.status = KLT_SMALL_DET;
    } else {
        result.dx = (gyy * ex - gxy * ey) / det;
        result.dy = (gxx * ey - gxy * ex) / det;
        result.status = KLT_TRACKED;
    }
    return result;
}

static KLT_BOOL _outOfBounds(
  float x,
  float y,
  int ncols,
  int nrows,
  int borderx,
  int bordery)
{
  return (x < borderx || x > ncols-1-borderx ||
          y < bordery || y > nrows-1-bordery );
}


/*********************************************************************
 * _trackFeature - OpenACC optimized with flat buffers
 */

static int _trackFeature(
  float x1, float y1,
  float *x2, float *y2,
  const float *img1_flat, int img1_ncols, int img1_nrows,
  const float *gradx1_flat, int gx1_ncols, int gx1_nrows,
  const float *grady1_flat, int gy1_ncols, int gy1_nrows,
  const float *img2_flat, int img2_ncols, int img2_nrows,
  const float *gradx2_flat, int gx2_ncols, int gx2_nrows,
  const float *grady2_flat, int gy2_ncols, int gy2_nrows,
  int width, int height,
  float step_factor,
  int max_iterations,
  float small,
  float th,
  float max_residue,
  int lightininsensitive
)
{
  FloatWindow imgdiff, gradx, grady;
  float gxx, gxy, gyy, ex, ey, dx, dy;
  int iteration = 0;
  int status;
  int val;
  int hw = width/2;
  int hh = height/2;
  float one_plus_eps = 1.001f;

  // Allocate windows
  imgdiff = (FloatWindow)malloc(width * height * sizeof(float));
  gradx = (FloatWindow)malloc(width * height * sizeof(float));
  grady = (FloatWindow)malloc(width * height * sizeof(float));

  #pragma acc data create(imgdiff[0:width*height], \
                         gradx[0:width*height], \
                         grady[0:width*height])
  {
    do {

       // Debug: print current positions and window
      //printf("[_trackFeature] Iteration %d: x1=%.2f y1=%.2f | x2=%.2f y2=%.2f | window=%dx%d\n",
             //iteration, x1, y1, *x2, *y2, width, height);
      
      
      //printf("[_trackFeature] Boundary check: x1=%.2f y1=%.2f x2=%.2f y2=%.2f hw=%d hh=%d img1_ncols=%d img1_nrows=%d img2_ncols=%d img2_nrows=%d\n",
            //x1, y1, *x2, *y2, hw, hh, img1_ncols, img1_nrows, img2_ncols, img2_nrows);

      // Boundary check
      if (x1 - hw < 0.0f || img1_ncols - (x1 + hw) < one_plus_eps ||
          *x2 - hw < 0.0f || img2_ncols - (*x2 + hw) < one_plus_eps ||
          y1 - hh < 0.0f || img1_nrows - (y1 + hh) < one_plus_eps ||
          *y2 - hh < 0.0f || img2_nrows - (*y2 + hh) < one_plus_eps) {
        status = KLT_OOB;
        printf("[_trackFeature] Dropped: OOB (out of bounds)\n");
        break;
      }

      //if (featureIdx==0){
        //printf("[_trackFeature] Tracking feature %d at position (%.2f, %.2f) in img1 and (%.2f, %.2f) in img2\n",
               //featureIdx, x1, y1, *x2, *y2);
        //verifyPyramidContents("img1", img1_flat, 1, 1, img1_ncols, img1_nrows);
        // verifyPyramidContents("img2", img2_flat, 1, 1, img2_ncols, img2_nrows);
        //verifyPyramidContents("gradx1", gradx1_flat, 1, 1, gx1_ncols, gx1_nrows);
        //verifyPyramidContents("grady1", grady1_flat, 1, 1, gy1_ncols, gy1_nrows);
        //verifyPyramidContents("gradx2", gradx2_flat, 1, 1, gx2_ncols, gx2_nrows);
        //verifyPyramidContents("grady2", grady2_flat, 1, 1, gy2_ncols, gy2_nrows);
      //}

      // Compute gradient and difference windows
      _computeIntensityDifference(img1_flat, img1_ncols, img1_nrows,
                                  img2_flat, img2_ncols, img2_nrows,
                                  x1, y1, *x2, *y2,
                                  width, height, imgdiff);
      _computeGradientSum(gradx1_flat, gx1_ncols, gx1_nrows,
                          grady1_flat, gy1_ncols, gy1_nrows,
                          gradx2_flat, gx2_ncols, gx2_nrows,
                          grady2_flat, gy2_ncols, gy2_nrows,
                          x1, y1, *x2, *y2,
                          width, height, gradx, grady);

      // Compute matrices
      //_compute2by2GradientMatrix(gradx, grady, width, height, &gxx, &gxy, &gyy);
      //_compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor, &ex, &ey);

      //if (featureIdx==5)
        //printf("[_trackFeature]feature %d gxx=%.6f gxy=%.6f gyy=%.6f ex=%.6f ey=%.6f\n", featureIdx, gxx, gxy, gyy, ex, ey);

      // Solve equation
      //status = _solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);

    
        // Compute matrices 
        GradientMatrix grad_matrix = _compute2by2GradientMatrixACC(gradx, grady, width, height);
        ErrorVector error_vec = _compute2by1ErrorVectorACC(imgdiff, gradx, grady, width, height, step_factor);

        // Solve equation
        SolutionResult solution = _solveEquationACC(grad_matrix.gxx, grad_matrix.gxy, grad_matrix.gyy,
                                              error_vec.ex, error_vec.ey, small);

      status = solution.status;
      dx = solution.dx;
      dy = solution.dy;
   
      if (status == KLT_SMALL_DET) {
        //printf("[_trackFeature] Dropped: SMALL_DET (small determinant)\n");
        break;
      }

      // Debug: print computed dx/dy
      //printf("[_trackFeature] dx=%.4f dy=%.4f\n", dx, dy);

      *x2 += dx;
      *y2 += dy;
      iteration++;


      //printf("[_trackFeature] Tracked from (%.2f, %.2f) to (%.2f, %.2f) | iteration %d | dx=%.4f dy=%.4f\n",x1, y1, *x2, *y2, iteration, dx, dy);

      if (iteration >= max_iterations) {
        //printf("[_trackFeature] Dropped: MAX_ITERATIONS\n");
      }


    } while ((fabs(dx) >= th || fabs(dy) >= th) && iteration < max_iterations);
  }

  /* Check whether window is out of bounds */
  if (*x2-hw < 0.0f || img1_ncols-(*x2+hw) < one_plus_eps || 
      *y2-hh < 0.0f || img1_nrows-(*y2+hh) < one_plus_eps)
    status = KLT_OOB;

  /* Check whether residue is too large */
  if (status == KLT_TRACKED)  {
    _computeIntensityDifference(img1_flat, img1_ncols, img1_nrows,
                                img2_flat, img2_ncols, img2_nrows,
                                x1, y1, *x2, *y2, 
                                width, height, imgdiff);
    if (_sumAbsFloatWindow(imgdiff, width, height)/(width*height) > max_residue) 
      status = KLT_LARGE_RESIDUE;
  }

  // Free memory
  free(imgdiff);
  free(gradx);
  free(grady);

  return (status == KLT_SMALL_DET) ? KLT_SMALL_DET :
         (status == KLT_OOB) ? KLT_OOB :
         (iteration >= max_iterations) ? KLT_MAX_ITERATIONS : KLT_TRACKED;
}


void initializeOpenACCBuffers(int ncols, int nrows, int subsampling, int nLevels)
{
    if (device_buffs_allocated) return;
    
    // Initialize pyramid structure
    initializePyramidMetadata(ncols, nrows, subsampling, nLevels);
    
    // Allocate HOST buffers
    size_t total_size = getPyramidTotalSize();
    pyramid1_flat = (float*)malloc(total_size * sizeof(float));
    pyramid1_gradx_flat = (float*)malloc(total_size * sizeof(float));
    pyramid1_grady_flat = (float*)malloc(total_size * sizeof(float));
    pyramid2_flat = (float*)malloc(total_size * sizeof(float));
    pyramid2_gradx_flat = (float*)malloc(total_size * sizeof(float));
    pyramid2_grady_flat = (float*)malloc(total_size * sizeof(float));
    
    img1_flat = (float*)malloc(ncols * nrows * sizeof(float));
    img2_flat = (float*)malloc(ncols * nrows * sizeof(float));
    smooth_img1_flat = (float*)malloc(ncols * nrows * sizeof(float));
    smooth_img2_flat = (float*)malloc(ncols * nrows * sizeof(float));

    // Copy pyramid_meta to DEVICE (ONCE at initialization)
    #pragma acc enter data copyin(pyramid_meta)
    #pragma acc enter data copyin(pyramid_meta.ncols[0:nLevels])
    #pragma acc enter data copyin(pyramid_meta.nrows[0:nLevels]) 
    #pragma acc enter data copyin(pyramid_meta.offsets[0:nLevels])
    
                        
    // Allocate DEVICE buffers and keep them there
    #pragma acc enter data create( \
        img1_flat[0:ncols*nrows], \
        img2_flat[0:ncols*nrows], \
        smooth_img1_flat[0:ncols*nrows], \
        smooth_img2_flat[0:ncols*nrows], \
        pyramid1_flat[0:total_size], \
        pyramid2_flat[0:total_size], \
        pyramid1_gradx_flat[0:total_size], \
        pyramid1_grady_flat[0:total_size], \
        pyramid2_gradx_flat[0:total_size], \
        pyramid2_grady_flat[0:total_size])

    device_buffs_allocated = true;
    
    printf("Initialized persistent OpenACC buffers\n");
}

/*********************************************************************
 * KLTTrackFeatures - Main optimized function with flat buffers
 */

void KLTTrackFeaturesACC(
  KLT_TrackingContext tc,
  KLT_PixelType *img1,
  KLT_PixelType *img2,
  int ncols,
  int nrows,
  KLT_FeatureList featurelist)
{

  float subsampling = (float) tc->subsampling;
    float xloc, yloc, xlocout, ylocout;
	int val;
	int indx, r;

  printf("KLTTrackFeaturesACC called for frame %d\n", frame_count);

  {
    // Process images using FLAT versions
    if (first_frame || !tc->sequentialMode) {
      // Compute pyramid for first image using flat functions
      _KLTToFloatImageACC(img1, ncols, nrows, img1_flat);

      _KLTComputeSmoothedImageACC(img1_flat, ncols, nrows,
                                  _KLTComputeSmoothSigma(tc),
                                  smooth_img1_flat, ncols, nrows);

      computePyramidOpenACC(smooth_img1_flat, ncols, nrows,
                           tc->pyramid_sigma_fact,
                           pyramid1_flat, (int)tc->subsampling, tc->nPyramidLevels);

      //verifyPyramidContents("KLTTrackFeaturesACC - pyramid1", pyramid1_flat, tc->nPyramidLevels, (int)tc->subsampling, ncols, nrows);

      // Compute gradients for each pyramid level using flat functions
      for (int i = 0; i < tc->nPyramidLevels; i++) {
        float *level_img = pyramid1_flat + pyramid_meta.offsets[i];
        float *level_gradx = pyramid1_gradx_flat + pyramid_meta.offsets[i];
        float *level_grady = pyramid1_grady_flat + pyramid_meta.offsets[i];
        int level_ncols = pyramid_meta.ncols[i];
        int level_nrows = pyramid_meta.nrows[i];
        
        _KLTComputeGradientsACC(level_img, level_ncols, level_nrows,
                                tc->grad_sigma,
                                level_gradx, level_ncols, level_nrows,
                                level_grady, level_ncols, level_nrows);
      }

        
      //verifyPyramidContents("KLTTrackFeaturesACC - pyramid1gradx", pyramid1_gradx_flat, tc->nPyramidLevels, (int)tc->subsampling, ncols, nrows);
      //verifyPyramidContents("KLTTrackFeaturesACC - pyramid1grady", pyramid1_grady_flat, tc->nPyramidLevels, (int)tc->subsampling, ncols, nrows);

      first_frame = false;
    }
    else{
      // Swap the pyramid buffers to reuse computed pyramids
      float *temp_pyramid = pyramid1_flat;
      float *temp_gradx = pyramid1_gradx_flat;
      float *temp_grady = pyramid1_grady_flat;
      
      pyramid1_flat = pyramid2_flat;
      pyramid1_gradx_flat = pyramid2_gradx_flat;
      pyramid1_grady_flat = pyramid2_grady_flat;

      //verifyPyramidContents("KLTTrackFeaturesACC - pyramid1", pyramid1_flat, tc->nPyramidLevels, (int)tc->subsampling, ncols, nrows);
      //verifyPyramidContents("KLTTrackFeaturesACC - pyramid1gradx", pyramid1_gradx_flat, tc->nPyramidLevels, (int)tc->subsampling, ncols, nrows);
      //verifyPyramidContents("KLTTrackFeaturesACC - pyramid1grady", pyramid1_grady_flat, tc->nPyramidLevels, (int)tc->subsampling, ncols, nrows);

      pyramid2_flat = temp_pyramid;
      pyramid2_gradx_flat = temp_gradx;
      pyramid2_grady_flat = temp_grady;

      
      //printf("Sequential mode: Swapped frame buffers (reusing previous frame2 as new frame1)\n");  

    }

    _KLTToFloatImageACC(img2, ncols, nrows, img2_flat);

    // Always compute pyramid for second image using flat functions
    _KLTComputeSmoothedImageACC(img2_flat, ncols, nrows,
                                _KLTComputeSmoothSigma(tc),
                                smooth_img2_flat, ncols, nrows);
    
    computePyramidOpenACC(smooth_img2_flat, ncols, nrows,
                         tc->pyramid_sigma_fact,
                         pyramid2_flat, (int)tc->subsampling, tc->nPyramidLevels);

    //verifyPyramidContents("KLTTrackFeaturesACC - pyramid2", pyramid2_flat, tc->nPyramidLevels, (int)tc->subsampling, ncols, nrows);

    // Compute gradients for second pyramid using flat functions
    for (int i = 0; i < tc->nPyramidLevels; i++) {
      float *level_img = pyramid2_flat + pyramid_meta.offsets[i];
      float *level_gradx = pyramid2_gradx_flat + pyramid_meta.offsets[i];
      float *level_grady = pyramid2_grady_flat + pyramid_meta.offsets[i];
      int level_ncols = pyramid_meta.ncols[i];
      int level_nrows = pyramid_meta.nrows[i];
      
      _KLTComputeGradientsACC(level_img, level_ncols, level_nrows,
                              tc->grad_sigma,
                              level_gradx, level_ncols, level_nrows,
                              level_grady, level_ncols, level_nrows);
    }

    //verifyPyramidContents("KLTTrackFeaturesACC - pyramid2gradx", pyramid2_gradx_flat, tc->nPyramidLevels, (int)tc->subsampling, ncols, nrows);
    //verifyPyramidContents("KLTTrackFeaturesACC - pyramid2grady", pyramid2_grady_flat, tc->nPyramidLevels, (int)tc->subsampling, ncols, nrows);


    // Track features using flat buffers
       #pragma acc kernels present( \
        pyramid1_flat, pyramid1_gradx_flat, pyramid1_grady_flat, \
        pyramid2_flat, pyramid2_gradx_flat, pyramid2_grady_flat, \
        pyramid_meta, pyramid_meta.ncols, pyramid_meta.nrows, pyramid_meta.offsets)
    for (indx = 0; indx < featurelist->nFeatures; indx++) {

      int val;

      if (featurelist->feature[indx]->val >= 0) {

        float xloc = featurelist->feature[indx]->x;
        float yloc = featurelist->feature[indx]->y;
        float xlocout = xloc, ylocout = yloc;

        int tracking_failed = 0;  // Flag to track failure


        /* Transform location to coarsest resolution */
        for (r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
          xloc /= subsampling;  yloc /= subsampling;
        }
        xlocout = xloc;  ylocout = yloc;


        // Track through pyramid levels (coarsest to finest)
        for (r = tc->nPyramidLevels - 1; r >= 0; r--) {

          if (tracking_failed) continue;

          float *img1_level = pyramid1_flat + pyramid_meta.offsets[r];
          float *gradx1_level = pyramid1_gradx_flat + pyramid_meta.offsets[r];
          float *grady1_level = pyramid1_grady_flat + pyramid_meta.offsets[r];
          float *img2_level = pyramid2_flat + pyramid_meta.offsets[r];
          float *gradx2_level = pyramid2_gradx_flat + pyramid_meta.offsets[r];
          float *grady2_level = pyramid2_grady_flat + pyramid_meta.offsets[r];
        
          /* Track feature at current resolution */
          xloc *= subsampling;  yloc *= subsampling;
          xlocout *= subsampling;  ylocout *= subsampling;
         
          int level_ncols = pyramid_meta.ncols[r];
          int level_nrows = pyramid_meta.nrows[r];

          if (indx==0){
            //printf("Tracking feature %d at pyramid level %d with image size %dx%d\n", indx, r, level_ncols, level_nrows);
            //verifyPyramidContents("img1_level", img1_level, 1, 1, level_ncols, level_nrows);
          }

          val = _trackFeature(xloc, yloc, &xlocout, &ylocout,
                                 img1_level, level_ncols, level_nrows,
                                 gradx1_level, level_ncols, level_nrows,
                                 grady1_level, level_ncols, level_nrows,
                                 img2_level, level_ncols, level_nrows,
                                 gradx2_level, level_ncols, level_nrows,
                                 grady2_level, level_ncols, level_nrows,
                                 tc->window_width, tc->window_height,
                                 tc->step_factor, tc->max_iterations,
                                 tc->min_determinant, tc->min_displacement,
                                 tc->max_residue, tc->lighting_insensitive
                                 );

          if (val==KLT_SMALL_DET || val==KLT_OOB)
            tracking_failed = 1;

        }

        /* Record feature */
        if (val == KLT_OOB) {
          featurelist->feature[indx]->x   = -1.0;
          featurelist->feature[indx]->y   = -1.0;
          featurelist->feature[indx]->val = KLT_OOB;
        } else if (_outOfBounds(xlocout, ylocout, ncols, nrows, tc->borderx, tc->bordery))  {
          featurelist->feature[indx]->x   = -1.0;
          featurelist->feature[indx]->y   = -1.0;
          featurelist->feature[indx]->val = KLT_OOB;
        } else if (val == KLT_SMALL_DET)  {
          featurelist->feature[indx]->x   = -1.0;
          featurelist->feature[indx]->y   = -1.0;
          featurelist->feature[indx]->val = KLT_SMALL_DET;
        } else if (val == KLT_LARGE_RESIDUE)  {
          featurelist->feature[indx]->x   = -1.0;
          featurelist->feature[indx]->y   = -1.0;
          featurelist->feature[indx]->val = KLT_LARGE_RESIDUE;
        } else if (val == KLT_MAX_ITERATIONS)  {
          featurelist->feature[indx]->x   = -1.0;
          featurelist->feature[indx]->y   = -1.0;
          featurelist->feature[indx]->val = KLT_MAX_ITERATIONS;
        } else  {
          featurelist->feature[indx]->x = xlocout;
          featurelist->feature[indx]->y = ylocout;
          featurelist->feature[indx]->val = KLT_TRACKED;
		}
      
      }
    }
  }

  frame_count++;
}

void cleanupOpenACCResources() {
    if (pyramid_meta_initialized) {
        #pragma acc exit data delete( \
            pyramid1_flat[0:pyramid_meta.total_size], \
            pyramid1_gradx_flat[0:pyramid_meta.total_size], \
            pyramid1_grady_flat[0:pyramid_meta.total_size], \
            pyramid2_flat[0:pyramid_meta.total_size], \
            pyramid2_gradx_flat[0:pyramid_meta.total_size], \
            pyramid2_grady_flat[0:pyramid_meta.total_size], \
            img1_flat[0:1], img2_flat[0:1], \
            smooth_img1_flat[0:1], smooth_img2_flat[0:1], \
            pyramid_meta.offsets[0:pyramid_meta.nLevels], \
            pyramid_meta.nrows[0:pyramid_meta.nLevels], \
            pyramid_meta.ncols[0:pyramid_meta.nLevels], \
            pyramid_meta)
        
        
        freePyramidMetadata();
        if (pyramid1_flat) free(pyramid1_flat);
        if (pyramid1_gradx_flat) free(pyramid1_gradx_flat);
        if (pyramid1_grady_flat) free(pyramid1_grady_flat);
        if (pyramid2_flat) free(pyramid2_flat);
        if (pyramid2_gradx_flat) free(pyramid2_gradx_flat);
        if (pyramid2_grady_flat) free(pyramid2_grady_flat);
        if (img1_flat) free(img1_flat);
        if (img2_flat) free(img2_flat);
        if (smooth_img1_flat) free(smooth_img1_flat);
        if (smooth_img2_flat) free(smooth_img2_flat);
        
        pyramid_meta_initialized = false;
    }
}