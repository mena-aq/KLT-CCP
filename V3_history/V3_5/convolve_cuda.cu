/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "base.h"
#include "error.h"
#include "klt_util.h"

#include "convolve_cuda.h"

#define CUDA_CHECK(call)                                                    \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)


/* External globals expected from other parts of the KLT codebase */
extern ConvolutionKernel gauss_kernel;
extern ConvolutionKernel gaussderiv_kernel;

/* Forward declarations for static helpers used below */
static void _computeKernels(float sigma, ConvolutionKernel *gauss, ConvolutionKernel *gaussderiv);

/* Small helper for CUDA errors */
static void checkCuda(cudaError_t err, const char *msg)
{
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
    exit(1);
  }
}

/* Horizontal 1D convolution kernel (reads from imgin, writes to imgout) */
__global__ void convolveHorizKernelold(const float *imgin, float *imgout,
                                    int ncols, int nrows,
                                    const float *kernel, int kwidth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= ncols || y >= nrows) return;

  int radius = kwidth / 2;

  /* Zero border */
  if (x < radius || x >= ncols - radius) {
    imgout[y * ncols + x] = 0.0f;
    return;
  }

  float sum = 0.0f;
  int start = x - radius;
  for (int k = 0; k < kwidth; k++) {
    /* kernel indexed reversed to match original CPU implementation */
    sum += imgin[y * ncols + (start + k)] * kernel[kwidth - 1 - k];
  }
  imgout[y * ncols + x] = sum;
}

/* Vertical 1D convolution kernel (reads from imgin, writes to imgout) */
__global__ void convolveVertKernelold(const float *imgin, float *imgout,
                                    int ncols, int nrows,
                                    const float *kernel, int kwidth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= ncols || y >= nrows) return;

  int radius = kwidth / 2;

  /* Zero border */
  if (y < radius || y >= nrows - radius) {
    imgout[y * ncols + x] = 0.0f;
    return;
  }

  float sum = 0.0f;
  int start = y - radius;
  for (int k = 0; k < kwidth; k++) {
    sum += imgin[(start + k) * ncols + x] * kernel[kwidth - 1 - k];
  }
  imgout[y * ncols + x] = sum;
}

// ------------------------------------------- v3.5 --------------------------------------


/* CPU: convert pixel values to float image */
void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + (size_t)ncols * (size_t)nrows;
  float *ptrout = floatimg->data;

  /* Output image must be large enough to hold result */
  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend)  *ptrout++ = (float) *img++;
}



/*********************************************************************
 * _computeKernels
 */

static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;   /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));
	
    /* Compute gauss and deriv */
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;
			
    den = 0.0;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}
	

/*********************************************************************
 * _KLTGetKernelWidths
 *
 */

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}


/*********************************************************************
 * _convolveImageHoriz
 */

static void _convolveImageHoriz(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  float *ptrrow = imgin->data;           /* Points to row's first pixel */
  float *ptrout = imgout->data, /* Points to next output pixel */
    *ppp;
  float sum;
  int radius = kernel.width / 2;
  int ncols = imgin->ncols, nrows = imgin->nrows;
  int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* For each row, do ... */
  for (j = 0 ; j < nrows ; j++)  {

    /* Zero leftmost columns */
    for (i = 0 ; i < radius ; i++)
      *ptrout++ = 0.0;

    /* Convolve middle columns with kernel */
    for ( ; i < ncols - radius ; i++)  {
      ppp = ptrrow + i - radius;
      sum = 0.0;
      for (k = kernel.width-1 ; k >= 0 ; k--)
        sum += *ppp++ * kernel.data[k];
      *ptrout++ = sum;
    }

    /* Zero rightmost columns */
    for ( ; i < ncols ; i++)
      *ptrout++ = 0.0;

    ptrrow += ncols;
  }
}


/*********************************************************************
 * _convolveImageVert
 */

static void _convolveImageVert(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  float *ptrcol = imgin->data;            /* Points to row's first pixel */
  float *ptrout = imgout->data,  /* Points to next output pixel */
    *ppp;
  float sum;
  int radius = kernel.width / 2;
  int ncols = imgin->ncols, nrows = imgin->nrows;
  int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* For each column, do ... */
  for (i = 0 ; i < ncols ; i++)  {

    /* Zero topmost rows */
    for (j = 0 ; j < radius ; j++)  {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    /* Convolve middle rows with kernel */
    for ( ; j < nrows - radius ; j++)  {
      ppp = ptrcol + ncols * (j - radius);
      sum = 0.0;
      for (k = kernel.width-1 ; k >= 0 ; k--)  {
        sum += *ppp * kernel.data[k];
        ppp += ncols;
      }
      *ptrout = sum;
      ptrout += ncols;
    }

    /* Zero bottommost rows */
    for ( ; j < nrows ; j++)  {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    ptrcol++;
    ptrout -= nrows * ncols - 1;
  }
}


/*********************************************************************
 * _convolveSeparate
 */

static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  /* Create temporary image */
  _KLT_FloatImage tmpimg;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
  
  /* Do convolution */
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);

  _convolveImageVert(tmpimg, vert_kernel, imgout);

  /* Free memory */
  _KLTFreeFloatImage(tmpimg);
}

	
/*********************************************************************
 * _KLTComputeGradients
 */

 // An image gradient is the pixel-wise rate of change of intensity in the x or y direction
void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
				
  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
	
  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);

}
	

/*********************************************************************
 * _KLTComputeSmoothedImage
 */

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}


// ------------------------------------------- v3.6 --------------------------------------

// Helper function to determine which sigma we're using
__host__ __device__ SigmaType getSigmaType(float sigma) {
    if (fabsf(sigma - 0.7f) < 0.05f) return SIGMA_07;
    else if (fabsf(sigma - 3.6f) < 0.05f) return SIGMA_36;
    else if (fabsf(sigma - 1.0f) < 0.05f) return SIGMA_10;
    else return SIGMA_OTHER;
}

__device__ const float* getKernelPtr(KernelType kernel_type, SigmaType sigma_type) {
    if (kernel_type == KERNEL_GAUSSIAN) {
        if (sigma_type == SIGMA_07) return &c_gauss_kernel_07[0];
        else if (sigma_type == SIGMA_36) return &c_gauss_kernel_36[0];
        else if (sigma_type == SIGMA_10) return &c_gauss_kernel_10[0];
        else return &c_gauss_kernel_10[0];  // Fallback to sigma 1.0
    } else {
        if (sigma_type == SIGMA_07) return &c_gaussderiv_kernel_07[0];
        else if (sigma_type == SIGMA_36) return &c_gaussderiv_kernel_36[0];
        else if (sigma_type == SIGMA_10) return &c_gaussderiv_kernel_10[0];
        else return &c_gaussderiv_kernel_10[0];  // Fallback to sigma 1.0
    }
}

__global__ void convolveHorizKernel(
    const float *imgin, float *imgout,
    int ncols, int nrows,
    int kwidth,
    KernelType kernel_type,
    float sigma)
{
    extern __shared__ float s_data[];

     SigmaType sigma_type;
    if (fabsf(sigma - 0.7f) < 0.05f) sigma_type = SIGMA_07;
    else if (fabsf(sigma - 3.6f) < 0.05f) sigma_type = SIGMA_36;
    else if (fabsf(sigma - 1.0f) < 0.05f) sigma_type = SIGMA_10;
    else sigma_type = SIGMA_10;  // Fallback
    const float *kernel = getKernelPtr(kernel_type,sigma_type);
    int radius = kwidth / 2;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate shared memory dimensions (block + halo regions)
    int shared_width = blockDim.x + 2 * radius;
    int shared_idx = threadIdx.y * shared_width + threadIdx.x;
    
    // Load main data to shared memory
    if (x < ncols && y < nrows) {
        s_data[shared_idx + radius] = imgin[y * ncols + x];
    }
    
    // Load left halo
    if (threadIdx.x < radius && x >= radius) {
        int left_x = x - radius;
        s_data[shared_idx] = imgin[y * ncols + left_x];
    }
    
    // Load right halo  
    if (threadIdx.x >= blockDim.x - radius && x < ncols - radius) {
        int right_x = x + radius;
        s_data[shared_idx + 2 * radius] = imgin[y * ncols + right_x];
    }
    
    __syncthreads();
    
    if (x >= ncols || y >= nrows) return;
    
    // Zero border handling
    if (x < radius || x >= ncols - radius) {
        imgout[y * ncols + x] = 0.0f;
        return;
    }
    
    // Convolve from shared memory
    float sum = 0.0f;
    for (int k = 0; k < kwidth; k++) {
        sum += s_data[shared_idx + k] * kernel[kwidth - 1 - k];
    }
    imgout[y * ncols + x] = sum;
}

__global__ void convolveVertKernel(
    const float *imgin, float *imgout,
    int ncols, int nrows,
    int kwidth,
    KernelType kernel_type,
    float sigma
)
{
    extern __shared__ float s_data[];

     SigmaType sigma_type;
    if (fabsf(sigma - 0.7f) < 0.05f) sigma_type = SIGMA_07;
    else if (fabsf(sigma - 3.6f) < 0.05f) sigma_type = SIGMA_36;
    else if (fabsf(sigma - 1.0f) < 0.05f) sigma_type = SIGMA_10;
    else sigma_type = SIGMA_10;  // Fallback
    const float *kernel = getKernelPtr(kernel_type,sigma_type);
    int radius = kwidth / 2;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate shared memory dimensions
    int shared_height = blockDim.y + 2 * radius;
    int shared_idx = threadIdx.x * shared_height + threadIdx.y;
    
    // Load main data to shared memory
    if (x < ncols && y < nrows) {
        s_data[shared_idx + radius] = imgin[y * ncols + x];
    }
    
    // Load top halo
    if (threadIdx.y < radius && y >= radius) {
        int top_y = y - radius;
        s_data[shared_idx] = imgin[top_y * ncols + x];
    }
    
    // Load bottom halo
    if (threadIdx.y >= blockDim.y - radius && y < nrows - radius) {
        int bottom_y = y + radius;
        s_data[shared_idx + 2 * radius] = imgin[bottom_y * ncols + x];
    }
    
    __syncthreads();
    
    if (x >= ncols || y >= nrows) return;
    
    // Zero border handling
    if (y < radius || y >= nrows - radius) {
        imgout[y * ncols + x] = 0.0f;
        return;
    }
    
    // Convolve from shared memory
    float sum = 0.0f;
    for (int k = 0; k < kwidth; k++) {
        sum += s_data[shared_idx + k] * kernel[kwidth - 1 - k];
    }
    imgout[y * ncols + x] = sum;
}

__host__ void convolveImageHorizCUDA(
  float *d_imgin,
  float *d_imgout,
  int ncols,
  int nrows,
  int kwidth,
  KernelType kernel_type,
  float sigma,
  cudaStream_t stream
)
{
  //assert(h_kernel.width % 2 == 1);
  assert(d_imgin != d_imgout);

  // Allocate and copy device kernel
  //float *d_kernel = NULL;
  //size_t kernel_bytes = (size_t)kwidth * sizeof(float);
  //CUDA_CHECK(cudaMalloc((void**)&d_kernel, kernel_bytes));
  //CUDA_CHECK(cudaMemcpyAsync(d_kernel, h_kernel.data, kernel_bytes, cudaMemcpyHostToDevice,stream));

  // Launch configuration
  dim3 blockSize(16, 16);  // You can tune this (32x8, 16x16, 32x4)
  dim3 gridSize((ncols + blockSize.x - 1)/blockSize.x, 
                (nrows + blockSize.y - 1)/blockSize.y);

  // Calculate required shared memory
  int radius = kwidth / 2;
  int shared_width = blockSize.x + 2 * radius;
  size_t shared_mem_size = (blockSize.y * shared_width) * sizeof(float);

  // Launch kernel with shared memory
  convolveHorizKernel<<<gridSize, blockSize, shared_mem_size,stream>>>(
      d_imgin, d_imgout, ncols, nrows, kwidth, kernel_type, sigma);
  
  CUDA_CHECK(cudaGetLastError());
  //CUDA_CHECK(cudaDeviceSynchronize());

  // Free device memory
  //cudaFree(d_kernel);
}

__host__ void convolveImageVertCUDA(
  float *d_imgin,
  float *d_imgout,
  int ncols,
  int nrows,
  int kwidth,
  KernelType kernel_type,
  float sigma,
  cudaStream_t stream
)
{
  //assert(h_kernel.width % 2 == 1);
  assert(d_imgin != d_imgout);

  // Allocate and copy device kernel
  //float *d_kernel = NULL;
  //size_t kernel_bytes = (size_t)kwidth * sizeof(float);
  //CUDA_CHECK(cudaMalloc((void**)&d_kernel, kernel_bytes));
  //CUDA_CHECK(cudaMemcpyAsync(d_kernel, h_kernel.data, kernel_bytes, cudaMemcpyHostToDevice,stream));

  // Launch configuration
  dim3 blockSize(16, 16);
  dim3 gridSize((ncols + blockSize.x - 1)/blockSize.x, 
                (nrows + blockSize.y - 1)/blockSize.y);

  // Calculate required shared memory (different for vertical)
  int radius = kwidth / 2;
  int shared_height = blockSize.y + 2 * radius;
  size_t shared_mem_size = (blockSize.x * shared_height) * sizeof(float);

  // Launch kernel with shared memory
  convolveVertKernel<<<gridSize, blockSize, shared_mem_size,stream>>>(
      d_imgin, d_imgout, ncols, nrows, kwidth, kernel_type, sigma);
  
  CUDA_CHECK(cudaGetLastError());
  //CUDA_CHECK(cudaDeviceSynchronize());

  // Free device memory
  //cudaFree(d_kernel);
}

__host__ void convolveSeparateCUDA(
  float *d_imgin,
  int horiz_kwidth,
  KernelType horiz_kernel_type,
  int vert_kwidth,
  KernelType vert_kernel_type,
  float sigma,
  float *d_imgout,
  int ncols,
  int nrows,
  cudaStream_t stream
)
{
   
  // Create temporary image
  float *d_tmpimg = NULL;
  size_t img_bytes = (size_t)ncols * (size_t)nrows * sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&d_tmpimg, img_bytes));
  CUDA_CHECK(cudaMemsetAsync(d_tmpimg, 0, img_bytes,stream));

  // Use shared memory versions
  convolveImageHorizCUDA(d_imgin, d_tmpimg, ncols, nrows, horiz_kwidth, horiz_kernel_type, sigma, stream);
  convolveImageVertCUDA(d_tmpimg, d_imgout, ncols, nrows, vert_kwidth, vert_kernel_type, sigma, stream);

  // Free memory
  cudaFree(d_tmpimg);
}

__host__ void computeSmoothedImageCUDA(
  float *d_img,
  float sigma,
  float *d_smooth_img,
  int ncols,
  int nrows,
  cudaStream_t stream
)
{
  assert(d_smooth_img != NULL);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  /*
  if (fabsf(sigma - sigma_last) > 0.05f)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  */

  SigmaType sigma_type = getSigmaType(sigma);
  if (fabsf(sigma - sigma_last) > 0.05f) {
      if (sigma_type == SIGMA_OTHER) {
          // Only recompute for non-precomputed sigmas
          computeKernelsConstant(sigma);
      } else {
          // For precomputed sigmas, just update widths
          ConvolutionKernel gauss, gaussderiv;
          _computeKernels(sigma, &gauss, &gaussderiv);
          gauss_width = gauss.width;
          gaussderiv_width = gaussderiv.width;
          sigma_last = sigma;
          //printf("Using precomputed kernels for sigma %f\n", sigma);
      }
  }

  convolveSeparateCUDA(d_img, gauss_width, KERNEL_GAUSSIAN, gauss_width, KERNEL_GAUSSIAN, 
                       sigma, d_smooth_img, ncols, nrows, stream);
}

__host__ void computeGradientsCUDA(
  float *d_img,
  float sigma,
  float *d_gradx,
  float *d_grady,
  int ncols,
  int nrows,
  cudaStream_t stream
)
{
  /* Compute kernels, if necessary */
  /*
  if (fabsf(sigma - sigma_last) > 0.05f)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  */
  
  SigmaType sigma_type = getSigmaType(sigma);
  if (fabsf(sigma - sigma_last) > 0.05f) {
      if (sigma_type == SIGMA_OTHER) {
          // Only recompute for non-precomputed sigmas
          computeKernelsConstant(sigma);
      } else {
          // For precomputed sigmas, just update widths
          ConvolutionKernel gauss, gaussderiv;
          _computeKernels(sigma, &gauss, &gaussderiv);
          gauss_width = gauss.width;
          gaussderiv_width = gaussderiv.width;
          sigma_last = sigma;
          //printf("Using precomputed kernels for sigma %f\n", sigma);
      }
  }

  convolveSeparateCUDA(d_img, gaussderiv_width, KERNEL_GAUSSIAN_DERIV,
                       gauss_width, KERNEL_GAUSSIAN, sigma, d_gradx, ncols, nrows,stream);
  convolveSeparateCUDA(d_img, gauss_width, KERNEL_GAUSSIAN,
                       gaussderiv_width, KERNEL_GAUSSIAN_DERIV, sigma, d_grady, ncols, nrows,stream);
}

__global__ void subsampleKernel(
    const float *d_src, float *d_dst,
    int src_ncols, int src_nrows,
    int dst_ncols, int dst_nrows,
    int subsampling, int subhalf)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dst_ncols && y < dst_nrows) {
        int src_x = subsampling * x + subhalf;
        int src_y = subsampling * y + subhalf;
        d_dst[y * dst_ncols + x] = d_src[src_y * src_ncols + src_x];
    }
}

__host__ void computePyramidCUDA(
  float *d_img,
  float *d_pyramid,
  float sigma_fact,
  int ncols,
  int nrows,
  int subsampling,
  int nLevels,
  cudaStream_t stream
)
{
  int subhalf = subsampling / 2;
  float sigma = subsampling * sigma_fact;

  if (subsampling != 2 && subsampling != 4 &&
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTComputePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  // Copy input image to level 0 of pyramid
  CUDA_CHECK(cudaMemcpyAsync(d_pyramid, d_img, ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice,stream));

  int current_ncols = ncols;
  int current_nrows = nrows;
  size_t current_offset = 0;

  float *d_smooth = NULL;
  CUDA_CHECK(cudaMalloc(&d_smooth, current_ncols * current_nrows * sizeof(float)));

  for (int i = 1; i < nLevels; ++i) {
    size_t next_offset = current_offset + (current_ncols * current_nrows);

    // Smooth current level using shared memory optimization
    computeSmoothedImageCUDA(d_pyramid + current_offset, sigma, d_smooth, current_ncols, current_nrows, stream);

    // Get new dimensions
    int next_ncols = current_ncols / subsampling;
    int next_nrows = current_nrows / subsampling;

    // Subsample smoothed image into next pyramid level
    dim3 blockSize(16, 16);
    dim3 gridSize((next_ncols + blockSize.x - 1) / blockSize.x,
                  (next_nrows + blockSize.y - 1) / blockSize.y);
    subsampleKernel<<<gridSize, blockSize, 0, stream>>>(
        d_smooth, d_pyramid + next_offset,
        current_ncols, current_nrows,
        next_ncols, next_nrows,
        subsampling, subhalf);
    CUDA_CHECK(cudaGetLastError());

    // Prepare for next level
    current_offset = next_offset;
    current_ncols = next_ncols;
    current_nrows = next_nrows;
  }

  CUDA_CHECK(cudaFree(d_smooth));
}


//////////////////// ------------ v3.8 ----------------- ////////////////////////
__host__ void computeKernelsConstant(float sigma) {
    
    // Only compute and upload if it's NOT one of our precomputed sigmas
    SigmaType sigma_type = getSigmaType(sigma);
    
    if (sigma_type != SIGMA_OTHER) {
        // This is one of our precomputed sigmas - no need to recompute!
        printf("Sigma %f is precomputed, using constant memory\n", sigma);
        
        // Just update the global widths for this sigma
        ConvolutionKernel gauss, gaussderiv;
        _computeKernels(sigma, &gauss, &gaussderiv);
        gauss_width = gauss.width;
        gaussderiv_width = gaussderiv.width;
        sigma_last = sigma;
        return;
    }
    
    // Only reach here for non-precomputed sigmas
    //printf("Computing kernels for non-precomputed sigma: %f\n", sigma);
    
    ConvolutionKernel gauss, gaussderiv;
    _computeKernels(sigma, &gauss, &gaussderiv);
    
    // Use the sigma 1.0 slots as fallback for other sigma values
    // MAYBE MAKE ANOTHER ONE MAYBE
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss_kernel_10, gauss.data, gauss.width * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_gaussderiv_kernel_10, gaussderiv.data, gaussderiv.width * sizeof(float)));
    
    // Update global widths
    gauss_width = gauss.width;
    gaussderiv_width = gaussderiv.width;
    sigma_last = sigma;
}

__host__ void initializePrecomputedKernels() {
    //printf("Precomputing kernels for common sigma values...\n");
    
    // Precompute and upload the three main sigmas
    ConvolutionKernel gauss, gaussderiv;
    
    // Sigma 0.7
    _computeKernels(0.7f, &gauss, &gaussderiv);
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss_kernel_07, gauss.data, gauss.width * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_gaussderiv_kernel_07, gaussderiv.data, gaussderiv.width * sizeof(float)));
    printf("Precomputed kernels for sigma 0.7 (widths: %d, %d)\n", gauss.width, gaussderiv.width);
    //
    // Sigma 3.6
    _computeKernels(3.6f, &gauss, &gaussderiv);
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss_kernel_36, gauss.data, gauss.width * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_gaussderiv_kernel_36, gaussderiv.data, gaussderiv.width * sizeof(float)));
    //printf("Precomputed kernels for sigma 3.6 (widths: %d, %d)\n", gauss.width, gaussderiv.width);
    
    // Sigma 1.0
    _computeKernels(1.0f, &gauss, &gaussderiv);
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss_kernel_10, gauss.data, gauss.width * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_gaussderiv_kernel_10, gaussderiv.data, gaussderiv.width * sizeof(float)));
    //printf("Precomputed kernels for sigma 1.0 (widths: %d, %d)\n", gauss.width, gaussderiv.width);
    
    //printf("Precomputation complete.\n");
}