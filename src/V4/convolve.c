/*********************************************************************
 * convolve.c
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>   /* malloc(), realloc() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"   /* printing */

#define MAX_KERNEL_WIDTH 	71


typedef struct  {
  int width;
  float data[MAX_KERNEL_WIDTH];
}  ConvolutionKernel;

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;


/*********************************************************************
 * _KLTToFloatImage
 *
 * Given a pointer to image data (probably unsigned chars), copy
 * data to a float image.
 */

void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + ncols*nrows;
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
  //float *ptrrow = imgin->data;           /* Points to row's first pixel */
  //register float *ptrout = imgout->data, /* Points to next output pixel */
  //  *ppp;
  register float sum;
  register int radius = kernel.width / 2;
  register int ncols = imgin->ncols, nrows = imgin->nrows;
  int kwidth = kernel.width;
  register int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* For each row, do ... */
  #pragma acc parallel loop gang vector collapse(2) \
  present(imgin, imgout, kernel) 
  for (int j = 0 ; j < nrows ; j++)  {
    for (int i = 0 ; i < ncols ; i++)  {

      // Cache the current row segment in shared memory
      #pragma acc cache(imgin->data[j*ncols + max(0, i-radius) : min(ncols, i+radius+1) - max(0, i-radius)])

      if (i < radius || i >= ncols - radius) {
        /* Zero border columns */
        imgout->data[j * ncols + i] = 0.0;
      } else {
        /* Convolve middle columns with kernel */
        float sum = 0.0;
        int base_idx = j * ncols + i - radius;
        
        //#pragma acc loop seq
        #pragma acc loop vector reduction(+:sum)
        for (int k = 0; k < kwidth; k++) {
          sum += imgin->data[base_idx + k] * kernel.data[k];
        }
        imgout->data[j * ncols + i] = sum;
      }
    }
  }
  /*
  for (j = 0 ; j < nrows ; j++)  {

    ///* Zero leftmost columns
    for (i = 0 ; i < radius ; i++)
      *ptrout++ = 0.0;

    ///* Convolve middle columns with kernel 
    for ( ; i < ncols - radius ; i++)  {
      ppp = ptrrow + i - radius;
      sum = 0.0;
      for (k = kernel.width-1 ; k >= 0 ; k--)
        sum += *ppp++ * kernel.data[k];
      *ptrout++ = sum;
    }

    ///* Zero rightmost columns
    for ( ; i < ncols ; i++)
      *ptrout++ = 0.0;

    ptrrow += ncols;
  }
  */
}

#pragma acc routine vector
static void _convolveImageHorizFlat(
  float *imgin_data, int ncols, int nrows,
  float *kernel,
  int kwidth,
  float *imgout_data)
{
  int radius = kwidth / 2;
  /* Kernel width must be odd */
  assert(kwidth % 2 == 1);

  /* For each row, do ... */
  #pragma acc parallel loop gang vector collapse(2) \
  present(imgin_data[0:ncols*nrows], imgout_data[0:ncols*nrows], kernel) \
  copyin(kernel.data[0:kwidth])
  for (int j = 0 ; j < nrows ; j++)  {
    for (int i = 0 ; i < ncols ; i++)  {

      // Cache the current row segment in shared memory
      #pragma acc cache(imgin_data[j*ncols + max(0, i-radius) : min(ncols, i+radius+1) - max(0, i-radius)])

      if (i < radius || i >= ncols - radius) {
        /* Zero border columns */
        imgout_data[j * ncols + i] = 0.0;
      } else {
        /* Convolve middle columns with kernel */
        float sum = 0.0;
        int base_idx = j * ncols + i - radius;
        
        #pragma acc loop vector reduction(+:sum)
        for (int k = 0; k < kwidth; k++) {
          sum += imgin_data[base_idx + k] * kernel[k];
        }
        imgout_data[j * ncols + i] = sum;
      }
    }
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
  //float *ptrcol = imgin->data;            /* Points to row's first pixel */
  //register float *ptrout = imgout->data,  /* Points to next output pixel */
  //  *ppp;
  register float sum;
  register int radius = kernel.width / 2;
  register int ncols = imgin->ncols, nrows = imgin->nrows;
  int kwidth = kernel.width;
  register int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* For each column, do ... */
  #pragma acc parallel loop gang vector collapse(2) \
  present(imgin, imgout, kernel) 
  for (int i = 0 ; i < ncols ; i++)  {
    for (int j = 0 ; j < nrows ; j++)  {

      #pragma acc cache(imgin->data[max(0, j-radius)*ncols + i : \
                                   (min(nrows, j+radius+1) - max(0, j-radius)) : \
                                   ncols])
      if (j < radius || j >= nrows - radius) {
        /* Zero border rows */
        imgout->data[j * ncols + i] = 0.0;
      } else {
        /* Convolve middle rows with kernel */
        float sum = 0.0;
        int base_idx = (j - radius) * ncols + i;
        
        //#pragma acc loop seq
        #pragma acc loop vector reduction(+:sum)
        for (int k = 0; k < kwidth; k++) {
          sum += imgin->data[base_idx + k * ncols] * kernel.data[k];
        }
        imgout->data[j * ncols + i] = sum;
      }
    }
  }
  /*
  for (i = 0 ; i < ncols ; i++)  {

    /* Zero topmost rows 
    for (j = 0 ; j < radius ; j++)  {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    /* Convolve middle rows with kernel 
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

    /* Zero bottommost rows 
    for ( ; j < nrows ; j++)  {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    ptrcol++;
    ptrout -= nrows * ncols - 1;
  }
  */
}

#pragma acc routine vector
static void _convolveImageVertFlat(
  float *imgin_data, int ncols, int nrows,
  float *kernel,
  int kwidth,
  float *imgout_data)
{
  int radius = kwidth / 2;

  /* Kernel width must be odd */
  assert(kwidth % 2 == 1);

  /* For each column, do ... */
  #pragma acc parallel loop gang vector collapse(2) \
  present(imgin_data[0:ncols*nrows], imgout_data[0:ncols*nrows], kernel) \
  copyin(kernel[0:kwidth])
  for (int i = 0 ; i < ncols ; i++)  {
    for (int j = 0 ; j < nrows ; j++)  {

      #pragma acc cache(imgin_data[max(0, j-radius)*ncols + i : \
                                   (min(nrows, j+radius+1) - max(0, j-radius)) : \
                                   ncols])
      if (j < radius || j >= nrows - radius) {
        /* Zero border rows */
        imgout_data[j * ncols + i] = 0.0;
      } else {
        /* Convolve middle rows with kernel */
        float sum = 0.0;
        int base_idx = (j - radius) * ncols + i;
        
        #pragma acc loop vector reduction(+:sum)
        for (int k = 0; k < kwidth; k++) {
          sum += imgin_data[base_idx + k * ncols] * kernel[k];
        }
        imgout_data[j * ncols + i] = sum;
      }
    }
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

#pragma acc routine vector
static void _convolveSeparateFlat(
  float *imgin_data, int ncols, int nrows,
  float *horiz_kernel,
  int horiz_kwidth,
  float *vert_kernel,
  int vert_kwidth,
  float *imgout_data)
{
  /* Create temporary image as flat array */
  float *tmpimg_data = (float*)malloc(ncols * nrows * sizeof(float));
  
  #pragma acc data create(tmpimg_data[0:ncols*nrows])
  {
    /* Do convolution */
    _convolveImageHorizFlat(imgin_data, ncols, nrows, horiz_kernel, horiz_kwidth,  tmpimg_data);
    _convolveImageVertFlat(tmpimg_data, ncols, nrows, vert_kernel, vert_kwidth, imgout_data);
  }

  /* Free memory */
  free(tmpimg_data);
}
	
/*********************************************************************
 * _KLTComputeGradients
 */

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
	
  #pragma acc data present(img, gradx, grady) \
  copyin(gauss_kernel, gaussderiv_kernel)
  {
    _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
    _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
  }

}
	
#pragma acc routine vector
void _KLTComputeGradientsFlat(
  float *img_data, int img_ncols, int img_nrows,
  float sigma,
  float *gradx_data, 
  float *grady_data)
{
  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  
  #pragma acc data present(img_data[0:img_ncols*img_nrows], \
                          gradx_data[0:img_ncols*img_nrows], \
                          grady_data[0:img_ncols*img_nrows]) \
                copyin(gauss_kernel.data[0:gauss_kernel.width], gaussderiv_kernel.data[0:gaussderiv_kernel.width])
  {
    // For gradx: gaussderiv horizontally, gauss vertically
    _convolveSeparateFlat(img_data, img_ncols, img_nrows, 
                         gaussderiv_kernel.data, gaussderiv_kernel.width, 
                         gauss_kernel.data, gauss_kernel.width, 
                         gradx_data);
    // For grady: gauss horizontally, gaussderiv vertically  
    _convolveSeparateFlat(img_data, img_ncols, img_nrows,
                         gauss_kernel.data, gauss_kernel.width, 
                         gaussderiv_kernel.data, gaussderiv_kernel.width, 
                         grady_data);
  }
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

  #pragma acc data present(img, smooth) \
  copyin(gauss_kernel)
  {
    _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
  }
  
}

#pragma acc routine vector
void _KLTComputeSmoothedImageFlat(
  float *img_data, int ncols, int nrows,
  float sigma,
  float *smooth_data)
{
  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  #pragma acc data present(img_data[0:ncols*nrows], smooth_data[0:ncols*nrows]) \
                copyin(gauss_kernel.data[0:gauss_kernel.width])
  {
    // Smoothing: gauss horizontally, gauss vertically
    _convolveSeparateFlat(img_data, ncols, nrows, 
                         gauss_kernel.data, gauss_kernel.width,
                         gauss_kernel.data, gauss_kernel.width,
                         smooth_data);
  }
}


