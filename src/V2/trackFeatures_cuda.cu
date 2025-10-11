#include "trackFeatures_cuda.h"

// NOT DONE i just copy pasted from the doc 
// understand and modify as needed
// need to add iterations with threshold like in _trackFeatures
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
	int nrows)
{

  int featureIdx = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int hw = window_width / 2;
  int hh = window_height / 2;

  extern __shared__ float shared[];
  float* imgdiff = shared;
  float* gradx = imgdiff + window_width * window_height;
  float* grady = gradx + window_width * window_height;

  // Load feature position
  float x1 = d_feature_x[featureIdx];
  float y1 = d_feature_y[featureIdx];
  float x2 = x1, y2 = y1;

  // For each pyramid level (coarse to fine)
  for (int r = nPyramidLevels - 1; r >= 0; r--) {

    // ------------- add a loop to converge like in _trackFeatures ---------------------

    // Each thread computes its window pixel
    int i = threadIdx.x - hw;
    int j = threadIdx.y - hh;
    int idx = threadIdx.y * window_width + threadIdx.x;

    // intensity difference
    float g1 = interpolateDevice(x1 + i, y1 + j, d_pyramid1[r], ncols, nrows);
    float g2 = interpolateDevice(x2 + i, y2 + j, d_pyramid2[r], ncols, nrows);
    imgdiff[idx] = g1 - g2;

    // gradient sum
    float gx1 = interpolateDevice(x1 + i, y1 + j, d_pyramid1_gradx[r], ncols, nrows);
    float gx2 = interpolateDevice(x2 + i, y2 + j, d_pyramid2_gradx[r], ncols, nrows);
    gradx[idx] = gx1 + gx2;

    float gy1 = interpolateDevice(x1 + i, y1 + j, d_pyramid1_grady[r], ncols, nrows);
    float gy2 = interpolateDevice(x2 + i, y2 + j, d_pyramid2_grady[r], ncols, nrows);
    grady[idx] = gy1 + gy2;

    __syncthreads();

    // Reduction for matrix and error vector (simplified, use warp reduction for performance)
    // compute2by2gradientmatrix,compute2by1errormatrix
    float gxx = 0, gxy = 0, gyy = 0, ex = 0, ey = 0;
    if (tid == 0) {
        for (int k = 0; k < window_width * window_height; ++k) {
            float gx = gradx[k];
            float gy = grady[k];
            float diff = imgdiff[k];
            gxx += gx * gx;
            gxy += gx * gy;
            gyy += gy * gy;
            ex += diff * gx;
            ey += diff * gy;
        }
        // Solve for dx, dy (as in _solveEquation)
        float det = gxx * gyy - gxy * gxy;
        float dx = 0, dy = 0;
        if (det > 1e-6) {
            dx = (gyy * ex - gxy * ey) / det;
            dy = (gxx * ey - gxy * ex) / det;
            x2 += dx;
            y2 += dy;
        }
        // Write back new position
        d_feature_x_out[featureIdx] = x2;
        d_feature_y_out[featureIdx] = y2;
    }
    __syncthreads();
  }
};

// NOT DONE YET DOESNT LAUNCH KERNELS
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
  KLT_FeatureList *d_fl)
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
			KLTCountRemainingFeatures(featurelist), ncols, nrows);
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
		_KLTToFloatImage(img1, ncols, nrows, tmpimg);

    // ------------- convolve_seperate with Gaussian kernel (blur) ---------------
		_KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(h_tc), floatimg1);
    //---------------------------------------------------------------------------

		pyramid1 = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);

		// ------------- calls ComputeSmoothedImage at each level which calls convolve_seperate ---------------
    _KLTComputePyramid(floatimg1, pyramid1, h_tc->pyramid_sigma_fact);
    // --------------------------------------------------------------------------

		pyramid1_gradx = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);
		pyramid1_grady = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);

    // ------ calls convolve_seperate for both gradx and grady for each level ---------
		for (i = 0 ; i < h_tc->nPyramidLevels ; i++)
			_KLTComputeGradients(pyramid1->img[i], h_tc->grad_sigma, 
			pyramid1_gradx->img[i],
			pyramid1_grady->img[i]);
    // --------------------------------------------------------------------------
    
	}

	/* Do the same thing with second image */
	floatimg2 = _KLTCreateFloatImage(ncols, nrows);
	_KLTToFloatImage(img2, ncols, nrows, tmpimg);

  // ------------- convolve_seperate with Gaussian kernel (blur) ---------------
	_KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(h_tc), floatimg2);
	// --------------------------------------------------------------------------

  pyramid2 = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);
	
  // ------------- calls ComputeSmoothedImage at each level which calls convolve_seperate ---------------
  _KLTComputePyramid(floatimg2, pyramid2, h_tc->pyramid_sigma_fact);
	// --------------------------------------------------------------------------

  pyramid2_gradx = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);
	pyramid2_grady = _KLTCreatePyramid(ncols, nrows, (int) subsampling, h_tc->nPyramidLevels);
	
  // ------ calls convolve_seperate for both gradx and grady for each level ---------
  for (i = 0 ; i < h_tc->nPyramidLevels ; i++)
		_KLTComputeGradients(pyramid2->img[i], h_tc->grad_sigma, 
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

  ////////////////////////////////////// in kernel ///////////////////////////////////////
  // ----- for understanding Feature struct go to klt.h ----------------
	/* For each feature, do ... */
	for (indx = 0 ; indx < featurelist->nFeatures ; indx++)  {

		/* Only track features that are not lost */

    // features in parallel but pyramid levels sequential
		if (featurelist->feature[indx]->val >= 0)  {

			xloc = featurelist->feature[indx]->x;
			yloc = featurelist->feature[indx]->y;

			/* Transform location to coarsest resolution */
			for (r = h_tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
				xloc /= subsampling;  yloc /= subsampling;
			}
			xlocout = xloc;  ylocout = yloc;

			/* Beginning with coarsest resolution, do ... */
			for (r = h_tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {

				/* Track feature at current resolution */
				xloc *= subsampling;  yloc *= subsampling;
				xlocout *= subsampling;  ylocout *= subsampling;

        /// ------------ calls interpolate ---------------
        // --- called nLevels times for each feature ----
				val = _trackFeature(xloc, yloc, 
					&xlocout, &ylocout, // position in second image
					pyramid1->img[r], 
					pyramid1_gradx->img[r], pyramid1_grady->img[r], 
					pyramid2->img[r], 
					pyramid2_gradx->img[r], pyramid2_grady->img[r],
					h_tc->window_width, h_tc->window_height,
					h_tc->step_factor,
					h_tc->max_iterations,
					h_tc->min_determinant,
					h_tc->min_displacement,
					h_tc->max_residue,
					h_tc->lighting_insensitive);

				if (val==KLT_SMALL_DET || val==KLT_OOB)
					break;
			}

			/* Record feature */
			if (val == KLT_OOB) {
				featurelist->feature[indx]->x   = -1.0;
				featurelist->feature[indx]->y   = -1.0;
				featurelist->feature[indx]->val = KLT_OOB;
				if( featurelist->feature[indx]->aff_img ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
				if( featurelist->feature[indx]->aff_img_gradx ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
				if( featurelist->feature[indx]->aff_img_grady ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
				featurelist->feature[indx]->aff_img = NULL;
				featurelist->feature[indx]->aff_img_gradx = NULL;
				featurelist->feature[indx]->aff_img_grady = NULL;

			} else if (_outOfBounds(xlocout, ylocout, ncols, nrows, h_tc->borderx, h_tc->bordery))  {
				featurelist->feature[indx]->x   = -1.0;
				featurelist->feature[indx]->y   = -1.0;
				featurelist->feature[indx]->val = KLT_OOB;
				if( featurelist->feature[indx]->aff_img ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
				if( featurelist->feature[indx]->aff_img_gradx ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
				if( featurelist->feature[indx]->aff_img_grady ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
				featurelist->feature[indx]->aff_img = NULL;
				featurelist->feature[indx]->aff_img_gradx = NULL;
				featurelist->feature[indx]->aff_img_grady = NULL;
			} else if (val == KLT_SMALL_DET)  {
				featurelist->feature[indx]->x   = -1.0;
				featurelist->feature[indx]->y   = -1.0;
				featurelist->feature[indx]->val = KLT_SMALL_DET;
				if( featurelist->feature[indx]->aff_img ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
				if( featurelist->feature[indx]->aff_img_gradx ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
				if( featurelist->feature[indx]->aff_img_grady ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
				featurelist->feature[indx]->aff_img = NULL;
				featurelist->feature[indx]->aff_img_gradx = NULL;
				featurelist->feature[indx]->aff_img_grady = NULL;
			} else if (val == KLT_LARGE_RESIDUE)  {
				featurelist->feature[indx]->x   = -1.0;
				featurelist->feature[indx]->y   = -1.0;
				featurelist->feature[indx]->val = KLT_LARGE_RESIDUE;
				if( featurelist->feature[indx]->aff_img ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
				if( featurelist->feature[indx]->aff_img_gradx ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
				if( featurelist->feature[indx]->aff_img_grady ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
				featurelist->feature[indx]->aff_img = NULL;
				featurelist->feature[indx]->aff_img_gradx = NULL;
				featurelist->feature[indx]->aff_img_grady = NULL;
			} else if (val == KLT_MAX_ITERATIONS)  {
				featurelist->feature[indx]->x   = -1.0;
				featurelist->feature[indx]->y   = -1.0;
				featurelist->feature[indx]->val = KLT_MAX_ITERATIONS;
				if( featurelist->feature[indx]->aff_img ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
				if( featurelist->feature[indx]->aff_img_gradx ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
				if( featurelist->feature[indx]->aff_img_grady ) _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
				featurelist->feature[indx]->aff_img = NULL;
				featurelist->feature[indx]->aff_img_gradx = NULL;
				featurelist->feature[indx]->aff_img_grady = NULL;
			} else  {
				featurelist->feature[indx]->x = xlocout;
				featurelist->feature[indx]->y = ylocout;
				featurelist->feature[indx]->val = KLT_TRACKED;
				if (h_tc->affineConsistencyCheck >= 0 && val == KLT_TRACKED)  { /*for affine mapping*/
					int border = 2; /* add border for interpolation */

#ifdef DEBUG_AFFINE_MAPPING	  
					glob_index = indx;
#endif

					if(!featurelist->feature[indx]->aff_img){
						/* save image and gradient for each feature at finest resolution after first successful track */
						featurelist->feature[indx]->aff_img = _KLTCreateFloatImage((h_tc->affine_window_width+border), (h_tc->affine_window_height+border));
						featurelist->feature[indx]->aff_img_gradx = _KLTCreateFloatImage((h_tc->affine_window_width+border), (h_tc->affine_window_height+border));
						featurelist->feature[indx]->aff_img_grady = _KLTCreateFloatImage((h_tc->affine_window_width+border), (h_tc->affine_window_height+border));
						_am_getSubFloatImage(pyramid1->img[0],xloc,yloc,featurelist->feature[indx]->aff_img);
						_am_getSubFloatImage(pyramid1_gradx->img[0],xloc,yloc,featurelist->feature[indx]->aff_img_gradx);
						_am_getSubFloatImage(pyramid1_grady->img[0],xloc,yloc,featurelist->feature[indx]->aff_img_grady);
						featurelist->feature[indx]->aff_x = xloc - (int) xloc + (h_tc->affine_window_width+border)/2;
						featurelist->feature[indx]->aff_y = yloc - (int) yloc + (h_tc->affine_window_height+border)/2;;
					}else{
						/* affine tracking */
						val = _am_trackFeatureAffine(featurelist->feature[indx]->aff_x, featurelist->feature[indx]->aff_y,
							&xlocout, &ylocout,
							featurelist->feature[indx]->aff_img, 
							featurelist->feature[indx]->aff_img_gradx, 
							featurelist->feature[indx]->aff_img_grady,
							pyramid2->img[0], 
							pyramid2_gradx->img[0], pyramid2_grady->img[0],
							h_tc->affine_window_width, h_tc->affine_window_height,
							h_tc->step_factor,
							h_tc->affine_max_iterations,
							h_tc->min_determinant,
							h_tc->min_displacement,
							h_tc->affine_min_displacement,
							h_tc->affine_max_residue, 
							h_tc->lighting_insensitive,
							h_tc->affineConsistencyCheck,
							h_tc->affine_max_displacement_differ,
							&featurelist->feature[indx]->aff_Axx,
							&featurelist->feature[indx]->aff_Ayx,
							&featurelist->feature[indx]->aff_Axy,
							&featurelist->feature[indx]->aff_Ayy 
							);
						featurelist->feature[indx]->val = val;
						if(val != KLT_TRACKED){
							featurelist->feature[indx]->x   = -1.0;
							featurelist->feature[indx]->y   = -1.0;
							featurelist->feature[indx]->aff_x = -1.0;
							featurelist->feature[indx]->aff_y = -1.0;
							/* free image and gradient for lost feature */
							_KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
							_KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
							_KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
							featurelist->feature[indx]->aff_img = NULL;
							featurelist->feature[indx]->aff_img_gradx = NULL;
							featurelist->feature[indx]->aff_img_grady = NULL;
						}else{
							/*featurelist->feature[indx]->x = xlocout;*/
							/*featurelist->feature[indx]->y = ylocout;*/
						}
					}
				}

			}
		}
	}
  ////////////////////////////////////////////////////////////////////////////////////////

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
