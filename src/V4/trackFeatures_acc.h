#ifndef _TRACKFEATURES_ACC_H_
#define _TRACKFEATURES_ACC_H_

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* Forward declarations */
#include "klt.h"

void initializePyramidMetadata(int base_ncols, int base_nrows, int subsampling, int nLevels);
void freePyramidMetadata();

void initializeOpenACCBuffers(int ncols, int nrows, int subsampling, int nLevels);

void KLTTrackFeaturesACC(
  KLT_TrackingContext tc,
  KLT_PixelType *img1,
  KLT_PixelType *img2,
  int ncols,
  int nrows,
  KLT_FeatureList featurelist);

void cleanupOpenACCResources();

float* getPyramidLevel(int level);
size_t getPyramidTotalSize();
int getPyramidLevelCols(int level);
int getPyramidLevelRows(int level);

#endif 


