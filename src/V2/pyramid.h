/*********************************************************************
 * pyramid.h
 *********************************************************************/

#ifndef _PYRAMID_H_
#define _PYRAMID_H_

#include "klt_util.h"

// pyramid basically contains multiple images at different scales
// each level is a downsampled version of the previous level
// subsampling is the factor by which the image is downsampled at each level
// ncols and nrows are arrays containing the dimensions of each level
typedef struct  {
  int subsampling;
  int nLevels;
  _KLT_FloatImage *img;
  int *ncols, *nrows;
}  _KLT_PyramidRec, *_KLT_Pyramid;


_KLT_Pyramid _KLTCreatePyramid(
  int ncols,
  int nrows,
  int subsampling,
  int nlevels);

void _KLTComputePyramid(
  _KLT_FloatImage floatimg, 
  _KLT_Pyramid pyramid,
  float sigma_fact);

void _KLTFreePyramid(
  _KLT_Pyramid pyramid);

#endif
