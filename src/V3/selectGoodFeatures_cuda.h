#ifndef _SELECTGOODFEATURES_CUDA_H_
#define _SELECTGOODFEATURES_CUDA_H_

#include "base.h"
#include "klt.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Selection mode enum */
typedef enum {SELECTING_ALL, REPLACING_SOME} selectionMode;

/* Main GPU-accelerated feature selection function */
void _KLTSelectGoodFeaturesCUDA(
    KLT_TrackingContext tc,
    KLT_PixelType *img, 
    int ncols, 
    int nrows,
    KLT_FeatureList featurelist,
    selectionMode mode);

/* Public wrapper functions (compatible with original KLT API) */
void KLTSelectGoodFeaturesCUDA(
    KLT_TrackingContext tc,
    KLT_PixelType *img, 
    int ncols, 
    int nrows,
    KLT_FeatureList featurelist);

void KLTReplaceLostFeaturesCUDA(
    KLT_TrackingContext tc,
    KLT_PixelType *img, 
    int ncols, 
    int nrows,
    KLT_FeatureList featurelist);

#ifdef __cplusplus
}
#endif

#endif /* _SELECTGOODFEATURES_CUDA_H_ */