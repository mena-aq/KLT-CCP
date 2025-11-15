/**********************************************************************
Finds the 150 best features in an image and tracks them through the 
next two images.  The sequential mode is set in order to speed
processing.  The features are stored in a feature table, which is then
saved to a text file; each feature list is also written to a PPM file.
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#include "pnmio.h"
#include "klt.h"

#include "trackFeatures_acc.h"  

// Function to count image files in dataset folder 
int count_image_files(const char* folder_path) {
  DIR *dir;
  struct dirent *entry;
  int count = 0;
  
  dir = opendir(folder_path);
  if (dir == NULL) {
    printf("Error: Cannot open directory %s\n", folder_path);
    return -1;
  }
  
  while ((entry = readdir(dir)) != NULL) {
    // Check if filename matches pattern img*.pgm
    if (strncmp(entry->d_name, "img", 3) == 0 && 
        strstr(entry->d_name, ".pgm") != NULL) {
      count++;
    }
  }
  
  closedir(dir);
  return count;
}

// Debug: Print what's defined
#ifdef KLT_USE_OPENACC
  #define USE_OPENACC 1
#else
  #define USE_OPENACC 0
#endif

#ifdef WIN32
int RunExample3()
#else
int main(int argc, char *argv[])
#endif
{

  printf("USE_OPENACC = %d\n", USE_OPENACC);
  printf("KLT_USE_OPENACC defined: %d\n", 
  #ifdef KLT_USE_OPENACC
      1
  #else
      0
  #endif
  );

  unsigned char *img1, *img2;
  char fnamein[100], fnameout[100];
  char dataset_folder[200] = "../../data/dataset3"; // default folder
  char output_folder[200] = "output";

  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_FeatureTable ft;
  int nFeatures = 150, nFrames;
  int ncols, nrows;
  int i;

  // create output directory if it doesn't exist
  mkdir(output_folder, 0755);

  // check if dataset folder is provided as command-line argument
  if (argc > 1) {
    sprintf(dataset_folder, "../../data/%s", argv[1]);
  }

  // count the number of image files in the dataset folder
  nFrames = count_image_files(dataset_folder);
  //for debug change nframe
  //nFrames = 10;

  if (nFrames <= 0) {
    printf("Error: No image files found in %s or cannot access folder\n", dataset_folder);
    return -1;
  }
  printf("Found %d image files in %s\n", nFrames, dataset_folder);

  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);
  ft = KLTCreateFeatureTable(nFrames, nFeatures);
  tc->sequentialMode = TRUE;
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */
 
  sprintf(fnamein, "%s/img0.pgm", dataset_folder);
  img1 = pgmReadFile(fnamein, NULL, &ncols, &nrows);
  img2 = (unsigned char *) malloc(ncols*nrows*sizeof(unsigned char));

  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLTStoreFeatureList(fl, ft, 0);
  sprintf(fnameout, "%s/feat0.ppm", output_folder);
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, fnameout);

  // Start timing the tracking loop
  clock_t start_time = clock();

#if USE_OPENACC
    initializeOpenACCBuffers(ncols,nrows,(int)tc->subsampling,tc->nPyramidLevels);
#endif
  
  for (i = 1 ; i < nFrames ; i++)  {
    sprintf(fnamein, "%s/img%d.pgm", dataset_folder, i);
    pgmReadFile(fnamein, img2, &ncols, &nrows);
#if USE_OPENACC
    KLTTrackFeaturesACC(tc, img1, img2, ncols, nrows, fl);
#else
    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
#endif

#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif
    KLTStoreFeatureList(fl, ft, i);
    sprintf(fnameout, "%s/feat%d.ppm", output_folder, i);
    KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);
  }

#if USE_OPENACC
  cleanupOpenACCResources();
#endif
  
  // End timing and calculate elapsed time
  clock_t end_time = clock();
  double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
  printf("Tracking time: %.3f ms\n", elapsed_time*1000);
  
  KLTWriteFeatureTable(ft, "features.txt", "%5.1f");
  KLTWriteFeatureTable(ft, "features.ft", NULL);

  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(fl);
  KLTFreeTrackingContext(tc);
  free(img1);
  free(img2);

  return 0;
}

