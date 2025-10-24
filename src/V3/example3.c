/**********************************************************************
This is the GPU version of example3.c from the KLT library.
It calls a cuda implementation of KLTTrackFeatures instead of the CPU version.
/********************************************************************/

/********************************************************************
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
#include "pnmio.h"
#include "klt.h"

#include "trackFeatures_cuda.h"
#include <cuda_runtime.h>


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

static cudaEvent_t start_event = NULL;
static cudaEvent_t stop_event = NULL;

#ifdef WIN32
int RunExample3()
#else
int main(int argc, char *argv[])
#endif
{
  unsigned char *img1, *img2;
  char fnamein[100], fnameout[100];
  char dataset_folder[200] = "dataset3";
  //char dataset_folder[200] = "/kaggle/input/dataset3/dataset3";
  char output_folder[200] = "output";

  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_FeatureTable ft;
  int nFeatures = 150, nFrames=10;
  int ncols, nrows;
  int i;

  // create output directory if it doesn't exist
  mkdir(output_folder, 0755);

  // check if dataset folder is provided as command-line argument
  if (argc > 1) {
    strcpy(dataset_folder, argv[1]);
  }

  // count the number of image files in the dataset folder
  nFrames = count_image_files(dataset_folder);
  if (nFrames <= 0) {
    printf("Error: No image files found in %s or cannot access folder\n", dataset_folder);
    return -1;
  }
  printf("Found %d image files in %s\n", nFrames, dataset_folder);

  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);
  ft = KLTCreateFeatureTable(nFrames, nFeatures);
  tc->sequentialMode = TRUE; // USES SEQUENTIAL MODE
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */
 
  sprintf(fnamein, "%s/img0.pgm", dataset_folder);
  img1 = pgmReadFile(fnamein, NULL, &ncols, &nrows);
  img2 = (unsigned char *) malloc(ncols*nrows*sizeof(unsigned char));

  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLTStoreFeatureList(fl, ft, 0);
  sprintf(fnameout, "%s/feat0.ppm", output_folder);
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, fnameout);

  // for each frame in the sequence... 

    // add cuda timings
  /*
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  */
  if (!start_event) {
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
  }

  allocateGPUResources(nFeatures, tc, ncols, nrows);

  cudaEventRecord(start_event, 0);
  for (i = 1 ; i < nFrames ; i++)  {
    sprintf(fnamein, "%s/img%d.pgm", dataset_folder, i);
    pgmReadFile(fnamein, img2, &ncols, &nrows);
    // track the features from img1 to img2 using CUDA implementation
    kltTrackFeaturesCUDA(tc, img1, img2, ncols, nrows, fl);

#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif
    KLTStoreFeatureList(fl, ft, i);
    sprintf(fnameout, "%s/feat%d.ppm", output_folder, i);
    KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);
  }

  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  
  float total_ms = 0;
  cudaEventElapsedTime(&total_ms, start_event, stop_event);
  printf("GPU tracking time for %d frames: %f ms\n", nFrames-1, total_ms);
  printf("Average per frame: %f ms\n", total_ms / (nFrames-1));

  freeGPUResources();
  /*
    cudaEventRecord(stop,0);   // Stop timing
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU tracking time for %d frames: %f ms\n", nFrames-1, milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  */
  KLTWriteFeatureTable(ft, "features.txt", "%5.1f");
  KLTWriteFeatureTable(ft, "features.ft", NULL);

  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(fl);
  KLTFreeTrackingContext(tc);
  free(img1);
  free(img2);

  return 0;
}