# include <cuda.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include "pnmio.h"
#include "klt.h"

#include "trackFeatures_cuda.h"

void KLTAllocateFeatureList(KLT_FeatureList** d_fl, int nFeatures){
  int nbytes = sizeof(KLT_FeatureListRec) +
    nFeatures * sizeof(KLT_Feature) +
    nFeatures * sizeof(KLT_FeatureRec);
  cudaMalloc((void**)d_fl, nbytes);
}

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

#ifdef WIN32
int RunExample3()
#else
int main(int argc, char *argv[])
#endif
{
  unsigned char *img1, *img2;
  char fnamein[100], fnameout[100];
  char dataset_folder[200] = "dataset3"; // default folder
  char output_folder[200] = "output";

  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_FeatureTable ft;
  int nFeatures = 550, nFrames;
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

  // allocate memory for feature list
  KLT_FeatureList* d_fl;
  KLTAllocateFeatureList(&d_fl, nFeatures);

  // allocate memory for tracking context on device & copy to device
  KLT_TrackingContext d_tc;
  cudaMalloc((void**)&d_tc, sizeof(KLT_TrackingContextRec));
  cudaMemcpy(d_tc, tc, sizeof(KLT_TrackingContextRec), cudaMemcpyHostToDevice);

  // allocate device memory for 2 sequential frames
  KLT_PixelType *d_img1, *d_img2;
  cudaMalloc((void**)&d_img1, ncols * nrows * sizeof(KLT_PixelType));
  cudaMalloc((void**)&d_img2, ncols * nrows * sizeof(KLT_PixelType));

  for (i = 1 ; i < nFrames ; i++)  {
    sprintf(fnamein, "%s/img%d.pgm", dataset_folder, i);
    pgmReadFile(fnamein, img2, &ncols, &nrows);

    // copy the 2 frames to track b/w to device
    cudaMemcpy(d_img1, img1_float, ncols * nrows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, img2_float, ncols * nrows * sizeof(float), cudaMemcpyHostToDevice);

    kltTrackFeaturesCUDA(
      tc, img1, img2, fl,
      d_tc, d_img1, d_img2, d_fl,
      ncols, nrows
    );
    // copy d_fl to fl
    cudaMemcpy(fl, d_fl, sizeof(KLT_FeatureListRec) + nFeatures * sizeof(KLT_Feature) + nFeatures * sizeof(KLT_FeatureRec), cudaMemcpyDeviceToHost);


#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif
    KLTStoreFeatureList(fl, ft, i);
    sprintf(fnameout, "%s/feat%d.ppm", output_folder, i);
    KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);
  }
  
  KLTWriteFeatureTable(ft, "features.txt", "%5.1f");
  KLTWriteFeatureTable(ft, "features.ft", NULL);

  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(fl);
  KLTFreeTrackingContext(tc);
  free(img1);
  free(img2);

  return 0;
}




