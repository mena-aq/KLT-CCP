#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>
#include "pnmio.h"
#include "klt.h"
#include "trackFeatures_cuda.h"
#include <cuda_runtime.h>

// Simple thread structure for async file reading
typedef struct {
    char filename[100];
    unsigned char* buffer;
    int ncols, nrows;
    int done;
} file_reader_t;

void* read_file_thread(void* arg) {
    file_reader_t* reader = (file_reader_t*)arg;
    unsigned char* temp = pgmReadFile(reader->filename, NULL, &reader->ncols, &reader->nrows);
    if (temp) {
        memcpy(reader->buffer, temp, reader->ncols * reader->nrows);
        free(temp);
    }
    reader->done = 1;
    return NULL;
}

int count_image_files(const char* folder_path) {
    DIR *dir;
    struct dirent *entry;
    int count = 0;
    dir = opendir(folder_path);
    if (dir == NULL) return -1;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "img", 3) == 0 && strstr(entry->d_name, ".pgm") != NULL) {
            count++;
        }
    }
    closedir(dir);
    return count;
}

static cudaEvent_t start_event = NULL;
static cudaEvent_t stop_event = NULL;

int main(int argc, char *argv[]) {
    unsigned char *img1, *img2, *img3;
    char fname[100], fnameout[100];
    char dataset_folder[200] = "/kaggle/input/dataset3/dataset3";
    char output_folder[200] = "output";

    KLT_TrackingContext tc;
    KLT_FeatureList fl;
    KLT_FeatureTable ft;
    int nFeatures = 150, nFrames;
    int ncols, nrows;
    int i;

    mkdir(output_folder, 0755);
    if (argc > 1) strcpy(dataset_folder, argv[1]);

    nFrames = count_image_files(dataset_folder);
    if (nFrames <= 0) {
        printf("Error: No image files found in %s\n", dataset_folder);
        return -1;
    }
    printf("Found %d image files in %s\n", nFrames, dataset_folder);

    tc = KLTCreateTrackingContext();
    fl = KLTCreateFeatureList(nFeatures);
    ft = KLTCreateFeatureTable(nFrames, nFeatures);
    tc->sequentialMode = TRUE;

    // Load initial frames
    sprintf(fname, "%s/img0.pgm", dataset_folder);
    img1 = pgmReadFile(fname, NULL, &ncols, &nrows);
    sprintf(fname, "%s/img1.pgm", dataset_folder); 
    img2 = pgmReadFile(fname, NULL, &ncols, &nrows);
    img3 = (unsigned char *)malloc(ncols * nrows * sizeof(unsigned char));

    // Start async read of frame 2
    file_reader_t reader;
    sprintf(reader.filename, "%s/img2.pgm", dataset_folder);
    reader.buffer = img3;
    reader.done = 0;
    pthread_t thread;
    if (nFrames > 2) {
        pthread_create(&thread, NULL, read_file_thread, &reader);
    }

    KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
    KLTStoreFeatureList(fl, ft, 0);
    sprintf(fnameout, "%s/feat0.ppm", output_folder);
    KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, fnameout);

    if (!start_event) {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    allocateGPUResources(nFeatures, tc, ncols, nrows);
    cudaEventRecord(start_event, 0);

    for (i = 1; i < nFrames; i++) {
        // Wait for async read to complete if needed
        if (i == 1 && nFrames > 2) {
            pthread_join(thread, NULL);
        }

        // Start async read of NEXT next frame (i+2) if available
        file_reader_t next_reader;
        unsigned char* next_buffer = (unsigned char*)malloc(ncols * nrows);
        if (i + 2 < nFrames) {
            sprintf(next_reader.filename, "%s/img%d.pgm", dataset_folder, i+2);
            next_reader.buffer = next_buffer;
            next_reader.done = 0;
            pthread_create(&thread, NULL, read_file_thread, &next_reader);
        }

        // Process tracking - file reading happens in parallel with GPU work!
        kltTrackFeaturesCUDA(tc, img1, img2, img3, ncols, nrows, fl);

        #ifdef REPLACE
        KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
        #endif
        
        KLTStoreFeatureList(fl, ft, i);
        sprintf(fnameout, "%s/feat%d.ppm", output_folder, i);
        KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);

        // Rotate buffers
        free(img1);  // Free the oldest frame
        img1 = img2;
        img2 = img3;
        
        // Use the asynchronously loaded frame if available
        if (i + 2 < nFrames) {
            pthread_join(thread, NULL);  // Wait for the async read to finish
            img3 = next_reader.buffer;
        } else {
            img3 = next_buffer;  // Just use the allocated buffer
        }
    }

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start_event, stop_event);
    printf("GPU tracking time for %d frames: %f ms\n", nFrames-1, total_ms);
    printf("Average per frame: %f ms\n", total_ms / (nFrames-1));

    freeGPUResources();
    KLTWriteFeatureTable(ft, "features.txt", "%5.1f");
    KLTWriteFeatureTable(ft, "features.ft", NULL);

    KLTFreeFeatureTable(ft);
    KLTFreeFeatureList(fl);
    KLTFreeTrackingContext(tc);
    free(img1);
    free(img2);
    free(img3);

    return 0;
}