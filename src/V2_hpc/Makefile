#####################################################################
# Compilers
CC = gcc
NVCC = nvcc

######################################################################
# Flags
FLAG1 = -DNDEBUG
# FLAG2 = -DKLT_USE_QSORT
CFLAGS = $(FLAG1) $(FLAG2)
NVCCFLAGS = -O3 -Wno-deprecated-gpu-targets -diag-suppress 2464
CUDA_LIB = -lcudart
CUDA_INCLUDE = -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/include
CUDA_LIBDIR = -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/targets/x86_64-linux/lib

######################################################################
# Source files
EXAMPLES = example1.c example2.c example3.c example4.c example5.c
CUDA_EXAMPLES = example3_cuda.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
CUDA_SRCS = convolve_cuda.cu trackFeatures_cuda.cu
OBJECTS = $(ARCH:.c=.o) $(CUDA_SRCS:.cu=.o)

.SUFFIXES: .c .cu .o

all: lib $(EXAMPLES:.c=) $(CUDA_EXAMPLES:.c=)

.c.o:
	$(CC) -c $(CFLAGS) $(CUDA_INCLUDE) $< -o $@

.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $(CUDA_INCLUDE) $< -o $@

lib: $(OBJECTS)
	rm -f libklt.a
	ar ruv libklt.a $(OBJECTS)
	ranlib libklt.a

example3: libklt.a
	$(CC) -O3 $(CFLAGS) $(CUDA_INCLUDE) -o $@ $@.c -L. -lklt $(CUDA_LIBDIR) -lm $(CUDA_LIB)

example3_cuda: libklt.a
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) -o $@ example3_cuda.cu -L. -lklt $(CUDA_LIBDIR) -lm $(CUDA_LIB)

run_example3: example3
	./example3

run_example3_cuda: example3_cuda
	./example3_cuda

depend:
	makedepend $(ARCH) $(EXAMPLES) $(CUDA_EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) $(CUDA_EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt gmon.out profile.txt callgraph.png

# Debug target to see variables
debug:
	@echo "OBJECTS: $(OBJECTS)"
	@echo "ARCH: $(ARCH)"
	@echo "EXAMPLES: $(EXAMPLES)"
	@echo "CUDA_EXAMPLES: $(CUDA_EXAMPLES)"
