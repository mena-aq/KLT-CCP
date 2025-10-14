%%writefile Makefile

#####################################################################
# Compilers
NVCC = nvcc

######################################################################
# Flags
FLAG1 = -DNDEBUG
CFLAGS = $(FLAG1)
NVCCFLAGS = -O3 -Wno-deprecated-gpu-targets -diag-suppress 2464
CUDA_LIB = -lcudart
CUDA_INCLUDE = -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/include
CUDA_LIBDIR = -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/targets/x86_64-linux/lib

######################################################################
# Source files
EXAMPLES = example3.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
CUDA_SRCS = convolve_cuda.cu trackFeatures_cuda.cu
OBJECTS = $(ARCH:.c=.o) $(CUDA_SRCS:.cu=.o)

.SUFFIXES: .c .cu .o

all: lib example3

.c.o:
	$(NVCC) -c $(NVCCFLAGS) $(CUDA_INCLUDE) $< -o $@

.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $(CUDA_INCLUDE) $< -o $@

lib: $(OBJECTS)
	rm -f libklt.a
	ar ruv libklt.a $(OBJECTS)
	ranlib libklt.a

example3: libklt.a
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) -o $@ example3.c -L. -lklt $(CUDA_LIBDIR) -lm $(CUDA_LIB)

run: example3
	./example3

clean:
	rm -f *.o *.a example3 *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt gmon.out profile.txt callgraph.png