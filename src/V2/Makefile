######################################################################
# Choose your favorite C compiler
CC = gcc

######################################################################
# CUDA compiler
NVCC = nvcc

######################################################################
# -DNDEBUG prevents the assert() statements from being included in 
# the code.  If you are having problems running the code, you might 
# want to comment this line to see if an assert() statement fires.
FLAG1 = -DNDEBUG

######################################################################
# -DKLT_USE_QSORT forces the code to use the standard qsort() 
# routine.  Otherwise it will use a quicksort routine that takes
# advantage of our specific data structure to greatly reduce the
# running time on some machines. Uncomment this line if for some
# reason you are unhappy with the special routine.
# FLAG2 = -DKLT_USE_QSORT

######################################################################
# Add your favorite C flags here.
CFLAGS = $(FLAG1) $(FLAG2) -pg

######################################################################
# CUDA compilation flags
NVCCFLAGS = -arch=sm_60 -O3

######################################################################
# CUDA source files for library (replace CPU versions)
CUDA_LIB_SRCS = convolve_cuda.cu trackFeatures_cuda.cu
CUDA_LIB_OBJS = $(CUDA_LIB_SRCS:.cu=.o)

######################################################################
# CUDA example source files
CUDA_EXAMPLE_SRCS = example3_cuda.c
CUDA_EXAMPLES = $(CUDA_EXAMPLE_SRCS:.cu=)

######################################################################
# There should be no need to modify anything below this line (but
# feel free to if you want).

EXAMPLES = example3.c
ARCH = error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c klt.c klt_util.c writeFeatures.c
# Note: convolve.c and trackFeatures.c removed from ARCH since we have CUDA versions

LIB = -L/usr/local/lib -L/usr/lib
CUDA_LIB = -lcudart -L/usr/local/cuda/lib64

.SUFFIXES:  .c .o .cu

all:  lib $(EXAMPLES:.c=) $(CUDA_EXAMPLES)

.c.o:
	$(CC) -c $(CFLAGS) $<

# NVCC compilation rule for CUDA files
.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

lib: $(ARCH:.c=.o) $(CUDA_LIB_OBJS)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o) $(CUDA_LIB_OBJS)
	ranlib libklt.a
	@echo "Library built with CUDA components: $(CUDA_LIB_SRCS)"

example3: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

# CUDA version of example3
example3_cuda: libklt.a
	$(NVCC) $(NVCCFLAGS) -o $@ $@.cu -L. -lklt $(LIB) $(CUDA_LIB) -lm

run: example3
ifeq ($(DEBUG),1)
	./example3 $(DATASET)
	@echo "Run complete (DEBUG mode, no profiling)"
else
	./example3 $(DATASET)
	gprof ./example3 > profile.txt
	gprof ./example3 | python3 gprof2dot.py | dot -Tpng -o callgraph.png
	@echo "Profiling complete: profile.txt and callgraph.png generated"
endif

# Run CUDA version
run_cuda: example3_cuda
ifeq ($(DEBUG),1)
	./example3_cuda $(DATASET)
	@echo "CUDA Run complete (DEBUG mode, no profiling)"
else
	./example3_cuda $(DATASET)
	@echo "CUDA Run complete"
endif

depend:
	makedepend $(ARCH) $(EXAMPLES) $(CUDA_LIB_SRCS) $(CUDA_EXAMPLE_SRCS)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) $(CUDA_EXAMPLES) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt gmon.out profile.txt callgraph.png output/*
	rm -rf output

# Explicit rules for CUDA files to ensure they compile correctly
convolve_cuda.o: convolve_cuda.cu
	$(NVCC) -c $(NVCCFLAGS) convolve_cuda.cu -o convolve_cuda.o

trackFeatures_cuda.o: trackFeatures_cuda.cu
	$(NVCC) -c $(NVCCFLAGS) trackFeatures_cuda.cu -o trackFeatures_cuda.o

example3_cuda.o: example3_cuda.cu
	$(NVCC) -c $(NVCCFLAGS) example3_cuda.cu -o example3_cuda.o
