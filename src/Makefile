CC = gcc

######################################################################
# -DNDEBUG prevents the assert() statements from being included in 
# the code.  If you are having problems running the code, you might 
# want to comment this line to see if an assert() statement fires.
FLAG1 = -DNDEBUG

######################################################################
# -DKLT_USE_QSORT forces the code to use the standard qsort() 
# routine.  Otherwise it will use a quicksort routine that takes
# advantage of our specific data structure to greatly reduce the
# running time on some machines.  Uncomment this line if for some
# reason you are unhappy with the special routine.
# FLAG2 = -DKLT_USE_QSORT

# CPU flags
CPU_CFLAGS = $(FLAG1) $(FLAG2) -pg 
CPU_CC = gcc

# GPU flags 
GPU_CFLAGS = $(FLAG1) $(FLAG2) -acc -gpu=cc86 -Minfo=accel -fast
GPU_LDFLAGS = -acc -gpu=cc70
GPU_CC = nvc

EXAMPLES = example3.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
LIB = -L/usr/local/lib -L/usr/lib

DATASET ?= dataset3

.SUFFIXES:  .c .o

# Default target - build both CPU and GPU
all: build

######################################################################
# BUILD RULES
######################################################################

# Build library for CPU
lib_cpu: 
	$(foreach src, $(ARCH), $(CPU_CC) -c $(CPU_CFLAGS) $(src) -o $(src:.c=.o);)
	rm -f libklt_cpu.a
	ar ruv libklt_cpu.a $(ARCH:.c=.o)
	rm -f *.o

# Build library for GPU
lib_gpu: 
	$(foreach src, $(ARCH), $(GPU_CC) -c $(GPU_CFLAGS) $(src) -o $(src:.c=.o);)
	rm -f libklt_gpu.a
	ar ruv libklt_gpu.a $(ARCH:.c=.o)
	rm -f *.o

# CPU executable
example3_cpu: lib_cpu
	$(CPU_CC) -O3 $(CPU_CFLAGS) -o $@ example3.c -L. -lklt_cpu $(LIB) -lm
	@echo "CPU build complete using gcc compiler"

# GPU executable
example3_gpu: lib_gpu
	$(GPU_CC) -O3 $(GPU_CFLAGS) $(GPU_LDFLAGS) -o $@ example3.c -L. -lklt_gpu $(LIB) -lm
	@echo "GPU build complete using nvc compiler"

# Build both CPU and GPU versions
build: example3_cpu example3_gpu
	@echo "Both CPU and GPU versions built"


# Run CPU
run_cpu: example3_cpu
	./example3_cpu $(DATASET)

# Run GPU
run_gpu: example3_gpu
	./example3_gpu $(DATASET)

# Run both and compare
run: build
	@echo "=== Running CPU version ==="
	@./example3_cpu $(DATASET) > /dev/null 2>&1
	@echo "=== Running GPU version ==="  
	@./example3_gpu $(DATASET) > /dev/null 2>&1


# CPU profiling
profile_cpu: example3_cpu
	./example3_cpu $(DATASET)
	gprof ./example3_cpu > profile_cpu.txt
	gprof ./example3_cpu | python3 gprof2dot.py | dot -Tpng -o callgraph_cpu.png
	@echo "CPU profiling complete: profile_cpu.txt and callgraph_cpu.png generated"

# GPU profiling
profile_gpu: example3_gpu
	nsys profile --stats=true ./example3_gpu $(DATASET)
	@echo "GPU profiling complete with NVIDIA Nsight Systems"


# Test GPU compilation with info
test-gpu-compile:
	@echo "=== Testing GPU compilation with OpenACC info ==="
	$(GPU_CC) $(GPU_CFLAGS) -c example3.c -o example3_gpu_test.o
	rm -f example3_gpu_test.o


depend:
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a *.tar *.tar.gz \
	      feat*.ppm features.ft features.txt gmon.out profile*.txt callgraph*.png output/* \
	      example3_cpu example3_gpu libklt_cpu.a libklt_gpu.a
	rm -rf output
