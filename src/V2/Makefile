#####################################################################
# Compilers
NVCC = nvcc
CC = gcc

######################################################################
# Flags
FLAG1 = -DNDEBUG
CFLAGS = $(FLAG1) -O3
NVCCFLAGS = -O3 -Wno-deprecated-gpu-targets -diag-suppress 2464 -gencode arch=compute_86,code=sm_86
GPROF= -pg
CUDA_LIB = -lcudart
CUDA_INCLUDE = -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/include
CUDA_LIBDIR = -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/targets/x86_64-linux/lib

DATASET ?= dataset3

######################################################################
# Source files
EXAMPLES_GPU = example3.c
EXAMPLES_CPU = example3_cpu.c  # We'll create this
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
CUDA_SRCS = convolve_cuda.cu trackFeatures_cuda.cu

# Object files for different builds
CPU_OBJECTS = $(ARCH:.c=.o) $(CUDA_SRCS:.cu=_cpu.o)
GPU_OBJECTS = $(ARCH:.c=.o) $(CUDA_SRCS:.cu=.o)

.SUFFIXES: .c .cu .o

# Default build (CPU if no NVCC, GPU if available)
all: 
	@if which nvcc >/dev/null 2>&1; then \
		$(MAKE) gpu; \
	else \
		$(MAKE) cpu; \
	fi

# GPU build (only if NVCC available)
gpu: check_nvcc libklt_gpu.a example3_gpu

# CPU build  
cpu: libklt_cpu.a example3_cpu

# Build both (GPU only if available)
both: cpu
	@if which nvcc >/dev/null 2>&1; then \
		$(MAKE) gpu; \
	fi

# Check if NVCC is available for GPU build
check_nvcc:
	@if ! which nvcc >/dev/null 2>&1; then \
		echo "Error: nvcc not found. Cannot build GPU version."; \
		echo "Please install CUDA toolkit or use 'make cpu'"; \
		exit 1; \
	fi

# CPU compilation rules (using GCC)
.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

# GPU compilation rules (using NVCC)
.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $(CUDA_INCLUDE) $< -o $@

# CPU version of CUDA files (use C versions from current directory)
convolve_cuda_cpu.o: convolve.c
	$(CC) -c $(CFLAGS) $< -o $@

trackFeatures_cuda_cpu.o: trackFeatures.c  
	$(CC) -c $(CFLAGS) $< -o $@

# Libraries
libklt_gpu.a: $(GPU_OBJECTS)
	rm -f libklt_gpu.a
	ar ruv libklt_gpu.a $(GPU_OBJECTS)
	ranlib libklt_gpu.a

libklt_cpu.a: $(CPU_OBJECTS)
	rm -f libklt_cpu.a
	ar ruv libklt_cpu.a $(CPU_OBJECTS)
	ranlib libklt_cpu.a

# Executables
example3_gpu: libklt_gpu.a
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) -o $@ $(EXAMPLES_GPU) -L. -lklt_gpu $(CUDA_LIBDIR) -lm $(CUDA_LIB)

example3_cpu: libklt_cpu.a
	$(CC) $(CFLAGS) $(GPROF) -o $@ $(EXAMPLES_CPU) -L. -lklt_cpu -lm

# Run targets
run_gpu: example3_gpu
	./example3_gpu $(DATASET)

run_cpu: example3_cpu
	./example3_cpu $(DATASET)
	

# Default run
run: 
	@if [ -f example3_gpu ]; then \
		./example3_gpu $(DATASET); \
	elif [ -f example3_cpu ]; then \
		./example3_cpu $(DATASET); \
	else \
		echo "No executable found. Run 'make' first."; \
	fi
	

profile_gpu:
	nsys profile --stats=true ./example3_gpu > analysis.txt
	
profile_cpu:
	gprof ./example3_cpu > profile.txt
	gprof ./example3_cpu | python3 gprof2dot.py | dot -Tpng -o callgraph.png
	@echo "Profiling complete: profile.txt and callgraph.png generated"

clean:
	rm -f *.o *.a example3_cpu example3_gpu *.tar *.tar.gz libklt*.a \
	      feat*.ppm features.ft features.txt gmon.out profile.txt callgraph.png
	rm -rf output

# Create CPU-compatible example file from the original
example3_cpu.c: example3.c
	@echo "Creating CPU-compatible version of example3.c..."
	@sed 's/#include.*cuda_runtime.h//g; s/trackFeatures_cuda/trackFeatures/g; s/convolve_cuda/convolve/g' $< > $@
	@echo "Note: You may need to manually adjust example3_cpu.c for full CPU compatibility"

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build automatically (GPU if available, else CPU)"
	@echo "  gpu         - Build GPU version (requires nvcc)"
	@echo "  cpu         - Build CPU version"
	@echo "  both        - Build both versions if possible"
	@echo "  run/run_gpu - Run GPU version"
	@echo "  run_cpu     - Run CPU version"
	@echo "  clean       - Clean all built files"
