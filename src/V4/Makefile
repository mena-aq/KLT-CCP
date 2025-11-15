######################################################################
# Build configuration
# COMPILER = gcc or nvc
# MODE = cpu or acc
COMPILER = gcc
MODE = cpu

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

######################################################################
# Compiler and flags configuration
ifeq ($(COMPILER)-$(MODE),nvc-acc)
  CC = nvc
  CFLAGS = $(FLAG1) $(FLAG2) -pg -acc -gpu=managed -Minfo=accel -DKLT_USE_OPENACC -DKLT_OPENACC_NVC
  ACC_SUFFIX = _acc
else ifeq ($(COMPILER)-$(MODE),gcc-acc)
  CC = gcc
  CFLAGS = $(FLAG1) $(FLAG2) -pg -fopenacc -fopt-info-optimized -DKLT_USE_OPENACC -DKLT_OPENACC_GCC
  ACC_SUFFIX = _acc
else
  # Default to GCC CPU mode
  CC = gcc
  CFLAGS = $(FLAG1) $(FLAG2) -pg
  ACC_SUFFIX =
endif

######################################################################
# Source files
EXAMPLES = example3.c

# dataset
DATASET ?= dataset3

# Common files (always included)
COMMON_SRCS = error.c pnmio.c pyramid.c selectGoodFeatures.c \
              storeFeatures.c klt.c klt_util.c writeFeatures.c

# CPU-specific files (always included for CPU, also for ACC as fallback)
CPU_SRCS = convolve.c trackFeatures.c

# ACC-specific files (only included for ACC builds)
ACC_SRCS = convolve$(ACC_SUFFIX).c trackFeatures$(ACC_SUFFIX).c

# Select source files based on mode
ifeq ($(MODE),acc)
  # ACC builds use ACC versions where available, but include all CPU sources too
  ARCH = $(ACC_SRCS) $(CPU_SRCS) $(COMMON_SRCS)
else
  # CPU builds only use CPU sources
  ARCH = $(CPU_SRCS) $(COMMON_SRCS)
endif

LIB = -L/usr/local/lib -L/usr/lib -lm

.SUFFIXES:  .c .o

all:  lib $(EXAMPLES:.c=)

.c.o:
	$(CC) -c $(CFLAGS) $<

lib: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)
	rm -f *.o

example3: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB)

# Predefined configurations
gcc-cpu:
	$(MAKE) COMPILER=gcc MODE=cpu all

gcc-acc:
	$(MAKE) COMPILER=gcc MODE=acc all

nvc-acc:
	$(MAKE) COMPILER=nvc MODE=acc all

# Run targets
run: example3
	./example3 $(DATASET)

run-gcc-cpu: gcc-cpu
	$(MAKE) COMPILER=gcc MODE=cpu run

run-gcc-acc: gcc-acc
	GOACC_DEBUG=1 ./example3 $(DATASET)

run-nvc-acc: nvc-acc
	$(MAKE) COMPILER=nvc MODE=acc run

# Valgrind debugging targets
valgrind: example3
	valgrind --tool=memcheck --leak-check=full --track-origins=yes --show-leak-kinds=all ./example3 $(DATASET)

valgrind-gcc-cpu: gcc-cpu
	$(MAKE) COMPILER=gcc MODE=cpu valgrind

valgrind-gcc-acc: gcc-acc
	$(MAKE) COMPILER=gcc MODE=acc valgrind

valgrind-nvc-acc: nvc-acc
	$(MAKE) COMPILER=nvc MODE=acc valgrind

# Configuration info
config:
	@echo "Build Configuration:"
	@echo "  COMPILER: $(COMPILER)"
	@echo "  MODE:     $(MODE)"
	@echo "  CC:       $(CC)"
	@echo "  CFLAGS:   $(CFLAGS)"
	@echo "  Sources:  $(ARCH)"

# Clean
clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt gmon.out profile.txt callgraph.png output/*
	rm -rf output

# Help
help:
	@echo "Available build configurations:"
	@echo "  gcc-cpu    - GCC compiler, CPU version (original code)"
	@echo "  gcc-acc    - GCC compiler, OpenACC version"
	@echo "  nvc-acc    - NVIDIA compiler, OpenACC version"
	@echo ""
	@echo "Run targets:"
	@echo "  run-gcc-cpu - Build and run GCC CPU version"
	@echo "  run-gcc-acc - Build and run GCC OpenACC version"  
	@echo "  run-nvc-acc - Build and run NVIDIA OpenACC version"
	@echo ""
	@echo "Valgrind debugging targets:"
	@echo "  valgrind-gcc-cpu - Build and run GCC CPU version with valgrind"
	@echo "  valgrind-gcc-acc - Build and run GCC OpenACC version with valgrind"
	@echo "  valgrind-nvc-acc - Build and run NVIDIA OpenACC version with valgrind"
	@echo ""
	@echo "Note: ACC builds include both ACC-optimized and CPU fallback sources"
