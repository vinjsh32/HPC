# ============================================================================
#  High-Performance OBDD Library Build System
#  
#  Multi-Backend Makefile for Sequential CPU, OpenMP Parallel, and CUDA GPU
#  
#  Author: @vijsh32
#  Date: August 11, 2024
#  Version: 2.1
#  Copyright: 2024 High Performance Computing Laboratory
# ============================================================================
#
#  OVERVIEW:
#  This sophisticated build system supports multiple computational backends
#  with automatic dependency management, GPU architecture detection, and
#  comprehensive test suite integration. The system enables selective
#  compilation of backend-specific code while maintaining compatibility
#  across different hardware configurations.
#
#  SUPPORTED BACKENDS:
#  - Sequential CPU: Optimized single-threaded baseline implementation
#  - OpenMP Parallel: Multi-threaded CPU parallelization with optimizations
#  - CUDA GPU: GPU acceleration with automatic architecture detection
#
#  KEY FEATURES:
#  - Configurable backend compilation (CUDA=0/1, OMP=0/1)
#  - Automatic GPU compute capability detection
#  - Comprehensive test suite with performance benchmarking
#  - Debug and release build configurations
#  - Advanced algorithm testing (reordering, mathematical applications)
#  - Large-scale performance analysis and crossover point detection
#
#  USAGE EXAMPLES:
#     make                        # Build all enabled backends
#     make run-seq                # Run sequential CPU tests
#     make OMP=1 run-omp          # Build and run OpenMP tests
#     make CUDA=1 run-cuda        # Build and run CUDA tests  
#     make run-reordering         # Test variable reordering algorithms
#     make run-advmath            # Test advanced mathematical applications
#     make run-performance        # Test CUDA performance optimizations
#     make run-empirical          # Test empirical analysis framework
#     make run-large-scale        # Test large-scale crossover analysis
#     make run-benchmark          # Run comprehensive benchmarks
#     make DEBUG=1                # Build with debug information
#     make clean                  # Clean all build artifacts
#
#  CONFIGURATION:
#  Set environment variables or make arguments to customize build:
#  - CUDA=1/0: Enable/disable CUDA GPU backend (default: 1)
#  - OMP=1/0: Enable/disable OpenMP parallel backend (default: 0)  
#  - DEBUG=1/0: Enable/disable debug build (default: 0)
#  - CXX: C++ compiler (default: g++)
#  - NVCC: CUDA compiler (default: nvcc)
# ============================================================================

# ------------------------ Configurazione ------------------------
CXX      ?= g++
NVCC     ?= nvcc

CUDA     ?= 1          # 1 = abilita backend CUDA, 0 = disabilita
OMP      ?= 0          # 1 = abilita backend OpenMP
DEBUG    ?= 0          # 1 = build di debug

SRC_DIR  := src
INC_DIR  := include
TEST_DIR := tests
OBJ_DIR  := build
BIN_DIR  := bin

CUDA_HOME ?= /usr/local/cuda
CUDA_LIB  ?= $(CUDA_HOME)/lib64

# ------------------------ Flags comuni --------------------------
CXXFLAGS   := -std=c++17 -O2 -I$(INC_DIR) -I$(INC_DIR)/core -I$(INC_DIR)/backends -I$(SRC_DIR)

NVCC_ARCH  ?= $(shell ./scripts/detect_gpu_arch.sh)
NVCCFLAGS  := -std=c++17 -O2 -I$(INC_DIR) -I$(INC_DIR)/core -I$(INC_DIR)/backends -I$(SRC_DIR) $(NVCC_ARCH) --expt-extended-lambda

LDFLAGS    :=
LDLIBS     := -lgtest -pthread

ifeq ($(DEBUG),1)
  CXXFLAGS  += -g -O0
  NVCCFLAGS += -G -O0
endif

ifeq ($(OMP),1)
  CXXFLAGS  += -fopenmp -DOBDD_ENABLE_OPENMP
  LDFLAGS   += -fopenmp
endif

ifeq ($(CUDA),1)
  CXXFLAGS  += -DOBDD_ENABLE_CUDA
  NVCCFLAGS += -DOBDD_ENABLE_CUDA
  LDLIBS    += -L$(CUDA_LIB) -lcudart
endif

# ------------------------ Sorgenti / Oggetti --------------------

# Core CPU
CPU_OBJS := \
  $(OBJ_DIR)/obdd_core.o \
  $(OBJ_DIR)/apply_cache_c_api.o \
  $(OBJ_DIR)/unique_table.o \
  $(OBJ_DIR)/obdd_reordering.o \
  $(OBJ_DIR)/obdd_advanced_math.o \
  $(OBJ_DIR)/performance_benchmark.o \
  $(OBJ_DIR)/realistic_problems.o \
  $(OBJ_DIR)/statistical_analysis.o

# OpenMP (opzionale)
ifeq ($(OMP),1)
  OMP_OBJS := \
    $(OBJ_DIR)/obdd_openmp.o \
    $(OBJ_DIR)/obdd_openmp_optim.o
else
  OMP_OBJS :=
endif

# CUDA (opzionale)
ifeq ($(CUDA),1)
  CUDA_OBJS := \
    $(OBJ_DIR)/obdd_cuda.o \
    $(OBJ_DIR)/obdd_cuda_optimized.o \
    $(OBJ_DIR)/obdd_cuda_algorithms.o \
    $(OBJ_DIR)/empirical_analysis.o \
    $(OBJ_DIR)/empirical_reporting.o \
    $(OBJ_DIR)/cuda_profiler.o
else
  CUDA_OBJS :=
endif

# Eseguibili (vengono costruiti se il backend è abilitato)
SEQ_EXE      := $(BIN_DIR)/test_seq
OMP_EXE      := $(BIN_DIR)/test_omp
CUDA_EXE     := $(BIN_DIR)/test_cuda
REORDER_EXE  := $(BIN_DIR)/test_reordering
ADVMATH_EXE  := $(BIN_DIR)/test_advanced_math
PERF_EXE     := $(BIN_DIR)/test_performance
EMPIRICAL_EXE := $(BIN_DIR)/test_empirical_analysis
OMP_EXT_EXE  := $(BIN_DIR)/test_openmp_extended
CUDA_EXT_EXE := $(BIN_DIR)/test_cuda_extended
COMP_EXE     := $(BIN_DIR)/test_backend_comparison
BENCH_EXE    := $(BIN_DIR)/test_performance_benchmark
ADV_BENCH_EXE := $(BIN_DIR)/test_advanced_benchmark
LARGE_SCALE_EXE := $(BIN_DIR)/test_large_scale_performance

# ------------------------ Target di default ----------------------
ALL_EXE := $(SEQ_EXE)
ifeq ($(OMP),1)
  ALL_EXE += $(OMP_EXE)
endif
ifeq ($(CUDA),1)
  ALL_EXE += $(CUDA_EXE)
endif

.PHONY: all
all: dirs $(ALL_EXE)

# ------------------------ Directory helper ----------------------
.PHONY: dirs
dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

# ------------------------ Regole di compilazione -----------------

# .cpp → .o da src/ (organized directories)
$(OBJ_DIR)/%.o: $(SRC_DIR)/core/%.cpp | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/openmp/%.cpp | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/advanced/%.cpp | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

# .cu  → .o (solo se CUDA=1) from cuda directory
ifeq ($(CUDA),1)
$(OBJ_DIR)/%.o: $(SRC_DIR)/cuda/%.cu | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif

# .cpp → .o dai tests/
$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ------------------------ Link degli eseguibili ------------------

# Sequenziale (test GoogleTest)
$(SEQ_EXE): $(CPU_OBJS) $(OBJ_DIR)/test_apply.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

# OpenMP (se abilitato)
ifeq ($(OMP),1)
$(OMP_EXE): $(CPU_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_openmp.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
endif

# CUDA (se abilitato)
ifeq ($(CUDA),1)
$(CUDA_EXE): $(CPU_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_cuda.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
endif

# Advanced Reordering Tests
$(REORDER_EXE): $(CPU_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_reordering.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Advanced Mathematical Tests
$(ADVMATH_EXE): $(CPU_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_advanced_math.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Performance Tests (CUDA required)
ifeq ($(CUDA),1)
$(PERF_EXE): $(CPU_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_performance.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
endif

# Empirical Analysis Tests (CUDA required)
ifeq ($(CUDA),1)
$(EMPIRICAL_EXE): $(CPU_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_empirical_analysis.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
endif

# ------------------------ Target di run --------------------------
.PHONY: run-seq run-omp run-cuda run-reordering run-advmath run-performance run-empirical

run-seq: $(SEQ_EXE)
	$<

run-omp: $(OMP_EXE)
	$<

run-cuda: $(CUDA_EXE)
	$<

run-reordering: $(REORDER_EXE)
	$<

run-advmath: $(ADVMATH_EXE)
	$<

run-performance: $(PERF_EXE)
	$<

run-empirical: $(EMPIRICAL_EXE)
	$<

# Extended test targets
$(OMP_EXT_EXE): $(CPU_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_openmp_extended.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

ifeq ($(CUDA),1)
$(CUDA_EXT_EXE): $(CPU_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_cuda_extended.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
endif

$(COMP_EXE): $(CPU_OBJS) $(OMP_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_backend_comparison.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Performance Benchmark Suite (includes all backends)
$(BENCH_EXE): $(CPU_OBJS) $(OMP_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_performance_benchmark.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Advanced Sophisticated Benchmark Suite
$(ADV_BENCH_EXE): $(CPU_OBJS) $(OMP_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_advanced_benchmark.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Large Scale OpenMP Performance Analysis
$(LARGE_SCALE_EXE): $(CPU_OBJS) $(OMP_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_large_scale_performance.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Run extended tests
.PHONY: run-omp-ext run-cuda-ext run-comparison run-extended run-benchmark run-advanced

run-omp-ext: $(OMP_EXT_EXE)
	$<

run-cuda-ext: $(CUDA_EXT_EXE)
	$<

run-comparison: $(COMP_EXE)
	$<

run-extended: run-omp-ext run-cuda-ext run-comparison

run-benchmark: $(BENCH_EXE)
	@echo "🚀 Running comprehensive performance benchmarks..."
	$<

run-advanced: $(ADV_BENCH_EXE)
	@echo "🎯 Running ADVANCED SOPHISTICATED benchmarks with statistical analysis..."
	$<

run-large-scale: $(LARGE_SCALE_EXE)
	@echo "🔬 Running LARGE-SCALE OpenMP vs Sequential crossover analysis..."
	$<

# ------------------------ Pulizia -------------------------------
.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
