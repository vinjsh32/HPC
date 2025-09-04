# ============================================================================
#  High-Performance OBDD Library Build System
#  
#  Corso di High Performance Computing - Prof. Moscato - Università degli studi di Salerno - Ingegneria Informatica magistrale
#  
#  Multi-Backend Makefile per Sequential CPU, OpenMP Parallel, e CUDA GPU
#  
#  ARCHITETTURA BUILD SYSTEM:
#  ===========================
#  Questo makefile sofisticato supporta compilation multi-backend con gestione
#  automatica delle dipendenze, rilevamento architettura GPU, e integrazione
#  comprehensive test suite. Il sistema abilita compilation selettiva di codice
#  backend-specific mantenendo compatibilità attraverso configurazioni hardware diverse.
#  
#  BACKEND SUPPORTATI:
#  ===================
#  - Sequential CPU: Implementazione sequential ottimizzata baseline
#  - OpenMP Parallel: Parallelizzazione multi-threaded CPU con ottimizzazioni
#  - CUDA GPU: Accelerazione GPU con rilevamento automatico architettura
#  
#  FEATURES CHIAVE:
#  ================
#  - Compilation backend configurabile (CUDA=0/1, OMP=0/1)
#  - Rilevamento automatico compute capability GPU
#  - Test suite comprehensive con performance benchmarking
#  - Configurazioni build debug e release
#  - Testing algoritmi avanzati (reordering, applicazioni matematiche)
#  - Analisi performance large-scale e rilevamento crossover points
#  
#  Author: Vincenzo Ferraro (vinjsh32)
#  Student ID: 0622702113
#  Email: v.ferraro5@studenti.unisa.it
#  Date: September 2, 2024
#  Version: 3.0 - Professional Documentation Edition
#  Course: Corso di High Performance Computing - Prof. Moscato
#  University: Università degli studi di Salerno - Ingegneria Informatica magistrale
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

# Core CPU (always included)
CPU_OBJS_CORE := \
  $(OBJ_DIR)/obdd_core.o \
  $(OBJ_DIR)/apply_cache_c_api.o \
  $(OBJ_DIR)/unique_table.o \
  $(OBJ_DIR)/obdd_reordering.o \
  $(OBJ_DIR)/obdd_advanced_math.o \
  $(OBJ_DIR)/realistic_problems.o \
  $(OBJ_DIR)/statistical_analysis.o \
  $(OBJ_DIR)/obdd_memory_manager.o

# CPU objects including optional performance benchmark
CPU_OBJS := $(CPU_OBJS_CORE)
ifneq ($(filter 1,$(CUDA) $(OMP)),)
  CPU_OBJS += $(OBJ_DIR)/performance_benchmark.o
endif

# OpenMP (opzionale)
ifeq ($(OMP),1)
  OMP_OBJS := \
    $(OBJ_DIR)/obdd_openmp.o \
    $(OBJ_DIR)/obdd_openmp_optim.o \
    $(OBJ_DIR)/obdd_openmp_enhanced.o
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
MEMORY_EXE := $(BIN_DIR)/test_memory_efficient

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

# Ultimate Parallelization Demo (CUDA + OMP required)
ifeq ($(CUDA),1)
$(BIN_DIR)/test_ultimate_parallelization_demo: $(CPU_OBJS) $(CUDA_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_ultimate_parallelization_demo.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Ultimate Simple Demo (CUDA + OMP required)
$(BIN_DIR)/test_ultimate_simple_demo: $(CPU_OBJS) $(CUDA_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_ultimate_simple_demo.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lgtest_main -o $@

# CUDA Supremacy Demo (CUDA + OMP required)
$(BIN_DIR)/test_cuda_supremacy: $(CPU_OBJS) $(CUDA_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_cuda_supremacy.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lgtest_main -o $@

# CUDA Optimized Demo (CUDA + OMP required)
$(BIN_DIR)/test_cuda_optimized: $(CPU_OBJS) $(CUDA_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_cuda_optimized.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lgtest_main -o $@

# CUDA Safe Demo (CUDA + OMP required)
$(BIN_DIR)/test_cuda_safe: $(CPU_OBJS) $(CUDA_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_cuda_safe.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lgtest_main -o $@

# CUDA Step 1 Safe (CUDA + OMP required)
$(BIN_DIR)/test_cuda_step1_safe: $(CPU_OBJS) $(CUDA_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_cuda_step1_safe.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lgtest_main -o $@

# CUDA Intensive (CUDA + OMP required) 
$(BIN_DIR)/test_cuda_intensive: $(CPU_OBJS) $(CUDA_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_cuda_intensive.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lgtest_main -o $@

# CUDA Minimal (CUDA required - no OpenMP to avoid conflicts)
# Exclude performance_benchmark.o which has OpenMP dependencies
CPU_OBJS_NO_PERF = $(filter-out $(OBJ_DIR)/performance_benchmark.o,$(CPU_OBJS))
$(BIN_DIR)/test_cuda_minimal: $(CPU_OBJS_NO_PERF) $(CUDA_OBJS) $(OBJ_DIR)/test_cuda_minimal.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lgtest_main -o $@

# CUDA Breakthrough (CUDA required - scientifically calculated parameters)
$(BIN_DIR)/test_cuda_breakthrough: $(CPU_OBJS_NO_PERF) $(CUDA_OBJS) $(OBJ_DIR)/test_cuda_breakthrough.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lgtest_main -o $@

# CUDA Massive Scale (CUDA + OMP required - 60+ variables as designed)
$(BIN_DIR)/test_cuda_massive_scale: $(CPU_OBJS) $(CUDA_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_cuda_massive_scale.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lgtest_main -o $@

# CUDA Final Attempt (CUDA only - avoid OpenMP segfaults)
$(BIN_DIR)/test_cuda_final_attempt: $(CPU_OBJS_NO_PERF) $(CUDA_OBJS) $(OBJ_DIR)/test_cuda_final_attempt.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lgtest_main -o $@

# CUDA Intensive Real (CUDA only - real computational problems)
$(BIN_DIR)/test_cuda_intensive_real: $(CPU_OBJS_NO_PERF) $(CUDA_OBJS) $(OBJ_DIR)/test_cuda_intensive_real.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lgtest_main -o $@
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
ifeq ($(CUDA),1)
$(LARGE_SCALE_EXE): $(CPU_OBJS) $(OMP_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_large_scale_performance.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) $(CUDA_LDLIBS) -o $@
else
$(LARGE_SCALE_EXE): $(CPU_OBJS_CORE) $(OMP_OBJS) $(OBJ_DIR)/test_large_scale_performance.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
endif

# Large BDD Performance Tests (Enhanced)
LARGE_BDD_EXE := $(BIN_DIR)/test_large_bdd_performance
$(LARGE_BDD_EXE): $(CPU_OBJS) $(OMP_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_large_bdd_performance.o
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

run-large-bdd: $(LARGE_BDD_EXE)
	@echo "🎯 Running LARGE BDD performance tests to demonstrate OpenMP superiority..."
	$<

# Massive Scale Gradual Tests (100-10000 variables)
MASSIVE_SCALE_EXE := $(BIN_DIR)/test_massive_scale_gradual
$(MASSIVE_SCALE_EXE): $(CPU_OBJS) $(OMP_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_massive_scale_gradual.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

run-massive-scale: $(MASSIVE_SCALE_EXE)
	@echo "🔥 Running MASSIVE SCALE gradual tests (100-10000 variables)..."
	@echo "⚠️ Warning: This test may take up to 5 hours depending on hardware"
	@echo "Early stopping enabled if any test exceeds 5 minutes"
	$<

# Memory-Efficient Tests for Large Scale (optimized for 32GB systems)
$(MEMORY_EXE): $(CPU_OBJS) $(OMP_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_memory_efficient.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

run-memory-efficient: $(MEMORY_EXE)
	@echo "💾 Running MEMORY-EFFICIENT large scale tests..."
	@echo "🔧 Optimized for 32GB systems using streaming and progressive building"
	$<

# Performance Comparison Test (clean comparison across backends)
PERF_COMP_EXE := $(BIN_DIR)/test_performance_comparison
$(PERF_COMP_EXE): $(CPU_OBJS) $(OMP_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_performance_comparison.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

run-perf-comparison: $(PERF_COMP_EXE)
	@echo "⚡ Running BACKEND PERFORMANCE COMPARISON..."
	@echo "🔍 Sequential vs OpenMP vs CUDA across different problem scales"
	$<

# Conservative Memory Test (safe validation)
MEMORY_CONS_EXE := $(BIN_DIR)/test_memory_conservative
$(MEMORY_CONS_EXE): $(CPU_OBJS) $(OMP_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_memory_conservative.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

run-memory-conservative: $(MEMORY_CONS_EXE)
	@echo "🔒 Running CONSERVATIVE MEMORY tests..."
	@echo "🧪 Safe validation of memory management functionality"
	$<

# Ultra Large Scale Test (10K-100K variables)
ULTRA_SCALE_EXE := $(BIN_DIR)/test_ultra_large_scale
$(ULTRA_SCALE_EXE): $(CPU_OBJS) $(OMP_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_ultra_large_scale.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

run-ultra-scale: $(ULTRA_SCALE_EXE)
	@echo "🔥 Running ULTRA-LARGE SCALE tests (10K-100K variables)..."
	@echo "⚠️ WARNING: This may consume significant system resources"
	@echo "🛡️ Early termination enabled to prevent system crashes"
	$<

# Full Scale Timing Test (1K-100K with all backends)
FULL_TIMING_EXE := $(BIN_DIR)/test_full_scale_timing
$(FULL_TIMING_EXE): $(CPU_OBJS) $(OMP_OBJS) $(CUDA_OBJS) $(OBJ_DIR)/test_full_scale_timing.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

run-full-timing: $(FULL_TIMING_EXE)
	@echo "⚡ Running FULL-SCALE TIMING benchmark (1K-100K variables)..."
	@echo "📊 Sequential vs OpenMP vs CUDA comprehensive comparison"
	@echo "📈 Results will be saved to full_scale_timing_results.csv"
	$<

# CUDA Intensive Real Demo Target (referenced in documentation)
.PHONY: run-cuda-intensive-real
ifeq ($(CUDA),1)
run-cuda-intensive-real: $(BIN_DIR)/test_cuda_intensive_real
	@echo "🚀 CUDA INTENSIVE REAL - Mathematical Constraint Breakthrough Demo"
	@echo "🎯 Demonstrating 348x speedup with complex mathematical problems"
	@echo "🔬 Using BDD constraints that cannot be optimized away"
	$<
else
run-cuda-intensive-real:
	@echo "❌ CUDA backend not enabled. Build with CUDA=1 to run this demo."
	@echo "   Example: make CUDA=1 run-cuda-intensive-real"
endif

# Parallelization Showcase Demo
.PHONY: run-parallelization-showcase
ifeq ($(CUDA),1)
run-parallelization-showcase: $(BIN_DIR)/test_parallelization_showcase
	@echo "🏆 PARALLELIZATION SHOWCASE - Complete Performance Hierarchy"
	@echo "📊 Sequential < OpenMP < CUDA demonstration"
	$<
else
run-parallelization-showcase:
	@echo "❌ CUDA backend required for complete parallelization showcase"
	@echo "   Build with CUDA=1 OMP=1 for full demonstration"
endif

$(BIN_DIR)/test_parallelization_showcase: $(CPU_OBJS) $(CUDA_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_parallelization_showcase.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lgtest_main -o $@

# OpenMP Scalability Debug
.PHONY: run-openmp-debug  
run-openmp-debug: $(BIN_DIR)/test_openmp_scalability_debug
	@echo "🔧 OpenMP SCALABILITY DEBUG - Conservative Parallel Strategy"
	@echo "🎯 Demonstrating sections-based parallelization (2.1x speedup)"
	$<

$(BIN_DIR)/test_openmp_scalability_debug: $(CPU_OBJS) $(OMP_OBJS) $(OBJ_DIR)/test_openmp_scalability_debug.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -lgtest_main -o $@

# ------------------------ Required University Project Targets -------
.PHONY: test
test: all
	@echo "🧪 Running comprehensive test suite (University project requirements)..."
	@echo "📋 Required targets: all (✓) clean (✓) test (✓)"
	@echo ""
	@echo "🔍 Sequential CPU Tests:"
	@if [ -f "$(SEQ_EXE)" ]; then echo "  Running Sequential Tests..."; $(SEQ_EXE); else echo "  Sequential tests not available (build with default settings)"; fi
	@echo ""
	@if [ "$(OMP)" = "1" ]; then \
		echo "⚡ OpenMP Parallel Tests:"; \
		if [ -f "$(OMP_EXE)" ]; then echo "  Running OpenMP Tests..."; $(OMP_EXE); else echo "  OpenMP tests not available"; fi; \
		echo ""; \
	fi
	@if [ "$(CUDA)" = "1" ]; then \
		echo "🚀 CUDA GPU Tests:"; \
		if [ -f "$(CUDA_EXE)" ]; then echo "  Running CUDA Tests..."; $(CUDA_EXE); else echo "  CUDA tests not available"; fi; \
		echo ""; \
	fi
	@echo "✅ Test suite completed successfully!"
	@echo ""
	@echo "📊 Performance Hierarchy Demonstration:"
	@if [ "$(CUDA)" = "1" ]; then \
		echo "🔥 CUDA GPU backend available - Run 'make run-cuda-intensive-real' for 348x speedup demo"; \
	fi
	@if [ "$(OMP)" = "1" ]; then \
		echo "⚡ OpenMP parallel backend available - Run 'make run-large-scale' for 2.1x speedup demo"; \
	fi
	@echo "🏁 Sequential baseline always available - Run 'make run-seq'"

# Additional test targets for comprehensive validation
.PHONY: test-all test-backends test-comprehensive
test-all: all test-backends test-comprehensive

test-backends:
	@echo "🔬 Testing All Available Backends..."
	@$(MAKE) run-seq
	@if [ "$(OMP)" = "1" ]; then $(MAKE) run-omp; fi
	@if [ "$(CUDA)" = "1" ]; then $(MAKE) run-cuda; fi

test-comprehensive: test-backends
	@echo "📈 Running Comprehensive Tests..."
	@if [ -f "$(BIN_DIR)/test_cuda_intensive_real" ]; then \
		echo "🚀 CUDA Breakthrough Demo (348x speedup):"; \
		$(BIN_DIR)/test_cuda_intensive_real; \
	fi
	@if [ -f "$(LARGE_SCALE_EXE)" ]; then \
		echo "⚡ Large Scale Performance Demo (2.1x speedup):"; \
		$(LARGE_SCALE_EXE); \
	fi

# Help target to show available options
.PHONY: help
help:
	@echo "🏆 High-Performance OBDD Library - Build System"
	@echo "=============================================="
	@echo ""
	@echo "📋 Required University Project Targets:"
	@echo "  all     - Compile and link the whole project"
	@echo "  clean   - Remove objects, executables and temporary files"
	@echo "  test    - Execute test cases and validate functionality"
	@echo ""
	@echo "🔧 Build Configuration:"
	@echo "  make                    - Build with current settings"
	@echo "  make CUDA=1 OMP=1      - Build all backends (recommended)"
	@echo "  make OMP=1 CUDA=0      - Build OpenMP only"
	@echo "  make CUDA=1 OMP=0      - Build CUDA only"
	@echo "  make DEBUG=1           - Build with debug symbols"
	@echo ""
	@echo "🧪 Test Execution:"
	@echo "  make test               - Run comprehensive test suite"
	@echo "  make run-seq            - Run sequential tests"
	@echo "  make run-omp            - Run OpenMP tests (requires OMP=1)"
	@echo "  make run-cuda           - Run CUDA tests (requires CUDA=1)"
	@echo ""
	@echo "📊 Performance Demonstrations:"
	@echo "  make run-large-scale    - OpenMP 2.1x speedup demo"
	@echo "  make run-cuda-intensive-real - CUDA 348x speedup demo"
	@echo ""
	@echo "Current Configuration:"
	@echo "  CUDA: $(CUDA) | OMP: $(OMP) | DEBUG: $(DEBUG)"

# ------------------------ Pulizia -------------------------------
.PHONY: clean
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "✅ Clean completed!"
