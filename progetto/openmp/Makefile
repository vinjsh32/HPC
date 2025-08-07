# ------------------------------------------------------------
#  Makefile – OBDD project (CPU + OpenMP + CUDA)
#  versione 2025-06-30   – fix link test_cuda
# ------------------------------------------------------------

CXX       := g++
NVCC      := nvcc
PY        := python3

INC       := -Iinclude                              # path header
CXXFLAGS  := -std=c++17 -O2 -fopenmp -DOBDD_ENABLE_OPENMP $(INC) $(EXTRA_FLAGS)

# -----------------------------------------------------------------
#  CORE libreria (CPU/OpenMP)
# -----------------------------------------------------------------
SRCS_CORE := \
    src/obdd_core.cpp \
    src/obdd_openmp.cpp \
    src/obdd_openmp_optim.cpp \
    src/apply_cache.cpp
OBJ_CORE  := $(SRCS_CORE:.cpp=.o)

# -----------------------------------------------------------------
#  Eseguibili CPU
# -----------------------------------------------------------------
RUN_BENCH_SRC  := examples/run_bench.cpp
RUN_BENCH_OBJ  := $(RUN_BENCH_SRC:.cpp=.o)
BIN_BENCH      := run_bench

RUN_STRESS_SRC := examples/run_stress.cpp
RUN_STRESS_OBJ := $(RUN_STRESS_SRC:.cpp=.o)
BIN_STRESS     := run_stress

# -----------------------------------------------------------------
#  CUDA objects + demo
# -----------------------------------------------------------------
CUDA_OBJ := src/obdd_cuda.o
src/obdd_cuda.o: ../src/obdd_cuda.cu
	$(NVCC) -std=c++17 -O3 -lineinfo $(INC) -c $< -o $@

examples/main_cuda.o: ../examples/main_cuda.cu
	$(NVCC) -std=c++17 -O3 -lineinfo $(INC) -c $< -o $@

# --- DEMO GPU -------------------------------------------------------
# • linkiamo $(OBJ_CORE) per includere apply_cache.o & friends
test_cuda: $(OBJ_CORE) $(CUDA_OBJ) examples/main_cuda.o
	$(NVCC) -lineinfo $(INC) \
	        -Xcompiler "-fopenmp $(INC)" \
	        $^ -o $@
.PHONY: test_cuda

# -----------------------------------------------------------------
#  Build CPU di default
# -----------------------------------------------------------------
all: $(BIN_BENCH) $(BIN_STRESS)

$(BIN_BENCH): $(OBJ_CORE) $(RUN_BENCH_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN_STRESS): $(OBJ_CORE) $(RUN_STRESS_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

# -----------------------------------------------------------------
#  Compilazione generica .cpp → .o
# -----------------------------------------------------------------
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# -----------------------------------------------------------------
#  Benchmark completo (bench_all.py)
# -----------------------------------------------------------------
bench: all ../scripts/bench_all.py
	$(PY) scripts/bench_all.py \
	    --run-bench  ./$(BIN_BENCH) \
	    --run-stress ./$(BIN_STRESS) \
	    --threads 1 2 4 8 12 16 20 \
	    --min 10 --max 24 --step 2 --rep 20
.PHONY: bench

# -----------------------------------------------------------------
#  Profiling macro (perf stat)  –  esempio:
#     make prof n=24 thr=20
# -----------------------------------------------------------------
prof:
	@if [ -z "$(n)" ]; then echo "❌  serve n=<variabili>"; exit 1; fi
	@if [ -z "$(thr)" ]; then thr=20; else thr=$(thr); fi; \
	echo "# perf Parity-$$n  $$thr-thr"; \
	OMP_NUM_THREADS=$$thr perf stat -d ./run_stress \
	    --min $$n --max $$n --step 2 --rep 1 --csv /dev/null
.PHONY: prof

# -----------------------------------------------------------------
#  Clean
# -----------------------------------------------------------------
clean:
	rm -f $(OBJ_CORE) $(RUN_BENCH_OBJ) $(RUN_STRESS_OBJ) \
	      $(CUDA_OBJ) examples/main_cuda.o \
	      $(BIN_BENCH) $(BIN_STRESS) test_cuda \
	      *.o *.csv *.png build_time.png nodes.png \
	      time_vs_vars.png speedup.png nodes_vs_vars.png
.PHONY: clean
