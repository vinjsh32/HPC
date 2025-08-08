# ============================================================
#  Makefile per OBDD + (opzionale) OpenMP + (opzionale) CUDA
#  Uso:
#     make                 # build di tutto (seq + eventuale omp/cuda)
#     make run-seq         # esegue il test sequenziale
#     make OMP=1 run-omp   # build+run test OpenMP
#     make CUDA=1 run-cuda # build+run test CUDA
#     make clean           # pulizia
# ============================================================

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
CXXFLAGS   := -std=c++17 -O2 -I$(INC_DIR) -I$(SRC_DIR)

NVCC_ARCH  ?= $(shell ./scripts/detect_gpu_arch.sh)
NVCCFLAGS  := -std=c++17 -O2 -I$(INC_DIR) -I$(SRC_DIR) $(NVCC_ARCH) --expt-extended-lambda

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
  $(OBJ_DIR)/unique_table.o

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
  CUDA_OBJS := $(OBJ_DIR)/obdd_cuda.o
else
  CUDA_OBJS :=
endif

# Eseguibili (vengono costruiti se il backend è abilitato)
SEQ_EXE   := $(BIN_DIR)/test_seq
OMP_EXE   := $(BIN_DIR)/test_omp
CUDA_EXE  := $(BIN_DIR)/test_cuda

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

# .cpp → .o da src/
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

# .cu  → .o (solo se CUDA=1)
ifeq ($(CUDA),1)
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

# ------------------------ Target di run --------------------------
.PHONY: run-seq run-omp run-cuda

run-seq: $(SEQ_EXE)
	$<

run-omp: $(OMP_EXE)
	$<

run-cuda: $(CUDA_EXE)
	$<

# ------------------------ Pulizia -------------------------------
.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
