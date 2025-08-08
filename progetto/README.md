# OBDD Project

This project implements an Ordered Binary Decision Diagram (OBDD) library with optional OpenMP and CUDA backends. It provides a sequential CPU implementation and optional parallel versions for multi-core CPUs and NVIDIA GPUs. Variable reordering now uses a parallel merge sort on the CPU and Thrust-based sorting on the GPU.

## Building

All builds are driven by the top-level `makefile`. By default the CUDA backend is enabled; disable it if you do not have the CUDA toolkit installed.

### CPU only

```bash
make CUDA=0
```
This produces `bin/test_seq`.

### OpenMP backend

```bash
make OMP=1 CUDA=0
```
This adds OpenMP support and builds `bin/test_omp`.

The OpenMP backend now mantiene una cache per-thread per la funzione `apply`
al fine di evitare contention sui lock. Ogni thread utilizza una `unordered_map`
locale e, al termine delle regioni parallele, le cache vengono fuse nel thread
master per riutilizzare i risultati nelle chiamate successive.

### CUDA backend

```bash
make CUDA=1
```
Requires `nvcc` in your PATH and produces `bin/test_cuda` in addition to the CPU binaries.

## Testing

Run the provided tests via make:

```bash
make CUDA=0 run-seq        # sequential CPU test
make OMP=1 CUDA=0 run-omp  # OpenMP test
make CUDA=1 run-cuda       # CUDA test
```

## Example usage

After building, execute the binaries in `bin/`:

```bash
./bin/test_seq                       # sequential example
OMP_NUM_THREADS=4 ./bin/test_omp     # OpenMP with 4 threads
./bin/test_cuda                      # CUDA example
```

These examples demonstrate evaluating and combining OBDDs on different backends.
