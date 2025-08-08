# OBDD Project

This project implements an Ordered Binary Decision Diagram (OBDD) library with optional OpenMP and CUDA backends. It provides a sequential CPU implementation and optional parallel versions for multi-core CPUs and NVIDIA GPUs. Variable reordering uses a parallel merge sort on the CPU and Thrust-based sorting on the GPU.

## Design choices

* **Memory management** – all dynamically created nodes are tracked in a global set so that `obdd_destroy` can reclaim every allocation. Constant leaves are singletons and freed when the last BDD handle is destroyed.
* **C/C++ interface** – the public API is defined in `include/obdd.hpp` and exposed to pure C code through the lightweight wrapper `include/obdd.h`, avoiding extra wrapper layers.
* **Automatic CUDA architecture** – `make` invokes `scripts/detect_gpu_arch.sh` to query `nvidia-smi` and pick the appropriate `-gencode` flag. This removes the need to manually edit `NVCCFLAGS` when compiling on different GPUs.

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

The OpenMP backend keeps a per-thread cache for the `apply` function to reduce lock contention. Each thread uses a local `unordered_map` and, at the end of parallel regions, caches are merged in the master thread.

### CUDA backend

```bash
make CUDA=1
```
Requires `nvcc` in your PATH and produces `bin/test_cuda` in addition to the CPU binaries. The GPU architecture is detected automatically.

## Testing

Run the provided tests via make:

```bash
make CUDA=0 run-seq        # sequential CPU test
make OMP=1 CUDA=0 run-omp  # OpenMP test
make CUDA=1 run-cuda       # CUDA test
```

## Example usage

After building, execute the binaries in `bin/` to run the GoogleTest-based test suites for each backend.
