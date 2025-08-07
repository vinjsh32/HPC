/**
 * @file obdd_cuda.cu
 * @brief Backend GPU (CUDA) per OBDD/ROBDD.
 *
 *  – Copia ("flatten") di un BDD host → vettore compatto di nodi NodeGPU.
 *  – Kernel Breadth-First (BFS) per le operazioni logiche AND / OR / XOR.
 *  – Kernel specializzato NOT (un solo BDD in ingresso).
 *  – Kernel odd–even transposition sort (bubble-sort GPU) per l’ordinamento del
 *    vettore varOrder.
 *  – Wrapper C-linkage: copy, AND, OR, XOR, NOT, var_ordering, free.
 *
 *  Il grafo risultante viene ora ridotto in una ROBDD canonica copiandolo su
 *  host e invocando obdd_reduce().
 */

#include "obdd_cuda.hpp"

#ifdef OBDD_ENABLE_CUDA

#include "obdd.hpp"
#include "obdd_cuda_types.cuh"
#include "cuda_utils.cuh"

#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <unordered_map>
#include <climits>
#include <cstdlib>

/* -------------------------------------------------------------------------- */
/*                     HOST → DEVICE  (flatten + copy)                        */
/* -------------------------------------------------------------------------- */

static void flatten_host(const OBDD* bdd, std::vector<NodeGPU>& out)
{
    out.clear();
    if (!bdd || !bdd->root) {
        out.resize(2);
        out[0] = { -1, 0, 0 };    // FALSE
        out[1] = { -1, 1, 1 };    // TRUE
        return;
    }

    out.reserve(1024);
    out.push_back({ -1, 0, 0 }); // 0 = FALSE
    out.push_back({ -1, 1, 1 }); // 1 = TRUE

    std::queue<const OBDDNode*> Q;
    std::unordered_map<const OBDDNode*, int> idx;

    Q.push(bdd->root);
    while (!Q.empty()) {
        const OBDDNode* cur = Q.front(); Q.pop();
        if (idx.count(cur)) continue;

        int id = static_cast<int>(out.size());
        idx[cur] = id;

        NodeGPU n;
        n.var = cur->varIndex;
        if (cur->varIndex < 0) {
            int v = (cur == obdd_constant(1)) ? 1 : 0;
            n.low = n.high = v;
        } else {
            Q.push(cur->lowChild);
            Q.push(cur->highChild);
            n.low  = -1;
            n.high = -1;
        }
        out.push_back(n);
    }

    for (auto& kv : idx) {
        const OBDDNode* node = kv.first;
        int id               = kv.second;
        if (node->varIndex < 0) continue;
        out[id].low  = (node->lowChild  == obdd_constant(1) || node->lowChild  == obdd_constant(0))
                       ? (node->lowChild == obdd_constant(1) ? 1 : 0)
                       : idx[node->lowChild];
        out[id].high = (node->highChild == obdd_constant(1) || node->highChild == obdd_constant(0))
                       ? (node->highChild == obdd_constant(1) ? 1 : 0)
                       : idx[node->highChild];
    }
}

static DeviceOBDD* copy_flat_to_device(const OBDD* bdd)
{
    std::vector<NodeGPU> host;
    flatten_host(bdd, host);

    DeviceOBDD hostDev{};
    hostDev.size  = static_cast<int>(host.size());
    hostDev.nVars = bdd ? bdd->numVars : 0;

    CUDA_CHECK(cudaMalloc(&hostDev.nodes, sizeof(NodeGPU) * hostDev.size));
    CUDA_CHECK(cudaMemcpy(hostDev.nodes, host.data(),
                          sizeof(NodeGPU) * hostDev.size,
                          cudaMemcpyHostToDevice));

    DeviceOBDD* dHandle = nullptr;
    CUDA_CHECK(cudaMalloc(&dHandle, sizeof(DeviceOBDD)));
    CUDA_CHECK(cudaMemcpy(dHandle, &hostDev, sizeof(DeviceOBDD), cudaMemcpyHostToDevice));
    return dHandle;
}

/* -------------------------------------------------------------------------- */
/*                                KERNELS                                     */
/* -------------------------------------------------------------------------- */

struct Pair { int u, v; };

template<int OP>
__device__ __forceinline__
int logic_op_bit(int a, int b)
{
    if constexpr (OP == 0) return a & b;   // AND
    if constexpr (OP == 1) return a | b;   // OR
    if constexpr (OP == 3) return a ^ b;   // XOR
    return 0;
}

template<int OP>
__global__ void apply_bfs_kernel(const NodeGPU* __restrict__ A,
                                 const NodeGPU* __restrict__ B,
                                 Pair* frontierIn,
                                 int   frontierSize,
                                 Pair* frontierOut,
                                 int*  nextCounter,
                                 NodeGPU* outNodes,
                                 int*  nodeCounter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontierSize) return;

    Pair cur = frontierIn[tid];
    int u = cur.u;
    int v = cur.v;

    if (A[u].var < 0 && B[v].var < 0) {
        int res = logic_op_bit<OP>(A[u].low, B[v].low);
        frontierIn[tid].u = frontierIn[tid].v = res;
        return;
    }

    int varU = (A[u].var < 0) ? INT_MAX : A[u].var;
    int varV = (B[v].var < 0) ? INT_MAX : B[v].var;
    int top  = (varU < varV) ? varU : varV;

    int uLow  = (varU==top) ? A[u].low  : u;
    int uHigh = (varU==top) ? A[u].high : u;
    int vLow  = (varV==top) ? B[v].low  : v;
    int vHigh = (varV==top) ? B[v].high : v;

    int pos = atomicAdd(nextCounter, 2);
    frontierOut[pos]   = { uLow,  vLow  };
    frontierOut[pos+1] = { uHigh, vHigh };

    int myIdx = atomicAdd(nodeCounter, 1);
    outNodes[myIdx] = { top, -1, -1 };

    frontierIn[tid].u = frontierIn[tid].v = myIdx;
}

__global__ void not_kernel(const NodeGPU* inNodes,
                           int*  frontierCur,
                           int   frontierSize,
                           int*  frontierNext,
                           int*  nextCounter,
                           NodeGPU* outNodes,
                           int*  nodeCounter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontierSize) return;

    int u = frontierCur[tid];

    if (inNodes[u].var < 0) {
        frontierCur[tid] = !inNodes[u].low;
        return;
    }

    int var   = inNodes[u].var;
    int uLow  = inNodes[u].low;
    int uHigh = inNodes[u].high;

    int pos = atomicAdd(nextCounter, 2);
    frontierNext[pos]   = uLow;
    frontierNext[pos+1] = uHigh;

    int myIdx = atomicAdd(nodeCounter, 1);
    outNodes[myIdx] = { var, -1, -1 };

    frontierCur[tid] = myIdx;
}

__global__ void oets_phase(int* dArr, int n, int phase)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i   = 2 * tid + phase;
    if (i + 1 >= n) return;

    int a = dArr[i];
    int b = dArr[i + 1];
    if (a > b) { dArr[i] = b; dArr[i + 1] = a; }
}

/* -------------------------------------------------------------------------- */
/*                              HELPERS HOST                                  */
/* -------------------------------------------------------------------------- */

namespace {

static OBDDNode* rebuild_host_bdd(const std::vector<NodeGPU>& nodes,
                                  int idx,
                                  std::vector<OBDDNode*>& cache)
{
    if (idx == 0) return obdd_constant(0);
    if (idx == 1) return obdd_constant(1);
    if (cache[idx]) return cache[idx];
    const NodeGPU& n = nodes[idx];
    OBDDNode* low  = rebuild_host_bdd(nodes, n.low,  cache);
    OBDDNode* high = rebuild_host_bdd(nodes, n.high, cache);
    cache[idx] = obdd_node_create(n.var, low, high);
    return cache[idx];
}

static void reduce_device_obdd(void** dHandle)
{
    if (!dHandle || !*dHandle) return;

    DeviceOBDD dev{};
    CUDA_CHECK(cudaMemcpy(&dev, *dHandle, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost));

    std::vector<NodeGPU> nodes(dev.size);
    CUDA_CHECK(cudaMemcpy(nodes.data(), dev.nodes,
                          sizeof(NodeGPU) * dev.size,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dev.nodes));
    CUDA_CHECK(cudaFree(*dHandle));

    std::vector<OBDDNode*> cache(dev.size, nullptr);
    int rootIdx = (dev.size > 2) ? 2 : 0;
    OBDDNode* root = rebuild_host_bdd(nodes, rootIdx, cache);
    OBDDNode* reduced = obdd_reduce(root);

    OBDD tmpUn{root, dev.nVars, static_cast<int*>(std::malloc(sizeof(int)*dev.nVars))};
    obdd_destroy(&tmpUn);

    OBDD tmpRed{reduced, dev.nVars, static_cast<int*>(std::malloc(sizeof(int)*dev.nVars))};
    DeviceOBDD* newDev = copy_flat_to_device(&tmpRed);
    obdd_destroy(&tmpRed);

    *dHandle = static_cast<void*>(newDev);
}

template<int OP>
void gpu_binary_apply(void* dA, void* dB, void** dOut)
{
    DeviceOBDD A{}, B{};
    CUDA_CHECK(cudaMemcpy(&A, dA, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&B, dB, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost));

    const int MAX_PAIRS = A.size * B.size;
    const int MAX_NODES = 2 + MAX_PAIRS * 2;

    NodeGPU* dNodes = nullptr;
    CUDA_CHECK(cudaMalloc(&dNodes, sizeof(NodeGPU) * MAX_NODES));

    const NodeGPU term0 = { -1, 0, 0 };
    const NodeGPU term1 = { -1, 1, 1 };
    CUDA_CHECK(cudaMemcpy(dNodes,     &term0, sizeof(NodeGPU), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNodes + 1, &term1, sizeof(NodeGPU), cudaMemcpyHostToDevice));

    Pair *dCur = nullptr, *dNext = nullptr;
    CUDA_CHECK(cudaMalloc(&dCur,  sizeof(Pair) * MAX_PAIRS));
    CUDA_CHECK(cudaMalloc(&dNext, sizeof(Pair) * MAX_PAIRS));

    Pair start = { 2, 2 };
    CUDA_CHECK(cudaMemcpy(dCur, &start, sizeof(Pair), cudaMemcpyHostToDevice));

    int *dCurSz=nullptr, *dNextSz=nullptr, *dNodeCnt=nullptr;
    CUDA_CHECK(cudaMalloc(&dCurSz,  sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dNextSz, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dNodeCnt,sizeof(int)));

    int one = 1, two = 2;
    CUDA_CHECK(cudaMemcpy(dCurSz,  &one, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNodeCnt,&two, sizeof(int), cudaMemcpyHostToDevice));

    while (true) {
        int hFront = 0;
        CUDA_CHECK(cudaMemcpy(&hFront, dCurSz, sizeof(int), cudaMemcpyDeviceToHost));
        if (hFront == 0) break;

        int blocks = (hFront + OBDD_CUDA_TPB - 1) / OBDD_CUDA_TPB;
        CUDA_CHECK(cudaMemset(dNextSz, 0, sizeof(int)));

        apply_bfs_kernel<OP><<<blocks, OBDD_CUDA_TPB>>>(A.nodes, B.nodes,
                                                        dCur, hFront,
                                                        dNext, dNextSz,
                                                        dNodes, dNodeCnt);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(dCur, dNext);
        CUDA_CHECK(cudaMemcpy(dCurSz, dNextSz, sizeof(int), cudaMemcpyDeviceToDevice));
    }

    int hCount = 0;
    CUDA_CHECK(cudaMemcpy(&hCount, dNodeCnt, sizeof(int), cudaMemcpyDeviceToHost));

    NodeGPU* dCompact = nullptr;
    CUDA_CHECK(cudaMalloc(&dCompact, sizeof(NodeGPU) * hCount));
    CUDA_CHECK(cudaMemcpy(dCompact, dNodes, sizeof(NodeGPU) * hCount,
                          cudaMemcpyDeviceToDevice));

    DeviceOBDD res{ dCompact, hCount, A.nVars };
    DeviceOBDD* dRes = nullptr;
    CUDA_CHECK(cudaMalloc(&dRes, sizeof(DeviceOBDD)));
    CUDA_CHECK(cudaMemcpy(dRes, &res, sizeof(DeviceOBDD), cudaMemcpyHostToDevice));

    *dOut = static_cast<void*>(dRes);

    CUDA_CHECK(cudaFree(dNodes));
    CUDA_CHECK(cudaFree(dCur));
    CUDA_CHECK(cudaFree(dNext));
    CUDA_CHECK(cudaFree(dCurSz));
    CUDA_CHECK(cudaFree(dNextSz));
    CUDA_CHECK(cudaFree(dNodeCnt));

    reduce_device_obdd(dOut);
}

} // anon

/* -------------------------------------------------------------------------- */
/*                              API PUBBLICHE                                 */
/* -------------------------------------------------------------------------- */

extern "C" {

void* obdd_cuda_copy_to_device(const OBDD* bdd)
{
    return static_cast<void*>(copy_flat_to_device(bdd));
}

void obdd_cuda_free_device(void* dHandle)
{
    if (!dHandle) return;
    DeviceOBDD tmp{};
    CUDA_CHECK(cudaMemcpy(&tmp, dHandle, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(tmp.nodes));
    CUDA_CHECK(cudaFree(dHandle));
}

void obdd_cuda_and(void* dA, void* dB, void** dOut)
{
    gpu_binary_apply<0>(dA, dB, dOut);
}

void obdd_cuda_or(void* dA, void* dB, void** dOut)
{
    gpu_binary_apply<1>(dA, dB, dOut);
}

void obdd_cuda_xor(void* dA, void* dB, void** dOut)
{
    gpu_binary_apply<3>(dA, dB, dOut);
}

void obdd_cuda_not(void* dA, void** dOut)
{
    DeviceOBDD A{};
    CUDA_CHECK(cudaMemcpy(&A, dA, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost));

    const int MAX_NODES = 2 + A.size * 2;

    NodeGPU* dNodes = nullptr;
    CUDA_CHECK(cudaMalloc(&dNodes, sizeof(NodeGPU) * MAX_NODES));
    const NodeGPU t0 = { -1, 0, 0 }, t1 = { -1, 1, 1 };
    CUDA_CHECK(cudaMemcpy(dNodes,     &t0, sizeof(NodeGPU), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNodes + 1, &t1, sizeof(NodeGPU), cudaMemcpyHostToDevice));

    int *dCur=nullptr, *dNext=nullptr;
    CUDA_CHECK(cudaMalloc(&dCur,  sizeof(int) * MAX_NODES));
    CUDA_CHECK(cudaMalloc(&dNext, sizeof(int) * MAX_NODES));

    int start = 2;
    CUDA_CHECK(cudaMemcpy(dCur, &start, sizeof(int), cudaMemcpyHostToDevice));

    int *dCurSz=nullptr, *dNextSz=nullptr, *dNodeCnt=nullptr;
    CUDA_CHECK(cudaMalloc(&dCurSz,  sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dNextSz, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dNodeCnt,sizeof(int)));

    int one = 1, two = 2;
    CUDA_CHECK(cudaMemcpy(dCurSz,  &one, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNodeCnt,&two, sizeof(int), cudaMemcpyHostToDevice));

    while (true) {
        int hFront = 0;
        CUDA_CHECK(cudaMemcpy(&hFront, dCurSz, sizeof(int), cudaMemcpyDeviceToHost));
        if (hFront == 0) break;

        int blocks = (hFront + OBDD_CUDA_TPB - 1) / OBDD_CUDA_TPB;
        CUDA_CHECK(cudaMemset(dNextSz, 0, sizeof(int)));

        not_kernel<<<blocks, OBDD_CUDA_TPB>>>(A.nodes,
                                              dCur, hFront,
                                              dNext, dNextSz,
                                              dNodes, dNodeCnt);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(dCur, dNext);
        CUDA_CHECK(cudaMemcpy(dCurSz, dNextSz, sizeof(int), cudaMemcpyDeviceToDevice));
    }

    int hCount = 0;
    CUDA_CHECK(cudaMemcpy(&hCount, dNodeCnt, sizeof(int), cudaMemcpyDeviceToHost));

    NodeGPU* dCompact = nullptr;
    CUDA_CHECK(cudaMalloc(&dCompact, sizeof(NodeGPU) * hCount));
    CUDA_CHECK(cudaMemcpy(dCompact, dNodes, sizeof(NodeGPU) * hCount,
                          cudaMemcpyDeviceToDevice));

    DeviceOBDD res{ dCompact, hCount, A.nVars };
    DeviceOBDD* dRes = nullptr;
    CUDA_CHECK(cudaMalloc(&dRes, sizeof(DeviceOBDD)));
    CUDA_CHECK(cudaMemcpy(dRes, &res, sizeof(DeviceOBDD), cudaMemcpyHostToDevice));

    *dOut = static_cast<void*>(dRes);

    CUDA_CHECK(cudaFree(dNodes));
    CUDA_CHECK(cudaFree(dCur));
    CUDA_CHECK(cudaFree(dNext));
    CUDA_CHECK(cudaFree(dCurSz));
    CUDA_CHECK(cudaFree(dNextSz));
    CUDA_CHECK(cudaFree(dNodeCnt));

    reduce_device_obdd(dOut);
}

void* obdd_cuda_apply(void* dA, void* dB, OBDD_Op op)
{
    void* out = nullptr;
    switch (op) {
        case OBDD_AND: obdd_cuda_and(dA, dB, &out); break;
        case OBDD_OR:  obdd_cuda_or (dA, dB, &out); break;
        case OBDD_XOR: obdd_cuda_xor(dA, dB, &out); break;
        case OBDD_NOT: obdd_cuda_not(dA, &out);     break;
        default: out = nullptr; break;
    }
    return out;
}

void obdd_cuda_var_ordering(int* hostVarOrder, int n)
{
    if (!hostVarOrder || n <= 1) return;

    int* dArr = nullptr;
    CUDA_CHECK(cudaMalloc(&dArr, sizeof(int) * n));
    CUDA_CHECK(cudaMemcpy(dArr, hostVarOrder, sizeof(int) * n, cudaMemcpyHostToDevice));

    int maxPairs = (n + 1) / 2;
    int blocks   = (maxPairs + OBDD_CUDA_TPB - 1) / OBDD_CUDA_TPB;

    for (int pass = 0; pass < n; ++pass) {
        int phase = pass & 1;
        oets_phase<<<blocks, OBDD_CUDA_TPB>>>(dArr, n, phase);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hostVarOrder, dArr, sizeof(int) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dArr));
}

} /* extern "C" */

#endif /* OBDD_ENABLE_CUDA */
