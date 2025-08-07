/* -----------------------------------------------------------------------
 *  run_stress.cpp  – Parity‑n sweep (OpenMP) – CLI‑compatible
 * -----------------------------------------------------------------------
 *  Flags (as expected by bench_all.py):
 *     --min N      starting n (default 10)
 *     --max N      ending n   (default 24)
 *     --step N     increment  (default 2)
 *     --rep  R     repetitions per n (default 5)
 *     --csv PATH   output CSV (default "results_stress.csv")
 *
 *  CSV columns: func,vars,threads,rep,nodes,ms
 * --------------------------------------------------------------------- */
#include "obdd.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <omp.h>

#if !defined(OBDD_ENABLE_OPENMP)
#   error "Compile with -DOBDD_ENABLE_OPENMP and -fopenmp"
#endif

#define APPLY_BIN  obdd_parallel_apply_omp

using Clock = std::chrono::high_resolution_clock;

static inline OBDDNode* var(int idx) {
    return obdd_node_create(idx, obdd_constant(0), obdd_constant(1));
}

static OBDDNode* parity_n(int n, const int *ord) {
    OBDD acc{var(0), n, const_cast<int*>(ord)};
    for (int i = 1; i < n; ++i) {
        OBDD xi{var(i), n, const_cast<int*>(ord)};
        acc.root = APPLY_BIN(&acc, &xi, OBDD_XOR);
    }
    return acc.root;
}

static double timed_build(int n, const int* ord, int &outNodes) {
    OBDD dummy{nullptr, n, const_cast<int*>(ord)};
    auto t0 = Clock::now();
    dummy.root = parity_n(n, ord);
    auto t1 = Clock::now();

    outNodes = 0;
    std::vector<const OBDDNode*> st{dummy.root};
    while (!st.empty()) {
        const OBDDNode* p = st.back(); st.pop_back();
        if (!p || p->varIndex < 0) continue;
        ++outNodes;
        st.push_back(p->lowChild);
        st.push_back(p->highChild);
    }
    return std::chrono::duration<double,std::milli>(t1 - t0).count();
}

int main(int argc, char **argv) {
    int vMin = 10, vMax = 24, step = 2, reps = 5;
    const char* csvFile = "results_stress.csv";

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--min")  && i+1<argc) vMin  = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max")  && i+1<argc) vMax  = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--step") && i+1<argc) step  = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--rep")  && i+1<argc) reps  = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--csv")  && i+1<argc) csvFile = argv[++i];
        else {
            std::cerr << "Unknown or incomplete flag: " << argv[i] << "\n";
            return 1;
        }
    }

    std::ofstream csv(csvFile, std::ios::app);
    if (csv.tellp() == 0) csv << "func,vars,threads,rep,nodes,ms\n";

    int threads = omp_get_max_threads();
    std::vector<int> order(vMax);
    for (int i = 0; i < vMax; ++i) order[i] = i;

    for (int n = vMin; n <= vMax; n += step) {
        for (int r = 0; r < reps; ++r) {
            int nodes = 0;
            double ms = timed_build(n, order.data(), nodes);
            csv << "Parity" << n << ',' << n << ',' << threads << ','
                << r << ',' << nodes << ',' << ms << '\n';
        }
    }
    return 0;
}
#if defined(OBDD_ENABLE_OPENMP) && defined(OBDD_USE_OPTIM)
#   define APPLY_BIN  obdd_parallel_apply_omp_optim
#else
#   define APPLY_BIN  obdd_parallel_apply_omp
#endif
