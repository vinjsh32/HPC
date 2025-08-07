/* --------------------------------------------------------------------
 *  run_bench.cpp  – micro‑benchmark runner (CLI‑friendly revision, C++17)
 * --------------------------------------------------------------------
 *  Adds **command‑line options** so that external wrappers (e.g. bench_all.py)
 *  can control precisely what gets benchmarked and where the CSV is written.
 *
 *  Supported flags (all optional, order‑independent):
 *      --func NAME       majority | mux | adder | equality | parity | all
 *      --bits N [...]    one or more bit‑sizes (overrides defaults per func)
 *      --threads T       call omp_set_num_threads(T) before running
 *      --repeat R        repeat each measurement R times (default 3)
 *      --csvout PATH     append results to PATH  (default "results.csv")
 *
 *  Behaviour with *no* flags is identical to the old version: runs the full
 *  matrix   FUNCS × BITS   once with 1 thread and appends to results.csv.
 * ------------------------------------------------------------------ */

#include "obdd.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#ifdef OBDD_ENABLE_OPENMP
#   include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/*  Helper aliases for seq / omp                                      */
/* ------------------------------------------------------------------ */
#if defined(OBDD_ENABLE_OPENMP)
#   define APPLY_BIN  obdd_parallel_apply_omp
#   define APPLY_NOT  obdd_parallel_not_omp
#else
#   define APPLY_BIN  obdd_apply
#   define APPLY_NOT  nullptr
#endif

using Clock   = std::chrono::high_resolution_clock;
using ApplyBin = OBDDNode *(*)(const OBDD *, const OBDD *, OBDD_Op);

static inline OBDD make_bdd(OBDDNode *root, int vars, const int *ord) {
    return {root, vars, const_cast<int *>(ord)};
}
static inline OBDDNode *var(int idx) {
    return obdd_node_create(idx, obdd_constant(0), obdd_constant(1));
}

/* ==============================================================
 *  Benchmark circuits  (same definitions as original file)
 * ==============================================================*/
static OBDDNode *majority3(const int *ord, int /*bits*/, ApplyBin AB) {
    OBDD a = make_bdd(var(0), 3, ord);
    OBDD b = make_bdd(var(1), 3, ord);
    OBDD c = make_bdd(var(2), 3, ord);
    OBDDNode *ab = AB(&a, &b, OBDD_AND);
    OBDDNode *ac = AB(&a, &c, OBDD_AND);
    OBDDNode *bc = AB(&b, &c, OBDD_AND);
    OBDD abB = make_bdd(ab, 3, ord);
    OBDD acB = make_bdd(ac, 3, ord);
    OBDD bcB = make_bdd(bc, 3, ord);
    OBDDNode *t1 = AB(&abB, &acB, OBDD_OR);
    OBDD t1B = make_bdd(t1, 3, ord);
    return AB(&t1B, &bcB, OBDD_OR);
}

static OBDDNode *mux3(const int *ord, int /*bits*/, ApplyBin AB) {
    OBDD s = make_bdd(var(0), 3, ord);
    OBDD a = make_bdd(var(1), 3, ord);
    OBDD b = make_bdd(var(2), 3, ord);
    OBDDNode *ns_root = AB(&s, nullptr, OBDD_NOT);
    OBDD nsB = make_bdd(ns_root, 3, ord);
    OBDDNode *lhs = AB(&nsB, &a, OBDD_AND);
    OBDD lhsB = make_bdd(lhs, 3, ord);
    OBDDNode *rhs = AB(&s, &b, OBDD_AND);
    OBDD rhsB = make_bdd(rhs, 3, ord);
    return AB(&lhsB, &rhsB, OBDD_OR);
}

static OBDDNode *parity_bits(const int *ord, int bits, ApplyBin AB) {
    OBDD acc = make_bdd(var(0), bits, ord);
    for (int i = 1; i < bits; ++i) {
        OBDD xi = make_bdd(var(i), bits, ord);
        acc.root = AB(&acc, &xi, OBDD_XOR);
    }
    return acc.root;
}

/* Placeholder stubs for adder / equality until real implementations exist */
static OBDDNode *adder_stub(const int *ord, int bits, ApplyBin AB)    { return parity_bits(ord, bits, AB); }
static OBDDNode *equality_stub(const int *ord, int bits, ApplyBin AB) { return parity_bits(ord, bits, AB); }

/* Map string → builder */
using Builder = OBDDNode *(*)(const int *, int, ApplyBin);
static const std::map<std::string, Builder> BUILDERS = {
    {"majority", majority3},
    {"mux",      mux3},
    {"parity",   parity_bits},
    {"adder",    adder_stub},
    {"equality", equality_stub}
};

/* ------------------------------------------------------------------ */
struct Result { std::string func; int bits; int nodes; double ms; };

static Result run_one(const std::string &func, int bits, ApplyBin AB) {
    std::vector<int> order(bits); for (int i = 0; i < bits; ++i) order[i] = i;
    Builder B = BUILDERS.at(func);

    /* build + count nodes */
    OBDD dummy{nullptr, bits, order.data()};
    auto t0 = Clock::now();
    dummy.root = B(order.data(), bits, AB);
    auto t1 = Clock::now();

    std::vector<const OBDDNode *> st{dummy.root};
    int nodes = 0;
    while (!st.empty()) {
        const auto *p = st.back(); st.pop_back();
        if (!p || p->varIndex < 0) continue;
        ++nodes; st.push_back(p->lowChild); st.push_back(p->highChild);
    }

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {func, bits, nodes, ms};
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    /* ---------- default configuration ---------------------------- */
    std::vector<std::string> funcs = {"majority", "mux", "adder", "equality", "parity"};
    std::map<std::string, std::vector<int>> bits_map = {
        {"majority", {3}},
        {"mux",      {3}},
        {"adder",    {8, 16}},
        {"equality", {8, 16}},
        {"parity",   {16, 20}}
    };
    int repeat = 3;
    int threads_cli = 1;
    std::string csvOut = "results.csv";

    /* ---------- CLI parsing -------------------------------------- */
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--func") == 0 && i + 1 < argc) {
            funcs = {argv[++i]};
        } else if (std::strcmp(argv[i], "--bits") == 0 && i + 1 < argc) {
            bits_map[funcs.front()].clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                bits_map[funcs.front()].push_back(std::atoi(argv[++i]));
            }
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads_cli = std::atoi(argv[++i]);
#ifdef OBDD_ENABLE_OPENMP
            omp_set_num_threads(threads_cli);
#endif
        } else if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--csvout") == 0 && i + 1 < argc) {
            csvOut = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete flag: " << argv[i] << '\n';
            return 1;
        }
    }

    /* ---------- run benchmarks ----------------------------------- */
    std::ofstream csv(csvOut, std::ios::app);
    if (csv.tellp() == 0) csv << "func,bits,threads,nodes,ms\n";

    for (const auto &f : funcs) {
        for (int bits : bits_map[f]) {
            for (int r = 0; r < repeat; ++r) {
                Result res = run_one(f, bits, APPLY_BIN);
                std::cout << std::left << std::setw(10) << res.func << "  "
                          << std::setw(4)  << res.bits << "  "
                          << std::setw(9) << res.nodes << "  "
                          << res.ms << " ms\n";

                csv << res.func << ',' << res.bits << ','
                    << threads_cli << ',' << res.nodes << ',' << res.ms << '\n';
            }
        }
    }
    return 0;
}
#if defined(OBDD_ENABLE_OPENMP) && defined(OBDD_USE_OPTIM)
#   define APPLY_BIN  obdd_parallel_apply_omp_optim
#else
#   define APPLY_BIN  obdd_parallel_apply_omp
#endif
