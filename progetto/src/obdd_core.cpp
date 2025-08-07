/**
 * @file obdd_core.cpp
 * @brief Implementazione sequenziale dell’OBDD/ROBDD.
 *
 * Fornisce tutte le primitive dichiarate in **obdd.hpp**:
 *   - create / destroy
 *   - constant / node_create
 *   - evaluate
 *   - apply (AND/OR/NOT/XOR)
 *   - reduce (→ ROBDD canonico)
 *   - reorder_sifting (sifting naïf)
 *   - build_demo_bdd (piccolo esempio)
 *
 * NOTA:
 *  - Non c’è alcun namespace: le funzioni hanno linkage C perché
 *    l’header è pensato per essere usato anche da C puro;
 *    i simboli devono quindi restare “globali”.
 *  - I dettagli di memoizzazione sono privati: la struct `ApplyEntry`
 *    e la cache globale sono definiti in **apply_cache.cpp** e dichiarati
 *    `extern` in **apply_cache.hpp** (header interno, NON incluso dagli
 *    utenti della libreria).
 */

#include "obdd.hpp"          /* API pubblica              */
#include "apply_cache.hpp"   /* cache + hash globali      */

#include <cstring>   /* memcpy, memset */
#include <cstdlib>   /* malloc, free, exit */
#include <cstdio>    /* fprintf */
#include <climits>   /* INT_MAX */
#include <cstdint>   /* uintptr_t */
#include <unordered_set> /* tracking nodi allocati */

/* =============================================================
 *  Helper & globals
 * ============================================================*/

/* (Niente più definizione di apply_cache qui: vive in apply_cache.cpp) */

/* abort su OOM */
static void* xmalloc(size_t sz)
{
    void* p = std::malloc(sz);
    if (!p) {
        std::fprintf(stderr, "[OBDD] out-of-memory (%zu bytes)\n", sz);
        std::exit(EXIT_FAILURE);
    }
    return p;
}

/* =============================================================
 *  Costanti foglia (singleton 0 / 1)
 * ============================================================*/
static OBDDNode* g_falseLeaf = nullptr;
static OBDDNode* g_trueLeaf  = nullptr;

/* insieme globale dei nodi creati dinamicamente */
static std::unordered_set<OBDDNode*> g_all_nodes;
/* numero di BDD attivi, usato per sapere quando liberare le foglie */
static int g_bdd_count = 0;

static void free_nodes_rec(OBDDNode* node,
                           std::unordered_set<OBDDNode*>& visited)
{
    if (!node || visited.count(node)) return;
    if (node == g_falseLeaf || node == g_trueLeaf) return;
    visited.insert(node);
    free_nodes_rec(node->lowChild, visited);
    free_nodes_rec(node->highChild, visited);
    g_all_nodes.erase(node);
    std::free(node);
}

/* =============================================================
 *  API pubblica (linkage C)
 * ============================================================*/

extern "C" {

/* ------------------------ create / destroy ------------------ */
OBDD* obdd_create(int numVars, const int* varOrder)
{
    OBDD* b = static_cast<OBDD*>(xmalloc(sizeof(OBDD)));
    b->root     = nullptr;
    b->numVars  = numVars;
    b->varOrder = static_cast<int*>(xmalloc(sizeof(int) * numVars));
    std::memcpy(b->varOrder, varOrder, sizeof(int) * numVars);
    ++g_bdd_count;
    return b;
}

void obdd_destroy(OBDD* bdd)
{
    if (!bdd) return;

    std::unordered_set<OBDDNode*> visited;
    free_nodes_rec(bdd->root, visited);

    std::free(bdd->varOrder);
    std::free(bdd);

    if (--g_bdd_count == 0) {
        if (g_falseLeaf) { g_all_nodes.erase(g_falseLeaf); std::free(g_falseLeaf); g_falseLeaf = nullptr; }
        if (g_trueLeaf)  { g_all_nodes.erase(g_trueLeaf);  std::free(g_trueLeaf);  g_trueLeaf  = nullptr; }
    }
}

/* --------------------- constant / node_create --------------- */
OBDDNode* obdd_constant(int value)
{
    if (!g_falseLeaf) {
        g_falseLeaf = static_cast<OBDDNode*>(xmalloc(sizeof(OBDDNode)));
        g_falseLeaf->varIndex  = -1;
        g_falseLeaf->lowChild  = nullptr;
        g_falseLeaf->highChild = nullptr;
        g_all_nodes.insert(g_falseLeaf);
    }
    if (!g_trueLeaf) {
        g_trueLeaf = static_cast<OBDDNode*>(xmalloc(sizeof(OBDDNode)));
        g_trueLeaf->varIndex  = -1;
        /* puntatori qualunque != nullptr per riconoscere la foglia */
        g_trueLeaf->lowChild  = reinterpret_cast<OBDDNode*>(0x1);
        g_trueLeaf->highChild = reinterpret_cast<OBDDNode*>(0x1);
        g_all_nodes.insert(g_trueLeaf);
    }
    return value ? g_trueLeaf : g_falseLeaf;
}

OBDDNode* obdd_node_create(int varIndex, OBDDNode* low, OBDDNode* high)
{
    OBDDNode* n = static_cast<OBDDNode*>(xmalloc(sizeof(OBDDNode)));
    n->varIndex  = varIndex;
    n->lowChild  = low;
    n->highChild = high;
    g_all_nodes.insert(n);
    return n;
}

/* --------------------------- utils -------------------------- */
int is_leaf(const OBDDNode* node)
{
    return (!node || node->varIndex < 0);
}

OBDDNode* apply_leaf(OBDD_Op op, int v1, int v2)
{
    switch (op) {
        case OBDD_AND: return obdd_constant(v1 && v2);
        case OBDD_OR:  return obdd_constant(v1 || v2);
        case OBDD_NOT: return obdd_constant(!v1);      /* v2 ignorato */
        case OBDD_XOR: return obdd_constant(v1 ^ v2);
        default:       return obdd_constant(0);
    }
}

/* -------------------------- evaluate ------------------------ */
int obdd_evaluate(const OBDD* bdd, const int* assignment)
{
    const OBDDNode* cur = bdd ? bdd->root : nullptr;
    while (cur && cur->varIndex >= 0) {
        const int val = assignment[cur->varIndex];
        cur = (val == 0) ? cur->lowChild : cur->highChild;
    }
    if (!cur) return 0;
    return (cur == g_trueLeaf) ? 1 : 0;
}

/* --------------------------- apply -------------------------- */
static OBDDNode* obdd_apply_internal(const OBDDNode* n1,
                                     const OBDDNode* n2,
                                     OBDD_Op         op);

static OBDDNode* obdd_apply_internal(const OBDDNode* n1,
                                     const OBDDNode* n2,
                                     OBDD_Op         op)
{
    /* NOT unario: n2 == nullptr */
    if (op == OBDD_NOT && !n2) {
        size_t h = apply_hash(n1, nullptr, op);
        if (apply_cache[h].a == n1 && apply_cache[h].b == nullptr && apply_cache[h].op == op)
            return apply_cache[h].result;

        if (is_leaf(n1)) {
            int v1 = (n1 == obdd_constant(1));
            OBDDNode* res = obdd_constant(!v1);
            apply_cache[h] = { n1, nullptr, op, res };
            return res;
        }
        OBDDNode* l  = obdd_apply_internal(n1->lowChild,  nullptr, op);
        OBDDNode* hN = obdd_apply_internal(n1->highChild, nullptr, op);
        OBDDNode* newN = obdd_node_create(n1->varIndex, l, hN);
        apply_cache[h] = { n1, nullptr, op, newN };
        return newN;
    }

    /* memo lookup */
    size_t h = apply_hash(n1, n2, op);
    if (apply_cache[h].a == n1 && apply_cache[h].b == n2 && apply_cache[h].op == op)
        return apply_cache[h].result;

    /* foglie */
    bool leaf1 = is_leaf(n1);
    bool leaf2 = (n2 && is_leaf(n2));

    if (leaf1 && leaf2) {
        int v1 = (n1 == obdd_constant(1));
        int v2 = (n2 == obdd_constant(1));
        OBDDNode* res = apply_leaf(op, v1, v2);
        apply_cache[h] = { n1, n2, op, res };
        return res;
    }

    int v1 = leaf1 ? INT_MAX : n1->varIndex;
    int v2 = (n2 && !leaf2) ? n2->varIndex : INT_MAX;

    int topVar = (v1 < v2) ? v1 : v2;

    const OBDDNode* n1_low  = (v1 == topVar) ? n1->lowChild  : n1;
    const OBDDNode* n1_high = (v1 == topVar) ? n1->highChild : n1;
    const OBDDNode* n2_low  = (v2 == topVar) ? n2->lowChild  : n2;
    const OBDDNode* n2_high = (v2 == topVar) ? n2->highChild : n2;

    OBDDNode* lowRes  = obdd_apply_internal(n1_low,  n2_low,  op);
    OBDDNode* highRes = obdd_apply_internal(n1_high, n2_high, op);

    if (lowRes == highRes) {
        apply_cache[h] = { n1, n2, op, lowRes };
        return lowRes;
    }

    OBDDNode* res = obdd_node_create(topVar, lowRes, highRes);
    apply_cache[h] = { n1, n2, op, res };
    return res;
}

OBDDNode* obdd_apply(const OBDD* bdd1, const OBDD* bdd2, OBDD_Op opType)
{
    apply_cache_clear();                       /* pulizia globale */
    const OBDDNode* n2 = bdd2 ? bdd2->root : nullptr;
    return obdd_apply_internal(bdd1->root, n2, opType);
}

/* --------------------------- reduce ------------------------- */
#define UNIQUE_SIZE 10007

struct UniqueEntry {
    int var;
    const OBDDNode* low;
    const OBDDNode* high;
    OBDDNode*       result;
};
static UniqueEntry unique_table[UNIQUE_SIZE];

static inline void unique_clear() { std::memset(unique_table, 0, sizeof(unique_table)); }
static inline size_t triple_hash(int var, const OBDDNode* l, const OBDDNode* h)
{
    uintptr_t a = reinterpret_cast<uintptr_t>(l) >> 3;
    uintptr_t b = reinterpret_cast<uintptr_t>(h) >> 3;
    return (a ^ b ^ var) % UNIQUE_SIZE;
}

static OBDDNode* reduce_rec(OBDDNode* root)
{
    if (root->varIndex < 0) return root;

    OBDDNode* l = reduce_rec(root->lowChild);
    OBDDNode* h = reduce_rec(root->highChild);
    if (l == h) return l;

    size_t idx = triple_hash(root->varIndex, l, h);
    for (;;) {
        if (!unique_table[idx].result) {
            unique_table[idx] = { root->varIndex, l, h, nullptr };
            unique_table[idx].result = obdd_node_create(root->varIndex, l, h);
            return unique_table[idx].result;
        }
        if (unique_table[idx].var == root->varIndex &&
            unique_table[idx].low == l &&
            unique_table[idx].high == h)
            return unique_table[idx].result;
        idx = (idx + 1) % UNIQUE_SIZE;
    }
}

OBDDNode* obdd_reduce(OBDDNode* root)
{
    if (!root) return nullptr;
    unique_clear();
    return reduce_rec(root);
}

/* ----------------------- reorder_sifting -------------------- */
static void moveVar(int* arr, int fromPos, int toPos, int n)
{
    if (fromPos == toPos) return;
    int tmp = arr[fromPos];
    if (fromPos < toPos) {
        for (int i = fromPos; i < toPos; ++i) arr[i] = arr[i + 1];
        arr[toPos] = tmp;
    } else {
        for (int i = fromPos; i > toPos; --i) arr[i] = arr[i - 1];
        arr[toPos] = tmp;
    }
}

OBDDNode* reorder_sifting(OBDD* bdd,
                          OBDDNode* (*buildFunc)(const int*, int),
                          OBDDNode* (*reduceFunc)(OBDDNode*),
                          int       (*sizeFunc)(const OBDDNode*))
{
    if (!bdd || bdd->numVars < 2) return bdd ? bdd->root : nullptr;

    OBDDNode* bestRoot = reduceFunc(buildFunc(bdd->varOrder, bdd->numVars));
    int bestGlobal = sizeFunc(bestRoot);
    const int n = bdd->numVars;

    for (int i = 0; i < n; ++i) {
        int bestPos = i, bestSize = bestGlobal;
        for (int newPos = 0; newPos < n; ++newPos) {
            if (newPos == i) continue;
            moveVar(bdd->varOrder, i, newPos, n);
            OBDDNode* tmp = reduceFunc(buildFunc(bdd->varOrder, n));
            int sz = sizeFunc(tmp);
            if (sz < bestSize) { bestSize = sz; bestPos = newPos; }
            moveVar(bdd->varOrder, newPos, i, n);
        }
        moveVar(bdd->varOrder, i, bestPos, n);
        bestRoot   = reduceFunc(buildFunc(bdd->varOrder, n));
        bestGlobal = bestSize;
    }
    bdd->root = bestRoot;
    return bestRoot;
}

/* ----------------------- build_demo_bdd --------------------- */
static OBDDNode* make_var(int idx)
{
    return (idx < 0) ? obdd_constant(0)
                     : obdd_node_create(idx, obdd_constant(0), obdd_constant(1));
}

OBDDNode* build_demo_bdd(const int* varOrder, int numVars)
{
    int idx0 = -1, idx1 = -1, idx2 = -1, idx4 = -1;
    for (int i = 0; i < numVars; ++i) {
        if (varOrder[i] == 0) idx0 = i;
        if (varOrder[i] == 1) idx1 = i;
        if (varOrder[i] == 2) idx2 = i;
        if (varOrder[i] == 4) idx4 = i;
    }

    OBDDNode* x0 = make_var(idx0);
    OBDDNode* x1 = make_var(idx1);
    OBDDNode* x2 = make_var(idx2);
    OBDDNode* x4 = make_var(idx4);

    OBDD bX0 = { x0, numVars, const_cast<int*>(varOrder) };
    OBDD bX1 = { x1, numVars, const_cast<int*>(varOrder) };
    OBDD bX2 = { x2, numVars, const_cast<int*>(varOrder) };
    OBDD bX4 = { x4, numVars, const_cast<int*>(varOrder) };

    /* A = x0 AND x1 */
    OBDDNode* A = obdd_apply(&bX0, &bX1, OBDD_AND);
    /* B = x2 XOR x4 */
    OBDDNode* B = obdd_apply(&bX2, &bX4, OBDD_XOR);

    OBDD bA = { A, numVars, const_cast<int*>(varOrder) };
    OBDD bB = { B, numVars, const_cast<int*>(varOrder) };

    /* f = A OR B */
    return obdd_apply(&bA, &bB, OBDD_OR);
}

} /* extern "C" */

extern "C" size_t obdd_nodes_tracked()
{
    return g_all_nodes.size();
}
