#pragma once
#ifndef OBDD_HPP
#define OBDD_HPP

/*
 * Interfaccia pubblica della libreria OBDD/ROBDD.
 *
 * ‑ Compatibile sia con C che con C++ (grazie al blocco extern "C").
 * ‑ Non espone dettagli interni di memoizzazione o strutture ausiliarie;
 *    chi usa la libreria vede solo i tipi e le API "user‑facing".
 */

#include <stddef.h>   /* size_t, NULL */

#ifdef __cplusplus
extern "C" {
#endif

/* =====================================================
   ENUM: operazioni logiche supportate
   ===================================================== */

typedef enum {
    OBDD_AND = 0,
    OBDD_OR  = 1,
    OBDD_NOT = 2,
    OBDD_XOR = 3
} OBDD_Op;

/* =====================================================
   STRUTTURE DATI PRINCIPALI
   ===================================================== */

/**
 * Nodo di un Ordered BDD.
 *  ‑ varIndex < 0  ⇒ foglia (costante 0 o 1)
 *  ‑ highChild punta al ramo TRUE
 *  ‑ lowChild  punta al ramo FALSE
 */
typedef struct OBDDNode {
    int             varIndex;   /* indice della variabile, -1 leaf */
    struct OBDDNode *highChild; /* branch "1" */
    struct OBDDNode *lowChild;  /* branch "0" */
    int             refCount;   /* reference count */
} OBDDNode;

/**
 * Handle dell'intero OBDD: radice + meta‑info.
 */
typedef struct OBDD {
    OBDDNode *root;     /* radice del grafo */
    int       numVars;  /* # variabili booleane */
    int      *varOrder; /* permutazione delle variabili (size = numVars) */
} OBDD;

/* Shortcut per accedere alle foglie costanti */
#define OBDD_FALSE obdd_constant(0)
#define OBDD_TRUE  obdd_constant(1)

/* =====================================================
   API SEQUENZIALI CORE
   ===================================================== */

OBDD       *obdd_create(int numVars, const int *varOrder);
void        obdd_destroy(OBDD *bdd);

OBDDNode   *obdd_constant(int value); /* 0 o 1 */
OBDDNode   *obdd_node_create(int varIndex,
                             OBDDNode *low,
                             OBDDNode *high);

int         obdd_evaluate(const OBDD *bdd,
                          const int  *assignment); /* ritorna 0/1 */

OBDDNode   *obdd_apply(const OBDD *bdd1,
                       const OBDD *bdd2,
                       OBDD_Op     opType);

OBDDNode   *obdd_reduce(OBDDNode *root);          /* ROBDD canonicalisation */

OBDDNode   *reorder_sifting(OBDD            *bdd,
                            OBDDNode *(*buildFunc)(const int *, int),
                            OBDDNode *(*reduceFunc)(OBDDNode *),
                            int       (*sizeFunc)(const OBDDNode*));

/* Builder di esempio usato nei test/benchmark */
OBDDNode   *build_demo_bdd(const int *varOrder, int numVars);

/* =====================================================
   API PARALLELE (OpenMP) — opzionali
   ===================================================== */
#ifdef OBDD_ENABLE_OPENMP
OBDDNode* obdd_parallel_apply_omp_optim(const OBDD* A,
                                        const OBDD* B,
                                        OBDD_Op      op,
                                        size_t       approx_nodes = 0);
void        obdd_parallel_var_ordering_omp(OBDD *bdd);         /* parallel merge sort */
OBDDNode   *obdd_parallel_and_omp(const OBDD *bdd1, const OBDD *bdd2);
OBDDNode   *obdd_parallel_or_omp (const OBDD *bdd1, const OBDD *bdd2);
OBDDNode   *obdd_parallel_not_omp(const OBDD *bdd);
OBDDNode   *obdd_parallel_xor_omp(const OBDD *bdd1, const OBDD *bdd2);
OBDDNode   *obdd_parallel_apply_omp(const OBDD *bdd1,
                                    const OBDD *bdd2,
                                    OBDD_Op     op);
#endif /* OBDD_ENABLE_OPENMP */

/* =====================================================
   HELPER UTILITIES (rese pubbliche per i test)
   ===================================================== */
int         is_leaf(const OBDDNode *node);
OBDDNode   *apply_leaf(OBDD_Op op, int v1, int v2);
size_t      obdd_nodes_tracked(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OBDD_HPP */
