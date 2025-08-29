/**
 * @file obdd.hpp
 * @brief High-Performance Ordered Binary Decision Diagrams (OBDD) Library - Core Public API
 * 
 * This header defines the primary public interface for the OBDD library, providing
 * a comprehensive set of operations for constructing, manipulating, and analyzing
 * Binary Decision Diagrams. The library supports multiple computational backends:
 * 
 * - Sequential CPU: Optimized single-threaded implementation
 * - OpenMP Parallel: Multi-threaded CPU parallelization  
 * - CUDA GPU: Massively parallel GPU acceleration
 * 
 * Key Features:
 * - Canonical ROBDD (Reduced Ordered BDD) representation
 * - Advanced memoization with apply cache and unique table
 * - Thread-safe operations with configurable backends
 * - C/C++ hybrid design for maximum compatibility
 * - Comprehensive memory management with reference counting
 * - Variable reordering algorithms for optimization
 * - Performance benchmarking and profiling integration
 * 
 * Design Philosophy:
 * The API maintains clean separation between public interface and internal 
 * implementation details. All operations preserve BDD canonicity and provide
 * deterministic results across different computational backends.
 * 
 * @author @vijsh32
 * @date August 15, 2024
 * @version 2.1
 * @copyright 2024 High Performance Computing Laboratory
 */

#pragma once
#ifndef OBDD_HPP
#define OBDD_HPP


#include <stddef.h>   /* size_t, NULL */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Logical operations supported by the OBDD library
 * 
 * This enumeration defines the fundamental Boolean operations that can be
 * performed on OBDD nodes. Each operation is assigned a specific integer
 * value for efficient memoization and cache indexing.
 * 
 * Operation Semantics:
 * - AND: Boolean conjunction (f ∧ g)
 * - OR:  Boolean disjunction (f ∨ g)  
 * - NOT: Boolean negation (¬f)
 * - XOR: Boolean exclusive-or (f ⊕ g)
 * 
 * Implementation Notes:
 * - Values are chosen for optimal hash distribution in apply cache
 * - NOT operation requires only one operand (second operand ignored)
 * - All operations preserve OBDD canonicity through reduction
 * 
 * @see obdd_apply() for operation implementation details
 */
typedef enum {
    OBDD_AND = 0,    /**< Boolean AND operation (conjunction) */
    OBDD_OR  = 1,    /**< Boolean OR operation (disjunction) */
    OBDD_NOT = 2,    /**< Boolean NOT operation (negation) */
    OBDD_XOR = 3     /**< Boolean XOR operation (exclusive-or) */
} OBDD_Op;

/**
 * @brief Core data structure representing a single node in an Ordered Binary Decision Diagram
 * 
 * This structure forms the fundamental building block of all OBDD operations.
 * Each node represents either a decision point (internal node) or a terminal
 * value (leaf node). The design prioritizes cache efficiency and memory locality.
 * 
 * Node Types:
 * - Decision Node: varIndex >= 0, represents a Boolean variable test
 * - Leaf Node: varIndex < 0, represents constant values (0 or 1)
 * 
 * Memory Layout:
 * - Optimized for 64-bit systems with natural alignment
 * - Total size: 24 bytes (3 words) on 64-bit architectures
 * - Cache-friendly linear layout for optimal performance
 * 
 * Thread Safety:
 * - Node creation and destruction are protected by global mutex
 * - Reference counting prevents premature deallocation
 * - Immutable after creation (functional data structure)
 * 
 * @note All nodes are managed through a global unique table to ensure
 *       canonical representation and efficient memory usage
 */
typedef struct OBDDNode {
    int             varIndex;   /**< Variable index for decision nodes, -1 for leaves */
    struct OBDDNode *highChild; /**< Right child (variable = 1 branch) */
    struct OBDDNode *lowChild;  /**< Left child (variable = 0 branch) */
    int             refCount;   /**< Reference count for memory management */
} OBDDNode;

/**
 * @brief Handle structure for complete OBDD representation with metadata
 * 
 * This structure encapsulates a complete Binary Decision Diagram along with
 * its associated metadata. It serves as the primary interface for all high-level
 * OBDD operations and maintains essential structural information.
 * 
 * Design Rationale:
 * - Separates tree structure (nodes) from metadata (variables, ordering)
 * - Enables multiple BDD views of the same node structure
 * - Supports efficient variable reordering without node reconstruction
 * - Provides clean abstraction for multi-backend implementations
 * 
 * Memory Management:
 * - Handle owns variable ordering array (must be freed)
 * - Root node follows standard reference counting protocol
 * - Structure itself allocated/freed by create/destroy functions
 * 
 * Thread Safety:
 * - Read-only operations are thread-safe after construction
 * - Modification requires external synchronization
 * - Variable reordering operations are not thread-safe
 * 
 * @see obdd_create() for construction details
 * @see obdd_destroy() for proper cleanup
 */
typedef struct OBDD {
    OBDDNode *root;     /**< Root node of the decision diagram */
    int       numVars;  /**< Number of Boolean variables in the function */
    int      *varOrder; /**< Variable ordering array (size = numVars) */
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
OBDDNode   *obdd_parallel_apply_omp_enhanced(const OBDD *bdd1,
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
