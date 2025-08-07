#pragma once
#ifndef APPLY_CACHE_HPP
#define APPLY_CACHE_HPP

#include "obdd.hpp"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 *  Memoization cache per la funzione apply (host e, in futuro, device).
 * -------------------------------------------------------------------------- */

/* Includiamo nuovamente la definizione di ApplyEntry se non già presente */
#ifndef APPLY_ENTRY_DECLARED
#define APPLY_ENTRY_DECLARED
typedef struct {
    const OBDDNode* a;
    const OBDDNode* b;  /* NULL when op==NOT */
    int             op; /* 0=AND 1=OR 2=NOT 3=XOR */
    OBDDNode*       result;
} ApplyEntry;
#endif

#define CACHE_SIZE 10007

extern ApplyEntry apply_cache[CACHE_SIZE];

/* Svuota la cache */
void apply_cache_clear(void);

/* Hash helper (uguale a quello che usavi in apply) */
size_t apply_hash(const OBDDNode* a, const OBDDNode* b, int op);

#ifdef __cplusplus
}
#endif

#endif /* APPLY_CACHE_HPP */
