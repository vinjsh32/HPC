#pragma once
#ifndef APPLY_CACHE_HPP
#define APPLY_CACHE_HPP

#include "obdd.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 *  Memoization cache per la funzione apply.
 *  Ora implementata con std::unordered_map e accesso thread-safe.
 * -------------------------------------------------------------------------- */

/* Svuota la cache globale */
void apply_cache_clear(void);

/* Lookup thread-safe: ritorna il risultato memorizzato oppure NULL */
OBDDNode* apply_cache_lookup(const OBDDNode* a, const OBDDNode* b, int op);

/* Inserisce nella cache in modo thread-safe */
void apply_cache_insert(const OBDDNode* a, const OBDDNode* b, int op, OBDDNode* result);

#ifdef __cplusplus
}
#endif

#endif /* APPLY_CACHE_HPP */
