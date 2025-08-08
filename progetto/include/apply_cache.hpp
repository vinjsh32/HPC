#pragma once
#ifndef APPLY_CACHE_HPP
#define APPLY_CACHE_HPP

#include "obdd.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 *  Memoization cache per la funzione apply.
 *  Ogni thread mantiene una cache locale (thread_local) senza lock.
 *  Prima di usare la cache in una regione parallela chiamare
 *  apply_cache_thread_init() e, al termine, unire le cache con
 *  apply_cache_merge().
 * -------------------------------------------------------------------------- */

/* Svuota la cache del thread corrente e azzera i registri delle TLS */
void apply_cache_clear(void);

/* Inizializza la cache locale del thread e la registra per la merge finale */
void apply_cache_thread_init(void);

/* Lookup nella cache locale: ritorna il risultato memorizzato oppure NULL */
OBDDNode* apply_cache_lookup(const OBDDNode* a, const OBDDNode* b, int op);

/* Inserisce nella cache locale */
void apply_cache_insert(const OBDDNode* a, const OBDDNode* b, int op, OBDDNode* result);

/* Merge di tutte le cache locali nel thread master */
void apply_cache_merge(void);

#ifdef __cplusplus
}
#endif

#endif /* APPLY_CACHE_HPP */
