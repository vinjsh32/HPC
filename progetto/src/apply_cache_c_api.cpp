// Piccolo "collante" C-linkage che fornisce i simboli
// richiesti da obdd_core.cpp (array apply_cache[], apply_hash, apply_cache_clear)
//
// Non tocca la tua implementazione C++ in apply_cache.cpp.
// obdd_core.cpp linkerà questi simboli (C linkage) e tutto compila.

#include "apply_cache.hpp"
#include "obdd.hpp"
#include <cstring>
#include <cstdint>

extern "C" {

// definizione reale dell'array globale atteso da obdd_core.cpp
ApplyEntry apply_cache[CACHE_SIZE];

// svuota la cache (qui: semplicemente azzeriamo l'array)
void apply_cache_clear(void)
{
    std::memset(apply_cache, 0, sizeof(apply_cache));
}

// hash helper identico a quello che usavi in apply (compatibile con l'header)
size_t apply_hash(const OBDDNode* a, const OBDDNode* b, int op)
{
    uintptr_t aa = (uintptr_t)a >> 3;
    uintptr_t bb = (uintptr_t)b >> 3;
    return (aa ^ (bb << 1) ^ (uintptr_t)op) % CACHE_SIZE;
}

} // extern "C"
