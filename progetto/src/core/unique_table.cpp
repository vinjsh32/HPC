/**
 * @file unique_table.cpp
 * @brief Tabella "unique" (host‑side) per garantire la canonicità dei nodi.
 *
 * Implementa:
 *   • array globale `unique_table[]` (open addressing, size = UNIQUE_SIZE)
 *   • `unique_table_clear()`        – azzera lo storage
 *   • `unique_table_get_or_create()` – lookup+insert su tripla (var,low,high)
 *
 *  Al momento esiste solo la variante host; se in futuro servirà un backend
 *  device (CUDA) si potrà allocare una tabella analoga in memoria globale GPU
 *  e passare il puntatore al kernel.
 */

#include "core/unique_table.hpp"
#include <cstring>   /* memset */
#include <cstdint>   /* uintptr_t */

/* --------------------------------------------------------------------------
 *  Storage reale (host).
 * -------------------------------------------------------------------------- */
UniqueEntry unique_table[UNIQUE_SIZE];

/* --------------------------------------------------------------------------
 *  API pubbliche (linkage C)
 * -------------------------------------------------------------------------- */
extern "C" {

void unique_table_clear(void)
{
    std::memset(unique_table, 0, sizeof(unique_table));
}

OBDDNode* unique_table_get_or_create(int var,
                                     OBDDNode* low,
                                     OBDDNode* high)
{
    /* Regola di riduzione: se i due figli coincidono ⇒ ritorna direttamente */
    if (low == high) return low;

    size_t h = triple_hash(var, low, high);
    for (;;) {
        UniqueEntry& slot = unique_table[h];
        if (!slot.result) {
            /* Slot vuoto → inserisce nuovo nodo canonico */
            slot.var  = var;
            slot.low  = low;
            slot.high = high;
            slot.result = obdd_node_create(var, low, high);
            return slot.result;
        }
        if (slot.var == var && slot.low == low && slot.high == high)
            return slot.result;          /* già presente */

        /* linear probing */
        h = (h + 1) % UNIQUE_SIZE;
    }
}

} /* extern "C" */
