/* ================================================================
 *  apply_cache.cpp  – memoisation table per obdd_apply
 *
 *  MODE 0 (default) : cache globale + mutex (vecchia versione)
 *  MODE 1 (OPT)     : cache thread_local  + merge finale
 *                     attiva se il compilatore vede -DOBDD_PER_THREAD_CACHE
 * ================================================================*/
#include "obdd.hpp"
#include <unordered_map>
#include <vector>
#include <mutex>

/* ---------- struttura chiave -----------------------------------*/
struct Key {
    const OBDDNode* a;
    const OBDDNode* b;
    int             op;
    bool operator==(const Key& k) const { return a==k.a && b==k.b && op==k.op; }
};

struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept {
        return std::hash<const void*>()(k.a) ^ (std::hash<const void*>()(k.b)<<1) ^ k.op;
    }
};

/* ---------- modalità GLOBAL vs THREAD_local --------------------*/
#if defined(DOBDD_PER_THREAD_CACHE)
/* ---- 1) cache privata al thread --------------------------------*/
static thread_local std::unordered_map<Key,OBDDNode*,KeyHash> tl_cache;

/*  In release si preferisce un vector per evitare malloc in merge */
static std::vector<std::unordered_map<Key,OBDDNode*,KeyHash>*> all_caches;
static std::mutex reg_mtx;    /* protegge all_caches */

static void register_cache() {
    std::lock_guard<std::mutex> g(reg_mtx);
    all_caches.push_back(&tl_cache);
}

/*  Merge: chiamata una sola volta nel thread master  */
void apply_cache_global_merge()
{
    std::unordered_map<Key,OBDDNode*,KeyHash> global;
    for (auto* pc : all_caches)
        global.insert(pc->begin(), pc->end());
    /* swap nei TLS dei worker = visibile anche in seq successivi  */
    for (auto* pc : all_caches)
        pc->swap(global);
}

/* ---- lookup / insert ------------------------------------------*/
OBDDNode* apply_cache_lookup(const OBDDNode* a,const OBDDNode* b,int op)
{
    if (tl_cache.empty()) register_cache();
    Key k{a,b,op};
    auto it = tl_cache.find(k);
    return (it==tl_cache.end()) ? nullptr : it->second;
}

void apply_cache_insert(const OBDDNode* a,const OBDDNode* b,int op,OBDDNode* r)
{
    tl_cache.emplace(Key{a,b,op}, r);
}

#else   /* ---------- vecchia modalità globale + mutex ---------- */

static std::unordered_map<Key,OBDDNode*,KeyHash> apply_cache;
static std::mutex mtx;

OBDDNode* apply_cache_lookup(const OBDDNode* a,const OBDDNode* b,int op)
{
    std::lock_guard<std::mutex> g(mtx);
    auto it = apply_cache.find(Key{a,b,op});
    return (it==apply_cache.end()) ? nullptr : it->second;
}
void apply_cache_insert(const OBDDNode* a,const OBDDNode* b,int op,OBDDNode* r)
{
    std::lock_guard<std::mutex> g(mtx);
    apply_cache.emplace(Key{a,b,op}, r);
}
void apply_cache_global_merge() {}          /* no-op */
#endif

/* helper per debug / test */
void apply_cache_clear()
{
#if defined(DOBDD_PER_THREAD_CACHE)
    tl_cache.clear();
#else
    std::lock_guard<std::mutex> g(mtx);
    apply_cache.clear();
#endif
}
