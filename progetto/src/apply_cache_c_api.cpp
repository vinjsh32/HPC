#include "apply_cache.hpp"
#include "obdd.hpp"
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstdint>

/* -------------------------- chiave hash ------------------------- */
struct ApplyKey {
    const OBDDNode* a;
    const OBDDNode* b;
    int             op;
    bool operator==(const ApplyKey& o) const {
        return a==o.a && b==o.b && op==o.op;
    }
};

struct ApplyKeyHash {
    std::size_t operator()(const ApplyKey& k) const noexcept {
        std::uintptr_t aa = reinterpret_cast<std::uintptr_t>(k.a) >> 3;
        std::uintptr_t bb = reinterpret_cast<std::uintptr_t>(k.b) >> 3;
        return aa ^ (bb << 1) ^ static_cast<std::uintptr_t>(k.op);
    }
};

using LocalCache = std::unordered_map<ApplyKey, OBDDNode*, ApplyKeyHash>;

/* TLS per ogni thread ------------------------------------------------------ */
static thread_local LocalCache tls_cache;

/* Elenco delle TLS da unire a fine regione parallela ---------------------- */
static std::vector<LocalCache*> g_tls;
static std::mutex               g_tls_mtx;

static void register_tls(LocalCache& c)
{
    std::lock_guard<std::mutex> g(g_tls_mtx);
    g_tls.push_back(&c);
}

extern "C" {

void apply_cache_clear(void)
{
    tls_cache.clear();
    std::lock_guard<std::mutex> g(g_tls_mtx);
    g_tls.clear();
}

void apply_cache_thread_init(void)
{
    tls_cache.clear();
    register_tls(tls_cache);
}

OBDDNode* apply_cache_lookup(const OBDDNode* a, const OBDDNode* b, int op)
{
    auto it = tls_cache.find({a,b,op});
    return (it==tls_cache.end()) ? nullptr : it->second;
}

void apply_cache_insert(const OBDDNode* a, const OBDDNode* b, int op, OBDDNode* result)
{
    tls_cache.emplace(ApplyKey{a,b,op}, result);
}

void apply_cache_merge(void)
{
    std::lock_guard<std::mutex> g(g_tls_mtx);
    if (g_tls.empty()) return;
    LocalCache& master = *g_tls.front();
    const std::size_t tls_size = g_tls.size();
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 1; i < tls_size; ++i) {
        LocalCache* c = g_tls[i];
        #pragma omp critical
        master.merge(*c);
    }
}

} // extern "C"

