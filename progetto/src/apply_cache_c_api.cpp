#include "apply_cache.hpp"
#include "obdd.hpp"
#include <unordered_map>
#include <mutex>
#include <cstdint>

struct ApplyKey {
    const OBDDNode* a;
    const OBDDNode* b;
    int op;
    bool operator==(const ApplyKey& other) const {
        return a == other.a && b == other.b && op == other.op;
    }
};

struct ApplyKeyHash {
    std::size_t operator()(const ApplyKey& k) const noexcept {
        std::uintptr_t aa = reinterpret_cast<std::uintptr_t>(k.a) >> 3;
        std::uintptr_t bb = reinterpret_cast<std::uintptr_t>(k.b) >> 3;
        return aa ^ (bb << 1) ^ static_cast<std::uintptr_t>(k.op);
    }
};

static std::unordered_map<ApplyKey, OBDDNode*, ApplyKeyHash> apply_cache;
static std::mutex cache_mtx;

extern "C" {

void apply_cache_clear(void) {
    std::lock_guard<std::mutex> g(cache_mtx);
    apply_cache.clear();
}

OBDDNode* apply_cache_lookup(const OBDDNode* a, const OBDDNode* b, int op) {
    std::lock_guard<std::mutex> g(cache_mtx);
    auto it = apply_cache.find({a, b, op});
    return (it == apply_cache.end()) ? nullptr : it->second;
}

void apply_cache_insert(const OBDDNode* a, const OBDDNode* b, int op, OBDDNode* result) {
    std::lock_guard<std::mutex> g(cache_mtx);
    apply_cache[{a, b, op}] = result;
}

} // extern "C"
