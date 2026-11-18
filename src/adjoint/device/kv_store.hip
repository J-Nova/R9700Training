// src/adjoint/device/kv_store.hip -- kv_store's device adjoint.
//
// The same op as src/adjoint/host/kv_store.cpp, on the card: the forward writes k and v into the
// pool at each token's slot, so the cotangent of the pool flows back to the k and v the step
// wrote, gathered at those slots. A slot no token of this step wrote sends nothing back; a slot
// two tokens wrote sends each token its own row.
//
// The pool is [n_blocks, kv_heads, block_size, 2*head_dim]: k in a row's first `head_dim` lanes
// and v in the second (docs/adjoints/qk.md). The contract gives a device adjoint f64 operands, so
// this call owns its device buffers; they are scratch, freed before it returns.
#include "r97/adjoint.h"

#include "scratch.h"

#include "rad_device.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <variant>
#include <vector>

namespace r97 {
namespace {

int64_t param_int(const OpRecord& rec, const char* name, int64_t fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (auto* i = std::get_if<int64_t>(&p.second)) return *i;
    }
    return fallback;
}

// One thread a (token, head, lane): dk and dv are that lane's pool row, or zero where the token
// wrote no slot (a padded row).
__global__ void kv_store_adjoint_kernel(const float* d_pool, const int32_t* slot, float* d_k,
                                        float* d_v, int64_t tokens, int64_t kv_heads,
                                        int64_t head_dim, int64_t block_size) {
    const int64_t per = kv_heads * head_dim;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * per) return;
    const int64_t t = i / per;
    const int64_t r = i - t * per;
    const int64_t h = r / head_dim;
    const int64_t d = r - h * head_dim;
    const int64_t s = slot[t];
    if (s < 0) {
        d_k[i] = 0.0f;
        d_v[i] = 0.0f;
        return;
    }
    const int64_t block = s / block_size;
    const int64_t offset = s - block * block_size;
    const int64_t base = ((block * kv_heads + h) * block_size + offset) * 2 * head_dim;
    d_k[i] = d_pool[base + d];
    d_v[i] = d_pool[base + head_dim + d];
}

RadStream adjoint_stream() {
    static RadStream s = [] {
        RadStream created = nullptr;
        if (rad_stream_create(&created, 0) != 0) return (RadStream) nullptr;
        return created;
    }();
    return s;
}

int64_t blocks(int64_t n) { return (n + 255) / 256; }

void kv_store_device_adjoint(const AdjointCall& call) {
    if (!call.dc || call.dc->size() < 2 || call.cot.empty() || call.fwd.size() < 4) return;
    const AdValue& k = call.fwd[0];
    const AdValue& v = call.fwd[1];
    const AdValue& slot = call.fwd[2];
    const AdValue& pool = call.fwd[3];
    const AdValue& d_pool = call.cot[0];

    const int64_t tokens = k.shape.empty() ? 0 : k.shape[0];
    const int64_t kv_heads = param_int(call.rec, "n_head_kv", 0);
    const int64_t head_dim = param_int(call.rec, "head_dim", 0);
    const int64_t block_size = param_int(call.rec, "block_size", 16);
    if (tokens <= 0 || kv_heads <= 0 || head_dim <= 0 || block_size <= 0) return;
    if ((int64_t)slot.value.size() < tokens || pool.value.empty() || d_pool.value.empty()) return;
    if (d_pool.value.size() < pool.value.size()) return;

    const int64_t n = tokens * kv_heads * head_dim;
    AdValue& d_k = call.dc->at(0);
    AdValue& d_v = call.dc->at(1);
    d_k.dtype = k.dtype;
    d_k.shape = k.shape;
    d_k.value.assign((size_t)n, 0.0);
    d_v.dtype = v.dtype;
    d_v.shape = v.shape;
    d_v.value.assign((size_t)n, 0.0);

    RadStream stream = adjoint_stream();
    if (!stream) return;

    const int64_t pool_values = (int64_t)d_pool.value.size();
    device::ScratchBlock d_dp(pool_values * (int64_t)sizeof(float));
    device::ScratchBlock d_slot(tokens * (int64_t)sizeof(int32_t));
    device::ScratchBlock d_kv(2 * n * (int64_t)sizeof(float));
    if (!d_dp || !d_slot || !d_kv) return;

    std::vector<float> h_dp((size_t)pool_values);
    for (int64_t i = 0; i < pool_values; ++i) h_dp[(size_t)i] = (float)d_pool.value[(size_t)i];
    std::vector<int32_t> h_slot((size_t)tokens);
    for (int64_t t = 0; t < tokens; ++t) h_slot[(size_t)t] = (int32_t)slot.value[(size_t)t];
    if (rad_memcpy_async(d_dp.p, h_dp.data(), pool_values * (int64_t)sizeof(float), stream) != 0)
        return;
    if (rad_memcpy_async(d_slot.p, h_slot.data(), tokens * (int64_t)sizeof(int32_t), stream) != 0)
        return;

    float* dk = (float*)d_kv.p;
    float* dv = dk + n;
    hipLaunchKernelGGL(kv_store_adjoint_kernel, dim3((unsigned int)blocks(n)), dim3(256), 0,
                       (hipStream_t)stream, (const float*)d_dp.p, (const int32_t*)d_slot.p, dk, dv,
                       tokens, kv_heads, head_dim, block_size);
    if (rad_stream_sync(stream) != 0) return;

    std::vector<float> h_kv((size_t)(2 * n));
    if (rad_memcpy_async(h_kv.data(), d_kv.p, 2 * n * (int64_t)sizeof(float), stream) != 0) return;
    if (rad_stream_sync(stream) != 0) return;

    // The tape's cotangents are f64 and are added to, never overwritten.
    for (int64_t i = 0; i < n; ++i) d_k.value[(size_t)i] += (double)h_kv[(size_t)i];
    for (int64_t i = 0; i < n; ++i) d_v.value[(size_t)i] += (double)h_kv[(size_t)(n + i)];

    // The in-place pool's own slot: the store overwrites the rows its slots name, so those rows of
    // the old pool are dead -- their cotangent is the k's and v's above. Only the rows the write
    // leaves alone carry the pool's cotangent on, and filling the slot here is what stops the
    // walk's passthrough from copying the whole pool into it (the covered rows included). The same
    // routing as the host row (src/adjoint/host/kv_store.cpp), by the same (block, offset)
    // arithmetic the forward scatters with.
    const int64_t n_blocks = pool.shape.empty() ? 0 : pool.shape[0];
    const int64_t pool_rows = n_blocks * block_size;
    if (pool_rows > 0) {
        std::vector<char> covered((size_t)pool_rows, 0);
        for (int64_t t = 0; t < tokens; ++t) {
            const int64_t s = (int64_t)slot.value[(size_t)t];
            if (s >= 0 && s < pool_rows) covered[(size_t)s] = 1;
        }
        for (size_t j = 0; j < call.rec.operands.size(); ++j) {
            const Operand& o = call.rec.operands[j];
            if (!o.written || !o.in_place || j >= call.dc->size()) continue;
            AdValue& dp = call.dc->at(j);
            if (dp.value.size() != d_pool.value.size()) continue;
            for (int64_t row = 0; row < pool_rows; ++row) {
                if (covered[(size_t)row]) continue;
                const int64_t block = row / block_size;
                const int64_t off = row % block_size;
                for (int64_t h = 0; h < kv_heads; ++h) {
                    const int64_t base =
                        ((block * kv_heads + h) * block_size + off) * 2 * head_dim;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        dp.value[(size_t)(base + d)] += d_pool.value[(size_t)(base + d)];
                        dp.value[(size_t)(base + head_dim + d)] +=
                            d_pool.value[(size_t)(base + head_dim + d)];
                    }
                }
            }
        }
    }
}

}  // namespace

R97_REGISTER_ADJOINT_FF("kv_store", device, kv_store_device_adjoint, fills_in_place_slots, 2)  // slots is discrete

}  // namespace r97
