// src/adjoint/host/qk_rows.cpp -- the tape rows for the qk family's two stateful ops, `rope` and
// `kv_store`. They live here rather than in qk.cpp because the family's f64 twins and adjoints are
// another card's file; this file only adds the rows the host step's walk needs. docs/adjoints/qk.md
// carries the formulas and says which live operand is which.
//
// A live record's INOUT operand keeps its input off the read list a TwinFn is handed, so the twins
// in qk.cpp cannot be called from the walk; these rows take the schema's operands directly.
#include "r97/adjoint.h"
#include "qk.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

// ------------------------------------------------------------------ the tape rows
//
// The family's f64 twins above are driven directly by tests/host/adjoint_qk.cpp, because an INOUT
// operand (rope's q|k, kv_store's pool) keeps its input off the read list a TwinFn is handed. The
// backward walk still needs an adjoint row for each op on the step's path.
namespace r97 {
namespace {

int64_t int_param(const OpRecord& rec, const char* name, int64_t fallback) {
    for (const OpParam& p : rec.params)
        if (p.first == name)
            if (const int64_t* v = std::get_if<int64_t>(&p.second)) return *v;
    return fallback;
}

std::string text_param(const OpRecord& rec, const char* name, const char* fallback) {
    for (const OpParam& p : rec.params)
        if (p.first == name)
            if (const std::string* v = std::get_if<std::string>(&p.second)) return *v;
    return fallback;
}

/* The schema names the rotary base `theta`. Read it so a record cannot silently take the kernel's
 * default when the container states another base. */
double theta_param(const OpRecord& rec) {
    for (const OpParam& p : rec.params) {
        if (p.first != "theta") continue;
        if (const int64_t* v = std::get_if<int64_t>(&p.second)) return static_cast<double>(*v);
        if (const std::string* v = std::get_if<std::string>(&p.second))
            return std::strtod(v->c_str(), nullptr);
    }
    return 10000.0;
}

std::vector<int32_t> ints_at(const AdValue& v, int64_t row_lo, int64_t n) {
    std::vector<int32_t> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const size_t at = static_cast<size_t>(row_lo + i);
        out[static_cast<size_t>(i)] =
            at < v.value.size() ? static_cast<int32_t>(v.value[at]) : 0;
    }
    return out;
}

/* rope rotates q|k in place: the cotangent of the written buffer is what the unrotated value
 * takes, so the adjoint puts the rotated-back cotangent in the write's slot. */
void rope_host_adjoint(const AdjointCall& call) {
    const OpRecord& rec = call.rec;
    if (call.fwd.size() < 2 || call.cot.empty()) return;
    const AdValue& x = call.fwd[0];
    const int64_t tokens = rec.n > 0 ? rec.n : (x.shape.empty() ? 0 : x.shape[0]);
    const int64_t head_dim = int_param(rec, "head_dim", x.shape.size() == 3 ? x.shape[2] : 0);
    const int64_t n_head = int_param(rec, "n_head", x.shape.size() == 3 ? x.shape[1] : 0);
    const int64_t n_head_kv = int_param(rec, "n_head_kv", 0);
    const int64_t rotary_dim = int_param(rec, "rotary_dim", head_dim);
    if (tokens <= 0 || head_dim <= 0 || n_head <= 0 || rotary_dim <= 0) return;

    const int64_t row_lo = rec.operands[0].row_lo;
    const std::vector<int32_t> positions = ints_at(call.fwd[1], rec.operands[1].row_lo, tokens);
    int32_t max_position = 0;
    for (int32_t p : positions) max_position = p > max_position ? p : max_position;

    const adjoint::qk::RopeShape shape{tokens, n_head, n_head_kv, head_dim, rotary_dim};
    const adjoint::qk::RopeTable table =
        adjoint::qk::rope_table_f64(max_position + 1, rotary_dim, theta_param(rec));

    std::vector<double> d_window(static_cast<size_t>(tokens) *
                                 static_cast<size_t>(n_head + 2 * n_head_kv) *
                                 static_cast<size_t>(head_dim));
    adjoint::qk::rope_adjoint_f64(call.cot[0].value.data(), positions.data(), shape, table,
                                  d_window.data());

    AdValue& d_x = (*call.dc)[0];
    const int64_t row = (n_head + 2 * n_head_kv) * head_dim;
    if (d_x.value.size() == d_window.size()) {
        for (size_t i = 0; i < d_window.size(); ++i) d_x.value[i] += d_window[i];
    } else if (static_cast<int64_t>(d_x.value.size()) >= (row_lo + tokens) * row) {
        for (int64_t t = 0; t < tokens; ++t)
            for (int64_t k = 0; k < row; ++k)
                d_x.value[static_cast<size_t>((row_lo + t) * row + k)] +=
                    d_window[static_cast<size_t>(t * row + k)];
    }
}

/* kv_store writes the paged pool and hands the pool's cotangent back to k and v. Its pool is
 * INOUT, but it replaces the slots it writes rather than reading their old values, so the write's
 * slot stays zero and the walk does not pass a cotangent to the previous pool. */
void kv_store_host_adjoint(const AdjointCall& call) {
    const OpRecord& rec = call.rec;
    if (call.fwd.size() < 4 || call.cot.empty()) return;
    const AdValue& k = call.fwd[0];
    const int64_t tokens = rec.n > 0 ? rec.n : (k.shape.empty() ? 0 : k.shape[0]);
    if (tokens <= 0) return;
    const int64_t kv_heads = int_param(rec, "n_head_kv", k.shape.size() == 3 ? k.shape[1] : 0);
    const int64_t head_dim = int_param(rec, "head_dim", k.shape.size() == 3 ? k.shape[2] : 0);
    const int64_t block_size = int_param(rec, "block_size", 16);
    const AdValue& pool = call.fwd[3];
    const int64_t n_blocks = pool.shape.empty() ? 0 : pool.shape[0];
    if (kv_heads <= 0 || head_dim <= 0 || n_blocks <= 0) return;

    const std::vector<int32_t> slots = ints_at(call.fwd[2], rec.operands[2].row_lo, tokens);
    const adjoint::qk::KvStoreShape shape{tokens, kv_heads, head_dim, block_size, n_blocks};
    const int64_t row = kv_heads * head_dim;
    std::vector<double> d_k(static_cast<size_t>(tokens * row), 0.0);
    std::vector<double> d_v(static_cast<size_t>(tokens * row), 0.0);
    adjoint::qk::kv_store_adjoint_f64(call.cot[0].value.data(), slots.data(), shape, d_k.data(),
                                      d_v.data());

    const int64_t row_lo = rec.operands[0].row_lo;
    for (int64_t which = 0; which < 2; ++which) {
        AdValue& d = (*call.dc)[static_cast<size_t>(which)];
        const std::vector<double>& src = which == 0 ? d_k : d_v;
        if (d.value.size() == src.size()) {
            for (size_t i = 0; i < src.size(); ++i) d.value[i] += src[i];
        } else if (static_cast<int64_t>(d.value.size()) >= (row_lo + tokens) * row) {
            for (int64_t t = 0; t < tokens; ++t)
                for (int64_t j = 0; j < row; ++j)
                    d.value[static_cast<size_t>((row_lo + t) * row + j)] +=
                        src[static_cast<size_t>(t * row + j)];
        }
    }
}

}  // namespace
}  // namespace r97

R97_REGISTER_ADJOINT_F("rope", host, r97::rope_host_adjoint, 1)
R97_REGISTER_ADJOINT_F("kv_store", host, r97::kv_store_host_adjoint, 2)
