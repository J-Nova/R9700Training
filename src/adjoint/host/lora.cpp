// src/adjoint/host/lora.cpp -- the adapter path of docs/ADAPTER.md: y += (alpha / r) * B * (A * x),
// in the two ops radiance serves for it, and the mm variant of the second.
//
//   lora_shrink  reads a, [a_scale], slot; writes t.  t[m, j] = sum_k x(m, k) * A[s, j, k]
//   lora_expand  reads t, slot, lora_b, scale; adds to y.
//                                                    y[m, n] += scale[s, n] * sum_j t[m, j] * B[s, n, j]
//
// with s = slot[m] and the value gemm_nt_q reads for that element of `a`: a code times its group's
// scale when the trunk is the rotated int8 one, the float itself when it is bf16 (docs/ADAPTER.md 1).
// A row whose slot is outside [0, S) is a base row: neither op reads or writes it, so neither the
// twin nor the adjoint touches it -- not even with a zero, which would turn -0 into +0.
//
// The gradients are the first trainable leaves in the tree: A under `<prefix>.lora_A` [r, K] and B
// under `<prefix>.lora_B` [N, r], f32, added to and never overwritten, so a step that reaches one
// module through two ops sums them the way a ParamSet wants.
//
// The shrink's schema is five operands in both forms -- a, a_scale, slot, lora_a, t
// (ref_registry.cpp: OPD("a"), OPD_O("a_scale"), OPD("slot"), OPD("lora_a"), OUT("t")). A float
// trunk's `a_scale` is an optional operand the call does not pass, and an absent operand keeps its
// place: radiance issues RAD_OPK_NONE there (issue.cpp), and Tape::record names the position
// `none:<i>` with dtype `invalid` (tape.cpp, rad_dtype_name(RAD_DT_INVALID)). So a bf16 record is
// five wide too, and `slot` at 2 and `lora_a` at 3 are frozen in both forms. The `a_scale` position
// at 1 is the one that differs: a float trunk leaves it absent, with no tensor and so no cotangent,
// and the row that serves a float record freezes it; a coded trunk passes a real scale, whose
// cotangent the adjoint gives, because the quantiser that made the codes reads both of its outputs'
// cotangents back. So the shrink registers two rows -- `R97_REGISTER_ADJOINT_IF_F` for the coded
// form with 2 and 3 frozen, `_F` for the float one with 1, 2 and 3 -- and the coded row comes
// first, because lookup applies the first row whose predicate accepts the record and the float row
// accepts every record. The expand's only cotangent is
// t's: `scale` is the adapter's alpha/r, a hyperparameter from the adapter file, and `lora_b` is a
// leaf whose gradient goes to the ParamSet.
//
// `lora_expand_mm` is the same op with t rounded to bf16 before the sum. Rounding has no
// derivative of its own, so its adjoint is this one's; its twin rounds, which is the only
// difference between the two rows.
#include "r97/adjoint.h"
#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace r97 {
namespace {

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

// The pair of rows the parallel for cuts the M-row loops into (a one-row call stays on the
// calling thread), and the pair of (column, row) pairs a factor-gradient pass splits over.
// The rows are the loops' outputs, and a factor's (column, row) is one of its: every unit's
// sums stay that unit's own serial order, so the partition writes nothing a neighbour reads.
constexpr int64_t kRowChunk = 2;
constexpr int64_t kPairChunk = 2;

/* The width of one scale group over K, read back from a stream of `groups` columns the way
 * radiance's make_scales does (grid_block in ref_gemm.h): the power of two whose ceiling division
 * gives `groups`, so a partial last group keeps the width its neighbours have. 300 over three
 * columns is groups of 128, 128 and 44 -- not three groups of 100, which would misassign
 * 100..127 -- and it is the 128 a coded checkpoint's `group` states. A single column spans the
 * whole row. */
int64_t group_width(int64_t k, int64_t groups) {
    if (k <= 0) return 1;
    if (groups <= 1) return k;
    const int64_t lo = (k + groups - 1) / groups;
    int64_t b = 1;
    while (b < lo) b <<= 1;
    return (k + b - 1) / b == groups ? b : lo;
}

/* An integer parameter, 0 when the record does not carry one (a parameter is an int64 or a
 * string; `group` is the only one this family reads as a number). */
int64_t param_int(const std::vector<OpParam>& params, const char* name) {
    for (const OpParam& p : params) {
        if (p.first != name) continue;
        if (const auto* i = std::get_if<int64_t>(&p.second)) return *i;
    }
    return 0;
}

/* Whether the record's activation is a code stream: the `dtype` parameter names one of the coded
 * dtypes radiance reads `a` as, so the record has an `a_scale` and the schema is one wider. */
bool coded_act(const std::vector<OpParam>& params) {
    for (const OpParam& p : params) {
        if (p.first != "dtype") continue;
        const auto* d = std::get_if<std::string>(&p.second);
        if (!d) return false;
        return *d == "i8" || *d == "fp8_e4m3" || *d == "fp8_e5m2";
    }
    return false;
}

/* The dtype the op writes, which the twin's output carries: the record is the only place it says. */
std::string write_dtype(const OpRecord& rec, const char* fallback) {
    for (const Operand& o : rec.operands)
        if (o.written && !o.dtype.empty()) return o.dtype;
    return fallback;
}

/* A cotangent of a trainable leaf, under the name the walk's prefix gives it. */
Param& grad_entry(ParamSet* grads, const std::string& name, const std::vector<int64_t>& shape) {
    for (Param& p : *grads)
        if (p.name == name) return p;
    Param p;
    p.name = name;
    p.shape = shape;
    p.value.assign((size_t)numel(shape), 0.0f);
    grads->push_back(std::move(p));
    return grads->back();
}

std::string grad_name(const char* prefix, const char* leaf) {
    if (!prefix || !*prefix) return leaf;
    return std::string(prefix) + "." + leaf;
}

/* bf16 rounding, which lora_expand_mm's twin needs and r97's own arithmetic never does. Ties go to
 * the even significand and a value below bf16's normal range flushes, as rad_f32_to_bf16 does. */
double rounded_bf16(double v) {
    float f = (float)v;
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof u);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    std::memcpy(&f, &u, sizeof f);
    return (double)f;
}

// ------------------------------------------------------------------ lora_shrink

/* The record's operands, in the schema's order: a, a_scale, slot, lora_a, t. A float activation
 * leaves `a_scale` absent, in place; a twin is handed the four reads and an adjoint the whole
 * schema, five wide. A record whose operands do not meet the op's contract lays out nothing: the
 * twin pushes no output and the adjoint adds nothing, which the checkers report against what they
 * expected. */
struct Shrink {
    const AdValue* a = nullptr;
    const AdValue* as = nullptr;  // null: a float activation, read as it stands
    const AdValue* slot = nullptr;
    const AdValue* la = nullptr;
    int64_t M = 0, K = 0, R = 0, S = 0;
    int64_t groups = 0, group = 0;
};

bool shrink_layout(const OpRecord& rec, const AdValue* v, size_t n, Shrink* sh) {
    const bool coded = coded_act(rec.params);
    if (n != 4u && n != 5u) return false;
    const AdValue& a = v[0];
    const AdValue& held = v[1];  // a_scale, or the place an absent one keeps
    const AdValue& slot = v[2];
    const AdValue& la = v[3];
    if (a.shape.size() != 2 || slot.shape.size() != 1 || la.shape.size() != 3) return false;
    /* The record's `dtype` and its operand 1 have to agree: a coded trunk passes a_scale, a float
     * trunk's position is the absent one the runtime left (radiance names it `invalid`). */
    const bool absent = held.dtype == "invalid";
    if (coded == absent) return false;
    const int64_t M = a.shape[0], K = a.shape[1], S = la.shape[0], R = la.shape[1];
    if (M <= 0 || K <= 0 || R <= 0 || S <= 0) return false;
    if (la.shape[2] != K || slot.shape[0] != M) return false;
    if ((int64_t)a.value.size() != M * K || (int64_t)slot.value.size() != M ||
        (int64_t)la.value.size() != S * R * K)
        return false;
    if (coded) {
        if (held.shape.size() != 2 || held.shape[0] != M || held.shape[1] <= 0) return false;
        if ((int64_t)held.value.size() != M * held.shape[1]) return false;
    }
    sh->a = &a;
    sh->as = coded ? &held : nullptr;
    sh->slot = &slot;
    sh->la = &la;
    sh->M = M;
    sh->K = K;
    sh->R = R;
    sh->S = S;
    sh->groups = coded ? held.shape[1] : 0;
    sh->group = coded ? group_width(K, sh->groups) : 0;
    /* The record's `group` is a consistency statement, not the width the kernel reads with: the
     * stream's column grid is what make_scales turns back into a width, and radiance's row refuses
     * a record whose `group` divides K into a different number of groups (ref_registry.cpp: K=300
     * with a `group` of 128 needs three columns, not two or four). So a record this twin lays out
     * is one the kernel would run, and it reads a partial last group the way the kernel does. */
    const int64_t stated = coded ? param_int(rec.params, "group") : 0;
    if (stated > 0 && (K + stated - 1) / stated != (K + sh->group - 1) / sh->group) return false;
    return true;
}

/* The pool row that row m is served by: 0..S-1, or -1 for a base row (a value below 0 and
 * anything past the pool are the two spellings a record uses). */
int64_t row_slot(const Shrink& sh, int64_t m) {
    const double v = sh.slot->value[(size_t)m];
    if (!(v >= 0.0)) return -1;
    const int64_t s = (int64_t)v;
    return s < sh.S ? s : -1;
}

/* The value the shrink reads for x(m, k): the activation itself, or the code times its group's
 * scale -- one product, as radiance's kernels take it. */
double x_at(const Shrink& sh, int64_t m, int64_t k) {
    const double v = sh.a->value[(size_t)(m * sh.K + k)];
    if (!sh.as) return v;
    return v * sh.as->value[(size_t)(m * sh.groups + k / sh.group)];
}

double a_at(const Shrink& sh, int64_t s, int64_t j, int64_t k) {
    return sh.la->value[(size_t)((s * sh.R + j) * sh.K + k)];
}

// A slot's trainable factor goes under the record's fused name when the caller splits a fused
// factor afterwards, under the slot's module when the walk names modules, and under the record's
// prefix otherwise. The shape's slot axis is the caller's business, and the writers place a slot's
// rows at its offset in that axis.
std::string factor_grad_name(const AdjointCall& call, const char* leaf, int64_t slot) {
    if (call.fused_slots) return grad_name(call.prefix, leaf);
    if (call.slot_prefix) return grad_name(call.slot_prefix(slot).c_str(), leaf);
    return grad_name(call.prefix, leaf);
}

void lora_shrink_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                      std::vector<AdValue>* out) {
    Shrink sh;
    if (!shrink_layout(rec, operands.data(), operands.size(), &sh)) return;
    AdValue t;
    t.dtype = write_dtype(rec, "f32");
    t.shape = {sh.M, sh.R};
    t.value.assign((size_t)(sh.M * sh.R), 0.0);
    parallel_for(sh.M, kRowChunk, [&](int64_t m) {
        const int64_t s = row_slot(sh, m);
        if (s < 0) return;  // a base row: nothing written
        for (int64_t j = 0; j < sh.R; ++j) {
            double acc = 0.0;
            for (int64_t k = 0; k < sh.K; ++k) acc += x_at(sh, m, k) * a_at(sh, s, j, k);
            t.value[(size_t)(m * sh.R + j)] = acc;
        }
    });
    out->push_back(std::move(t));
}

/* The adjoint of the shrink, in f64.
 *
 * With g[m, k] = sum_j cot t[m, j] * A[s, j, k] -- the cotangent of the shrunk row m in the value
 * space x lives in -- the three routes are:
 *
 *   a float activation:  dc a[m, k]    += g[m, k]
 *   a code stream:       dc a[m, k]    += g[m, k] * scale[m, k / group]
 *                        dc a_scale    += sum_{k in the group} g[m, k] * a[m, k]
 *   the weight:          dA[j, k]      += sum_{m: slot[m] = s} cot t[m, j] * x[m, k]
 *
 * i.e. dx is the value-space cotangent and the coded form splits it over the two operands its
 * dequantisation is made of (docs/ADAPTER.md 1: x = code * scale, so d/d(code) is the scale and
 * d/d(scale) the code).
 */
void lora_shrink_host_adjoint(const AdjointCall& call) {
    Shrink sh;
    if (!shrink_layout(call.rec, call.fwd.data(), call.fwd.size(), &sh)) return;
    if (call.cot.empty()) return;
    const AdValue& t_cot = call.cot[0];
    if ((int64_t)t_cot.value.size() != sh.M * sh.R) return;

    AdValue& a_cot = call.dc->at(0);
    if ((int64_t)a_cot.value.size() != sh.M * sh.K) return;
    AdValue* s_cot = nullptr;
    if (sh.as) {
        s_cot = &call.dc->at(1);
        if ((int64_t)s_cot->value.size() != sh.M * sh.groups) return;
    }

    // `grad_entry` hands a reference into the ParamSet, which a later entry may move: the
    // entries this call touches are made first, in the order the first rows find their slots,
    // and the pointers are held until the gradient pass is done.
    const auto da_for = [&](int64_t s) -> Param* {
        const std::vector<int64_t> shape =
            call.fused_slots ? std::vector<int64_t>{sh.S * sh.R, sh.K}
                             : std::vector<int64_t>{sh.R, sh.K};
        return &grad_entry(call.grads, factor_grad_name(call, "lora_A", s), shape);
    };
    std::vector<Param*> da(sh.S, nullptr);
    if (call.grads)
        for (int64_t m = 0; m < sh.M; ++m) {
            const int64_t s = row_slot(sh, m);
            if (s >= 0 && da[s] == nullptr) da[s] = da_for(s);
        }

    // The row's own outputs -- g, a_cot's row and s_cot's row -- are row-local, so the rows
    // partition the work. dA sums over the rows, and the sum's order is the rows' per
    // (slot, j, k): the pass over it splits over the (j, k) pairs, each keeping the rows
    // serial.
    parallel_for(sh.M, kRowChunk, [&](int64_t m) {
        const int64_t s = row_slot(sh, m);
        if (s < 0) return;  // a base row: no cotangent, no gradient
        std::vector<double> g((size_t)sh.K, 0.0);
        for (int64_t j = 0; j < sh.R; ++j) {
            const double c = t_cot.value[(size_t)(m * sh.R + j)];
            if (c == 0.0) continue;
            for (int64_t k = 0; k < sh.K; ++k) g[(size_t)k] += c * a_at(sh, s, j, k);
        }
        for (int64_t k = 0; k < sh.K; ++k) {
            const size_t e = (size_t)(m * sh.K + k);
            if (!sh.as) {
                a_cot.value[e] += g[(size_t)k];
                continue;
            }
            const size_t ge = (size_t)(m * sh.groups + k / sh.group);
            const double scale = sh.as->value[ge];
            a_cot.value[e] += g[(size_t)k] * scale;
            s_cot->value[ge] += g[(size_t)k] * sh.a->value[e];
        }
    });
    if (call.grads)
        parallel_for(sh.R * sh.K, kPairChunk, [&](int64_t u) {
            const int64_t j = u / sh.K, k = u % sh.K;
            for (int64_t m = 0; m < sh.M; ++m) {
                const int64_t s = row_slot(sh, m);
                if (s < 0) continue;
                const double c = t_cot.value[(size_t)(m * sh.R + j)];
                if (c == 0.0) continue;
                const int64_t row0 = call.fused_slots ? s * sh.R : 0;
                da[s]->value[(size_t)((row0 + j) * sh.K + k)] += (float)(c * x_at(sh, m, k));
            }
        });
}

// ------------------------------------------------------------------ lora_expand

/* The record's operands, in the schema's order: t, slot, lora_b, scale, y, laid out the way the
 * shrink's are. */
struct Expand {
    const AdValue* t = nullptr;
    const AdValue* slot = nullptr;
    const AdValue* lb = nullptr;
    const AdValue* sc = nullptr;  // null: a float trunk's record, whose y is the fourth operand
    int64_t M = 0, N = 0, R = 0, S = 0;
};

/* The record's write is marked in place: the tape's INOUT mark, the shape
 * `src/join_common.h` gives every lora_expand record, because the engine adds the
 * adapter's branch to the buffer the trunk's GEMM already wrote. One schema slot
 * then holds two values -- the buffer's old one the op reads and the new one it
 * writes -- and the caller's read list carries the old one last. */
bool expand_in_place(const OpRecord& rec) {
    for (const Operand& o : rec.operands)
        if (o.written && o.in_place) return true;
    return false;
}

/* The scale the expand reads for y(m, n), or the 1 a float trunk's record leaves it at. */
double expand_scale(const Expand& ex, int64_t s, int64_t n) {
    return ex.sc ? ex.sc->value[(size_t)(s * ex.N + n)] : 1.0;
}

bool expand_layout(const OpRecord& rec, const AdValue* v, size_t n, Expand* ex) {
    /* A record with a scale operand carries it as t, slot, lora_b, scale, y (five), or as the
     * fourth read operand when the write is not among the reads (four). A float trunk's record has
     * no scale at all, and its fourth is the y the op writes -- the one operand there that the
     * record marks as written. Reading that y as a scale multiplies every row by whatever the
     * buffer held, and, when the pool has more than one slot, reads past it. */
    if (n != 4u && n != 5u) return false;
    const AdValue& t = v[0];
    const AdValue& slot = v[1];
    const AdValue& lb = v[2];
    const bool has_scale = n == 5u || !rec.operands[3].written;
    const AdValue* sc = has_scale ? &v[3] : nullptr;
    if (t.shape.size() != 2 || slot.shape.size() != 1 || lb.shape.size() != 3)
        return false;
    const int64_t M = t.shape[0], R = t.shape[1], S = lb.shape[0], N = lb.shape[1];
    if (M <= 0 || N <= 0 || R <= 0 || S <= 0) return false;
    if (lb.shape[2] != R || slot.shape[0] != M) return false;
    if ((int64_t)t.value.size() != M * R || (int64_t)slot.value.size() != M ||
        (int64_t)lb.value.size() != S * N * R)
        return false;
    if (sc && (sc->shape.size() != 2 || sc->shape[0] != S || sc->shape[1] != N ||
               (int64_t)sc->value.size() != S * N))
        return false;
    // The in-place write's old value is the read list's last operand, the twin's, and it has to
    // be the buffer the write names: [M, N] of the write's own width.
    if (expand_in_place(rec)) {
        const AdValue& base = v[n - 1];
        // The slot's value is the write's own rows, dense in `M * N`. Its shape may be the
        // walk's lane split ([M, N / lanes]) rather than the write's own [M, N], so the rows
        // and the element count are the check, not the second dimension.
        if (base.shape.empty() || base.shape[0] != M || (int64_t)base.value.size() != M * N)
            return false;
    }
    ex->t = &t;
    ex->slot = &slot;
    ex->lb = &lb;
    ex->sc = sc;
    ex->M = M;
    ex->N = N;
    ex->R = R;
    ex->S = S;
    return true;
}

int64_t row_slot(const Expand& ex, int64_t m) {
    const double v = ex.slot->value[(size_t)m];
    if (!(v >= 0.0)) return -1;
    const int64_t s = (int64_t)v;
    return s < ex.S ? s : -1;
}

/* The expand writes y's increment and leaves y's base value alone: radiance's row rounds
 * `y + scale * sum_j t * B` into y in one step and skips a base row entirely. A plain write has
 * no forward value of y to start from (the record writes it, so it is not among the operands),
 * and every caller hands the write a zeroed buffer, so the twin writes the increment. An
 * in-place write is also a read: the caller hands the buffer's old value as the last operand,
 * and the slot's post-op value the tape records is `old + increment`, the value the kernel
 * leaves in the buffer, with a base row keeping its old value unrounded. */
void lora_expand_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                      std::vector<AdValue>* out) {
    Expand ex;
    if (!expand_layout(rec, operands.data(), operands.size(), &ex)) return;
    const bool round_t = rec.op == "lora_expand_mm";
    const AdValue* base = expand_in_place(rec) ? &operands.back() : nullptr;
    AdValue y;
    y.dtype = write_dtype(rec, "bf16");
    y.shape = {ex.M, ex.N};
    y.value.assign((size_t)(ex.M * ex.N), 0.0);
    if (base) y.value = base->value;
    parallel_for(ex.M, kRowChunk, [&](int64_t m) {
        const int64_t s = row_slot(ex, m);
        if (s < 0) return;  // a base row: not read, not written
        for (int64_t n = 0; n < ex.N; ++n) {
            double acc = 0.0;
            for (int64_t j = 0; j < ex.R; ++j) {
                const double tv = ex.t->value[(size_t)(m * ex.R + j)];
                acc += (round_t ? rounded_bf16(tv) : tv) *
                       ex.lb->value[(size_t)((s * ex.N + n) * ex.R + j)];
            }
            y.value[(size_t)(m * ex.N + n)] += expand_scale(ex, s, n) * acc;
        }
    });
    out->push_back(std::move(y));
}

/* The adjoint of the expand, in f64:
 *
 *   dt[m, j]   += sum_n cot y[m, n] * scale[s, n] * B[s, n, j]
 *   dB[n, j]   += sum_{m: slot[m] = s} cot y[m, n] * scale[s, n] * t[m, j]
 *
 * The cotangent y's base value would take is the walk's to pass on, not this op's: the record
 * writes y, and a write's slot in `dc` is not one an adjoint fills.
 */
void lora_expand_host_adjoint(const AdjointCall& call) {
    Expand ex;
    if (!expand_layout(call.rec, call.fwd.data(), call.fwd.size(), &ex)) return;
    if (call.cot.empty()) return;
    const AdValue& y_cot = call.cot[0];
    if ((int64_t)y_cot.value.size() != ex.M * ex.N) return;
    AdValue& t_cot = call.dc->at(0);
    if ((int64_t)t_cot.value.size() != ex.M * ex.R) return;

    // The direct path of the in-place write: the buffer the write replaces reads its own
    // cotangent back, unchanged. The walk passes it on when the row leaves the slot empty,
    // so filling it here says the same thing the passthrough would -- and a check that calls
    // the row without the walk, as the finite differences do, sees it either way.
    for (size_t i = 0; i < call.rec.operands.size(); ++i)
        if (call.rec.operands[i].written && call.rec.operands[i].in_place) {
            AdValue& d_base = call.dc->at(i);
            if (d_base.value.size() == y_cot.value.size())
                for (size_t e = 0; e < y_cot.value.size(); ++e) d_base.value[e] += y_cot.value[e];
        }

    const auto db_for = [&](int64_t s) -> Param* {
        const std::vector<int64_t> shape =
            call.fused_slots ? std::vector<int64_t>{ex.N, ex.S * ex.R}
                             : std::vector<int64_t>{ex.N, ex.R};
        return &grad_entry(call.grads, factor_grad_name(call, "lora_B", s), shape);
    };
    // The entries this call touches, made first in the order the first rows find their slots,
    // the way the ParamSet's move rule wants; n_end is the slot's, read once from the entry.
    std::vector<Param*> db(ex.S, nullptr);
    if (call.grads)
        for (int64_t m = 0; m < ex.M; ++m) {
            const int64_t s = row_slot(ex, m);
            if (s >= 0 && db[s] == nullptr) db[s] = db_for(s);
        }
    const int64_t cols = call.fused_slots ? ex.S * ex.R : ex.R;
    std::vector<int64_t> n_end(ex.S, ex.N);
    if (call.grads)
        for (int64_t s = 0; s < ex.S; ++s)
            if (db[s] && cols > 0 && db[s]->value.size() < (size_t)(ex.N * cols))
                n_end[s] = (int64_t)db[s]->value.size() / cols;

    // dt's row m is row-local, so the rows partition the work. dB sums over the rows, and the
    // sum's order is the rows' per (slot, n, j): the pass over it splits over the (n, j)
    // pairs, each keeping the rows serial.
    parallel_for(ex.M, kRowChunk, [&](int64_t m) {
        const int64_t s = row_slot(ex, m);
        if (s < 0) return;
        for (int64_t n = 0; n < n_end[s]; ++n) {
            const double c = y_cot.value[(size_t)(m * ex.N + n)];
            if (c == 0.0) continue;
            const double cs = c * expand_scale(ex, s, n);
            const size_t b0 = (size_t)((s * ex.N + n) * ex.R);
            for (int64_t j = 0; j < ex.R; ++j)
                t_cot.value[(size_t)(m * ex.R + j)] += cs * ex.lb->value[b0 + (size_t)j];
        }
    });
    if (call.grads)
        parallel_for(ex.N * ex.R, kPairChunk, [&](int64_t u) {
            const int64_t n = u / ex.R, j = u % ex.R;
            for (int64_t m = 0; m < ex.M; ++m) {
                const int64_t s = row_slot(ex, m);
                if (s < 0 || n >= n_end[s]) continue;
                const double c = y_cot.value[(size_t)(m * ex.N + n)];
                if (c == 0.0) continue;
                const double cs = c * expand_scale(ex, s, n);
                const int64_t col0 = call.fused_slots ? s * ex.R : 0;
                db[s]->value[(size_t)(n * cols + col0 + j)] +=
                    (float)(cs * ex.t->value[(size_t)(m * ex.R + j)]);
            }
        });
}

}  // namespace

R97_REGISTER_TWIN("lora_shrink", lora_shrink_twin)
R97_REGISTER_TWIN("lora_expand", lora_expand_twin)
R97_REGISTER_TWIN("lora_expand_mm", lora_expand_twin)

/* Two rows, because the a_scale position means different things in the two forms: a coded trunk
 * passes a real scale, and its cotangent is part of the int8 chain, so only slot (2) and lora_a (3)
 * are frozen there; a float trunk leaves the position absent -- no tensor, no cotangent -- so 1 is
 * frozen with them. The coded row goes first: the registry applies the first row whose predicate
 * accepts the record, and the float row's predicate is null, which accepts everything. */
R97_REGISTER_ADJOINT_IF_F("lora_shrink", host, lora_shrink_host_adjoint, coded_act, 2, 3)
R97_REGISTER_ADJOINT_F("lora_shrink", host, lora_shrink_host_adjoint, 1, 2, 3)

R97_REGISTER_ADJOINT_FF("lora_expand", host, lora_expand_host_adjoint, fills_in_place_slots, 1, 2, 3)
R97_REGISTER_ADJOINT_F("lora_expand_mm", host, lora_expand_host_adjoint, 1, 2, 3)

}  // namespace r97
