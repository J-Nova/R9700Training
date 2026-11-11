// src/adjoint/host/attention.cpp -- the adjoint of `attn_paged`: causal scaled-dot-product
// attention over a paged key/value pool, as prefill runs it.
//
// Schema (third_party/radiance/libref/ref_registry.cpp): reads q, kv_cache, block_table,
// seqused, k_descale, v_descale, cu_seqlens; writes out. Every (query row, query head) is one
// softmax over the keys its mask allows, read out of the pool at the slot the block table gives
// for that sequence and key position:
//
//     slot(s, j) = block_table[s, j / block_size] * block_size + j % block_size
//     kv_cache[block, kv head, slot][d]              -- the key,   the first head_dim doubles
//     kv_cache[block, kv head, slot][head_dim + d]   -- the value, the next head_dim
//
// A step of training runs each sequence as one prefill from position 0, so the pool rows an op
// reads are the rows that step wrote, addressed by sequence and key position. The adjoint
// therefore returns the cotangent of the pool laid out like the operand and non-zero on exactly
// those rows: the same slots `kv_store` (the `qk` family) writes, which is how the two chain.
//
// Parameters: head_dim (derived from q's last dimension), gqa (query heads per key/value head),
// block_size (derived from kv_cache), causal (1 by default), window (0 by default: the sliding
// mask a nonzero window gives), scale (1/sqrt(head_dim) by default). k_descale and v_descale are
// the fp8 pool's dequantisation scales and are frozen: the plan makes a scale a constant, so the
// adjoint returns no cotangent for them. block_table, seqused and cu_seqlens address rows and are
// frozen for the same reason -- a different integer there is a different op, not a small step.
//
// Every formula is in docs/adjoints/attention.md.

#include "r97/adjoint.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace r97 {
namespace {

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (const int64_t d : shape) n *= d;
    return n;
}

/* A parameter as an integer: the record carries an integer, or a string that is one. A value the
 * caller may legitimately set to zero (window, causal) is a value, not a missing parameter, so this
 * never falls back on a zero -- only on an absent or unreadable one. */
int64_t param_int(const OpRecord& rec, const char* name, int64_t fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (const auto* i = std::get_if<int64_t>(&p.second)) return *i;
        const std::string& s = std::get<std::string>(p.second);
        char* end = nullptr;
        const long long v = std::strtoll(s.c_str(), &end, 10);
        if (end && end != s.c_str() && *end == '\0') return v;
        return fallback;
    }
    return fallback;
}

double param_f64(const OpRecord& rec, const char* name, double fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (const auto* i = std::get_if<int64_t>(&p.second)) return (double)*i;
        const std::string& s = std::get<std::string>(p.second);
        char* end = nullptr;
        const double v = std::strtod(s.c_str(), &end);
        if (end && end != s.c_str() && *end == '\0' && std::isfinite(v)) return v;
        return fallback;
    }
    return fallback;
}

/* The read operands' slots, in schema order. */
enum Slot : size_t {
    kQ = 0,
    kKv = 1,
    kBlockTable = 2,
    kSeqUsed = 3,
    kKDescale = 4,
    kVDescale = 5,
    kCuSeq = 6,
    kOut = 7,
};

/* An operand the record leaves out carries an empty shape; one it fills in carries a shape whose
 * element count is the operand's. Reads sit at their schema positions 0..n_reads-1 in both the
 * twin's operand list (the reads alone) and the adjoint's (every operand), so an optional one past
 * the reads is left out rather than read off a written operand. */
bool listed(const std::vector<AdValue>& fwd, size_t n_reads, size_t i) {
    return i < n_reads && i < fwd.size() && !fwd[i].shape.empty() &&
           fwd[i].value.size() == (size_t)std::max<int64_t>(numel(fwd[i].shape), 0);
}

size_t read_count(const OpRecord& rec) {
    size_t n = 0;
    for (const Operand& o : rec.operands)
        if (!o.written) ++n;
    return n;
}

int64_t int_at(const AdValue& a, size_t i) {
    return i < a.value.size() ? (int64_t)std::llround(a.value[i]) : 0;
}

double f64_at(const AdValue& a, size_t i, double fallback) {
    return i < a.value.size() ? a.value[i] : fallback;
}

/* One call's geometry and its integer operands, resolved once and shared by the twin and the
 * adjoint so that the two cannot disagree about which keys a query row read. */
struct Call {
    const AdValue* q = nullptr;
    const AdValue* kv = nullptr;

    int64_t head_dim = 0;
    int64_t kv_heads = 0;
    int64_t q_heads = 0;
    int64_t gqa = 1;
    int64_t block_size = 0;
    int64_t n_seq = 0;
    int64_t rows = 0;
    int64_t max_blocks = 0;
    int64_t jcap = 0;
    int64_t causal = 1;
    int64_t window = 0;
    double scale = 1.0;

    std::vector<int64_t> ctx;       // sequence s's key count, seqused
    std::vector<int64_t> cu;        // cu_seqlens[n_seq + 1]; empty when the record leaves it out
    std::vector<int64_t> table;     // block_table, row-major [n_seq][max_blocks]
    std::vector<double> k_des;      // [n_seq][kv head], 1 where the record leaves it out
    std::vector<double> v_des;

    /* The sequence a dense query row belongs to, the length of that sequence's query, and the row's
     * index inside it. */
    void row_of(int64_t tok, int64_t* s, int64_t* qlen, int64_t* i) const {
        if (!cu.empty()) {
            int64_t seq = 0;
            while (seq + 1 < n_seq && cu[(size_t)seq + 1] <= tok) ++seq;
            *s = seq;
            *qlen = cu[(size_t)seq + 1] - cu[(size_t)seq];
            *i = tok - cu[(size_t)seq];
        } else {
            const int64_t per = n_seq > 0 ? rows / n_seq : 0;
            *s = per > 0 ? tok / per : 0;
            *i = per > 0 ? tok % per : 0;
            *qlen = per;
        }
    }

    /* Key j's slot in the pool (its k at the slot itself, its v head_dim doubles on), or -1 when
     * the mask hides the key or its block is unmapped. */
    int64_t key_slot(int64_t s, int64_t kh, int64_t pos, int64_t j) const {
        if (j < 0 || j >= ctx[(size_t)s] || j >= jcap) return -1;
        if (causal) {
            if (j > pos) return -1;
            if (window > 0 && pos - j >= window) return -1;
        } else if (window > 0 && (pos - j >= window || j - pos >= window)) {
            return -1;
        }
        const int64_t blk = table[(size_t)(s * max_blocks + j / block_size)];
        if (blk < 0 || blk >= kv->shape[0]) return -1;
        return ((blk * kv_heads + kh) * block_size + (j % block_size)) * (2 * head_dim);
    }

    double k_at(int64_t slot, int64_t d) const { return kv->value[(size_t)(slot + d)]; }
    double v_at(int64_t slot, int64_t d) const {
        return kv->value[(size_t)(slot + head_dim + d)];
    }
    double k_des_at(int64_t s, int64_t kh) const { return k_des[(size_t)(s * kv_heads + kh)]; }
    double v_des_at(int64_t s, int64_t kh) const { return v_des[(size_t)(s * kv_heads + kh)]; }
};

/* Resolve a record's geometry. False when the call is not one this op can compute, so the twin
 * writes no output and the adjoint adds no cotangent -- neither throws. */
bool resolve(const OpRecord& rec, const std::vector<AdValue>& fwd, Call* c) {
    const size_t n_reads = read_count(rec);
    if (!listed(fwd, n_reads, kQ) || !listed(fwd, n_reads, kKv) ||
        !listed(fwd, n_reads, kBlockTable) || !listed(fwd, n_reads, kSeqUsed))
        return false;
    c->q = &fwd[kQ];
    c->kv = &fwd[kKv];
    /* q is [rows][heads][head_dim], or one rank above with a batch dimension the rows flatten
     * across, or -- the shape the engine's own record carries -- [rows][heads * head_dim], with
     * the heads flattened into the columns. The pool is radiance's [block][key/value head]
     * [slot][2 * head_dim], so a 2-D q takes its head width from there. */
    if (c->kv->shape.size() != 4) return false;
    c->kv_heads = c->kv->shape[1];
    if (c->kv_heads <= 0) return false;
    c->gqa = param_int(rec, "gqa", 0);
    int64_t heads = 0;
    if (c->q->shape.size() >= 3) {
        c->head_dim = c->q->shape.back();
        heads = c->q->shape[c->q->shape.size() - 2];
        /* head_dim is a derived parameter: when the record spells it out it must say the same. */
        if (param_int(rec, "head_dim", c->head_dim) != c->head_dim) return false;
    } else if (c->q->shape.size() == 2 && c->gqa > 0) {
        c->head_dim = c->kv->shape[3] / 2;
        heads = c->gqa * c->kv_heads;
        if (c->head_dim <= 0 || c->q->shape.back() != heads * c->head_dim) return false;
    } else {
        return false;
    }
    if (c->head_dim <= 0) return false;
    if (c->kv->shape[3] < 2 * c->head_dim) return false;
    if (c->gqa <= 0) c->gqa = (heads % c->kv_heads == 0) ? heads / c->kv_heads : 1;
    c->q_heads = c->gqa * c->kv_heads;
    if (c->q_heads != heads) return false;
    if (c->q->value.size() % (size_t)(c->q_heads * c->head_dim) != 0) return false;
    c->rows = (int64_t)c->q->value.size() / (c->q_heads * c->head_dim);
    if (c->rows <= 0) return false;

    c->block_size = param_int(rec, "block_size", c->kv->shape[2]);
    if (c->block_size <= 0 || c->kv->shape[2] != c->block_size) return false;

    const AdValue& su = fwd[kSeqUsed];
    c->n_seq = (int64_t)su.value.size();
    if (c->n_seq <= 0 || c->n_seq > c->rows) return false;
    c->ctx.resize((size_t)c->n_seq);
    for (int64_t s = 0; s < c->n_seq; ++s) {
        c->ctx[(size_t)s] = int_at(su, (size_t)s);
        if (c->ctx[(size_t)s] <= 0) return false;
    }

    const AdValue& bt = fwd[kBlockTable];
    c->max_blocks = bt.shape.empty() ? (int64_t)bt.value.size() / c->n_seq : bt.shape.back();
    if (c->max_blocks <= 0 || bt.value.size() < (size_t)(c->n_seq * c->max_blocks)) return false;
    c->jcap = c->max_blocks * c->block_size;
    c->table.resize((size_t)(c->n_seq * c->max_blocks));
    for (size_t i = 0; i < c->table.size(); ++i) c->table[i] = int_at(bt, i);

    if (listed(fwd, n_reads, kCuSeq)) {
        const AdValue& cus = fwd[kCuSeq];
        if ((int64_t)cus.value.size() != c->n_seq + 1) return false;
        c->cu.resize((size_t)(c->n_seq + 1));
        for (int64_t s = 0; s <= c->n_seq; ++s) c->cu[(size_t)s] = int_at(cus, (size_t)s);
        if (c->cu[0] != 0 || c->cu[(size_t)c->n_seq] != c->rows) return false;
        for (int64_t s = 0; s < c->n_seq; ++s)
            if (c->cu[(size_t)s + 1] < c->cu[(size_t)s]) return false;
    } else if (c->rows % c->n_seq != 0) {
        return false;
    }

    c->k_des.assign((size_t)(c->n_seq * c->kv_heads), 1.0);
    c->v_des.assign((size_t)(c->n_seq * c->kv_heads), 1.0);
    if (listed(fwd, n_reads, kKDescale))
        for (size_t i = 0; i < c->k_des.size(); ++i)
            c->k_des[i] = f64_at(fwd[kKDescale], i, 1.0);
    if (listed(fwd, n_reads, kVDescale))
        for (size_t i = 0; i < c->v_des.size(); ++i)
            c->v_des[i] = f64_at(fwd[kVDescale], i, 1.0);

    c->causal = param_int(rec, "causal", 1) != 0 ? 1 : 0;
    c->window = param_int(rec, "window", 0);
    c->scale = param_f64(rec, "scale", 1.0 / std::sqrt((double)c->head_dim));
    return true;
}

}  // namespace

/* The forward in f64: every (query row, head) is a softmax over its visible keys, and its output
 * row is the weighted mean of their values. */
void attention_twin(const OpRecord& rec, const std::vector<AdValue>& fwd,
                    std::vector<AdValue>* out) {
    Call c;
    if (!resolve(rec, fwd, &c)) return;

    AdValue y;
    y.dtype = c.q->dtype;
    y.shape = c.q->shape;
    y.value.assign(c.q->value.size(), 0.0);

    std::vector<int64_t> keys;
    std::vector<double> score;
    for (int64_t tok = 0; tok < c.rows; ++tok) {
        int64_t s = 0, qlen = 0, i = 0;
        c.row_of(tok, &s, &qlen, &i);
        const int64_t ctx = c.ctx[(size_t)s];
        const int64_t pos = ctx - qlen + i;
        const int64_t jend = std::min(ctx, c.jcap);
        for (int64_t h = 0; h < c.q_heads; ++h) {
            const int64_t kh = h / c.gqa;
            const double kds = c.k_des_at(s, kh);
            const double vds = c.v_des_at(s, kh);
            keys.clear();
            for (int64_t j = 0; j < jend; ++j) {
                const int64_t slot = c.key_slot(s, kh, pos, j);
                if (slot >= 0) keys.push_back(slot);
            }
            if (keys.empty()) continue;  // an empty row: all weights zero, output zero
            const size_t base = (size_t)((tok * c.q_heads + h) * c.head_dim);

            double mx = -INFINITY;
            score.resize(keys.size());
            for (size_t u = 0; u < keys.size(); ++u) {
                const int64_t slot = keys[u];
                double dot = 0.0;
                for (int64_t d = 0; d < c.head_dim; ++d)
                    dot += c.q->value[base + (size_t)d] * kds * c.k_at(slot, d);
                score[u] = dot * c.scale;
                mx = std::max(mx, score[u]);
            }
            double z = 0.0;
            for (double& sc : score) {
                sc = std::exp(sc - mx);
                z += sc;
            }
            if (!(z > 0.0)) continue;
            for (size_t u = 0; u < keys.size(); ++u) {
                const int64_t slot = keys[u];
                const double w = score[u] / z * vds;
                for (int64_t d = 0; d < c.head_dim; ++d)
                    y.value[base + (size_t)d] += w * c.v_at(slot, d);
            }
        }
    }
    out->push_back(std::move(y));
}

/* The adjoint. For one (query row, head), with p the softmax weights of its visible keys and
 * g_j = dL/ds_j the score's cotangent, g_j = p_j (G_j - sum_k p_k G_k) where
 * G_j = sum_d ycot[d] * V_j[d] * v_descale: the query's row, the score's cotangent and the pool row
 * of each key all follow from one pass over the keys. */
void attention_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || !call.dc || call.dc->size() < 2) return;
    Call c;
    if (!resolve(call.rec, call.fwd, &c)) return;
    const AdValue& ycot = call.cot[0];
    AdValue& q_cot = call.dc->at(kQ);
    AdValue& kv_cot = call.dc->at(kKv);
    if (ycot.value.size() != c.q->value.size()) return;
    if (q_cot.value.size() != c.q->value.size()) return;
    if (kv_cot.value.size() != c.kv->value.size()) return;

    std::vector<int64_t> keys;
    std::vector<double> soft;  // the key's unnormalised weight, then its p
    std::vector<double> g;     // G_j, the row's value cotangent dotted with the key's value
    for (int64_t tok = 0; tok < c.rows; ++tok) {
        int64_t s = 0, qlen = 0, i = 0;
        c.row_of(tok, &s, &qlen, &i);
        const int64_t ctx = c.ctx[(size_t)s];
        const int64_t pos = ctx - qlen + i;
        const int64_t jend = std::min(ctx, c.jcap);
        for (int64_t h = 0; h < c.q_heads; ++h) {
            const int64_t kh = h / c.gqa;
            const double kds = c.k_des_at(s, kh);
            const double vds = c.v_des_at(s, kh);
            keys.clear();
            for (int64_t j = 0; j < jend; ++j) {
                const int64_t slot = c.key_slot(s, kh, pos, j);
                if (slot >= 0) keys.push_back(slot);
            }
            if (keys.empty()) continue;
            const size_t base = (size_t)((tok * c.q_heads + h) * c.head_dim);

            double mx = -INFINITY;
            soft.resize(keys.size());
            g.resize(keys.size());
            for (size_t u = 0; u < keys.size(); ++u) {
                const int64_t slot = keys[u];
                double dot = 0.0;
                for (int64_t d = 0; d < c.head_dim; ++d)
                    dot += c.q->value[base + (size_t)d] * kds * c.k_at(slot, d);
                soft[u] = dot * c.scale;
                mx = std::max(mx, soft[u]);
            }
            double z = 0.0;
            for (double& sc : soft) {
                sc = std::exp(sc - mx);
                z += sc;
            }
            if (!(z > 0.0)) continue;
            for (double& sc : soft) sc /= z;

            /* The row's value cotangent against each key's value, and its mean under the same
             * weights: the softmax's Jacobian is p_j (e_j - p). */
            double mean = 0.0;
            for (size_t u = 0; u < keys.size(); ++u) {
                const int64_t slot = keys[u];
                double acc = 0.0;
                for (int64_t d = 0; d < c.head_dim; ++d)
                    acc += ycot.value[base + (size_t)d] * c.v_at(slot, d);
                g[u] = acc * vds;
                mean += soft[u] * g[u];
            }

            for (size_t u = 0; u < keys.size(); ++u) {
                const int64_t slot = keys[u];
                const double ds = soft[u] * (g[u] - mean);
                for (int64_t d = 0; d < c.head_dim; ++d) {
                    /* d score / d q[d] = K[d] * k_descale * scale: the query head's cotangent. */
                    q_cot.value[base + (size_t)d] += ds * c.scale * kds * c.k_at(slot, d);
                    /* ... and by symmetry the key row's, on the same slot. */
                    kv_cot.value[(size_t)(slot + d)] +=
                        ds * c.scale * kds * c.q->value[base + (size_t)d];
                    /* d y[d] / d V[d] = p_j * v_descale for every visible key. */
                    kv_cot.value[(size_t)(slot + c.head_dim + d)] +=
                        soft[u] * vds * ycot.value[base + (size_t)d];
                }
            }
        }
    }
    /* Frozen: block_table, seqused, k_descale, v_descale, cu_seqlens (positions 2..6), and no
     * trainable leaves, so call.grads stays untouched. */
}

R97_REGISTER_TWIN("attn_paged", attention_twin)
// Frozen operands, counted in schema order: block_table 2, seqused 3, k_descale 4, v_descale 5,
// cu_seqlens 6. A record that leaves an optional one out keeps the positions of the ones it has.
R97_REGISTER_ADJOINT_F("attn_paged", host, attention_host_adjoint, 2, 3, 4, 5, 6)

}  // namespace r97
