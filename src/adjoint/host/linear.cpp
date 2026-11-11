// r97/adjoint/host/linear.cpp -- host twins and adjoints for the frozen linears.
//
// Every op here has a frozen weight: only the input cotangent is produced. The int8 linear's
// cotangent is in the dequantised (rotated) activation basis, so the weight is dequantised in the
// adjoint but the activation scale is not applied again.

#include "r97/adjoint.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace r97 {
namespace {

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

const OpParam* find_param(const OpRecord& rec, const char* name) {
    for (const OpParam& p : rec.params) {
        if (p.first == name) return &p;
    }
    return nullptr;
}

int64_t param_int(const OpRecord& rec, const char* name, int64_t fallback) {
    const OpParam* p = find_param(rec, name);
    if (p == nullptr) return fallback;
    if (const int64_t* v = std::get_if<int64_t>(&p->second)) return *v;
    return fallback;
}

double param_double(const OpRecord& rec, const char* name, double fallback) {
    const OpParam* p = find_param(rec, name);
    if (p == nullptr) return fallback;
    if (const int64_t* v = std::get_if<int64_t>(&p->second)) return static_cast<double>(*v);
    try {
        return std::stod(std::get<std::string>(p->second));
    } catch (...) {
        return fallback;
    }
}

// The schema position of the operand a read or a write names.
int operand_pos(const OpRecord& rec, const char* name) {
    for (std::size_t i = 0; i < rec.operands.size(); ++i) {
        if (rec.operands[i].buffer == name) return static_cast<int>(i);
    }
    return -1;
}

// A recorded buffer may be named for the model's tensor rather than the schema's operand (the
// live model records `h` and `output.weight` where the schema says `x` and `lm_head`), so fall
// back to the schema position. The position is fixed by the op; the name is only a convenience.
int operand_pos_or(const OpRecord& rec, const char* name, int fallback) {
    const int by_name = operand_pos(rec, name);
    return by_name >= 0 ? by_name : fallback;
}

// The read operand at a schema position; the twin's `operands` holds reads only, in schema order.
const AdValue* read_at(const OpRecord& rec, const std::vector<AdValue>& operands, int schema_pos) {
    if (schema_pos < 0) return nullptr;
    int read = 0;
    for (std::size_t i = 0; i < rec.operands.size(); ++i) {
        if (rec.operands[i].written) continue;
        if (static_cast<int>(i) == schema_pos) {
            return read < static_cast<int>(operands.size()) ? &operands[static_cast<std::size_t>(read)]
                                                            : nullptr;
        }
        ++read;
    }
    return nullptr;
}

// The shape of the first written operand, which is the shape a twin's result must have.
const Operand* first_write(const OpRecord& rec) {
    for (const Operand& o : rec.operands) {
        if (o.written) return &o;
    }
    return nullptr;
}

AdValue make_value(const std::string& dtype, const std::vector<int64_t>& shape) {
    AdValue v;
    v.dtype = dtype;
    v.shape = shape;
    v.value.assign(static_cast<std::size_t>(numel(shape)), 0.0);
    return v;
}

// A grouped scale for row `row` and group `g` of K. The int8 grid stores one scale per row and
// block of K; a whole-row and a whole-matrix scale are accepted as well.
double scale_at(const AdValue* scale, int64_t row, int64_t group, int64_t rows, int64_t groups) {
    if (scale == nullptr) return 1.0;
    const int64_t n = static_cast<int64_t>(scale->value.size());
    if (n == 0) return 1.0;
    if (n == 1) return scale->value[0];
    if (n == rows * groups) return scale->value[static_cast<std::size_t>(row * groups + group)];
    if (n == groups) return scale->value[static_cast<std::size_t>(group)];
    if (n == rows) return scale->value[static_cast<std::size_t>(row)];
    const int64_t at = std::min<int64_t>(row * groups + group, n - 1);
    return scale->value[static_cast<std::size_t>(at)];
}

// y[M,N] = a[M,K] * b[N,K]^T, with the f32 accumulation of the host kernel done in f64.
void gemm_nt_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                  std::vector<AdValue>* out) {
    const AdValue* a = read_at(rec, operands, operand_pos_or(rec, "a", 0));
    const AdValue* b = read_at(rec, operands, operand_pos_or(rec, "b", 1));
    if (a == nullptr || b == nullptr) return;
    const int64_t K = param_int(rec, "K", a->shape.size() >= 2 ? a->shape[1] : 0);
    const int64_t N = param_int(rec, "N", b->shape.size() >= 2 ? b->shape[0] : 0);
    const Operand* w = first_write(rec);
    const int64_t M = w != nullptr && !w->shape.empty() ? w->shape[0]
                                                        : (a->shape.empty() ? 0 : a->shape[0]);
    out->push_back(make_value(w != nullptr ? w->dtype : a->dtype, {M, N}));
    AdValue& y = out->back();
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                acc += a->value[static_cast<std::size_t>(m * K + k)] *
                       b->value[static_cast<std::size_t>(n * K + k)];
            }
            y.value[static_cast<std::size_t>(m * N + n)] = acc;
        }
    }
}

void gemm_nt_adjoint(const AdjointCall& call) {
    const int ia = operand_pos_or(call.rec, "a", 0);
    const int ib = operand_pos_or(call.rec, "b", 1);
    if (ia < 0 || ib < 0 || call.cot.empty()) return;
    const AdValue& a = call.fwd[static_cast<std::size_t>(ia)];
    const AdValue& b = call.fwd[static_cast<std::size_t>(ib)];
    const AdValue& dy = call.cot[0];
    const int64_t K = param_int(call.rec, "K", a.shape.size() >= 2 ? a.shape[1] : 0);
    const int64_t N = param_int(call.rec, "N", b.shape.size() >= 2 ? b.shape[0] : 0);
    const Operand* w = first_write(call.rec);
    const int64_t M = w != nullptr && !w->shape.empty() ? w->shape[0]
                                                        : (a.shape.empty() ? 0 : a.shape[0]);
    AdValue& dx = (*call.dc)[static_cast<std::size_t>(ia)];
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            double acc = 0.0;
            for (int64_t n = 0; n < N; ++n) {
                acc += dy.value[static_cast<std::size_t>(m * N + n)] *
                       b.value[static_cast<std::size_t>(n * K + k)];
            }
            dx.value[static_cast<std::size_t>(m * K + k)] += acc;
        }
    }
}

// y[M,N] = (a codes * a_scale) * (b codes * b_scale)^T, grouped as the host kernel groups.
void gemm_nt_q_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                    std::vector<AdValue>* out) {
    const AdValue* a = read_at(rec, operands, operand_pos_or(rec, "a", 0));
    const AdValue* b = read_at(rec, operands, operand_pos_or(rec, "b", 1));
    const AdValue* as = read_at(rec, operands, operand_pos_or(rec, "a_scale", 2));
    const AdValue* bs = read_at(rec, operands, operand_pos_or(rec, "b_scale", 3));
    if (a == nullptr || b == nullptr) return;
    const int64_t K = param_int(rec, "K", a->shape.size() >= 2 ? a->shape[1] : 0);
    const int64_t N = param_int(rec, "N", b->shape.size() >= 2 ? b->shape[0] : 0);
    const Operand* w = first_write(rec);
    const int64_t M = w != nullptr && !w->shape.empty() ? w->shape[0]
                                                        : (a->shape.empty() ? 0 : a->shape[0]);
    const int64_t group = param_int(rec, "group", K > 0 ? K : 1);
    const int64_t groups = group > 0 ? (K + group - 1) / group : 1;
    out->push_back(make_value(w != nullptr ? w->dtype : a->dtype, {M, N}));
    AdValue& y = out->back();
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t g = 0; g < groups; ++g) {
                const int64_t k0 = g * group;
                const int64_t k1 = std::min(K, k0 + group);
                double sub = 0.0;
                for (int64_t k = k0; k < k1; ++k) {
                    sub += a->value[static_cast<std::size_t>(m * K + k)] *
                           b->value[static_cast<std::size_t>(n * K + k)];
                }
                acc += sub * scale_at(as, m, g, M, groups) * scale_at(bs, n, g, N, groups);
            }
            y.value[static_cast<std::size_t>(m * N + n)] = acc;
        }
    }
}

// dx in the dequantised activation basis: dy * W, where W dequantises the int8 weight codes.
void gemm_nt_q_adjoint(const AdjointCall& call) {
    const int ia = operand_pos_or(call.rec, "a", 0);
    const int ib = operand_pos_or(call.rec, "b", 1);
    const int ibs = operand_pos_or(call.rec, "b_scale", 3);
    if (ia < 0 || ib < 0 || call.cot.empty()) return;
    const AdValue& a = call.fwd[static_cast<std::size_t>(ia)];
    const AdValue& b = call.fwd[static_cast<std::size_t>(ib)];
    const AdValue* bs = ibs >= 0 ? &call.fwd[static_cast<std::size_t>(ibs)] : nullptr;
    const AdValue& dy = call.cot[0];
    const int64_t K = param_int(call.rec, "K", a.shape.size() >= 2 ? a.shape[1] : 0);
    const int64_t N = param_int(call.rec, "N", b.shape.size() >= 2 ? b.shape[0] : 0);
    const Operand* w = first_write(call.rec);
    const int64_t M = w != nullptr && !w->shape.empty() ? w->shape[0]
                                                        : (a.shape.empty() ? 0 : a.shape[0]);
    const int64_t group = param_int(call.rec, "group", K > 0 ? K : 1);
    const int64_t groups = group > 0 ? (K + group - 1) / group : 1;
    AdValue& dx = (*call.dc)[static_cast<std::size_t>(ia)];
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            const int64_t g = group > 0 ? k / group : 0;
            double acc = 0.0;
            for (int64_t n = 0; n < N; ++n) {
                acc += dy.value[static_cast<std::size_t>(m * N + n)] *
                       b.value[static_cast<std::size_t>(n * K + k)] *
                       scale_at(bs, n, g, N, groups);
            }
            dx.value[static_cast<std::size_t>(m * K + k)] += acc;
        }
    }
}

// logits_gemm: the same transposed product, with the head's own parameter names.
void logits_gemm_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                      std::vector<AdValue>* out) {
    const AdValue* x = read_at(rec, operands, operand_pos_or(rec, "x", 0));
    const AdValue* w = read_at(rec, operands, operand_pos_or(rec, "lm_head", 1));
    if (x == nullptr || w == nullptr) return;
    const int64_t K = param_int(rec, "n_embd", x->shape.size() >= 2 ? x->shape[1] : 0);
    const int64_t N = param_int(rec, "n_vocab", w->shape.size() >= 2 ? w->shape[0] : 0);
    const int64_t M = x->shape.empty() ? 0 : x->shape[0];
    const Operand* y = first_write(rec);
    out->push_back(make_value(y != nullptr ? y->dtype : x->dtype, {M, N}));
    AdValue& logits = out->back();
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                acc += x->value[static_cast<std::size_t>(m * K + k)] *
                       w->value[static_cast<std::size_t>(n * K + k)];
            }
            logits.value[static_cast<std::size_t>(m * N + n)] = acc;
        }
    }
}

void logits_gemm_adjoint(const AdjointCall& call) {
    const int ix = operand_pos_or(call.rec, "x", 0);
    const int iw = operand_pos_or(call.rec, "lm_head", 1);
    if (ix < 0 || iw < 0 || call.cot.empty()) return;
    const AdValue& x = call.fwd[static_cast<std::size_t>(ix)];
    const AdValue& w = call.fwd[static_cast<std::size_t>(iw)];
    const AdValue& dy = call.cot[0];
    const int64_t K = param_int(call.rec, "n_embd", x.shape.size() >= 2 ? x.shape[1] : 0);
    const int64_t N = param_int(call.rec, "n_vocab", w.shape.size() >= 2 ? w.shape[0] : 0);
    const int64_t M = x.shape.empty() ? 0 : x.shape[0];
    AdValue& dx = (*call.dc)[static_cast<std::size_t>(ix)];
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            double acc = 0.0;
            for (int64_t n = 0; n < N; ++n) {
                acc += dy.value[static_cast<std::size_t>(m * N + n)] *
                       w.value[static_cast<std::size_t>(n * K + k)];
            }
            dx.value[static_cast<std::size_t>(m * K + k)] += acc;
        }
    }
}

// gather_rows: y[i,:] = x[idx[i],:] and a negative index selects a zero row.
void gather_rows_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                      std::vector<AdValue>* out) {
    const AdValue* x = read_at(rec, operands, operand_pos_or(rec, "x", 0));
    const AdValue* idx = read_at(rec, operands, operand_pos_or(rec, "idx", 1));
    if (x == nullptr || idx == nullptr) return;
    const int64_t width = param_int(rec, "n", x->shape.size() >= 2 ? x->shape[1] : 1);
    const Operand* y = first_write(rec);
    const int64_t rows = std::min<int64_t>(
        y != nullptr && !y->shape.empty() ? y->rows : static_cast<int64_t>(idx->value.size()),
        static_cast<int64_t>(idx->value.size()));
    out->push_back(make_value(y != nullptr ? y->dtype : x->dtype, {rows, width}));
    AdValue& dst = out->back();
    const int64_t source_rows = x->shape.empty() ? 0 : x->shape[0];
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t r = static_cast<int64_t>(idx->value[static_cast<std::size_t>(i)]);
        if (r < 0 || r >= source_rows) continue;
        for (int64_t j = 0; j < width; ++j) {
            dst.value[static_cast<std::size_t>(i * width + j)] =
                x->value[static_cast<std::size_t>(r * width + j)];
        }
    }
}

void gather_rows_adjoint(const AdjointCall& call) {
    const int ix = operand_pos_or(call.rec, "x", 0);
    const int ii = operand_pos_or(call.rec, "idx", 1);
    if (ix < 0 || ii < 0 || call.cot.empty()) return;
    const AdValue& idx = call.fwd[static_cast<std::size_t>(ii)];
    const AdValue& dy = call.cot[0];
    const int64_t width = param_int(call.rec, "n", dy.shape.size() >= 2 ? dy.shape[1] : 1);
    const AdValue& x = call.fwd[static_cast<std::size_t>(ix)];
    const int64_t source_rows = x.shape.empty() ? 0 : x.shape[0];
    const Operand* y = first_write(call.rec);
    const int64_t dy_lo = y != nullptr ? y->row_lo : 0;
    // The cotangent is the whole written buffer; only the rows this record wrote are live, and
    // the index stream has one entry per one of those rows.
    const int64_t rows = std::min<int64_t>(
        y != nullptr ? y->rows : static_cast<int64_t>(idx.value.size()),
        static_cast<int64_t>(idx.value.size()));
    AdValue& dx = (*call.dc)[static_cast<std::size_t>(ix)];
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t r = static_cast<int64_t>(idx.value[static_cast<std::size_t>(i)]);
        if (r < 0 || r >= source_rows) continue;
        const int64_t d = dy_lo + i;
        if (d >= dy.shape.empty() ? 0 : dy.shape[0]) break;
        for (int64_t j = 0; j < width; ++j) {
            dx.value[static_cast<std::size_t>(r * width + j)] +=
                dy.value[static_cast<std::size_t>(d * width + j)];
        }
    }
}

// embed_lookup: tokens select rows of the (frozen) embedding table. It is a stop, so its adjoint
// writes nothing: neither the table nor the discrete token ids have a cotangent.
void embed_lookup_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                       std::vector<AdValue>* out) {
    const AdValue* tokens = read_at(rec, operands, operand_pos_or(rec, "tokens", 0));
    const AdValue* table = read_at(rec, operands, operand_pos_or(rec, "wte", 1));
    if (tokens == nullptr || table == nullptr) return;
    const int64_t width = param_int(rec, "n_embd", table->shape.size() >= 2 ? table->shape[1] : 1);
    const int64_t vocab = param_int(rec, "n_vocab", table->shape.size() >= 2 ? table->shape[0] : 0);
    const int64_t offset = param_int(rec, "vocab_offset", 0);
    const double mult = param_double(rec, "mult", 1.0);
    const Operand* x = first_write(rec);
    const int64_t rows = static_cast<int64_t>(tokens->value.size());
    out->push_back(make_value(x != nullptr ? x->dtype : table->dtype, {rows, width}));
    AdValue& dst = out->back();
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t id = static_cast<int64_t>(tokens->value[static_cast<std::size_t>(i)]);
        const int64_t r = id - offset;
        if (r < 0 || r >= vocab) continue;
        for (int64_t j = 0; j < width; ++j) {
            dst.value[static_cast<std::size_t>(i * width + j)] =
                mult * table->value[static_cast<std::size_t>(r * width + j)];
        }
    }
}

void embed_lookup_adjoint(const AdjointCall&) {}

}  // namespace

R97_REGISTER_TWIN("gemm_nt", gemm_nt_twin)
R97_REGISTER_ADJOINT_F("gemm_nt", host, gemm_nt_adjoint, 1)

R97_REGISTER_TWIN("gemm_nt_q", gemm_nt_q_twin)
R97_REGISTER_ADJOINT_F("gemm_nt_q", host, gemm_nt_q_adjoint, 1, 2, 3)

R97_REGISTER_TWIN("logits_gemm", logits_gemm_twin)
R97_REGISTER_ADJOINT_F("logits_gemm", host, logits_gemm_adjoint, 1)

R97_REGISTER_TWIN("gather_rows", gather_rows_twin)
R97_REGISTER_ADJOINT_F("gather_rows", host, gather_rows_adjoint, 1)

R97_REGISTER_TWIN("embed_lookup", embed_lookup_twin)
R97_REGISTER_ADJOINT_F("embed_lookup", host, embed_lookup_adjoint, 0, 1)

}  // namespace r97
