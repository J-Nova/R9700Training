// src/adjoint/host/elementwise.cpp -- the elementwise family: the smooth pointwise ops and the
// RMS norm, as the Qwen3.5 plugins issue them on the host (arch/common/rad_block_mlp*.h,
// rad_block_attn_gated*.h, rad_fp8.h).
//
// Schema positions, from libref/ref_registry.cpp:
//
//   mul      a, b -> y          y = a * b, with b allowed to carry one row of a   (oBinary)
//   scale    x, s -> y          y = x * s[0]                                      (oScale)
//   sigmoid  x -> y             y = 1 / (1 + exp(-x))                             (oUnary)
//   silu     x -> y             y = x * sigmoid(x)                                (oUnary)
//   silu_mul gate_up -> y       gate_up is [rows, 2n], y is [rows, n]             (oGateUp)
//   act_mul  a, b -> y          y = act(a) * b                                    (oBinary)
//   cast     x -> y             y = x, read in the output's dtype                 (oUnary)
//   rmsnorm  x, w -> y          y = x / rms(x) * (w + wadd) * mult                (oRmsnorm)
//
// Each twin is the op's forward in f64 and each adjoint is that forward's exact derivative, so
// finite differences of the twin reproduce the adjoint to rounding. The kernels round where they
// store -- act_mul rounds the activation to y's dtype before the product, and quick_gelu rounds
// 1.702 x -- so a twin and a kernel agree only to the dtype's epsilon; the kernel check budgets
// for that, and the adjoint stays the derivative of the smooth twin.
//
// The two weights are frozen, not trained: rmsnorm's gain and scale's factor. Their adjoints are
// registered with the frozen operand, so the walk never asks them for a cotangent.
#include "r97/adjoint.h"

#include "parallel.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <variant>
#include <vector>

namespace r97 {
namespace {
// The unit a flat elementwise pass splits into, and the unit a row-wise pass uses: one output
// element, or one output row. A call below the row threshold runs on the calling thread.
constexpr int64_t kElemChunk = 4096;
constexpr int64_t kRowChunk = 2;

/* ------------------------------------------------------------------ reading a record */

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return shape.empty() ? 0 : n;
}

/* A numeric parameter. A real one reaches a record as its decimal spelling (the record's value is
 * an int or a string), so a string parses and anything unparsable leaves the fallback. Zero is a
 * value here: `wadd=0` is the plain norm. */
double param_num(const OpRecord& rec, const char* name, double fallback) {
    for (const auto& p : rec.params) {
        if (p.first != name) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) return (double)*i;
        const std::string& s = std::get<std::string>(p.second);
        char* end = nullptr;
        const double v = std::strtod(s.c_str(), &end);
        return (end && end != s.c_str() && *end == '\0') ? v : fallback;
    }
    return fallback;
}

/* A string parameter, or nullptr when the record leaves it out. */
const std::string* param_str(const OpRecord& rec, const char* name) {
    for (const auto& p : rec.params) {
        if (p.first != name) continue;
        if (const std::string* s = std::get_if<std::string>(&p.second)) return s;
    }
    return nullptr;
}

/* The written operand's dtype and shape, its values zeroed. The record names both, so the twin
 * does not have to guess what the op writes. */
AdValue write_value(const OpRecord& rec, size_t slot, const std::vector<int64_t>& shape) {
    AdValue v;
    if (slot < rec.operands.size()) v.dtype = rec.operands[slot].dtype;
    v.shape = shape;
    v.value.assign((size_t)numel(shape), 0.0);
    return v;
}

/* The row width: the record's `n`, else the last dimension of the shape given. */
int64_t width_of(const OpRecord& rec, const std::vector<int64_t>& shape) {
    const double n = param_num(rec, "n", 0.0);
    if (n > 0.0) return (int64_t)n;
    return shape.empty() ? 0 : shape.back();
}

/* The rows of a [rows, n] read, or 0 when the read does not divide into rows of n. */
int64_t rows_of(const AdValue& v, int64_t n) {
    const int64_t total = (int64_t)v.value.size();
    if (n <= 0 || total <= 0 || total % n != 0) return 0;
    return total / n;
}

/* silu_mul's output width, which the record's `n` names; its input is the fused [rows, 2n] row. */
int64_t gate_width(const OpRecord& rec, const std::vector<int64_t>& in_shape) {
    const double n = param_num(rec, "n", 0.0);
    if (n > 0.0) return (int64_t)n;
    return in_shape.empty() ? 0 : in_shape.back() / 2;
}

/* ------------------------------------------------------------------ the activations */

/* The six activations `act_mul`'s schema enumerates, in its order. */
enum Act { kSilu = 0, kGelu, kGeluTanh, kSigmoid, kQuickGelu, kRelu };

bool act_of(const std::string& name, Act* act) {
    static const char* kNames[] = {"silu", "gelu", "gelu_tanh", "sigmoid", "quick_gelu", "relu"};
    for (int i = 0; i < 6; ++i) {
        if (name != kNames[i]) continue;
        *act = (Act)i;
        return true;
    }
    return false;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double act_value(Act act, double x) {
    switch (act) {
        case kSilu: return x * sigmoid(x);
        case kGelu: return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440));
        case kGeluTanh: {
            const double k0 = 0.79788456080286535588, k1 = 0.044715;
            return 0.5 * x * (1.0 + std::tanh(k0 * (x + k1 * x * x * x)));
        }
        case kSigmoid: return sigmoid(x);
        /* quick_gelu's act_mul mode is the sigmoid alone: the kernel leaves the x to the
         * multiplier operand (its b is the same tensor, the unfused mul-then-act_mul pair), so
         * act_mul(a, a) is x*sigmoid(1.702x) and act(a) here is sigmoid(1.702x). */
        case kQuickGelu: return sigmoid(1.702 * x);
        case kRelu: return x > 0.0 ? x : 0.0;
    }
    return 0.0;
}

double act_slope(Act act, double x) {
    switch (act) {
        case kSilu: {
            const double s = sigmoid(x);
            return s * (1.0 + x * (1.0 - s));
        }
        case kGelu: return 0.5 * (1.0 + std::erf(x * 0.70710678118654752440)) +
                            x * 0.39894228040143267794 * std::exp(-0.5 * x * x);
        case kGeluTanh: {
            const double k0 = 0.79788456080286535588, k1 = 0.044715;
            const double t = std::tanh(k0 * (x + k1 * x * x * x));
            return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * k0 * (1.0 + 3.0 * k1 * x * x);
        }
        case kSigmoid: {
            const double s = sigmoid(x);
            return s * (1.0 - s);
        }
        case kQuickGelu: {
            const double s = sigmoid(1.702 * x);
            return 1.702 * s * (1.0 - s);
        }
        case kRelu: return x > 0.0 ? 1.0 : 0.0;
    }
    return 0.0;
}

/* ------------------------------------------------------------------ mul: y = a * b */

void mul_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.size() < 2) return;
    const AdValue &a = op[0], &b = op[1];
    const int64_t na = (int64_t)a.value.size(), nb = (int64_t)b.value.size();
    /* b is the whole block, or one row of it that every row of a multiplies. */
    const bool row = nb > 0 && nb != na && na % nb == 0 && nb == width_of(rec, a.shape);
    if (!row && nb != na) return;
    if (nb == 0) return;

    AdValue y = write_value(rec, 2, a.shape);
    parallel_for(na, kElemChunk, [&](int64_t i) {
        y.value[(size_t)i] = a.value[(size_t)i] * b.value[row ? (size_t)(i % nb) : (size_t)i];
    });
    out->push_back(std::move(y));
}

void mul_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.size() < 2 || call.dc->size() < 2) return;
    const AdValue &a = call.fwd[0], &b = call.fwd[1], &dy = call.cot[0];
    AdValue &da = call.dc->at(0), &db = call.dc->at(1);
    const size_t na = dy.value.size(), nb = b.value.size();
    if (nb == 0 || nb > na) return;

    if (nb == na) {
        parallel_for((int64_t)na, kElemChunk, [&](int64_t i) {
            da.value[(size_t)i] += dy.value[(size_t)i] * b.value[(size_t)i];
            db.value[(size_t)i] += dy.value[(size_t)i] * a.value[(size_t)i];
        });
        return;
    }
    /* b is one row every row of a multiplies: db[j] sums the rows, so the pass splits over its
     * columns, each keeping the rows serial -- the sum's order is the rows' own. */
    parallel_for((int64_t)nb, kElemChunk, [&](int64_t j) {
        for (size_t i = (size_t)j; i < na; i += nb) {
            da.value[i] += dy.value[i] * b.value[(size_t)j];
            db.value[(size_t)j] += dy.value[i] * a.value[i];
        }
    });
}

/* ------------------------------------------------------------------ scale: y = x * s[0] */

void scale_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.size() < 2 || op[1].value.empty()) return;
    const AdValue& x = op[0];
    const double s = op[1].value[0];
    AdValue y = write_value(rec, 2, x.shape);
    parallel_for((int64_t)y.value.size(), kElemChunk, [&](int64_t i) {
        y.value[(size_t)i] = x.value[(size_t)i] * s;
    });
    out->push_back(std::move(y));
}

void scale_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.size() < 2 || call.fwd[1].value.empty() || call.dc->empty())
        return;
    const double s = call.fwd[1].value[0];
    AdValue& dx = call.dc->at(0);
    parallel_for((int64_t)call.cot[0].value.size(), kElemChunk, [&](int64_t i) {
        dx.value[(size_t)i] += call.cot[0].value[(size_t)i] * s;
    });
}

/* ------------------------------------------------------------------ sigmoid: y = 1/(1+e^-x) */

void sigmoid_twin(const OpRecord&, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.size() < 1) return;
    const AdValue& x = op[0];
    AdValue y = x;
    parallel_for((int64_t)y.value.size(), kElemChunk, [&](int64_t i) {
        y.value[(size_t)i] = sigmoid(x.value[(size_t)i]);
    });
    out->push_back(std::move(y));
}

void sigmoid_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.size() < 2 || call.dc->empty()) return;
    const AdValue &y = call.fwd[1], &dy = call.cot[0];  /* y, the op's own output */
    AdValue& dx = call.dc->at(0);
    parallel_for((int64_t)dy.value.size(), kElemChunk, [&](int64_t i) {
        dx.value[(size_t)i] += dy.value[(size_t)i] * y.value[(size_t)i] * (1.0 - y.value[(size_t)i]);
    });
}

/* ------------------------------------------------------------------ silu: y = x * sigmoid(x) */

void silu_twin(const OpRecord&, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.size() < 1) return;
    const AdValue& x = op[0];
    AdValue y = x;
    parallel_for((int64_t)y.value.size(), kElemChunk, [&](int64_t i) {
        y.value[(size_t)i] = act_value(kSilu, x.value[(size_t)i]);
    });
    out->push_back(std::move(y));
}

void silu_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.empty() || call.dc->empty()) return;
    const AdValue &x = call.fwd[0], &dy = call.cot[0];
    AdValue& dx = call.dc->at(0);
    parallel_for((int64_t)dy.value.size(), kElemChunk, [&](int64_t i) {
        dx.value[(size_t)i] += dy.value[(size_t)i] * act_slope(kSilu, x.value[(size_t)i]);
    });
}

/* ------------------------------------------------------------------ silu_mul: y = silu(g) * u */

/* gate_up is [rows, 2n]: the gate in its first half, the up projection in its second, the order
 * rad-convert fuses them in. */
void silu_mul_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.empty()) return;
    const AdValue& x = op[0];
    const int64_t n = gate_width(rec, x.shape);
    const int64_t rows = rows_of(x, 2 * n);
    if (n <= 0 || rows <= 0) return;

    AdValue y = write_value(rec, 1, {rows, n});
    parallel_for(rows, kRowChunk, [&](int64_t r) {
        const int64_t xb = r * 2 * n, yb = r * n;
        for (int64_t i = 0; i < n; ++i)
            y.value[(size_t)(yb + i)] = act_value(kSilu, x.value[(size_t)(xb + i)]) *
                                        x.value[(size_t)(xb + n + i)];
    });
    out->push_back(std::move(y));
}

void silu_mul_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.empty() || call.dc->empty()) return;
    const AdValue &x = call.fwd[0], &dy = call.cot[0];
    const int64_t n = gate_width(call.rec, x.shape);
    const int64_t rows = rows_of(x, 2 * n);
    if (n <= 0 || rows <= 0 || rows_of(dy, n) != rows) return;

    AdValue& dx = call.dc->at(0);
    parallel_for(rows, kRowChunk, [&](int64_t r) {
        const int64_t xb = r * 2 * n, yb = r * n;
        for (int64_t i = 0; i < n; ++i) {
            const double g = x.value[(size_t)(xb + i)], u = x.value[(size_t)(xb + n + i)];
            const double d = dy.value[(size_t)(yb + i)];
            dx.value[(size_t)(xb + i)] += d * u * act_slope(kSilu, g);
            dx.value[(size_t)(xb + n + i)] += d * act_value(kSilu, g);
        }
    });
}

/* ------------------------------------------------------------------ act_mul: y = act(a) * b */

void act_mul_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.size() < 2) return;
    const std::string* name = param_str(rec, "act");
    Act act;
    if (!name || !act_of(*name, &act) || op[0].value.size() != op[1].value.size()) return;

    AdValue y = write_value(rec, 2, op[0].shape);
    parallel_for((int64_t)y.value.size(), kElemChunk, [&](int64_t i) {
        y.value[(size_t)i] = act_value(act, op[0].value[(size_t)i]) * op[1].value[(size_t)i];
    });
    out->push_back(std::move(y));
}

void act_mul_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.size() < 2 || call.dc->size() < 2) return;
    const std::string* name = param_str(call.rec, "act");
    Act act;
    if (!name || !act_of(*name, &act)) return;

    const AdValue &a = call.fwd[0], &b = call.fwd[1], &dy = call.cot[0];
    AdValue &da = call.dc->at(0), &db = call.dc->at(1);
    size_t len = dy.value.size();
    if (a.value.size() < len) len = a.value.size();
    if (b.value.size() < len) len = b.value.size();
    parallel_for((int64_t)len, kElemChunk, [&](int64_t i) {
        da.value[(size_t)i] += dy.value[(size_t)i] * act_slope(act, a.value[(size_t)i]) * b.value[(size_t)i];
        db.value[(size_t)i] += dy.value[(size_t)i] * act_value(act, a.value[(size_t)i]);
    });
}

/* ------------------------------------------------------------------ cast: y = x in to */

/* Between float types a cast only rounds, so the twin is the identity and the cotangent passes
 * through unchanged. */
void cast_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.empty()) return;
    AdValue y = write_value(rec, 1, op[0].shape);
    y.value = op[0].value;
    out->push_back(std::move(y));
}

void cast_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.dc->empty()) return;
    AdValue& dx = call.dc->at(0);
    parallel_for((int64_t)call.cot[0].value.size(), kElemChunk, [&](int64_t i) {
        dx.value[(size_t)i] += call.cot[0].value[(size_t)i];
    });
}

/* ------------------------------------------------------------------ rmsnorm */

/* The rmsnorm family's gain: the one non-written operand after x that is a row of `n` columns,
 * alone or materialised once per row. The plain record is x, w{n}, y; the engine's fused producer
 * is h, x, w{n or rows x n}, q{rows,n}, scale{rows,n/128}, out?, and its gain arrives either way.
 *
 * A shape test alone is not enough on that record: the norm's own *input* `x` is a `[rows, n]`
 * read sitting beside the gain with the same shape, so a `[rows, n]` operand is a gain only when
 * every row is the same one. `gain_row` decides that, and it is also what keeps a positional rule
 * from taking the input for the gain -- which is what both rows did before it existed: the device
 * row took the record's second read (the input) and this one the first shape match (also the
 * input), so the norm's adjoint used a per-row activation as its gain.
 *
 * A gain the tape gave no values for is absent: the schema's `wadd` says whether the record needs
 * one at all, so a plain norm still computes and a record that needs a gain refuses by name. */
bool gain_row(const AdValue& v, int64_t n) {
    if (n <= 0 || v.value.empty()) return false;
    if (v.shape.size() == 1) return v.shape[0] == n && static_cast<int64_t>(v.value.size()) >= n;
    if (v.shape.size() != 2 || v.shape[1] != n || v.shape[0] <= 0) return false;
    return static_cast<int64_t>(v.value.size()) >= n * v.shape[0];
}

// The tape materialises a fused producer's gain as `[rows, n]`, and the rows are supposed to be
// the same one. On the live record they are not: `blk.63.ffn_norm.weight` arrives as 2621440
// values that differ row to row, for a weight the container holds as `[1, 5120]`. Both rows index
// row 0, which is the gain if the tape's first row is the weight's -- so this says once per
// process that the tape's form is not what it claims, rather than refusing a walk that has no
// other way to read the weight.
void note_gain_rows(const AdValue& v, int64_t n, const std::string& buffer, int64_t position) {
    if (v.shape.size() != 2 || n <= 0) return;
    const size_t row = static_cast<size_t>(n);
    if (v.value.size() <= row) return;
    for (size_t i = row; i < v.value.size(); ++i) {
        if (v.value[i] == v.value[i % row]) continue;
        static bool warned = false;
        if (warned) return;
        warned = true;
        std::fprintf(stderr,
                     "r97: rmsnorm adjoint: operand '%s' at %lld is [%lld, %lld] and its rows are "
                     "not the same one, so the gain is read from row 0\n",
                     buffer.c_str(), (long long)position, (long long)v.shape[0], (long long)n);
        std::fflush(stderr);
        return;
    }
}

const AdValue* norm_gain(const OpRecord& rec, const std::vector<AdValue>& op, int64_t n,
                        const char* who) {
    const size_t count = std::min(op.size(), rec.operands.size());
    for (size_t j = 1; j < count; ++j) {
        if (rec.operands[j].written) continue;
        if (!gain_row(op[j], n)) continue;
        note_gain_rows(op[j], n, rec.operands[j].buffer, rec.position);
        // R97_GAIN_POS=1: which operand was taken as the gain, and with what shape and size. The
        // schema's shape and the snapshot's can differ, and a gain that is `[rows, n]` in the record
        // but `[n]` in the snapshot (or the reverse) decides whether row 0 is the whole gain.
        if (std::getenv("R97_GAIN_POS") != nullptr) {
            std::string schema;
            for (int64_t d : rec.operands[j].shape)
                schema += (schema.empty() ? "" : ",") + std::to_string(d);
            // The values themselves, not just the shape: the gain was exonerated on buffer, shape
            // and rank, but never on the numbers. A rank-2 snapshot whose rows differ from each
            // other is not a broadcast of the rank-1 vector, and a rank-1 vector that is not the
            // container's weight makes every elementwise comparison against the kernel wrong.
            std::string vals;
            for (size_t i = 0; i < std::min<size_t>(op[j].value.size(), 4); ++i)
                vals += (vals.empty() ? "" : ",") + std::to_string(op[j].value[i]);
            double row0_vs_row1 = -1.0;
            if (op[j].shape.size() == 2 && op[j].shape[0] > 1 && op[j].shape[1] > 0) {
                const int64_t wcol = op[j].shape[1];
                row0_vs_row1 = 0.0;
                for (int64_t i = 0; i < wcol; ++i)
                    row0_vs_row1 = std::max(row0_vs_row1,
                                            std::fabs(op[j].value[(size_t)i] -
                                                      op[j].value[(size_t)(wcol + i)]));
            }
            std::fprintf(stderr, "GAINPOS pos=%lld who=%s j=%zu buffer=%s schema=[%s] value_rank=%zu "
                                 "value0=%lld values=%zu n=%lld vals=[%s] row0_vs_row1=%.6g\n",
                         (long long)rec.position, who, j, rec.operands[j].buffer.c_str(),
                         schema.c_str(), op[j].shape.size(),
                         op[j].shape.empty() ? -1LL : (long long)op[j].shape[0],
                         op[j].value.size(), (long long)n, vals.c_str(), row0_vs_row1);
            std::fflush(stderr);
        }
        return &op[j];
    }
    return nullptr;
}

// The fused norm's residual (docs/OPS.md's `residual`:inout?): operand 1, written and in place,
// of operand 0's logical shape. On the live tape the tape snapshots the operand after the op, so
// its value is the post-update sum x + residual_old, and the reference reads that value back as
// the norm's input (libref/ref_quant.cpp's `src(i)` narrows the sum to the residual's dtype).
// -1 when the record declares no residual: the plain `rmsnorm` record (x, w, y) never does, and
// that path stays byte-identical.
int64_t rmsnorm_residual(const OpRecord& rec) {
    if (rec.operands.size() <= 1) return -1;
    const Operand& o = rec.operands[1];
    if (!o.written || !o.in_place) return -1;
    if (o.shape != rec.operands[0].shape) return -1;
    return 1;
}

/* y[r, i] = x[r, i] * rsqrt(mean(x[r]^2) + eps) * (w[i] + wadd) * mult, the gain formed in f64
 * (the kernel forms it in f32, before the store's rounding). `wadd` without a gain is refused by
 * the kernel and left to nothing here. A gain the record names but the tape could not give values
 * for cannot be served: the twin would compute a different norm, and a wrong intermediate is worse
 * than a refusal. */
void rmsnorm_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.empty()) return;
    const int64_t res_at = rmsnorm_residual(rec);
    // The norm's input is operand 1's snapshot when the record declares a residual: the tape
    // snapshots it after the op, so it already holds the sum x + residual_old. Refuse rather than
    // sum the operands: the reference reads the snapshot back, and adding them here would build a
    // value the kernel never wrote.
    if (res_at >= 0) {
        if (op.size() <= (size_t)res_at)
            throw std::invalid_argument("r97: rmsnorm twin: op '" + rec.op + "' at " +
                                        std::to_string(rec.position) +
                                        " declares a residual the walk gave no value for");
        if (op[(size_t)res_at].value.size() != op[0].value.size())
            throw std::invalid_argument("r97: rmsnorm twin: op '" + rec.op + "' at " +
                                        std::to_string(rec.position) +
                                        ": the residual operand 1 has " +
                                        std::to_string(op[(size_t)res_at].value.size()) +
                                        " values, not the input's " +
                                        std::to_string(op[0].value.size()));
    }
    const AdValue& x = res_at >= 0 ? op[(size_t)res_at] : op[0];
    const int64_t n = width_of(rec, x.shape);
    const AdValue* w = norm_gain(rec, op, n, "twin");
    const int64_t rows = rows_of(x, n);
    if (rows <= 0) return;
    if (w && (int64_t)w->value.size() < n) return;
    const double eps = param_num(rec, "eps", 1e-6);
    // R97_NORM_WADD0=1: force the offset off, as a control only. The record's `wadd=1` is *right*
    // for this container: `docs/OPS-qwen35-device.md` declares `eps=1e-06 wadd=1` on every
    // `rmsnorm_had_quant_i8g` row, and `docs/adjoints/elementwise.md` records that Qwen3.5 uses
    // GemmaRMSNorm's `w + 1` with a zero-centred gain. Forcing 0 therefore makes `w + 0` near zero
    // and collapses the twin's activation, which is a negative control, not a candidate fix.
    const double wadd = std::getenv("R97_NORM_WADD0") != nullptr ? 0.0
                                                                : param_num(rec, "wadd", 0.0);
    const double mult = param_num(rec, "mult", 1.0);
    if (!w && wadd != 0.0) return;

    // R97_GAIN_ROW=1: index a `[rows, n]` gain by the row instead of reading row 0 for all of
    // them. The tape materialises a fused producer's gain once per row and the rows differ, so
    // whether the kernel reads row `r` or row 0 is a question only the tape can answer: with the
    // flag on, `DUMPTWIN`'s activation comparison either collapses to bf16 rounding (per-row is
    // right) or stays where it is (row 0 is).
    const bool gain_row_flag = std::getenv("R97_GAIN_ROW") != nullptr;
    const bool per_row = w != nullptr && w->shape.size() == 2 && w->shape[0] == rows && gain_row_flag;
    if (gain_row_flag) {
        static bool said = false;
        if (!said) {
            said = true;
            std::fprintf(stderr,
                         "FDRMS pos=%lld rows=%lld w_rank=%zu w0=%lld per_row=%d\n",
                         (long long)rec.position, (long long)rows,
                         w != nullptr ? w->shape.size() : 0,
                         w != nullptr && !w->shape.empty() ? (long long)w->shape[0] : -1,
                         per_row ? 1 : 0);
            std::fflush(stderr);
        }
    }
    AdValue y = write_value(rec, 2, x.shape);
    // R97_DUMP_QUANT=1: the twin's own normalisation constants for row 0. `DUMPTWIN`'s `best_of`
    // field is a *relative error* (`max|f - y| / max|f|`), not a match ratio: `1:x:1` is a 100%
    // error, and an identity would print `1:x:0`. These constants are what tell the norm's
    // disagreement from the quantiser's.
    if (std::getenv("R97_DUMP_QUANT") != nullptr) {
        double ss = 0.0, xmax = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double v = x.value[(size_t)i];
            ss += v * v;
            xmax = std::max(xmax, std::fabs(v));
        }
        const double s0 = 1.0 / std::sqrt(ss / (double)n + eps);
        std::fprintf(stderr,
                     "NORMTWIN pos=%lld n=%lld rows=%lld w=%s eps=%.6g wadd=%.6g mult=%.6g "
                     "xmax=%.6g ss_mean=%.6g s=%.6g\n",
                     (long long)rec.position, (long long)n, (long long)rows,
                     w != nullptr ? "yes" : "NULL", eps, wadd, mult, xmax, ss / (double)n, s0);
        std::fflush(stderr);
    }
    parallel_for(rows, kRowChunk, [&](int64_t r) {
        const int64_t b = r * n;
        double ss = 0.0;
        for (int64_t i = 0; i < n; ++i) ss += x.value[(size_t)(b + i)] * x.value[(size_t)(b + i)];
        const double s = 1.0 / std::sqrt(ss / (double)n + eps);
        for (int64_t i = 0; i < n; ++i)
            y.value[(size_t)(b + i)] = x.value[(size_t)(b + i)] * s *
                                       (w ? w->value[per_row ? (size_t)(b + i) : (size_t)i] + wadd
                                          : 1.0) *
                                       mult;
    });
    out->push_back(std::move(y));
}

/* dx_j = mult * (s * c_j * dy_j - (s^3 x_j / n) * sum_i dy_i x_i c_i), c = (w + wadd).
 *
 * The second term is the row's shared rms: every element of the row moves it, so every dy_i
 * weights x_j.
 *
 * The two-way rule (r97/adjoint.h's AdjointCall::read_rows): a row a consumer reads is
 * differentiated, and a non-finite value in it refuses -- the analytic `0 * inf` over a live row
 * is a bug, not a gradient. A row no consumer reads is provably dead, so it is left zero and
 * counted: the zero is the correct cotangent for a value nothing reads. When the caller could not
 * prove the read set every row is readable, and the refusal is what it was before the read set
 * existed. */
void rmsnorm_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.empty() || call.dc->empty()) return;
    const int64_t res_at = rmsnorm_residual(call.rec);
    // The norm's input is operand 1's snapshot when the record declares a residual (the tape
    // snapshots it after the op, so it holds the sum x + residual_old). The row reads the snapshot,
    // never a sum of its own, as the reference does (libref's `src(i)`).
    if (res_at >= 0) {
        if (call.fwd.size() <= (size_t)res_at)
            throw std::runtime_error("r97: rmsnorm adjoint: op '" + call.rec.op + "' at " +
                                     std::to_string(call.rec.position) +
                                     " declares a residual the walk gave no value for");
        if (call.fwd[(size_t)res_at].value.size() != call.fwd[0].value.size())
            throw std::runtime_error("r97: rmsnorm adjoint: op '" + call.rec.op + "' at " +
                                     std::to_string(call.rec.position) +
                                     ": the residual operand 1 has " +
                                     std::to_string(call.fwd[(size_t)res_at].value.size()) +
                                     " values, not the input's " +
                                     std::to_string(call.fwd[0].value.size()));
    }
    const AdValue& x = res_at >= 0 ? call.fwd[(size_t)res_at] : call.fwd[0];
    // The intermediate's cotangent is the part's last write; the residual's incoming cotangent
    // (when declared) is the write before it -- the part record's writes are the residual in
    // place, then the elementwise output the quantiser rotates.
    const AdValue& dy = call.cot.back();
    const AdValue* cot_res = (res_at >= 0 && call.cot.size() >= 2) ? &call.cot.front() : nullptr;
    const int64_t n = width_of(call.rec, x.shape);
    const AdValue* w = norm_gain(call.rec, call.fwd, n, "adjoint");
    // R97_GAIN_POS=1: which snapshot the *adjoint* took. `GAINPOS` shows two at the same position --
    // rank 1 `[n]` and rank 2 `[rows,n]` -- and `FDRMS` shows the twin takes the rank-1 one, so this
    // names what the row itself differentiates with. A rank-2 gain read as row 0 is a per-row tensor
    // used as a single vector, and it lands on `dc[0]`.
    if (std::getenv("R97_GAIN_POS") != nullptr && w != nullptr)
        std::fprintf(stderr, "ADJGAIN pos=%lld buffer=%s w_rank=%zu w0=%lld values=%zu n=%lld\n",
                     (long long)call.rec.position,
                     call.rec.operands.empty() ? "" : call.rec.operands[0].buffer.c_str(),
                     w->shape.size(), w->shape.empty() ? -1LL : (long long)w->shape[0],
                     w->value.size(), (long long)n);
    const int64_t rows = rows_of(x, n);
    if (rows <= 0) return;
    // A gain the record names but the tape could not give values for cannot be served: the row
    // would leave the input's cotangent untouched (a silent zero where the analytic answer is not
    // zero). Refusing names it instead, as every other unservable operand on this path does.
    if (w && (int64_t)w->value.size() < n)
        throw std::runtime_error("r97: rmsnorm adjoint: op '" + call.rec.op + "' at " +
                                 std::to_string(call.rec.position) + ": the gain names " +
                                 std::to_string(n) + " columns and the tape gave " +
                                 std::to_string(w->value.size()));
    const double eps = param_num(call.rec, "eps", 1e-6);
    const double wadd = param_num(call.rec, "wadd", 0.0);
    const double mult = param_num(call.rec, "mult", 1.0);
    if (!w && wadd != 0.0) {
        // Name every operand the gain lookup passed over, and why: the refusal is otherwise a
        // statement that something is absent with no way to tell a written operand from an empty
        // one from one of the wrong shape.
        std::string seen;
        for (size_t j = 0; j < std::min(call.fwd.size(), call.rec.operands.size()); ++j) {
            const Operand& o = call.rec.operands[j];
            std::string dims;
            for (size_t d = 0; d < call.fwd[j].shape.size(); ++d) {
                if (d != 0) dims += 'x';
                dims += std::to_string(call.fwd[j].shape[d]);
            }
            seen += " [" + std::to_string(j) + "] '" + o.buffer + "' written=" +
                    std::to_string(o.written ? 1 : 0) +
                    " in_place=" + std::to_string(o.in_place ? 1 : 0) + " shape=[" + dims +
                    "] values=" + std::to_string(call.fwd[j].value.size()) +
                    " raw=" + std::to_string(call.fwd[j].raw != nullptr ? 1 : 0);
        }
        throw std::runtime_error("r97: rmsnorm adjoint: op '" + call.rec.op + "' at " +
                                 std::to_string(call.rec.position) +
                                 ": the record adds a gain offset and the tape names no gain"
                                 " (n=" + std::to_string(n) + "):" + seen);
    }

    AdValue& dx = call.dc->at(0);
    AdValue* dx_res = res_at >= 0 ? &call.dc->at((size_t)res_at) : nullptr;
    std::vector<char> readable((size_t)rows, 0);
    int64_t zeroed = 0, admitted = 0, refused = 0;
    for (int64_t r = 0; r < rows; ++r) {
        if (!row_readable(call, r)) {
            ++zeroed;
            continue;
        }
        readable[(size_t)r] = 1;
        ++admitted;
        const int64_t b = r * n;
        double ss = 0.0;
        for (int64_t i = 0; i < n && std::isfinite(ss); ++i)
            ss += x.value[(size_t)(b + i)] * x.value[(size_t)(b + i)];
        if (!std::isfinite(ss)) ++refused;
    }
    count_read_rows(call.row_count, call.read_rows, zeroed, admitted, refused);
    if (refused > 0)
        throw std::runtime_error(
            "r97: rmsnorm adjoint: op '" + call.rec.op + "' at " +
            std::to_string(call.rec.position) + ": " + std::to_string(refused) + " of " +
            std::to_string(admitted) +
            " rows a consumer reads have a non-finite activation; refusing rather than zeroing a "
            "live row's gradient");
    parallel_for(rows, kRowChunk, [&](int64_t r) {
        if (!readable[(size_t)r]) return;
        const int64_t b = r * n;
        double ss = 0.0;
        for (int64_t i = 0; i < n; ++i) ss += x.value[(size_t)(b + i)] * x.value[(size_t)(b + i)];
        const double s = 1.0 / std::sqrt(ss / (double)n + eps);
        double t = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double c = w ? w->value[(size_t)i] + wadd : 1.0;
            t += dy.value[(size_t)(b + i)] * x.value[(size_t)(b + i)] * c;
        }
        for (int64_t i = 0; i < n; ++i) {
            const double c = w ? w->value[(size_t)i] + wadd : 1.0;
            const double shared = s * s * s * x.value[(size_t)(b + i)] * t / (double)n;
            // v = x + residual_old's cotangent, carried in full by BOTH summands (the `add` rule,
            // docs/reports/2026-11-15-device-backward-cotangent-growth.md): the norm's input term
            // `dv` plus the residual write's own incoming cotangent. With `dv = 0` the row reduces
            // to `add`, both slots getting the residual's cotangent once.
            const double d = mult * (s * c * dy.value[(size_t)(b + i)] - shared) +
                             (cot_res ? cot_res->value[(size_t)(b + i)] : 0.0);
            dx.value[(size_t)(b + i)] += d;
            if (dx_res) dx_res->value[(size_t)(b + i)] += d;
        }
    });
    // R97_GAIN_POS=1: the per-row scale both ways for a fused norm that has a residual. The row's
    // input cotangent is `s = 1/rms(input)` to first order, so a wrong input scales every per-norm
    // cotangent by the rms ratio between the branch (operand 0) and the residual (operand 1), and
    // two norms per block square it. Printing both ranges makes that ratio the test: if the
    // cotangent table moved by the predicted factor the input was the defect; if it did not, the
    // ratio says the growth lives elsewhere. Print only, over the rows the row visits.
    if (std::getenv("R97_GAIN_POS") != nullptr && res_at >= 0) {
        double s0_min = 0.0, s0_max = 0.0, s1_min = 0.0, s1_max = 0.0;
        for (int64_t r = 0; r < rows; ++r) {
            if (!readable[(size_t)r]) continue;
            const int64_t b = r * n;
            double ss0 = 0.0, ss1 = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                ss0 += call.fwd[0].value[(size_t)(b + i)] * call.fwd[0].value[(size_t)(b + i)];
                ss1 += x.value[(size_t)(b + i)] * x.value[(size_t)(b + i)];
            }
            const double s0 = 1.0 / std::sqrt(ss0 / (double)n + eps);
            const double s1 = 1.0 / std::sqrt(ss1 / (double)n + eps);
            if (s0_min == 0.0 || s0 < s0_min) s0_min = s0;
            if (s0 > s0_max) s0_max = s0;
            if (s1_min == 0.0 || s1 < s1_min) s1_min = s1;
            if (s1 > s1_max) s1_max = s1;
        }
        std::fprintf(stderr, "NORMSCALE pos=%lld op=%s s_in0=[%.6g,%.6g] s_in1=[%.6g,%.6g]\n",
                     (long long)call.rec.position, call.rec.op.c_str(), s0_min, s0_max, s1_min,
                     s1_max);
        std::fflush(stderr);
    }

    // R97_GAIN_POS=1: the positive control the fused record cannot give. At a plain `rmsnorm`
    // there is no quantiser and no rotation, so the row's own formula `y_i = x_i*s*(w+wadd)*mult`
    // can be compared against the value the kernel wrote (`call.fwd.back()`, the in-place write's
    // own tensor). A match says the norm formula and the gain handling are right, and so that the
    // fused record's 100% disagreement is its input or its fused stage; a miss says the norm
    // formula itself is the defect -- no earlier diagnosis compared the twin against an exactly
    // known, unquantised write.
    if (std::getenv("R97_GAIN_POS") != nullptr && call.rec.op == "rmsnorm" && !call.fwd.empty()) {
        const AdValue& wr = call.fwd.back();
        if (wr.value.size() == x.value.size()) {
            double worst = 0.0, wscale = 0.0;
            int64_t wr_r = -1, wr_i = -1;
            for (int64_t r = 0; r < rows; ++r) {
                const int64_t b = r * n;
                double ss = 0.0;
                for (int64_t i = 0; i < n; ++i) ss += x.value[(size_t)(b + i)] * x.value[(size_t)(b + i)];
                const double s = 1.0 / std::sqrt(ss / (double)n + eps);
                for (int64_t i = 0; i < n; ++i) {
                    const double c = w ? w->value[(size_t)i] + wadd : 1.0;
                    const double yv = x.value[(size_t)(b + i)] * s * c * mult;
                    const double d = std::fabs(yv - wr.value[(size_t)(b + i)]);
                    if (std::fabs(wr.value[(size_t)(b + i)]) > wscale)
                        wscale = std::fabs(wr.value[(size_t)(b + i)]);
                    if (d > worst) {
                        worst = d;
                        wr_r = r;
                        wr_i = i;
                    }
                }
            }
            std::fprintf(stderr,
                         "NORMELEM pos=%lld op=%s rows=%lld n=%lld gain=%s worst=%.6g wmax=%.6g "
                         "rel=%.6g at r=%lld i=%lld\n",
                         (long long)call.rec.position, call.rec.op.c_str(), (long long)rows,
                         (long long)n, w != nullptr ? "yes" : "none", worst, wscale,
                         wscale > 0.0 ? worst / wscale : worst, (long long)wr_r, (long long)wr_i);
            std::fflush(stderr);
        }
    }

    // R97_DUMP_QUANT=1: the two numbers that decide what this row's Jacobian is -- `s` is the
    // per-row 1/rms of the activation and `gain` the largest `|w + wadd|` -- beside the sums it was
    // handed and left. A branch that grows by two decades per block shows here as an `s` or a
    // `gain` the forward does not imply, or as a `dx`/`dy` ratio that is neither. The range is
    // printed rather than a mean: one row with a tiny rms is enough to make the maximum large, and
    // that is a different story from every row being large.
    if (const char* dq = std::getenv("R97_DUMP_QUANT"); dq != nullptr && dq[0] == '1') {
        double s_min = 0.0, s_max = 0.0, gain_max = 0.0, dy_sum = 0.0, dx_sum = 0.0;
        for (int64_t r = 0; r < rows; ++r) {
            if (!readable[(size_t)r]) continue;
            const int64_t b = r * n;
            double ss = 0.0;
            for (int64_t i = 0; i < n; ++i)
                ss += x.value[(size_t)(b + i)] * x.value[(size_t)(b + i)];
            const double s = 1.0 / std::sqrt(ss / (double)n + eps);
            if (s_min == 0.0 || s < s_min) s_min = s;
            if (s > s_max) s_max = s;
        }
        if (w)
            for (int64_t i = 0; i < n && i < (int64_t)w->value.size(); ++i)
                gain_max = std::max(gain_max, std::fabs(w->value[(size_t)i] + wadd));
        for (const double v : dy.value) dy_sum += std::fabs(v);
        for (const double v : dx.value) dx_sum += std::fabs(v);
        std::fprintf(stderr,
                     "DUMPNORM pos=%lld op=%s rows=%lld n=%lld s_min=%.6g s_max=%.6g gain_max=%.6g "
                     "dy_sum=%.6g dx_sum=%.6g\n",
                     (long long)call.rec.position, call.rec.op.c_str(), (long long)rows,
                     (long long)n, s_min, s_max, gain_max, dy_sum, dx_sum);
        std::fflush(stderr);
    }
}

}  // namespace

R97_REGISTER_TWIN("mul", mul_twin)
R97_REGISTER_ADJOINT("mul", host, mul_host_adjoint)
R97_REGISTER_TWIN("scale", scale_twin)
R97_REGISTER_ADJOINT_F("scale", host, scale_host_adjoint, 1)
R97_REGISTER_TWIN("sigmoid", sigmoid_twin)
R97_REGISTER_ADJOINT("sigmoid", host, sigmoid_host_adjoint)
R97_REGISTER_TWIN("silu", silu_twin)
R97_REGISTER_ADJOINT("silu", host, silu_host_adjoint)
R97_REGISTER_TWIN("silu_mul", silu_mul_twin)
R97_REGISTER_ADJOINT("silu_mul", host, silu_mul_host_adjoint)
R97_REGISTER_TWIN("act_mul", act_mul_twin)
R97_REGISTER_ADJOINT("act_mul", host, act_mul_host_adjoint)
R97_REGISTER_TWIN("cast", cast_twin)
R97_REGISTER_ADJOINT("cast", host, cast_host_adjoint)
R97_REGISTER_TWIN("rmsnorm", rmsnorm_twin)
R97_REGISTER_ADJOINT_F("rmsnorm", host, rmsnorm_host_adjoint, 1)

}  // namespace r97
