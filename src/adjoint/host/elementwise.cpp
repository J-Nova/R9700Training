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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <variant>
#include <vector>

namespace r97 {
namespace {

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
    for (size_t i = 0; i < (size_t)na; ++i) y.value[i] = a.value[i] * b.value[row ? i % nb : i];
    out->push_back(std::move(y));
}

void mul_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.size() < 2 || call.dc->size() < 2) return;
    const AdValue &a = call.fwd[0], &b = call.fwd[1], &dy = call.cot[0];
    AdValue &da = call.dc->at(0), &db = call.dc->at(1);
    const size_t na = dy.value.size(), nb = b.value.size();
    if (nb == 0 || nb > na) return;

    for (size_t i = 0; i < na; ++i) {
        da.value[i] += dy.value[i] * b.value[nb == na ? i : i % nb];
        db.value[nb == na ? i : i % nb] += dy.value[i] * a.value[i];
    }
}

/* ------------------------------------------------------------------ scale: y = x * s[0] */

void scale_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.size() < 2 || op[1].value.empty()) return;
    const AdValue& x = op[0];
    const double s = op[1].value[0];
    AdValue y = write_value(rec, 2, x.shape);
    for (size_t i = 0; i < y.value.size(); ++i) y.value[i] = x.value[i] * s;
    out->push_back(std::move(y));
}

void scale_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.size() < 2 || call.fwd[1].value.empty() || call.dc->empty())
        return;
    const double s = call.fwd[1].value[0];
    AdValue& dx = call.dc->at(0);
    for (size_t i = 0; i < call.cot[0].value.size(); ++i) dx.value[i] += call.cot[0].value[i] * s;
}

/* ------------------------------------------------------------------ sigmoid: y = 1/(1+e^-x) */

void sigmoid_twin(const OpRecord&, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.size() < 1) return;
    const AdValue& x = op[0];
    AdValue y = x;
    for (size_t i = 0; i < y.value.size(); ++i) y.value[i] = sigmoid(x.value[i]);
    out->push_back(std::move(y));
}

void sigmoid_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.size() < 2 || call.dc->empty()) return;
    const AdValue &y = call.fwd[1], &dy = call.cot[0];  /* y, the op's own output */
    AdValue& dx = call.dc->at(0);
    for (size_t i = 0; i < dy.value.size(); ++i)
        dx.value[i] += dy.value[i] * y.value[i] * (1.0 - y.value[i]);
}

/* ------------------------------------------------------------------ silu: y = x * sigmoid(x) */

void silu_twin(const OpRecord&, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.size() < 1) return;
    const AdValue& x = op[0];
    AdValue y = x;
    for (size_t i = 0; i < y.value.size(); ++i) y.value[i] = act_value(kSilu, x.value[i]);
    out->push_back(std::move(y));
}

void silu_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.empty() || call.dc->empty()) return;
    const AdValue &x = call.fwd[0], &dy = call.cot[0];
    AdValue& dx = call.dc->at(0);
    for (size_t i = 0; i < dy.value.size(); ++i)
        dx.value[i] += dy.value[i] * act_slope(kSilu, x.value[i]);
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
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = r * 2 * n, yb = r * n;
        for (int64_t i = 0; i < n; ++i)
            y.value[(size_t)(yb + i)] = act_value(kSilu, x.value[(size_t)(xb + i)]) *
                                        x.value[(size_t)(xb + n + i)];
    }
    out->push_back(std::move(y));
}

void silu_mul_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.empty() || call.dc->empty()) return;
    const AdValue &x = call.fwd[0], &dy = call.cot[0];
    const int64_t n = gate_width(call.rec, x.shape);
    const int64_t rows = rows_of(x, 2 * n);
    if (n <= 0 || rows <= 0 || rows_of(dy, n) != rows) return;

    AdValue& dx = call.dc->at(0);
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = r * 2 * n, yb = r * n;
        for (int64_t i = 0; i < n; ++i) {
            const double g = x.value[(size_t)(xb + i)], u = x.value[(size_t)(xb + n + i)];
            const double d = dy.value[(size_t)(yb + i)];
            dx.value[(size_t)(xb + i)] += d * u * act_slope(kSilu, g);
            dx.value[(size_t)(xb + n + i)] += d * act_value(kSilu, g);
        }
    }
}

/* ------------------------------------------------------------------ act_mul: y = act(a) * b */

void act_mul_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.size() < 2) return;
    const std::string* name = param_str(rec, "act");
    Act act;
    if (!name || !act_of(*name, &act) || op[0].value.size() != op[1].value.size()) return;

    AdValue y = write_value(rec, 2, op[0].shape);
    for (size_t i = 0; i < y.value.size(); ++i)
        y.value[i] = act_value(act, op[0].value[i]) * op[1].value[i];
    out->push_back(std::move(y));
}

void act_mul_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.size() < 2 || call.dc->size() < 2) return;
    const std::string* name = param_str(call.rec, "act");
    Act act;
    if (!name || !act_of(*name, &act)) return;

    const AdValue &a = call.fwd[0], &b = call.fwd[1], &dy = call.cot[0];
    AdValue &da = call.dc->at(0), &db = call.dc->at(1);
    for (size_t i = 0; i < dy.value.size() && i < a.value.size() && i < b.value.size(); ++i) {
        da.value[i] += dy.value[i] * act_slope(act, a.value[i]) * b.value[i];
        db.value[i] += dy.value[i] * act_value(act, a.value[i]);
    }
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
    for (size_t i = 0; i < call.cot[0].value.size(); ++i) dx.value[i] += call.cot[0].value[i];
}

/* ------------------------------------------------------------------ rmsnorm */

/* y[r, i] = x[r, i] * rsqrt(mean(x[r]^2) + eps) * (w[i] + wadd) * mult, the gain formed in f64
 * (the kernel forms it in f32, before the store's rounding). `wadd` without a gain is refused by
 * the kernel and left to nothing here. */
void rmsnorm_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.empty()) return;
    const AdValue& x = op[0];
    const AdValue* w = op.size() > 1 && !op[1].value.empty() ? &op[1] : nullptr;
    const int64_t n = width_of(rec, x.shape);
    const int64_t rows = rows_of(x, n);
    if (rows <= 0) return;
    if (w && (int64_t)w->value.size() < n) return;
    const double eps = param_num(rec, "eps", 1e-6);
    // A live record may leave `wadd` out; the qwen35 fixture's norms all shift the gain by one,
    // which is what the engine ran, so an absent parameter means the fixture's 1 and not zero.
    bool has_wadd = false;
    for (const auto& p : rec.params) has_wadd = has_wadd || p.first == "wadd";
    const double wadd = has_wadd ? param_num(rec, "wadd", 0.0) : 1.0;
    const double mult = param_num(rec, "mult", 1.0);
    if (!w && wadd != 0.0) return;

    AdValue y = write_value(rec, 2, x.shape);
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t b = r * n;
        double ss = 0.0;
        for (int64_t i = 0; i < n; ++i) ss += x.value[(size_t)(b + i)] * x.value[(size_t)(b + i)];
        const double s = 1.0 / std::sqrt(ss / (double)n + eps);
        for (int64_t i = 0; i < n; ++i)
            y.value[(size_t)(b + i)] =
                x.value[(size_t)(b + i)] * s * (w ? w->value[(size_t)i] + wadd : 1.0) * mult;
    }
    out->push_back(std::move(y));
}

/* dx_j = mult * (s * c_j * dy_j - (s^3 x_j / n) * sum_i dy_i x_i c_i), c = (w + wadd).
 *
 * The second term is the row's shared rms: every element of the row moves it, so every dy_i
 * weights x_j. */
void rmsnorm_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || call.fwd.empty() || call.dc->empty()) return;
    const AdValue& x = call.fwd[0];
    const AdValue* w = call.fwd.size() > 1 && !call.fwd[1].value.empty() ? &call.fwd[1] : nullptr;
    const AdValue& dy = call.cot[0];
    const int64_t n = width_of(call.rec, x.shape);
    // The write is the step's rows; a whole-buffer read can be longer than it on the engine.
    const int64_t rows = std::min(rows_of(x, n), rows_of(dy, n));
    if (rows <= 0) return;
    if (w && (int64_t)w->value.size() < n) return;
    const double eps = param_num(call.rec, "eps", 1e-6);
    bool has_wadd = false;
    for (const auto& p : call.rec.params) has_wadd = has_wadd || p.first == "wadd";
    const double wadd = has_wadd ? param_num(call.rec, "wadd", 0.0) : 1.0;
    const double mult = param_num(call.rec, "mult", 1.0);
    if (!w && wadd != 0.0) return;

    AdValue& dx = call.dc->at(0);
    for (int64_t r = 0; r < rows; ++r) {
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
            dx.value[(size_t)(b + i)] += mult * (s * c * dy.value[(size_t)(b + i)] - shared);
        }
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
