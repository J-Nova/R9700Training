// src/adjoint/host/quant.cpp -- the quant family's host twins and host adjoints. quant_act_i8g
// turns groups of an activation into int8 codes and had_quant_act_i8g does the same after a
// width-group Hadamard rotation; rmsnorm_quant_fp8 and gated_quant_fp8 are the int8 trunk's fused
// E4M3 producers, which carry the residual add and the norm or the gated product in the same op.
//
// Rounding has zero derivative almost everywhere, so the adjoint is a definition rather than a
// derivative of the forward: the straight-through estimator of docs/adjoints/quant.md. What that
// doc says, and this file implements:
//
//   the cotangent of the codes passes through the rounding where the code is inside the grid
//   (|u| <= 127, u = v/s), is dropped where the clamp engaged, and is then rotated back with
//   W^T = W; the scale's cotangent is dropped, the scale being a constant.
//
// Finite differences cannot differentiate a step function, so each op also registers a *frozen*
// twin under `<op>_ste`: the forward linearised at a base point carried as the record's second read
// operand. Its slope is the adjoint above, which is what makes the check of docs/adjoints/README.md
// possible; no walk appends a `<op>_ste` record.
#include "r97/adjoint.h"

#include "parallel.h"

#include "rad_plugin.h"  // bf16 and fp8 e4m3, the fused quantiser's two grids

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace r97 {
namespace {

// The unit the parallel for cuts a quantiser pass into: one activation row, whose scale, codes and
// rotation are its own. A call with fewer rows than this stays on the calling thread.
constexpr int64_t kRowChunk = 2;

// The top of the int8 code grid. The scale is the group's amax over it, so the largest value of
constexpr double kTop = 127.0;

// The ops this file serves: the two of the family, and the linearised twin of each.
enum class Op { quant, had, quant_ste, had_ste };

Op op_of(const std::string& name) {
    if (name == "quant_act_i8g") return Op::quant;
    if (name == "had_quant_act_i8g") return Op::had;
    if (name == "quant_act_i8g_ste") return Op::quant_ste;
    if (name == "had_quant_act_i8g_ste") return Op::had_ste;
    throw std::invalid_argument("quant: no op of this family is named '" + name + "'");
}

bool is_rotated(Op op) { return op == Op::had || op == Op::had_ste; }
bool is_frozen(Op op) { return op == Op::quant_ste || op == Op::had_ste; }

// -- the record's parameters -------------------------------------------------

const int64_t* int_param(const OpRecord& rec, const char* key) {
    for (const OpParam& p : rec.params) {
        if (p.first != key) continue;
        const int64_t* v = std::get_if<int64_t>(&p.second);
        if (v == nullptr)
            throw std::invalid_argument("quant: parameter '" + std::string(key) +
                                        "' is not an integer");
        return v;
    }
    return nullptr;
}

int64_t required(const OpRecord& rec, const char* key) {
    const int64_t* v = int_param(rec, key);
    if (v == nullptr)
        throw std::invalid_argument("quant: the record has no '" + std::string(key) +
                                    "' parameter");
    return *v;
}

int64_t optional(const OpRecord& rec, const char* key, int64_t fallback) {
    const int64_t* v = int_param(rec, key);
    return v != nullptr ? *v : fallback;
}

// -- the fused producers' predicates -----------------------------------------
//
// The rows of the int8 trunk's three fused producers are keyed by a predicate on the record's
// parameters: the rotation the record can take and, for the two gated ones, the activation it
// serves. The device side keys its rows with the same three (src/adjoint/device/quant.hip), so one
// record reaches one row on both sides. A predicate reads the parameters only, because the row
// lookup has no record.

// A numeric parameter: an int, or a decimal string that is the whole of the string. Missing or
// non-numeric leaves false rather than throwing -- a lookup refuses a record it cannot judge.
bool num_param(const std::vector<OpParam>& params, const char* name, double* out) {
    for (const OpParam& p : params) {
        if (p.first != name) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) {
            *out = static_cast<double>(*i);
            return true;
        }
        const std::string* s = std::get_if<std::string>(&p.second);
        if (s == nullptr || s->empty()) return false;
        char* end = nullptr;
        const double v = std::strtod(s->c_str(), &end);
        if (end == nullptr || *end != '\0') return false;
        *out = v;
        return true;
    }
    return false;
}

// The rotation and the per-group scale are the record's `n` and `group`: a power of two that
// divides the width, no wider than one block's threads. A record outside that is refused at the
// row lookup rather than served approximately.
bool rotated(const std::vector<OpParam>& params) {
    double n = 0.0, group = 0.0;
    if (!num_param(params, "n", &n) || !num_param(params, "group", &group)) return false;
    if (n <= 0.0 || n != std::floor(n)) return false;
    if (group <= 0.0 || group != std::floor(group) || group > 1024.0) return false;
    const int64_t g = static_cast<int64_t>(group);
    if ((g & (g - 1)) != 0 || static_cast<int64_t>(n) % g != 0) return false;
    return true;
}

// The gated producer's activation: silu, its documented first mode and the one the card's swiglu
// sites issue. The sigmoid form would take a different second half and is not served yet.
bool serves_gated(const std::vector<OpParam>& params) {
    if (!rotated(params)) return false;
    for (const OpParam& p : params) {
        if (p.first != "act") continue;
        const std::string* act = std::get_if<std::string>(&p.second);
        return act != nullptr && *act == "silu";
    }
    return true;
}

// The gate's activation: sigmoid, the mode the card's narrow-head gate issues. It is required,
// because the record's default is the schema's to say, not this row's to guess.
bool serves_gate(const std::vector<OpParam>& params) {
    if (!rotated(params)) return false;
    for (const OpParam& p : params) {
        if (p.first != "act") continue;
        const std::string* act = std::get_if<std::string>(&p.second);
        return act != nullptr && *act == "sigmoid";
    }
    return false;
}

// The shapes of one call, read the way the kernel reads them.
struct Geometry {
    int64_t n = 0;     // columns of the activation
    int64_t group = 0; // the effective group width
    int64_t ngrp = 0;  // groups per row
    bool rotated = false;
};

Geometry geometry(Op op, const OpRecord& rec) {
    Geometry g;
    g.rotated = is_rotated(op);
    g.n = required(rec, "n");
    if (g.n <= 0) throw std::invalid_argument("quant: 'n' must be positive");
    const int64_t t = optional(rec, "group", 0);
    if (g.rotated) {
        // A rotation exists only for a power of two, and a group must not straddle the end of a
        // row: the kernel refuses both, and so does the twin.
        if (t <= 0 || (t & (t - 1)) != 0 || g.n % t != 0)
            throw std::invalid_argument(
                "had_quant_act_i8g: 'group' must be a power of two that divides 'n'");
        g.group = t;
    } else {
        // One scale per row for a group that is absent or wider than the row, and a short last
        // group when the group does not divide the row.
        g.group = (t <= 0 || t > g.n) ? g.n : t;
    }
    g.ngrp = (g.n + g.group - 1) / g.group;
    return g;
}

// -- the quantiser -----------------------------------------------------------

// The unnormalised Walsh-Hadamard transform of one group, the kernel's own loop: W W = t I.
void fwht(double* v, int64_t t) {
    for (int64_t h = 1; h < t; h <<= 1)
        for (int64_t i0 = 0; i0 < t; i0 += 2 * h)
            for (int64_t j = i0; j < i0 + h; ++j) {
                const double a = v[j];
                const double b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
}

// Everything the forward knows about one call: the values the quantiser saw (`v`), the scale of
// every group, the codes, and where the clamp engaged. `v`, `code` and `live` are
// piecewise constant in the activation, which is why the frozen twin below holds them at a base
// point and why the adjoint reads them off the forward's own values.
struct Point {
    int64_t rows = 0;
    int64_t n = 0;
    int64_t group = 0;
    int64_t ngrp = 0;
    bool rotated = false;
    std::vector<double> v;       // [rows, n], rotated for the Hadamard op
    std::vector<double> scale;   // [rows, ngrp]
    std::vector<double> code;    // [rows, n]
    std::vector<uint8_t> live;   // 1 where the clamp did not engage
};

Point quantise(const std::vector<double>& x, int64_t rows, const Geometry& geo) {
    Point p;
    p.rows = rows;
    p.n = geo.n;
    p.group = geo.group;
    p.ngrp = geo.ngrp;
    p.rotated = geo.rotated;
    p.v = x;
    p.scale.assign(static_cast<size_t>(rows * geo.ngrp), 0.0);
    p.code.assign(static_cast<size_t>(rows * geo.n), 0.0);
    p.live.assign(static_cast<size_t>(rows * geo.n), 0);
    parallel_for(rows, kRowChunk, [&](int64_t r) {
        for (int64_t g = 0; g < geo.ngrp; ++g) {
            const int64_t k0 = g * geo.group;
            const int64_t w = std::min(geo.group, geo.n - k0);  // a short last group
            double* v = p.v.data() + r * geo.n + k0;
            if (geo.rotated) fwht(v, geo.group);  // n % group == 0, so every block is full
            double amax = 0.0;
            for (int64_t i = 0; i < w; ++i) amax = std::max(amax, std::fabs(v[i]));
            // The zero group has no direction to scale: one scale, codes zero, nothing clamped.
            const double s = amax > 0.0 ? amax * (1.0 / kTop) : 1.0;
            p.scale[static_cast<size_t>(r * geo.ngrp + g)] = s;
            for (int64_t i = 0; i < w; ++i) {
                const size_t j = static_cast<size_t>(r * geo.n + k0 + i);
                const double u = v[i] * (1.0 / s);
                const double c = std::nearbyint(u);  // nearest, ties to even
                p.code[j] = std::min(kTop, std::max(-kTop, c));
                p.live[j] = std::fabs(u) <= kTop ? 1 : 0;
            }
        }
    });
    return p;
}

// The scale a column is stored against.
double scale_of(const Point& p, int64_t r, int64_t col) {
    return p.scale[static_cast<size_t>(r * p.ngrp + col / p.group)];
}

// -- outputs -----------------------------------------------------------------

std::string write_dtype(const OpRecord& rec, size_t index, const char* fallback) {
    size_t w = 0;
    for (const Operand& o : rec.operands) {
        if (!o.written) continue;
        if (w++ == index) return o.dtype.empty() ? std::string(fallback) : o.dtype;
    }
    return fallback;
}

AdValue make_value(const OpRecord& rec, size_t index, const char* dtype, int64_t rows, int64_t cols,
                   std::vector<double> value) {
    AdValue a;
    a.dtype = write_dtype(rec, index, dtype);
    a.shape = {rows, cols};
    a.value = std::move(value);
    return a;
}

// -- the twin ----------------------------------------------------------------

// The forward in f64, and (under `<op>_ste`) its linearisation at a base point: with `operands`
// = (x, x0), every output is held at what the forward wrote at x0 except for the rounding
// residual, which is taken straight through in the units of the codes,
//
//     codes(x)  =  c0 + live0 ⊙ (v(x) − v0) / s0,        scale(x)  =  s0,
//
// so d codes / dx = live0 ⊙ dv/dx / s0 at x0. The product s·codes is the issue's twin
// x_hat = W x + const; it is divided by the scale here because the cotangent the tape hands back
// is the codes' own (docs/adjoints/quant.md).
void quant_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                std::vector<AdValue>* out) {
    const Op op = op_of(rec.op);
    const Geometry geo = geometry(op, rec);
    if (operands.empty()) throw std::invalid_argument("quant: the op reads an activation");
    const std::vector<double>& x = operands[0].value;
    if (x.size() % static_cast<size_t>(geo.n) != 0)
        throw std::invalid_argument("quant: the activation is not a whole number of rows of 'n'");
    const int64_t rows = static_cast<int64_t>(x.size()) / geo.n;

    if (!is_frozen(op)) {
        const Point fwd = quantise(x, rows, geo);
        out->push_back(make_value(rec, 0, "i8", rows, geo.n, fwd.code));
        out->push_back(make_value(rec, 1, "f32", rows, geo.ngrp, fwd.scale));
        return;
    }

    if (operands.size() < 2)
        throw std::invalid_argument(
            "quant: the straight-through twin reads the base point as its second operand");
    if (operands[1].value.size() != x.size())
        throw std::invalid_argument("quant: the base point and the activation have different sizes");
    const Point base = quantise(operands[1].value, rows, geo);
    const Point at = quantise(x, rows, geo);  // only `at.v`, the rotated activation, is used

    std::vector<double> codes(static_cast<size_t>(rows * geo.n), 0.0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < geo.n; ++j) {
            const size_t i = static_cast<size_t>(r * geo.n + j);
            const double s = scale_of(base, r, j);
            codes[i] = base.live[i] != 0 ? base.code[i] + (at.v[i] - base.v[i]) / s : base.code[i];
        }
    }
    out->push_back(make_value(rec, 0, "i8", rows, geo.n, std::move(codes)));
    out->push_back(make_value(rec, 1, "f32", rows, geo.ngrp, base.scale));
}

// -- the adjoint -------------------------------------------------------------

// dL/dx = W^T (live ⊙ cot_codes / s), the codes' cotangent taken through the rounding and dropped
// where the code saturated, then rotated back. W^T = W for the Walsh order, and the scale's
// cotangent is dropped: the scale is a constant.
void quant_adjoint(const AdjointCall& call) {
    const Op op = op_of(call.rec.op);
    const Geometry geo = geometry(op, call.rec);
    size_t reads = 0;
    for (const Operand& o : call.rec.operands)
        if (!o.written) ++reads;
    // The schema: the activation, the codes the op wrote, the scales it wrote. The frozen twin
    // reads its base point between the activation and the writes, so neither index is written
    // down here -- the writes are found from the record.
    if (reads == 0 || call.fwd.size() < reads + 2)
        throw std::invalid_argument(
            "quant: the record must read the activation and write the codes and the scale");
    if (call.cot.size() < 2)
        throw std::invalid_argument(
            "quant: the record writes the codes and the scale, so the walk hands two cotangents");
    if (call.dc == nullptr || call.dc->size() < reads)
        throw std::invalid_argument("quant: the walk handed no room for the activation's cotangent");

    const AdValue& x = call.fwd[0];
    const AdValue& scale = call.fwd[reads + 1];
    const AdValue& g = call.cot[0];  // the codes' cotangent; the scale's, cot[1], is dropped
    if (x.value.size() % static_cast<size_t>(geo.n) != 0)
        throw std::invalid_argument("quant: the activation is not a whole number of rows of 'n'");
    const int64_t rows = static_cast<int64_t>(x.value.size()) / geo.n;
    if (g.value.size() != x.value.size())
        throw std::invalid_argument("quant: the codes' cotangent has the activation's shape");
    if (scale.value.size() != static_cast<size_t>(rows * geo.ngrp))
        throw std::invalid_argument("quant: the scale does not have one value per (row, group)");

    // The values the quantiser saw, for the mask, from the activation the walk hands back.
    std::vector<double> v(x.value);
    std::vector<double> d(g.value);
    parallel_for(rows, kRowChunk, [&](int64_t r) {
        for (int64_t gi = 0; gi < geo.ngrp; ++gi) {
            const int64_t k0 = gi * geo.group;
            const int64_t w = std::min(geo.group, geo.n - k0);
            if (geo.rotated) fwht(v.data() + r * geo.n + k0, geo.group);
            const double s = scale.value[static_cast<size_t>(r * geo.ngrp + gi)];
            // The family's scale is amax/127, or 1 for the zero group: always positive.
            if (!(s > 0.0))
                throw std::invalid_argument("quant: the scale of a group must be positive");
            for (int64_t i = 0; i < w; ++i) {
                const size_t j = static_cast<size_t>(r * geo.n + k0 + i);
                // The clamp's own derivative: zero where it engaged, one inside the grid.
                d[j] = std::fabs(v[j] * (1.0 / s)) > kTop ? 0.0 : d[j] / s;
            }
            if (geo.rotated) fwht(d.data() + r * geo.n + k0, geo.group);
        }
    });

    AdValue& dx = call.dc->at(0);
    for (size_t i = 0; i < dx.value.size() && i < d.size(); ++i) dx.value[i] += d[i];
}

// -- the fused fp8 producers -------------------------------------------------
//
// The int8 trunk's norms and its gated MLP do their rounding inside the producer:
// `rmsnorm_quant_fp8` writes the normalised bf16 activation and its E4M3 codes, and
// `gated_quant_fp8` writes the bf16 product and its codes. Both quantise per 128 columns with the
// same scale the int8 rows use above, so the straight-through rule is the same too; what is new is
// the differentiable part around it, a norm or an activation gate. The code cotangent is divided
// by the group's scale because the tape hands back the codes' own cotangent, not `s * codes`.
constexpr double kFp8Top = 448.0;  // E4M3's largest finite magnitude

double bf16_round(double x) {
    return static_cast<double>(rad_bf16_to_f32(rad_f32_to_bf16(static_cast<float>(x))));
}

std::string string_param(const OpRecord& rec, const char* key, const char* fallback) {
    for (const OpParam& p : rec.params)
        if (p.first == key) {
            if (const std::string* s = std::get_if<std::string>(&p.second)) return *s;
            throw std::invalid_argument("quant: parameter '" + std::string(key) +
                                        "' is not a string");
        }
    return fallback;
}

double double_param(const OpRecord& rec, const char* key, double fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != key) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) return static_cast<double>(*i);
        if (const std::string* s = std::get_if<std::string>(&p.second)) {
            try {
                return std::stod(*s);
            } catch (const std::exception&) {
                break;
            }
        }
        throw std::invalid_argument("quant: parameter '" + std::string(key) +
                                    "' is not a number");
    }
    return fallback;
}

struct Fp8Geometry {
    int64_t n = 0;
    int64_t group = 0;
    int64_t ngrp = 0;
};

Fp8Geometry fp8_geometry(const OpRecord& rec) {
    Fp8Geometry g;
    g.n = required(rec, "n");
    if (g.n <= 0) throw std::invalid_argument("quant: 'n' must be positive");
    const int64_t t = optional(rec, "group", 0);
    g.group = (t <= 0 || t > g.n) ? g.n : t;
    g.ngrp = (g.n + g.group - 1) / g.group;
    return g;
}

struct Fp8Point {
    int64_t rows = 0;
    int64_t n = 0;
    int64_t group = 0;
    int64_t ngrp = 0;
    std::vector<double> value;  // what the quantiser saw, bf16-rounded
    std::vector<double> scale;  // [rows, ngrp]
    std::vector<double> code;   // [rows, n], E4M3 values (not the raw bytes)
    std::vector<uint8_t> live;  // 1 where the clamp did not engage
};

Fp8Point quantise_fp8(const std::vector<double>& v, int64_t rows, const Fp8Geometry& geo) {
    Fp8Point p;
    p.rows = rows;
    p.n = geo.n;
    p.group = geo.group;
    p.ngrp = geo.ngrp;
    p.value = v;
    p.scale.assign(static_cast<size_t>(rows * geo.ngrp), 0.0);
    p.code.assign(static_cast<size_t>(rows * geo.n), 0.0);
    p.live.assign(static_cast<size_t>(rows * geo.n), 0);
    // The scale and the code are float arithmetic, as `q8_group` does them: a group's amax times
    // one over the format's top, then the reciprocal. Two codes that sit within a float ulp of a
    // rounding boundary are one code apart, so the twin has to take the reciprocal the way the
    // kernel does or the check would see a whole E4M3 step of difference rather than a bit.
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t gi = 0; gi < geo.ngrp; ++gi) {
            const int64_t k0 = gi * geo.group;
            const int64_t w = std::min(geo.group, geo.n - k0);
            float amax = 0.0f;
            for (int64_t i = 0; i < w; ++i)
                amax = std::max(amax, std::fabs(static_cast<float>(
                                           v[static_cast<size_t>(r * geo.n + k0 + i)])));
            const float top = static_cast<float>(kFp8Top);
            const float s = amax > 0.0f ? amax * (1.0f / top) : 1.0f;
            const float inv = 1.0f / s;
            p.scale[static_cast<size_t>(r * geo.ngrp + gi)] = static_cast<double>(s);
            for (int64_t i = 0; i < w; ++i) {
                const size_t j = static_cast<size_t>(r * geo.n + k0 + i);
                const float u = static_cast<float>(v[j]) * inv;
                const uint8_t bits = rad_f32_to_fp8e4m3(u);
                p.code[j] = static_cast<double>(rad_fp8e4m3_to_f32(bits));
                p.live[j] = std::fabs(u) <= top ? 1 : 0;
            }
        }
    }
    return p;
}

double fp8_scale_of(const Fp8Point& p, int64_t r, int64_t col) {
    return p.scale[static_cast<size_t>(r * p.ngrp + col / p.group)];
}

int schema_index(const OpRecord& rec, const char* name) {
    for (size_t i = 0; i < rec.operands.size(); ++i)
        if (rec.operands[i].buffer == name) return static_cast<int>(i);
    return -1;
}

int write_index(const OpRecord& rec, int at) {
    int w = 0;
    for (const Operand& o : rec.operands)
        if (o.written && w++ == at) return w - 1;
    return -1;
}

const AdValue* cot_of(const AdjointCall& call, int write) {
    if (write < 0 || static_cast<size_t>(write) >= call.cot.size()) return nullptr;
    return &call.cot[static_cast<size_t>(write)];
}

// The bf16 activation the norm or the gate produced, and the fp8 point around it.
struct Fp8Activation {
    Fp8Point point;
    std::vector<double> value;  // [rows, n], in the bf16 activation's own units
};

// The scale and the clamp mask are the only parts of the quantiser the adjoint reads; they are
// properties of the forward, so they come back from the written scale and the activation.
void add_fp8_codes(const AdValue& codes, const AdValue& scale, int64_t rows, const Fp8Geometry& geo,
                   const std::vector<double>& activation, std::vector<double>* d) {
    if (codes.value.size() != static_cast<size_t>(rows * geo.n))
        throw std::invalid_argument("quant: the fp8 codes do not have the activation's shape");
    if (scale.value.size() != static_cast<size_t>(rows * geo.ngrp))
        throw std::invalid_argument("quant: the fp8 scale does not have one value per (row, group)");
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t gi = 0; gi < geo.ngrp; ++gi) {
            const int64_t k0 = gi * geo.group;
            const int64_t w = std::min(geo.group, geo.n - k0);
            const double s = scale.value[static_cast<size_t>(r * geo.ngrp + gi)];
            if (!(s > 0.0))
                throw std::invalid_argument("quant: the scale of a group must be positive");
            for (int64_t i = 0; i < w; ++i) {
                const size_t j = static_cast<size_t>(r * geo.n + k0 + i);
                if (std::fabs(activation[j] / s) <= kFp8Top)
                    (*d)[j] += codes.value[j] / s;
            }
        }
    }
}

// Whether the record asks for the linearised twin: `<op>_ste`, whose base point is the record's
// last read operand (the schema is frozen, so the base point is appended after everything else).
// The forward is then held at the base point's scale and taken straight through in the units of its
// codes,
//
//     codes(x) = c0 + live(x) * (v(x) - v0) / s0,   scale(x) = s0,
//
// with v the activation and no rounding step; the residual and the activation come back as their
// f64 values, so a finite difference sees the norm's or the gate's Jacobian and the quantiser's
// slope rather than a bf16 step.
bool is_linearised(const std::string& op) {
    return op.size() > 4 && op.compare(op.size() - 4, 4, "_ste") == 0;
}

// `rmsnorm_quant_fp8`: y = bf16(rmsnorm(r) * gain), r = x or bf16(x + residual), then y's E4M3
// codes. The residual is in-place, so its output cotangent and the norm's input cotangent both go
// back to its old value and to x.
void rmsnorm_quant_fp8_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                            std::vector<AdValue>* out) {
    const Fp8Geometry geo = fp8_geometry(rec);
    const bool linearised = is_linearised(rec.op);
    const bool has_residual = rec.operands.size() > 1 &&
                              rad_dtype_bits(rad_dtype_parse(rec.operands[1].dtype.c_str())) > 0 &&
                              !rec.operands[1].shape.empty();
    if (operands.empty()) throw std::invalid_argument("quant: rmsnorm_quant_fp8 reads an activation");
    const std::vector<double>& x = operands[0].value;
    if (x.size() % static_cast<size_t>(geo.n) != 0)
        throw std::invalid_argument("quant: the activation is not a whole number of rows of 'n'");
    const int64_t rows = static_cast<int64_t>(x.size()) / geo.n;
    const std::vector<double>* resid = has_residual ? &operands[1].value : nullptr;
    const std::vector<double>& weight = operands[has_residual ? 2 : 1].value;
    if (weight.size() != static_cast<size_t>(geo.n))
        throw std::invalid_argument("quant: the norm gain does not have 'n' values");
    if (resid != nullptr && resid->size() != x.size())
        throw std::invalid_argument("quant: the residual and the activation have different sizes");

    const double eps = double_param(rec, "eps", 1e-6);
    const double wadd = double_param(rec, "wadd", 0.0);
    // The linearised twin has no rounding step at all, so its finite differences are the norm's
    // Jacobian and the quantiser's slope; the forward's own twin rounds as the kernel does.
    std::vector<double> r(x.size());
    for (size_t i = 0; i < r.size(); ++i) {
        const double sum = resid != nullptr ? x[i] + (*resid)[i] : x[i];
        r[i] = linearised ? sum : bf16_round(sum);
    }

    std::vector<double> y(x.size());
    for (int64_t m = 0; m < rows; ++m) {
        double ss = 0.0;
        for (int64_t j = 0; j < geo.n; ++j) {
            const double v = r[static_cast<size_t>(m * geo.n + j)];
            ss += v * v;
        }
        const double rs = 1.0 / std::sqrt(ss / static_cast<double>(geo.n) + eps);
        for (int64_t j = 0; j < geo.n; ++j) {
            const size_t i = static_cast<size_t>(m * geo.n + j);
            const double ny = r[i] * rs * (weight[static_cast<size_t>(j)] + wadd);
            y[i] = linearised ? ny : bf16_round(ny);
        }
    }
    if (linearised) {
        const std::vector<double>& base_y = operands.back().value;
        if (base_y.size() != y.size())
            throw std::invalid_argument("quant: the base point has the activation's shape");
        const Fp8Point b = quantise_fp8(base_y, rows, geo);
        std::vector<double> codes(y.size(), 0.0);
        for (int64_t m = 0; m < rows; ++m) {
            for (int64_t j = 0; j < geo.n; ++j) {
                const size_t i = static_cast<size_t>(m * geo.n + j);
                const double s = fp8_scale_of(b, m, j);
                // The clamp's mask is the adjoint's: it reads where the activation sits now,
                // against the base point's scale, so the two agree on the elements whose code
                // stops responding.
                codes[i] = std::fabs(y[i] / s) <= kFp8Top ? b.code[i] + (y[i] - base_y[i]) / s
                                                          : b.code[i];
            }
        }
        out->push_back(has_residual ? make_value(rec, 0, "bf16", rows, geo.n, r)
                                    : AdValue{"invalid", {}, {}});
        out->push_back(make_value(rec, 1, "fp8_e4m3", rows, geo.n, std::move(codes)));
        out->push_back(make_value(rec, 2, "f32", rows, geo.ngrp, b.scale));
        out->push_back(make_value(rec, 3, "bf16", rows, geo.n, y));
        return;
    }
    const Fp8Point p = quantise_fp8(y, rows, geo);
    out->push_back(has_residual ? make_value(rec, 0, "bf16", rows, geo.n, r)
                                : AdValue{"invalid", {}, {}});
    out->push_back(make_value(rec, 1, "fp8_e4m3", rows, geo.n, p.code));
    out->push_back(make_value(rec, 2, "f32", rows, geo.ngrp, p.scale));
    out->push_back(make_value(rec, 3, "bf16", rows, geo.n, y));
}

void rmsnorm_quant_fp8_adjoint(const AdjointCall& call) {
    const Fp8Geometry geo = fp8_geometry(call.rec);
    // The schema is x, residual, weight, codes, scale, y: the read activation's buffer is the
    // residual stream's name (`h`), not `x`, which is the residual operand it is added into.
    const int ix = 0;
    const int ires = 1;
    const int iw = 2;
    if (call.fwd.size() <= static_cast<size_t>(iw) || call.dc == nullptr ||
        call.dc->size() <= static_cast<size_t>(iw))
        throw std::invalid_argument("quant: rmsnorm_quant_fp8 needs its activation and gain");
    const AdValue& x = call.fwd[static_cast<size_t>(ix)];
    const AdValue& weight = call.fwd[static_cast<size_t>(iw)];
    if (x.value.size() % static_cast<size_t>(geo.n) != 0)
        throw std::invalid_argument("quant: the activation is not a whole number of rows of 'n'");
    const int64_t rows = static_cast<int64_t>(x.value.size()) / geo.n;
    if (weight.value.size() != static_cast<size_t>(geo.n))
        throw std::invalid_argument("quant: the norm gain does not have 'n' values");

    const bool has_residual = call.rec.operands.size() > static_cast<size_t>(ires) &&
                              rad_dtype_bits(rad_dtype_parse(call.rec.operands[ires].dtype.c_str())) > 0 &&
                              !call.rec.operands[ires].shape.empty();
    const AdValue& residual_out = call.fwd[static_cast<size_t>(ires)];
    std::vector<double> r(x.value);
    if (has_residual) {
        if (residual_out.value.size() != r.size())
            throw std::invalid_argument("quant: the residual output has the activation's shape");
        r = residual_out.value;
    }
    const AdValue& y = call.fwd[5];
    if (y.value.size() != r.size())
        throw std::invalid_argument("quant: the normalised activation has the activation's shape");

    const double eps = double_param(call.rec, "eps", 1e-6);
    const double wadd = double_param(call.rec, "wadd", 0.0);
    std::vector<double> rs(static_cast<size_t>(rows), 0.0);
    std::vector<double> d(r.size(), 0.0);
    for (int64_t m = 0; m < rows; ++m) {
        double ss = 0.0;
        for (int64_t j = 0; j < geo.n; ++j) {
            const double v = r[static_cast<size_t>(m * geo.n + j)];
            ss += v * v;
        }
        rs[static_cast<size_t>(m)] =
            1.0 / std::sqrt(ss / static_cast<double>(geo.n) + eps);
    }

    // The output activation's cotangent is already in y's units; the codes' is divided by their
    // scale and kept only where the fp8 clamp did not engage.
    const AdValue* dy = cot_of(call, 3);
    const AdValue* dq = cot_of(call, 1);
    if (dy != nullptr) {
        if (dy->value.size() != d.size())
            throw std::invalid_argument("quant: the activation's cotangent has y's shape");
        for (size_t i = 0; i < d.size(); ++i) d[i] += dy->value[i];
    }
    if (dq != nullptr) add_fp8_codes(*dq, call.fwd[4], rows, geo, y.value, &d);

    // `cot` is indexed by WRITE order, not schema order (include/r97/adjoint.h:76), and the twin
    // writes residual first: residual=0, codes=1, scale=2, y=3. So write 0 is the residual, and
    // write 1 -- which `dq` already consumed above -- is the codes. `ires` is a schema index and
    // belongs to `fwd`/`dc` only.
    const AdValue* dres = cot_of(call, 0);
    std::vector<double> dres_out;
    if (has_residual && dres != nullptr) {
        if (dres->value.size() != d.size())
            throw std::invalid_argument("quant: the residual cotangent has the activation's shape");
        dres_out = dres->value;
    }

    std::vector<double> dx(d.size(), 0.0);
    std::vector<double> dw(static_cast<size_t>(geo.n), 0.0);
    for (int64_t m = 0; m < rows; ++m) {
        const double scale = rs[static_cast<size_t>(m)];
        double drs = 0.0;
        for (int64_t j = 0; j < geo.n; ++j) {
            const size_t i = static_cast<size_t>(m * geo.n + j);
            const double gain = weight.value[static_cast<size_t>(j)] + wadd;
            drs += d[i] * r[i] * gain;
            dx[i] += d[i] * scale * gain;
            dw[static_cast<size_t>(j)] += d[i] * r[i] * scale;
        }
        const double rs3 = scale * scale * scale;
        for (int64_t j = 0; j < geo.n; ++j) {
            const size_t i = static_cast<size_t>(m * geo.n + j);
            dx[i] -= drs * rs3 * r[i] / static_cast<double>(geo.n);
        }
    }
    for (size_t i = 0; i < dx.size(); ++i) {
        call.dc->at(static_cast<size_t>(ix)).value[i] += dx[i];
        if (has_residual) {
            call.dc->at(static_cast<size_t>(ires)).value[i] += dx[i];
            if (!dres_out.empty()) {
                call.dc->at(static_cast<size_t>(ix)).value[i] += dres_out[i];
                call.dc->at(static_cast<size_t>(ires)).value[i] += dres_out[i];
            }
        }
    }
    for (size_t j = 0; j < dw.size(); ++j)
        call.dc->at(static_cast<size_t>(iw)).value[j] += dw[j];
}

// `gated_quant_fp8`: p = bf16(act(gate) * up), then p's E4M3 codes. The activation is two halves
// of one row, so both cotangents go back into the same buffer.
void gated_quant_fp8_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                          std::vector<AdValue>* out) {
    const Fp8Geometry geo = fp8_geometry(rec);
    const bool linearised = is_linearised(rec.op);
    if (operands.empty()) throw std::invalid_argument("quant: gated_quant_fp8 reads an activation");
    const std::vector<double>& gu = operands[0].value;
    const int64_t width = 2 * geo.n;
    if (gu.size() % static_cast<size_t>(width) != 0)
        throw std::invalid_argument("quant: the gate|up activation is not a whole number of rows");
    const int64_t rows = static_cast<int64_t>(gu.size()) / width;
    const bool sigmoid = string_param(rec, "act", "silu") == "sigmoid";

    std::vector<double> p(static_cast<size_t>(rows * geo.n), 0.0);
    for (int64_t m = 0; m < rows; ++m) {
        for (int64_t j = 0; j < geo.n; ++j) {
            const double g = gu[static_cast<size_t>(m * width + j)];
            const double u = gu[static_cast<size_t>(m * width + geo.n + j)];
            const double s = 1.0 / (1.0 + std::exp(-g));
            const double a = sigmoid ? s : g * s;
            p[static_cast<size_t>(m * geo.n + j)] = linearised ? a * u : bf16_round(a * u);
        }
    }
    if (linearised) {
        const std::vector<double>& base = operands.back().value;
        if (base.size() != p.size())
            throw std::invalid_argument("quant: the base point has the product's shape");
        const Fp8Point b = quantise_fp8(base, rows, geo);
        std::vector<double> codes(p.size(), 0.0);
        for (int64_t m = 0; m < rows; ++m) {
            for (int64_t j = 0; j < geo.n; ++j) {
                const size_t i = static_cast<size_t>(m * geo.n + j);
                const double s = fp8_scale_of(b, m, j);
                // As above: the mask follows the activation, not the base point's.
                codes[i] = std::fabs(p[i] / s) <= kFp8Top ? b.code[i] + (p[i] - base[i]) / s
                                                          : b.code[i];
            }
        }
        out->push_back(make_value(rec, 0, "fp8_e4m3", rows, geo.n, std::move(codes)));
        out->push_back(make_value(rec, 1, "f32", rows, geo.ngrp, b.scale));
        out->push_back(make_value(rec, 2, "bf16", rows, geo.n, p));
        return;
    }
    const Fp8Point q = quantise_fp8(p, rows, geo);
    out->push_back(make_value(rec, 0, "fp8_e4m3", rows, geo.n, q.code));
    out->push_back(make_value(rec, 1, "f32", rows, geo.ngrp, q.scale));
    out->push_back(make_value(rec, 2, "bf16", rows, geo.n, p));
}

void gated_quant_fp8_adjoint(const AdjointCall& call) {
    const Fp8Geometry geo = fp8_geometry(call.rec);
    if (call.fwd.empty() || call.dc == nullptr || call.dc->empty())
        throw std::invalid_argument("quant: gated_quant_fp8 needs its activation");
    const AdValue& gu = call.fwd[0];
    const int64_t width = 2 * geo.n;
    if (gu.value.size() % static_cast<size_t>(width) != 0)
        throw std::invalid_argument("quant: the gate|up activation is not a whole number of rows");
    const int64_t rows = static_cast<int64_t>(gu.value.size()) / width;
    const bool sigmoid = string_param(call.rec, "act", "silu") == "sigmoid";

    std::vector<double> p(static_cast<size_t>(rows * geo.n), 0.0);
    for (int64_t m = 0; m < rows; ++m) {
        for (int64_t j = 0; j < geo.n; ++j) {
            const double g = gu.value[static_cast<size_t>(m * width + j)];
            const double u = gu.value[static_cast<size_t>(m * width + geo.n + j)];
            const double s = 1.0 / (1.0 + std::exp(-g));
            p[static_cast<size_t>(m * geo.n + j)] = bf16_round((sigmoid ? s : g * s) * u);
        }
    }
    const AdValue& ffn = call.fwd[3];
    if (ffn.value.size() != p.size())
        throw std::invalid_argument("quant: the gated product does not have the activation's shape");

    std::vector<double> dp(p.size(), 0.0);
    const AdValue* dy = cot_of(call, 2);
    if (dy != nullptr) {
        if (dy->value.size() != dp.size())
            throw std::invalid_argument("quant: the product cotangent has the product's shape");
        dp = dy->value;
    }
    const AdValue* dq = cot_of(call, 0);
    if (dq != nullptr) add_fp8_codes(*dq, call.fwd[2], rows, geo, ffn.value, &dp);

    AdValue& dgu = call.dc->at(0);
    for (int64_t m = 0; m < rows; ++m) {
        for (int64_t j = 0; j < geo.n; ++j) {
            const size_t i = static_cast<size_t>(m * geo.n + j);
            const double g = gu.value[static_cast<size_t>(m * width + j)];
            const double u = gu.value[static_cast<size_t>(m * width + geo.n + j)];
            const double s = 1.0 / (1.0 + std::exp(-g));
            const double a = sigmoid ? s : g * s;
            const double da = sigmoid ? s * (1.0 - s) : s * (1.0 + g * (1.0 - s));
            dgu.value[static_cast<size_t>(m * width + j)] += dp[i] * u * da;
            dgu.value[static_cast<size_t>(m * width + geo.n + j)] += dp[i] * a;
        }
    }
}

// -- the fused int8 producers ------------------------------------------------
//
// The int8 trunk's norm and its gates round inside the producer: `rmsnorm_had_quant_i8g`,
// `gated_had_quant_i8g` and `gate_had_quant_i8g` apply an elementwise part and hand the result to
// the quantiser above. Their forward is the parts' forwards composed and their adjoint the parts'
// host adjoints composed, in the order the kernel undoes them (docs/adjoints/quant.md). The parts
// are the elementwise family's rows (src/adjoint/host/elementwise.cpp) and the quantiser's own
// under `had_quant_act_i8g`, found through the registry so there is one definition of each.

// The elementwise op a fused producer applies before the rotation.
enum class Part { rmsnorm, silu_mul, sigmoid_mul };

const char* part_op(Part p) {
    switch (p) {
        case Part::rmsnorm: return "rmsnorm";
        case Part::silu_mul: return "silu_mul";
        case Part::sigmoid_mul: return "act_mul";
    }
    return "";
}

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return shape.empty() ? 0 : n;
}

// A shape as `[a,b]`, for the refusals that have to say what the tape carried.
std::string shape_text(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t d = 0; d < shape.size(); ++d) s += (d == 0 ? "" : ",") + std::to_string(shape[d]);
    return s + "]";
}

// The fused norm's residual, by role: operand 1, written and in place, of operand 0's logical
// shape (the device slice's test, src/adjoint/device/quant.hip:704-716); -1 when absent. On the
// live tape its snapshot is the post-update sum x + residual_old, which the elementwise part
// reads as its norm's input and whose incoming cotangent both summands carry.
int64_t rmsnorm_residual(const OpRecord& rec) {
    if (rec.operands.size() <= 1) return -1;
    const Operand& o = rec.operands[1];
    if (!o.written || !o.in_place) return -1;
    if (o.shape != rec.operands[0].shape) return -1;
    return 1;
}

// The quantiser's own record over the part's output: the fused producer's intermediate is the
// quantiser's activation, so the part is served by the family's existing twin and adjoint under
// had_quant_act_i8g. Under `<op>_ste` the quantiser's own linearised row is used instead, and its
// schema is the activation, the base point it was linearised at, the codes, the scale
// (`quant_twin` reads the base point as the record's second operand, :296).
OpRecord had_quant_record(int64_t rows, int64_t n, int64_t group, bool linearised) {
    OpRecord q;
    q.op = linearised ? "had_quant_act_i8g_ste" : "had_quant_act_i8g";
    q.params = {{"M", rows}, {"n", n}, {"group", group}, {"dtype", std::string("bf16")}};
    q.operands = {
        Operand{"x", "bf16", {rows, n}, 0, rows, false},
    };
    if (linearised)
        q.operands.push_back(Operand{"base", "bf16", {rows, n}, 0, rows, false});
    q.operands.push_back(Operand{"q", "i8", {rows, n}, 0, rows, true});
    q.operands.push_back(Operand{"scale", "f32", {rows, n / group}, 0, rows, true});
    return q;
}

// The cotangent slot of a written operand: `cot` holds one entry per write, in the record's write
// order, so this is the write's index among the writes -- the device row's rule
// (`src/adjoint/device/quant.hip:531`). Reading `cot[0]` positionally is wrong on the live fused
// rmsnorm record, which writes the normed value in place *before* the codes: cot[0] there is a slot
// no consumer fills, and the quantiser's part then contributes nothing.
int64_t cot_slot(const OpRecord& rec, int64_t operand) {
    int64_t at = 0;
    for (int64_t j = 0; j < (int64_t)rec.operands.size(); ++j) {
        if (!rec.operands[j].written) continue;
        if (j == operand) return at;
        ++at;
    }
    return -1;
}

// The codes and the scale among the record's writes, found by dtype and shape as the device row
// finds them (`src/adjoint/device/quant.hip:492-523`). A position would be wrong under `_ste`,
// whose base point is appended after the writes and so moves what counting the reads returns.
void written_codes_and_scale(const AdjointCall& call, int64_t* codes_at, int64_t* scale_at) {
    *codes_at = -1;
    *scale_at = -1;
    const std::vector<Operand>& ops = call.rec.operands;
    for (int64_t j = 0; j < (int64_t)ops.size(); ++j) {
        const Operand& o = ops[(size_t)j];
        if (!o.written || o.dtype.empty() || o.shape.empty()) continue;
        if (*codes_at < 0 && (o.dtype == "i8" || o.dtype == "fp8_e4m3" || o.dtype == "fp8_e5m2") &&
            o.shape.size() == 2)
            *codes_at = j;
        else if (*scale_at < 0 && o.dtype == "f32" && o.shape.size() == 2)
            *scale_at = j;
    }
}

// The forward of each fused producer, as the parts' twins composed: the elementwise part over the
// record's read operands, then the quantiser over its output.
void fused_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                std::vector<AdValue>* out, Part part) {
    const int64_t group = required(rec, "group");
    const bool linearised = is_linearised(rec.op);

    std::vector<AdValue> y;
    TwinFn elem = twin_find(part_op(part), rec.params, nullptr);
    if (elem == nullptr)
        throw std::invalid_argument(std::string("quant: no twin for ") + part_op(part));
    elem(rec, operands, &y);
    if (y.empty() || y[0].shape.empty() || y[0].shape[0] <= 0)
        throw std::invalid_argument(std::string("quant: ") + part_op(part) +
                                    " wrote no [rows, n] intermediate for the quantiser");
    // The quantiser's width is the intermediate's value count over its rows, not the record's `n`:
    // the live gate record declares `n=256` while its codes are 6144 wide, and the elementwise
    // part hands back its own shape (`[rows, heads, dim]`) whose flattened width is the codes'
    // width. The device row reads the same fact off the codes; this is the forward's half of it.
    const int64_t rows = y[0].shape[0];
    if ((int64_t)y[0].value.size() % rows != 0)
        throw std::invalid_argument(std::string("quant: ") + part_op(part) + " wrote " +
                                    std::to_string(y[0].value.size()) + " values over " +
                                    std::to_string(rows) + " rows, which is not a whole width");
    const int64_t n = (int64_t)y[0].value.size() / rows;

    const OpRecord q = had_quant_record(rows, n, group, linearised);
    TwinFn quant = twin_find(q.op, q.params, nullptr);
    if (quant == nullptr)
        throw std::invalid_argument(std::string("quant: no twin for ") + q.op);
    // The linearised quantiser reads its base point as the second operand; the fused record names
    // it last, and `operands` arrives in read order, so it is `operands.back()`.
    std::vector<AdValue> qin = {y[0]};
    if (linearised) {
        if (operands.size() < 2 || operands.back().value.size() != y[0].value.size())
            throw std::invalid_argument(std::string("quant: ") + rec.op +
                                        ": the base point does not have the activation's shape");
        qin.push_back(operands.back());
    }
    std::vector<AdValue> qout;
    quant(q, qin, &qout);
    // The record's writes, in the schema's write order. The codes and the scale are the
    // quantiser's, found by dtype and shape as the row finds them. An in-place write is the
    // elementwise part's output over the read buffer it covers -- the buffer's bytes are the
    // int8 codes over a bf16 allocation, a reinterpretation nothing differentiates -- and a
    // further write of the part's own width is that output to a fresh buffer, which the live
    // tape's fused producers carry beside the codes. A write of neither role is the engine's
    // direct path or no-value placeholder: over a buffer the record also reads it carries the
    // read's value (the backward differentiates it at the identity), else it carries the
    // operand's shape and no forward fact.
    int64_t codes_at = -1, scale_at = -1;
    for (size_t j = 0; j < rec.operands.size(); ++j) {
        const Operand& o = rec.operands[j];
        if (!o.written || o.dtype.empty() || o.shape.empty()) continue;
        if (codes_at < 0 && (o.dtype == "i8" || o.dtype == "fp8_e4m3" || o.dtype == "fp8_e5m2") &&
            o.shape.size() == 2)
            codes_at = (int64_t)j;
        else if (scale_at < 0 && o.dtype == "f32" && o.shape.size() == 2)
            scale_at = (int64_t)j;
    }
    // The residual's slot carries the sum the tape snapshots (its own operand), not the part's
    // output: the kernel writes x + residual_old back, and the tape's snapshot is exactly that.
    const int64_t residual_at = (part == Part::rmsnorm) ? rmsnorm_residual(rec) : -1;
    size_t residual_read = 0;
    if (residual_at >= 0)
        for (size_t k = 0; k < (size_t)residual_at; ++k)
            if (!rec.operands[k].written || rec.operands[k].in_place) ++residual_read;
    for (size_t j = 0; j < rec.operands.size(); ++j) {
        const Operand& o = rec.operands[j];
        if (!o.written) continue;
        if (j == (size_t)codes_at || j == (size_t)scale_at) {
            out->push_back(std::move(qout[(size_t)(j == (size_t)codes_at ? 0 : 1)]));
            continue;
        }
        AdValue slot;
        slot.shape = o.shape;
        if (!o.dtype.empty()) slot.dtype = o.dtype;
        int64_t n_slot = 1;
        for (int64_t d : o.shape) n_slot *= d;
        if (o.in_place || n_slot == (int64_t)y[0].value.size()) {
            // The residual's post-op value is the sum the tape snapshots (its own operand), not the
            // part's output; the other in-place and part-width writes carry the part's output.
            slot.value = (j == (size_t)residual_at) ? operands[residual_read].value : y[0].value;
        }
        else {
            // The direct path: the write over a buffer the record also reads carries the read's
            // value, and the backward differentiates it at the identity.
            slot.value.assign((size_t)n_slot, 0.0);
            size_t read_index = 0;
            for (size_t k = 0; k < j; ++k) {
                const Operand& p = rec.operands[k];
                const bool in_list = !p.written || p.in_place;
                if (!p.written && p.buffer == o.buffer &&
                    (int64_t)operands[read_index].value.size() == n_slot)
                    slot.value = operands[read_index].value;
                if (in_list) ++read_index;
            }
        }
        out->push_back(std::move(slot));
    }
}

// The adjoint of each fused producer: the parts' host adjoints, in the order the kernel undoes
// them. One read operand's cotangent is the quantiser's output cotangent; the elementwise part's
// adjoint then folds it into the read operands' slots, which is where the device kernel must land
// too.
void fused_adjoint(const AdjointCall& call, Part part) {
    const bool linearised = is_linearised(call.rec.op);
    // The record's reads, and an in-place write is one: its slot's pre-op value is what the
    // walk hands the row, so the elementwise part's read set is the record's reads plus the
    // in-place write, in schema order.
    size_t reads = 0;
    for (const Operand& o : call.rec.operands)
        if (!o.written || o.in_place) ++reads;
    if (reads == 0 || call.fwd.size() < reads + 2)
        throw std::invalid_argument(
            "quant: the record must read the activation and write the codes and the scale");
    if (call.cot.size() < 2)
        throw std::invalid_argument(
            "quant: the record writes the codes and the scale, so the walk hands two cotangents");
    if (call.dc == nullptr || call.dc->size() < reads)
        throw std::invalid_argument("quant: the walk handed no room for the activation's cotangent");
    // The codes and the scale by dtype and shape, never by position -- under `_ste` the base point
    // is appended after the codes and the scale, so the reads and the writes are interleaved and
    // a counted position is wrong.
    int64_t codes_at = -1, scale_at = -1;
    written_codes_and_scale(call, &codes_at, &scale_at);
    if (codes_at < 0 || scale_at < 0)
        throw std::invalid_argument(std::string("quant: ") + call.rec.op +
                                    ": the record does not write the codes and the scale");
    const AdValue& codes = call.fwd[static_cast<size_t>(codes_at)];
    const AdValue& scale = call.fwd[static_cast<size_t>(scale_at)];

    // The intermediate the elementwise part wrote, which the quantiser's adjoint differentiates.
    std::vector<AdValue> y;
    TwinFn elem = twin_find(part_op(part), call.rec.params, nullptr);
    if (elem == nullptr)
        throw std::invalid_argument(std::string("quant: no twin for ") + part_op(part));
    elem(call.rec, call.fwd, &y);

    // The widths are the operands', as the device row reads them: the live gate record declares
    // `n=256` while its codes are 6144 wide over a 48-group scale, and its elementwise
    // intermediate carries its own shape (`[rows, heads, dim]`) with the codes' value count. A
    // rank test on that intermediate refused a record the device row serves; the value count is
    // the fact both sides need.
    if (codes.shape.size() != 2 || codes.shape[0] <= 0 || codes.shape[1] <= 0)
        throw std::invalid_argument(std::string("quant: ") + call.rec.op + ": the codes are " +
                                    shape_text(codes.shape) + ", not [rows, n]");
    if (scale.shape.size() != 2 || scale.shape[0] != codes.shape[0] || scale.shape[1] <= 0 ||
        codes.shape[1] % scale.shape[1] != 0)
        throw std::invalid_argument(std::string("quant: ") + call.rec.op + ": the codes are " +
                                    shape_text(codes.shape) + " and the scale " +
                                    shape_text(scale.shape) +
                                    ", which is not a group scale over the codes");
    const int64_t rows = codes.shape[0];
    const int64_t n = codes.shape[1];
    const int64_t group = n / scale.shape[1];
    const int64_t total = rows * n;
    if (y.empty() || (int64_t)y[0].value.size() != total) {
        std::string detail = " intermediate=" + (y.empty() ? std::string("none")
                                                           : shape_text(y[0].shape) + " with " +
                                                                 std::to_string(y[0].value.size()) +
                                                                 " values") +
                             " codes=" + shape_text(codes.shape) + " (" + std::to_string(total) +
                             " values)";
        for (size_t j = 0; j < call.rec.operands.size(); ++j) {
            const Operand& o = call.rec.operands[j];
            detail += " [" + std::to_string(j) + "] '" + o.buffer + "' " + o.dtype + " " +
                      shape_text(o.shape) + (o.written ? " written" : "");
        }
        throw std::invalid_argument(std::string("quant: ") + call.rec.op + ": the " +
                                    part_op(part) +
                                    " intermediate does not carry the codes' values" + detail);
    }

    const OpRecord q = had_quant_record(rows, n, group, linearised);
    std::vector<AdValue> qfwd = {y[0]};
    // The base point, when the row is linearised: the record's last read, the operand the check
    // appends for the STE. Found by that role, not by `fwd.back()` -- the output's write is
    // appended after the base in the producer's record, and `back()` would take it.
    int64_t base_at = -1;
    if (linearised) {
        for (int64_t k = (int64_t)call.rec.operands.size() - 1; k >= 0; --k)
            if (!call.rec.operands[(size_t)k].written) {
                base_at = k;
                break;
            }
        if (base_at < 0 || (size_t)base_at >= call.fwd.size() ||
            call.fwd[(size_t)base_at].value.size() != (size_t)y[0].value.size())
            throw std::invalid_argument(std::string("quant: ") + call.rec.op +
                                        ": the base point does not have the activation's shape");
        qfwd.push_back(call.fwd[(size_t)base_at]);
    }
    qfwd.push_back(codes);
    qfwd.push_back(scale);
    // The cotangents by the writes' schema order: an in-place write is one of the writes, and it
    // precedes the codes, so counting only the codes and the scale would hand the write's
    // cotangent to the codes.
    const int64_t codes_cot_at = cot_slot(call.rec, codes_at);
    const int64_t scale_cot_at = cot_slot(call.rec, scale_at);
    if (codes_cot_at < 0 || (size_t)codes_cot_at >= call.cot.size() || scale_cot_at < 0 ||
        (size_t)scale_cot_at >= call.cot.size())
        throw std::invalid_argument(std::string("quant: ") + call.rec.op + ": the codes' write " +
                                    std::to_string(codes_at) + " and the scale's " +
                                    std::to_string(scale_at) + " need cotangent slots " +
                                    std::to_string(codes_cot_at) + " and " +
                                    std::to_string(scale_cot_at) + " of the " +
                                    std::to_string(call.cot.size()) + " the walk handed");
    std::vector<AdValue> qcot = {call.cot[(size_t)codes_cot_at], call.cot[(size_t)scale_cot_at]};
    std::vector<AdValue> qdc;
    for (const Operand& o : q.operands) {
        AdValue v;
        v.dtype = o.dtype;
        v.shape = o.shape;
        v.value.assign(static_cast<size_t>(numel(o.shape)), 0.0);
        qdc.push_back(std::move(v));
    }
    AdjointCall qcall{q, qfwd, qcot, &qdc, nullptr, ""};
    AdjointFn quant = adjoint_find(q.op, Side::host, q.params, nullptr);
    if (quant == nullptr)
        throw std::invalid_argument("quant: no host adjoint for had_quant_act_i8g");
    quant(qcall);

    // R97_DUMP_QUANT=1: the twin's intermediate against the codes the tape's own kernel wrote.
    // Radiance's kernel table has no host entry for the int8 fused producers (`rmsnorm_had_quant_i8g`
    // and its two siblings), so `check_kernel` cannot check their twins and their FD differentiates a
    // row against the twin it is testing. The tape is the oracle instead: re-quantising the twin's
    // intermediate must reproduce the codes and the scale the kernel actually wrote, and `code_err`
    // is the largest disagreement. A non-zero one means the row's mask and its dx are taken at an
    // activation the kernel never had.
    if (const char* dq = std::getenv("R97_DUMP_QUANT"); dq != nullptr && dq[0] == '1') {
        std::vector<AdValue> twin_out;
        std::vector<AdValue> qin = {y[0]};
        if (linearised && base_at >= 0 && (size_t)base_at < call.fwd.size())
            qin.push_back(call.fwd[(size_t)base_at]);
        const TwinFn qt = twin_find(q.op, q.params, nullptr);
        double code_err = 0.0, scale_rel = 0.0;
        int64_t code_bad = 0;
        int64_t hist[10] = {0};
        if (qt != nullptr) {
            qt(q, qin, &twin_out);
            // `code_bad` counts any nonzero difference, so it cannot tell a uniform sub-LSB
            // mismatch (which flips nearly every code) from a structural one. The histogram can: a
            // tight 1-2 with a few outliers is a small systematic difference in the gain or the
            // precision, a broad spread is what a layout difference would need.
            if (!twin_out.empty() && twin_out[0].value.size() == codes.value.size())
                for (size_t i = 0; i < codes.value.size(); ++i) {
                    const double d = std::fabs(twin_out[0].value[i] - codes.value[i]);
                    if (d > 0.0) ++code_bad;
                    if (d > code_err) code_err = d;
                    ++hist[d == 0.0 ? 0
                                    : d <= 1.0 ? 1
                                    : d <= 2.0 ? 2
                                    : d <= 4.0 ? 3
                                    : d <= 8.0 ? 4
                                    : d <= 16.0 ? 5
                                    : d <= 32.0 ? 6
                                    : d <= 64.0 ? 7
                                    : d <= 128.0 ? 8 : 9];
                }
            // A *second* Hadamard on top of the one the twin already applies: `geometry()` sets
            // `rotated = is_rotated(op)` and `Op::had` is rotated, so the plain comparison above is
            // the single-rotation arm. This arm is therefore a control for "twice rotated", not an
            // "unrotated" arm -- the gated producers collapse here (99%) and match above (5.9%),
            // which is what makes it useful.
            double rot_err = 0.0;
            int64_t rot_bad = 0;
            {
                std::vector<AdValue> rot = {y[0]};
                const int64_t rn = n;
                const int64_t rgrp = group;
                for (int64_t r = 0; r < rows; ++r)
                    for (int64_t gi = 0; gi < rn / rgrp; ++gi) {
                        const int64_t k0 = gi * rgrp;
                        const int64_t w = std::min(rgrp, rn - k0);
                        if (w == rgrp) fwht(rot[0].value.data() + r * rn + k0, rgrp);
                    }
                std::vector<AdValue> rot_out;
                qt(q, rot, &rot_out);
                if (!rot_out.empty() && rot_out[0].value.size() == codes.value.size())
                    for (size_t i = 0; i < codes.value.size(); ++i) {
                        const double d = std::fabs(rot_out[0].value[i] - codes.value[i]);
                        if (d > 0.0) ++rot_bad;
                        if (d > rot_err) rot_err = d;
                    }
            }
            // The kernel's own activation, reconstructed from its codes and scale: `codes * scale` is
            // the rotated activation and `fwht` is unnormalised (H*H = n*I), so the inverse is
            // `fwht(.)/group`. Comparing this to `y[0]` tests the twin's *activation* without going
            // through the twin's quantiser at all, which is what separates "the norm is wrong" from
            // "the quantiser's scale is wrong".
            double deq_err = 0.0, deq_scale = 0.0;
            {
                const int64_t ngrp = n / group;
                std::vector<double> dq(codes.value.size());
                for (size_t i = 0; i < dq.size(); ++i) dq[i] = (double)codes.value[i];
                for (int64_t r = 0; r < rows; ++r)
                    for (int64_t gi = 0; gi < ngrp; ++gi) {
                        const int64_t k0 = gi * group;
                        const double sc = scale.value[(size_t)(r * ngrp + gi)];
                        for (int64_t i = 0; i < group; ++i)
                            dq[(size_t)(r * n + k0 + i)] *= sc;
                        fwht(dq.data() + r * n + k0, group);
                        for (int64_t i = 0; i < group; ++i)
                            dq[(size_t)(r * n + k0 + i)] /= (double)group;
                    }
                for (size_t i = 0; i < dq.size() && i < y[0].value.size(); ++i) {
                    const double d = std::fabs(dq[i] - y[0].value[i]);
                    if (d > deq_err) deq_err = d;
                    if (std::fabs(dq[i]) > deq_scale) deq_scale = std::fabs(dq[i]);
                }
                // The kernel's activation against the tensors around it: its own input (`op[0]`) and
                // the in-place write. Whichever it matches names what the kernel actually computed,
                // and whichever it does not names the snapshot the twin should have been given.
                double vs_in = 0.0, vs_wr = 0.0, in_vs_wr = 0.0, in_scale = 0.0;
                for (size_t i = 0; i < dq.size() && i < call.fwd[0].value.size(); ++i) {
                    const double a = std::fabs(dq[i] - call.fwd[0].value[i]);
                    if (a > vs_in) vs_in = a;
                    if (std::fabs(call.fwd[0].value[i]) > in_scale)
                        in_scale = std::fabs(call.fwd[0].value[i]);
                    if (i < call.fwd.back().value.size()) {
                        const double b = std::fabs(dq[i] - call.fwd.back().value[i]);
                        if (b > vs_wr) vs_wr = b;
                        const double c = std::fabs(call.fwd[0].value[i] - call.fwd.back().value[i]);
                        if (c > in_vs_wr) in_vs_wr = c;
                    }
                }
                // The twin's activation against the kernel's by magnitude: if the twin's is a scalar
                // multiple of the kernel's, `ymax/deqmax` is that multiple and the difference is one
                // factor (a scale or a gain), not a different tensor.
                double ymax = 0.0, dqmax = 0.0;
                for (size_t i = 0; i < dq.size() && i < y[0].value.size(); ++i) {
                    if (std::fabs(y[0].value[i]) > ymax) ymax = std::fabs(y[0].value[i]);
                    if (std::fabs(dq[i]) > dqmax) dqmax = std::fabs(dq[i]);
                }
                // Per-row magnitude ratio. A `s` that differs per row shows as a spread here; a
                // single elementwise difference (a gain vector) leaves every row's ratio the same
                // only when the max sits in the same column, so also report the ratio of the row
                // sums, which a per-row scale moves exactly.
                double rmin = 1e300, rmax = -1e300, sum_y = 0.0, sum_d = 0.0;
                for (int64_t r = 0; r < rows; ++r) {
                    double ry = 0.0, rd = 0.0;
                    for (int64_t i = 0; i < n; ++i) {
                        ry += std::fabs(y[0].value[(size_t)(r * n + i)]);
                        rd += std::fabs(dq[(size_t)(r * n + i)]);
                    }
                    sum_y += ry;
                    sum_d += rd;
                    if (rd > 0.0) {
                        const double q = ry / rd;
                        rmin = std::min(rmin, q);
                        rmax = std::max(rmax, q);
                    }
                }
                // Two tests that `row_ratio` cannot make, because it is an L1 ratio: an exact
                // permutation preserves it, a Hadamard preserves it approximately, and a different
                // gain with similar L1 also fits. (1) Sorting each row of `y` and of `deq` separates
                // a permutation (sorted rows equal) from everything else. (2) `deq` under a
                // gain-after-rotation kernel is `R^-1(g * R(n))` where `n = x*s` is the *ungained*
                // normalised activation, so building `y2 = R^-1(g * R(n))` from the twin's own `y`
                // and `x` and comparing it to `deq` tests that kernel shape directly.
                double sort_rel = 0.0, y2_rel = 0.0, y2_scale = 0.0;
                {
                    std::vector<double> a(n), b(n);
                    for (int64_t r = 0; r < rows; ++r) {
                        for (int64_t i = 0; i < n; ++i) {
                            a[(size_t)i] = y[0].value[(size_t)(r * n + i)];
                            b[(size_t)i] = dq[(size_t)(r * n + i)];
                        }
                        std::sort(a.begin(), a.end());
                        std::sort(b.begin(), b.end());
                        for (int64_t i = 0; i < n; ++i) {
                            const double d = std::fabs(a[(size_t)i] - b[(size_t)i]);
                            if (d > sort_rel) sort_rel = d;
                        }
                    }
                    // n = x * s, g = y / n, y2 = R^-1(g * R(n)).
                    std::vector<double> nn(n), gg(n), tmp(n);
                    for (int64_t r = 0; r < rows; ++r) {
                        double ss = 0.0;
                        for (int64_t i = 0; i < n; ++i) {
                            const double v = call.fwd[0].value[(size_t)(r * n + i)];
                            ss += v * v;
                        }
                        const double sr = 1.0 / std::sqrt(ss / (double)n + 1e-6);
                        for (int64_t i = 0; i < n; ++i) {
                            const double xv = call.fwd[0].value[(size_t)(r * n + i)];
                            nn[(size_t)i] = xv * sr;
                            tmp[(size_t)i] = nn[(size_t)i];
                        }
                        for (int64_t gi = 0; gi < n / group; ++gi)
                            fwht(tmp.data() + gi * group, group);
                        for (int64_t i = 0; i < n; ++i) {
                            const double nv = nn[(size_t)i];
                            gg[(size_t)i] = nv != 0.0 ? y[0].value[(size_t)(r * n + i)] / nv : 0.0;
                        }
                        for (int64_t gi = 0; gi < n / group; ++gi) {
                            const int64_t k0 = gi * group;
                            for (int64_t i = 0; i < group; ++i)
                                tmp[(size_t)(k0 + i)] *= gg[(size_t)(k0 + i)];
                            fwht(tmp.data() + k0, group);
                            for (int64_t i = 0; i < group; ++i)
                                tmp[(size_t)(k0 + i)] /= (double)group;
                        }
                        for (int64_t i = 0; i < n; ++i) {
                            const double d = std::fabs(tmp[(size_t)i] - dq[(size_t)(r * n + i)]);
                            if (d > y2_rel) y2_rel = d;
                            if (std::fabs(dq[(size_t)(r * n + i)]) > y2_scale)
                                y2_scale = std::fabs(dq[(size_t)(r * n + i)]);
                        }
                    }
                }
                std::fprintf(stderr,
                             "DEQORACLE pos=%lld deq_vs_in=%.6g deq_vs_write=%.6g in_vs_write=%.6g "
                             "in_max=%.6g ymax=%.6g deqmax=%.6g y_over_deq=%.6g row_ratio=[%.6g,%.6g] "
                             "sum_ratio=%.6g sort_rel=%.6g y2_rel=%.6g\n",
                             (long long)call.rec.position,
                             in_scale > 0.0 ? vs_in / in_scale : vs_in,
                             in_scale > 0.0 ? vs_wr / in_scale : vs_wr,
                             in_scale > 0.0 ? in_vs_wr / in_scale : in_vs_wr, in_scale, ymax, dqmax,
                             dqmax > 0.0 ? ymax / dqmax : 0.0, rmin, rmax,
                             sum_d > 0.0 ? sum_y / sum_d : 0.0,
                             y2_scale > 0.0 ? sort_rel / y2_scale : sort_rel,
                             y2_scale > 0.0 ? y2_rel / y2_scale : y2_rel);
                // R97_GAIN_POS=1: the elements, not the maxima and not the first four. `deq_rel`
                // is a maximum over all 2621440 elements, so indices 0..3 can be the well-behaved
                // ones and a consistent reading there would be uninformative; the elements worth
                // printing are the four with the largest |y - dq|, which is where the disagreement
                // lives. `deq_vs_write` establishes that `dq` IS the kernel's own activation, so a
                // `dq/n` that is near-constant across elements means the scale term diverges, a
                // scattered one means the gain or the vector does.
                if (std::getenv("R97_GAIN_POS") != nullptr) {
                    struct Hit {
                        double d = -1.0;
                        int64_t r = -1, i = -1;
                    } top[4];
                    double dqmax_seen = 0.0;
                    // The per-row scale, once: the row-alignment search below needs it for every
                    // candidate row.
                    std::vector<double> srow((size_t)rows, 0.0);
                    for (int64_t r = 0; r < rows; ++r) {
                        double ss = 0.0;
                        for (int64_t j = 0; j < n; ++j) {
                            const double v = call.fwd[0].value[(size_t)(r * n + j)];
                            ss += v * v;
                        }
                        srow[(size_t)r] = 1.0 / std::sqrt(ss / (double)n + 1e-6);
                    }
                    for (int64_t r = 0; r < rows; ++r) {
                        for (int64_t i = 0; i < n; ++i) {
                            const double a = y[0].value[(size_t)(r * n + i)];
                            const double b = dq[(size_t)(r * n + i)];
                            const double d = std::fabs(a - b);
                            if (std::fabs(b) > dqmax_seen) dqmax_seen = std::fabs(b);
                            for (int k = 0; k < 4; ++k) {
                                if (d <= top[k].d) continue;
                                for (int m = 3; m > k; --m) top[m] = top[m - 1];
                                top[k].d = d;
                                top[k].r = r;
                                top[k].i = i;
                                break;
                            }
                        }
                    }
                    for (int k = 0; k < 4; ++k) {
                        if (top[k].i < 0) continue;
                        const int64_t r = top[k].r, i = top[k].i;
                        const double s = srow[(size_t)r];
                        const double xv = call.fwd[0].value[(size_t)(r * n + i)];
                        const double gv =
                            call.fwd.size() > 2 && (int64_t)call.fwd[2].value.size() > i
                                ? call.fwd[2].value[(size_t)i]
                                : 0.0;
                        const double nv = xv * s;
                        const double dv = dq[(size_t)(r * n + i)];
                        const double yv = y[0].value[(size_t)(r * n + i)];
                        // The one hypothesis a per-element gain cannot produce: if the kernel's
                        // value at COLUMN i is x_{r',i} * s_{r'} * (g_i+1) for a DIFFERENT row r',
                        // the snapshot handed to this row is row-misaligned, which is a defect of
                        // an in-place read served from a boundary, not of the norm's arithmetic.
                        int64_t best_r = -1;
                        double best_d = 0.0;
                        for (int64_t rx = 0; rx < rows; ++rx) {
                            const double cand =
                                call.fwd[0].value[(size_t)(rx * n + i)] * srow[(size_t)rx] *
                                (gv + 1.0);
                            const double d = std::fabs(cand - dv);
                            if (best_r < 0 || d < best_d) {
                                best_d = d;
                                best_r = rx;
                            }
                        }
                        std::fprintf(stderr,
                                     "DEQELEM pos=%lld rank=%d r=%lld i=%lld dq=%.6g y=%.6g "
                                     "x=%.6g n=%.6g g=%.6g dq_over_n=%.6g g_plus_1=%.6g "
                                     "y_over_dq=%.6g bestrow=%lld bestval=%.6g\n",
                                     (long long)call.rec.position, k, (long long)r, (long long)i, dv,
                                     yv, xv, nv, gv, nv != 0.0 ? dv / nv : 0.0, gv + 1.0,
                                     dv != 0.0 ? yv / dv : 0.0, (long long)best_r,
                                     best_r >= 0
                                         ? call.fwd[0].value[(size_t)(best_r * n + i)] *
                                               srow[(size_t)best_r] * (gv + 1.0)
                                         : 0.0);
                    }
                    std::fprintf(stderr,
                                 "DEQELEM pos=%lld op=%s worst=%.6g deqmax=%.6g deq_err=%.6g "
                                 "deq_rel=%.6g\n",
                                 (long long)call.rec.position, call.rec.op.c_str(), top[0].d,
                                 dqmax_seen, deq_err, deq_scale > 0.0 ? deq_err / deq_scale : deq_err);
                    // The one thing the plain `rmsnorm` (which NORMELEM shows matches to 0.28%)
                    // does not share with a fused norm: which operand is the elementwise input.
                    // The plain norm's input is operand 0 and the row takes `call.fwd[0]`; the
                    // fused record reads operand 0 (`h`) and operand 1 (`x`, in place), and the
                    // elementwise part may take the other one. Rebuild `y` from operand 1 and
                    // compare over the WHOLE tensor, so the answer is a scalar and not a
                    // max-biased sample. A match here names the input operand; no match leaves
                    // the fused stage itself.
                    if (call.fwd.size() > 1 &&
                        call.fwd[1].value.size() == call.fwd[0].value.size()) {
                        double w1 = 0.0, sc1 = 0.0, fwd1_vs_dq = 0.0;
                        for (int64_t r = 0; r < rows; ++r) {
                            double ss = 0.0;
                            for (int64_t j = 0; j < n; ++j) {
                                const double v = call.fwd[1].value[(size_t)(r * n + j)];
                                ss += v * v;
                            }
                            const double s1 = 1.0 / std::sqrt(ss / (double)n + 1e-6);
                            for (int64_t i = 0; i < n; ++i) {
                                const double gv =
                                    call.fwd.size() > 2 &&
                                            (int64_t)call.fwd[2].value.size() > i
                                        ? call.fwd[2].value[(size_t)i]
                                        : 0.0;
                                const double y1 = call.fwd[1].value[(size_t)(r * n + i)] * s1 *
                                                  (gv + 1.0);
                                const double d =
                                    std::fabs(y1 - dq[(size_t)(r * n + i)]);
                                if (d > w1) w1 = d;
                                const double dq1 =
                                    std::fabs(call.fwd[1].value[(size_t)(r * n + i)] -
                                              dq[(size_t)(r * n + i)]);
                                if (dq1 > fwd1_vs_dq) fwd1_vs_dq = dq1;
                                if (std::fabs(dq[(size_t)(r * n + i)]) > sc1)
                                    sc1 = std::fabs(dq[(size_t)(r * n + i)]);
                            }
                        }
                        std::fprintf(stderr,
                                     "DEQELEM pos=%lld op=%s operand1_worst=%.6g operand1_rel=%.6g "
                                     "(operand0_rel=%.6g) fwd1_vs_dq_rel=%.6g\n",
                                     (long long)call.rec.position, call.rec.op.c_str(), w1,
                                     sc1 > 0.0 ? w1 / sc1 : w1,
                                     deq_scale > 0.0 ? deq_err / deq_scale : deq_err,
                                     fwd1_vs_dq > 0.0 ? fwd1_vs_dq / (sc1 > 0.0 ? sc1 : 1.0) : 0.0);
                    }
                    std::fflush(stderr);
                }
            }
            std::fprintf(stderr, "DUMPROT pos=%lld op=%s rot_err=%.6g rot_bad=%lld/%zu deq_rel=%.6g\n",
                         (long long)call.rec.position, call.rec.op.c_str(), rot_err,
                         (long long)rot_bad, codes.value.size(),
                         deq_scale > 0.0 ? deq_err / deq_scale : deq_err);
            if (twin_out.size() > 1 && twin_out[1].value.size() == scale.value.size())
                for (size_t i = 0; i < scale.value.size(); ++i) {
                    const double t = std::fabs(twin_out[1].value[i] - scale.value[i]);
                    const double r = std::fabs(scale.value[i]) > 0.0
                                         ? t / std::fabs(scale.value[i])
                                         : t;
                    if (r > scale_rel) scale_rel = r;
                }
        }
        // The sharper comparison: the activation the kernel wrote. A fused producer writes the
        // part's output in place (the norm's `h`, the gate's product), and that IS the intermediate
        // the twin recomputes, so the two must agree up to bf16 rounding.
        // The activation the kernel wrote, at *every* written operand of the same size: a record
        // carries the residual and the branch at the same shape, so reporting the largest error
        // would name the wrong tensor. The best match is the part's own output, and each candidate
        // is printed so the pairing is visible rather than assumed.
        double act_err = 0.0, act_scale = 0.0;
        int64_t act_at = -1;
        std::string act_all;
        for (size_t i = 0; i < call.fwd.size(); ++i) {
            const AdValue& f = call.fwd[i];
            if (i >= call.rec.operands.size() || !call.rec.operands[i].written) continue;
            if (f.value.size() != y[0].value.size()) continue;
            double e = 0.0, sc = 0.0;
            for (size_t j = 0; j < f.value.size(); ++j) {
                const double d = std::fabs((double)f.value[j] - y[0].value[j]);
                if (d > e) e = d;
                if (std::fabs((double)f.value[j]) > sc) sc = std::fabs((double)f.value[j]);
            }
            char one[96];
            std::snprintf(one, sizeof(one), "%zu:%s:%.3g ", i,
                          call.rec.operands[i].buffer.c_str(), sc > 0.0 ? e / sc : e);
            act_all += one;
            if (act_at < 0 || e < act_err) { act_err = e; act_scale = sc; act_at = (int64_t)i; }
        }
        std::fprintf(stderr,
                     "DUMPTWIN pos=%lld op=%s twin=%d codes=%zu code_err=%.6g code_bad=%lld/%zu "
                     "scale_rel=%.6g act_at=%lld act_err=%.6g act_rel=%.6g "
                     "hist=[%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld] best_of=[%s]\n",
                     (long long)call.rec.position, call.rec.op.c_str(), qt != nullptr ? 1 : 0,
                     codes.value.size(), code_err, (long long)code_bad, codes.value.size(),
                     scale_rel, (long long)act_at, act_err,
                     act_scale > 0.0 ? act_err / act_scale : act_err,
                     (long long)hist[0], (long long)hist[1], (long long)hist[2],
                     (long long)hist[3], (long long)hist[4], (long long)hist[5],
                     (long long)hist[6], (long long)hist[7], (long long)hist[8],
                     (long long)hist[9], act_all.c_str());
        std::fflush(stderr);
    }

    // The elementwise adjoint takes the parts' operands and the intermediate they wrote; it adds
    // its cotangents into the slots the parts' read operands hold. The two-way rule lives in that
    // row, so the read set the caller proved travels with it, as the device row hands its own.
    // The read operands are NOT a prefix of the record's: the live fused norm reads 0 and 2 and
    // writes 1, 3, 4 and 5, so taking the first `reads` handed the elementwise row operand 1 -- a
    // written activation -- in place of the weight, and put the intermediate in the weight's slot.
    // `norm_gain` then took the intermediate as the gain: `[512, 5120]` where the operand is
    // `[5120]`, row 0 of the norm's own output, and the row differentiated `dc[h]` with it.
    //
    // The part is called with its own operand list -- the record's read operands and in-place
    // writes, in schema order, then the intermediate it wrote -- and the slots it fills are
    // mapped back onto the fused record's.
    std::vector<size_t> slot;
    std::vector<AdValue> efwd;
    OpRecord erec = call.rec;
    erec.operands.clear();
    for (size_t j = 0; j < call.rec.operands.size(); ++j) {
        if (call.rec.operands[j].written && !call.rec.operands[j].in_place) continue;
        slot.push_back(j);
        efwd.push_back(call.fwd[j]);
        erec.operands.push_back(call.rec.operands[j]);
    }
    Operand yop;
    yop.buffer = part_op(part);
    yop.shape = y[0].shape;
    yop.written = true;
    slot.push_back(call.rec.operands.size());
    efwd.push_back(y[0]);
    erec.operands.push_back(yop);
    std::vector<AdValue> edc(slot.size());
    for (size_t i = 0; i < edc.size(); ++i) {
        edc[i].dtype = efwd[i].dtype;
        edc[i].shape = efwd[i].shape;
        edc[i].value.assign(static_cast<size_t>(numel(efwd[i].shape)), 0.0);
    }
    // The part's output's own cotangent, summed in before the part's adjoint runs -- the host
    // half of the device row's post-pass over the record's writes
    // (src/adjoint/device/quant.hip:817-880), which this mirrors write for write. Every write
    // that is not the codes or the scale is the part's output: one of the part's own width joins
    // the quantiser's output cotangent the part differentiates, and a write over a buffer the
    // record also reads is the direct path, added to that read's slot. Both are `+=`, never an
    // assignment -- a slot the codes' path already wrote keeps its term. The slot's post-op
    // value the twin writes is the part's output, so its cotangent reaches the part's input
    // through the part, exactly as the quantiser's does -- a passthrough would differentiate
    // the slot at the identity, and the slot holds `part(x)`, not `x`.
    // The residual's incoming cotangent (the new residual value's consumers) is the part record's
    // write #0, before the intermediate. The elementwise part adds it to both summands' slots with
    // the norm's input cotangent (the `add` rule), so it is handed as its own cotangent -- not
    // folded into the intermediate's, which would run the external consumer's cotangent through
    // the norm's Jacobian, where it does not belong.
    const int64_t residual_at = (part == Part::rmsnorm) ? rmsnorm_residual(call.rec) : -1;
    std::vector<AdValue> ecot;
    if (residual_at >= 0) {
        const int64_t res_cot = cot_slot(call.rec, residual_at);
        if (res_cot < 0 || (size_t)res_cot >= call.cot.size())
            throw std::invalid_argument(std::string("quant: ") + call.rec.op +
                                        ": the residual write '" +
                                        call.rec.operands[(size_t)residual_at].buffer +
                                        "' has no cotangent slot");
        ecot.push_back(call.cot[(size_t)res_cot]);
    }
    // The intermediate's cotangent: the quantiser's, joined below by the record's other writes of
    // the part's width (the `out_bf16` junction).
    ecot.push_back(qdc[0]);
    // The direct path's width, the elementwise input: both halves for the gated producer, one
    // row of it for the rest -- the device row's `out_slot` (src/adjoint/device/quant.hip:700).
    const int64_t out_slot = part == Part::silu_mul ? 2 * total : total;
    for (int64_t j = 0; j < (int64_t)call.rec.operands.size(); ++j) {
        const Operand& o = call.rec.operands[(size_t)j];
        if (!o.written || j == codes_at || j == scale_at || j == residual_at) continue;
        const int64_t at = cot_slot(call.rec, j);
        if (at < 0 || (size_t)at >= call.cot.size())
            throw std::invalid_argument(std::string("quant: ") + call.rec.op +
                                        ": the write '" + o.buffer + "' has no cotangent slot");
        const AdValue& oc = call.cot[(size_t)at];
        const int64_t out_values = (int64_t)oc.value.size();
        if (out_values == 0 || out_values == 1)
            continue;  // no consumers, or the engine's no-value placeholder: nothing to add
        if (out_values == total) {
            for (size_t e = 0; e < oc.value.size(); ++e) ecot.back().value[e] += oc.value[e];
            continue;
        }
        int64_t read_at = -1;
        for (int64_t k = 0; k < (int64_t)call.rec.operands.size() && read_at < 0; ++k)
            if (!call.rec.operands[(size_t)k].written &&
                call.rec.operands[(size_t)k].buffer == o.buffer)
                read_at = k;
        if (read_at < 0 || out_values != out_slot)
            throw std::invalid_argument(
                std::string("quant: ") + call.rec.op + ": the write '" + o.buffer + "' carries " +
                std::to_string(out_values) + " values, not the part's " + std::to_string(total) +
                " or the direct path's " + std::to_string(out_slot));
        AdValue& into = call.dc->at((size_t)read_at);
        if ((int64_t)into.value.size() != out_values)
            throw std::invalid_argument(
                std::string("quant: ") + call.rec.op + ": the direct path's slot for '" +
                call.rec.operands[(size_t)read_at].buffer +
                "' is not the size of its cotangent");
        for (size_t e = 0; e < oc.value.size(); ++e) into.value[e] += oc.value[e];
    }
    AdjointCall ecall{erec, efwd, ecot, &edc, nullptr, ""};
    ecall.read_rows = call.read_rows;
    ecall.row_count = call.row_count;
    AdjointFn elem_adj = adjoint_find(part_op(part), Side::host, erec.params, nullptr);
    if (elem_adj == nullptr)
        throw std::invalid_argument(std::string("quant: no host adjoint for ") + part_op(part));
    elem_adj(ecall);
    // Fold the part's slots back onto the fused record's. The intermediate's own slot has no fused
    // counterpart: it is not on the tape, and its cotangent belongs to the quantiser's output.
    for (size_t i = 0; i < slot.size(); ++i) {
        if (slot[i] >= call.dc->size()) continue;
        AdValue& dst = call.dc->at(slot[i]);
        if (dst.value.size() != edc[i].value.size()) continue;
        for (size_t k = 0; k < dst.value.size(); ++k) dst.value[k] += edc[i].value[k];
    }

    // R97_DUMP_QUANT=1: the same stage sums the device row prints, so a host gradient that came
    // back zero names its stage instead of the whole row. `quant_dx` is the cotangent the quantiser
    // handed the elementwise part; `dx` is what the walk's slot holds after both parts ran.
    if (const char* dq = std::getenv("R97_DUMP_QUANT"); dq != nullptr && dq[0] == '1') {
        const auto sum = [](const AdValue& v) {
            double s = 0.0;
            for (double x : v.value) s += std::fabs(x);
            return s;
        };
        std::fprintf(stderr,
                     "DUMPQUANT side=host op=%s pos=%lld y=%.6g codes_cot=%.6g scale_cot=%.6g "
                     "quant_dx=%.6g dx=%.6g\n",
                     call.rec.op.c_str(), (long long)call.rec.position, sum(y[0]),
                     sum(call.cot[(size_t)codes_cot_at]), sum(call.cot[(size_t)scale_cot_at]),
                     sum(qdc[0]),
                     call.dc->empty() ? 0.0 : sum(call.dc->at(0)));
    }
}

void rmsnorm_had_quant_i8g_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                                std::vector<AdValue>* out) {
    fused_twin(rec, operands, out, Part::rmsnorm);
}
void gated_had_quant_i8g_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                              std::vector<AdValue>* out) {
    fused_twin(rec, operands, out, Part::silu_mul);
}
void gate_had_quant_i8g_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                             std::vector<AdValue>* out) {
    fused_twin(rec, operands, out, Part::sigmoid_mul);
}

void rmsnorm_had_quant_i8g_adjoint(const AdjointCall& call) { fused_adjoint(call, Part::rmsnorm); }
void gated_had_quant_i8g_adjoint(const AdjointCall& call) { fused_adjoint(call, Part::silu_mul); }
void gate_had_quant_i8g_adjoint(const AdjointCall& call) { fused_adjoint(call, Part::sigmoid_mul); }

}  // namespace

// The family registers one twin and one adjoint function under each op's name, and the linearised
// twin under `<op>_ste` with its base point (schema position 1) frozen, for the finite-difference
// check. A walk appends records under the two real names only.
R97_REGISTER_TWIN("quant_act_i8g_ste", quant_twin)
R97_REGISTER_ADJOINT_F("quant_act_i8g_ste", host, quant_adjoint, 1)
R97_REGISTER_TWIN("had_quant_act_i8g_ste", quant_twin)
R97_REGISTER_ADJOINT_F("had_quant_act_i8g_ste", host, quant_adjoint, 1)
R97_REGISTER_TWIN("quant_act_i8g", quant_twin)
R97_REGISTER_ADJOINT("quant_act_i8g", host, quant_adjoint)
R97_REGISTER_TWIN("had_quant_act_i8g", quant_twin)
R97_REGISTER_ADJOINT("had_quant_act_i8g", host, quant_adjoint)

// The fused int8 producers: the parts' twins and host adjoints composed, keyed by the same
// predicates the device side keys its rows with (src/adjoint/device/quant.hip). The norm's gain is
// frozen (schema position 1), so the walk asks for x's cotangent only, as the elementwise row
// does.
R97_REGISTER_TWIN_IF("rmsnorm_had_quant_i8g", rmsnorm_had_quant_i8g_twin, rotated)
R97_REGISTER_ADJOINT_IF_F("rmsnorm_had_quant_i8g", host, rmsnorm_had_quant_i8g_adjoint, rotated, 1)
R97_REGISTER_TWIN_IF("gated_had_quant_i8g", gated_had_quant_i8g_twin, serves_gated)
R97_REGISTER_ADJOINT_IF("gated_had_quant_i8g", host, gated_had_quant_i8g_adjoint, serves_gated)
R97_REGISTER_TWIN_IF("gate_had_quant_i8g", gate_had_quant_i8g_twin, serves_gate)
R97_REGISTER_ADJOINT_IF("gate_had_quant_i8g", host, gate_had_quant_i8g_adjoint, serves_gate)

// The three producers' linearised rows, for the finite-difference check. The base point is the
// activation the quantiser saw -- the part's own output -- appended after the codes and the
// scale, and the output's write after it, so one frozen index is that appended operand's schema
// position: 4 for the norm (`x, w, codes, scale, base, out`), 3 for the gated one
// (`gate_up, codes, scale, base, out`), 4 for the gate (`a, b, codes, scale, base, out`). The
// in-place fixture's record carries no separate output write -- the in-place slot is the output --
// so the base sits at the same position there, and one registration serves both layouts.
// The norm also freezes its gain at 1, as the plain row above does: the walk never asks for it, so
// the row leaves that cotangent at zero and the finite differences must not measure it either
// (`check_adjoint` skips frozen slots). The quantiser the composition reaches is then its
// own `_ste` row, which reads the base point as its second operand.
R97_REGISTER_TWIN_IF("rmsnorm_had_quant_i8g_ste", rmsnorm_had_quant_i8g_twin, rotated)
R97_REGISTER_ADJOINT_IF_F("rmsnorm_had_quant_i8g_ste", host, rmsnorm_had_quant_i8g_adjoint, rotated, 1, 4)
R97_REGISTER_TWIN_IF("gated_had_quant_i8g_ste", gated_had_quant_i8g_twin, serves_gated)
R97_REGISTER_ADJOINT_IF_F("gated_had_quant_i8g_ste", host, gated_had_quant_i8g_adjoint, serves_gated, 3)
R97_REGISTER_TWIN_IF("gate_had_quant_i8g_ste", gate_had_quant_i8g_twin, serves_gate)
R97_REGISTER_ADJOINT_IF_F("gate_had_quant_i8g_ste", host, gate_had_quant_i8g_adjoint, serves_gate, 4)
R97_REGISTER_TWIN("rmsnorm_quant_fp8", rmsnorm_quant_fp8_twin)
R97_REGISTER_ADJOINT("rmsnorm_quant_fp8", host, rmsnorm_quant_fp8_adjoint)
R97_REGISTER_TWIN("gated_quant_fp8", gated_quant_fp8_twin)
R97_REGISTER_ADJOINT("gated_quant_fp8", host, gated_quant_fp8_adjoint)
// The linearised twins append their base point to the schema -- the activation the quantiser saw --
// because positions 1 and 2 are the residual and the written codes there; the frozen index is that
// appended operand's schema position.
R97_REGISTER_TWIN("rmsnorm_quant_fp8_ste", rmsnorm_quant_fp8_twin)
R97_REGISTER_ADJOINT_F("rmsnorm_quant_fp8_ste", host, rmsnorm_quant_fp8_adjoint, 6)
R97_REGISTER_TWIN("gated_quant_fp8_ste", gated_quant_fp8_twin)
R97_REGISTER_ADJOINT_F("gated_quant_fp8_ste", host, gated_quant_fp8_adjoint, 4)

}  // namespace r97
