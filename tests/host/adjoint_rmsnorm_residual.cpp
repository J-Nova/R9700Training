// tests/host/adjoint_rmsnorm_residual.cpp -- the fused norm's residual operand (issue 55).
//
// The live `rmsnorm_had_quant_i8g` records read the branch (operand 0) and the residual stream
// (operand 1, in place) and write the sum back into the residual: the reference (libref/
// ref_quant.cpp) forms v = x + residual_old in f32, writes v into the residual, and normalises v,
// reading the sum back after narrowing to the residual's dtype. The tape snapshots operand 1 after
// the op, so its value IS that sum, and the rows now read it as the norm's input rather than
// operand 0. Before that, the rows differentiated rmsnorm(x) where the kernel computed
// rmsnorm(x + residual).
//
// This file drives the residual layout directly -- operand 1 written and in place, a non-zero
// residual -- with no host kernel (the op is device only), so the twin is the forward oracle and
// the row is checked against a finite difference of the same decomposition. Both summands carry
// the norm's full input cotangent (the `add` rule), so the x slot and the residual slot read the
// same directional derivative.

#include "support/adjoint_check.h"
#include "support/test.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

const double kEps = 1e-6;  // the container's declared eps (docs/OPS-qwen35-device.md)
const double kWadd = 1.0;  // the container's declared wadd (GemmaRMSNorm's w + 1)

double as_bf16(double v) { return (double)rad_bf16_to_f32(rad_f32_to_bf16((float)v)); }

AdValue values(const char* dtype, std::vector<int64_t> shape, std::vector<double> v) {
    AdValue a;
    a.dtype = dtype;
    a.shape = std::move(shape);
    a.value = std::move(v);
    return a;
}

// The fused norm's residual layout: the branch read, the residual stream in place, the gain, and
// the quantiser's writes plus the normed to a fresh buffer.
OpRecord residual_rec(int64_t rows, int64_t n, int64_t group) {
    const int64_t ngrp = (n + group - 1) / group;
    OpRecord rec;
    rec.op = "rmsnorm_had_quant_i8g";
    rec.params = {{"M", rows}, {"n", n}, {"group", group}, {"dtype", std::string("bf16")},
                  {"eps", std::to_string(kEps)}, {"wadd", std::to_string(kWadd)}};
    rec.n = rows;
    rec.operands = {
        {"x", "bf16", {rows, n}, 0, rows, false},
        {"residual", "bf16", {rows, n}, 0, rows, true},
        {"w", "f32", {n}, 0, 1, false},
        {"codes", "i8", {rows, n}, 0, rows, true},
        {"scale", "f32", {rows, ngrp}, 0, rows, true},
        {"out", "bf16", {rows, n}, 0, rows, true},
    };
    rec.operands[1].in_place = true;
    return rec;
}

// The reference decomposition over one tensor: v is the sum, sc is 1/rms(v) per row, and the
// normed is v * sc * (w + wadd) in f64 -- the formula the twin and the row both form.
std::vector<double> reference_normed(const std::vector<double>& v, int64_t n,
                                     const std::vector<double>& w) {
    const int64_t rows = (int64_t)v.size() / (size_t)n;
    std::vector<double> out((size_t)v.size(), 0.0);
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t b = r * n;
        double ss = 0.0;
        for (int64_t i = 0; i < n; ++i) ss += v[(size_t)(b + i)] * v[(size_t)(b + i)];
        const double sc = 1.0 / std::sqrt(ss / (double)n + kEps);
        for (int64_t i = 0; i < n; ++i)
            out[(size_t)(b + i)] = v[(size_t)(b + i)] * sc * (w[(size_t)i] + kWadd);
    }
    return out;
}

double max_rel(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - b[i]);
        const double denom = std::max(std::fabs(a[i]), std::fabs(b[i]));
        if (denom > 1e-12) m = std::max(m, d / denom);
    }
    return m;
}

// One residual record, the sum v, and the gain, built from a seed. v = x + residual_old is held
// in f64: the tape's snapshot of the in-place operand is the post-update sum, so that is the
// value operand 1 carries.
struct Case {
    OpRecord rec;
    std::vector<double> x;  // the branch, operand 0
    std::vector<double> v;  // the sum, operand 1's snapshot
    std::vector<double> w;  // the gain, operand 2
};

Case make_case(uint64_t seed) {
    const int64_t rows = 5, n = 128, group = 128;
    std::vector<double> x((size_t)(rows * n));
    for (double& t : x) t = as_bf16(urand(&seed));
    std::vector<double> r_old((size_t)(rows * n));
    for (double& t : r_old) t = as_bf16(urand(&seed));
    std::vector<double> v((size_t)(rows * n));
    for (size_t i = 0; i < v.size(); ++i) v[i] = x[i] + r_old[i];  // non-zero, x and v differ
    std::vector<double> w((size_t)n);
    for (double& t : w) t = (double)(float)(1.0 + 0.5 * urand(&seed));
    return {residual_rec(rows, n, group), std::move(x), std::move(v), std::move(w)};
}

R97_TEST(rmsnorm_residual_forward) {
    const Case c = make_case(0x55311da1ull);
    std::string err;
    TwinFn twin = twin_find(c.rec.op, c.rec.params, &err);
    if (twin == nullptr) fail("no twin for the fused norm: " + err);
    const std::vector<AdValue> in = {
        values("bf16", c.rec.operands[0].shape, c.x),
        values("bf16", c.rec.operands[1].shape, c.v),
        values("f32", c.rec.operands[2].shape, c.w),
    };
    std::vector<AdValue> out;
    twin(c.rec, in, &out);
    // The writes are the residual, the codes, the scale and the normed, in the record's order.
    if (out.size() != 4)
        fail("fused norm twin wrote " + std::to_string(out.size()) + " outputs, expected 4");
    // (1) The residual slot carries the sum the tape snapshots, not the normed. Before the fix
    // the slot held the norm over operand 0, so this is the forward half of the defect.
    const double res_err = max_rel(out[0].value, c.v);
    // (2) The normed is the reference decomposition v * sc * (w + wadd).
    const std::vector<double> want = reference_normed(c.v, 128, c.w);
    const double norm_err = max_rel(out[3].value, want);
    std::printf("rmsnorm_residual forward: residual rel %.3g, normed rel %.3g\n", res_err,
                norm_err);
    if (res_err > 1e-12)
        fail("residual slot is not the sum: rel err " + std::to_string(res_err));
    if (norm_err > 1e-9)
        fail("normed is not v*sc*(w+wadd): rel err " + std::to_string(norm_err));
}

R97_TEST(rmsnorm_residual_gradient) {
    const Case c = make_case(0x716011d2ull);
    std::string err;
    TwinFn twin = twin_find(c.rec.op, c.rec.params, &err);
    if (twin == nullptr) fail("no twin for the fused norm: " + err);
    const std::vector<AdValue> in = {
        values("bf16", c.rec.operands[0].shape, c.x),
        values("bf16", c.rec.operands[1].shape, c.v),
        values("f32", c.rec.operands[2].shape, c.w),
    };
    // The forward the row's quantiser stage differentiates from: the twin's own writes.
    std::vector<AdValue> fwd_out;
    twin(c.rec, in, &fwd_out);
    std::vector<AdValue> fwd(c.rec.operands.size());
    fwd[0] = in[0];
    fwd[1] = in[1];  // the residual's snapshot is the sum
    fwd[2] = in[2];
    fwd[3] = fwd_out[1];  // codes
    fwd[4] = fwd_out[2];  // scale
    fwd[5] = fwd_out[3];  // the normed
    // A random cotangent for every write, in the record's write order: residual, codes, scale,
    // normed. The residual's is the term the external consumer of the stream hands the slot.
    uint64_t cs = 0x716011d2ull ^ 0xc0ffeeull;
    std::vector<AdValue> cot;
    for (size_t j = 0; j < c.rec.operands.size(); ++j) {
        if (!c.rec.operands[j].written) continue;
        int64_t cnt = 1;
        for (int64_t d : c.rec.operands[j].shape) cnt *= d;
        AdValue co;
        co.shape = c.rec.operands[j].shape;
        co.value.resize((size_t)cnt);
        for (double& t : co.value) t = urand(&cs);
        cot.push_back(std::move(co));
    }
    // The codes and the scale are the quantiser's outputs: piecewise constant in the input, so a
    // finite difference of them is zero almost everywhere and a spike where a cell boundary is
    // crossed. No FD can reproduce the row's straight-through treatment of that path, and with
    // their cotangents live the numeric measures a different function from the one the row
    // differentiates (it failed at 5.5% for exactly that reason). This test is about the residual
    // split, so they are zeroed and the FD then measures the path being checked: the norm's
    // derivative on the sum, plus the residual's own term from its external consumer. The
    // straight-through path has its own coverage (tests/host/adjoint_quant.cpp).
    if (cot.size() > 2) {
        for (double& t : cot[1].value) t = 0.0;  // codes
        for (double& t : cot[2].value) t = 0.0;  // scale
    }
    std::vector<AdValue> dc(c.rec.operands.size());
    for (size_t i = 0; i < dc.size(); ++i) {
        dc[i].shape = c.rec.operands[i].shape;
        int64_t cnt = 1;
        for (int64_t d : c.rec.operands[i].shape) cnt *= d;
        dc[i].value.assign((size_t)cnt, 0.0);
    }
    AdjointCall call{c.rec, fwd, cot, &dc, nullptr, ""};
    AdjointFn adj = adjoint_find(c.rec.op, Side::host, c.rec.params, &err);
    if (adj == nullptr) fail("no host adjoint for the fused norm: " + err);
    adj(call);

    // The finite difference of the same decomposition, along one random direction on the norm's
    // input (operand 1, the sum). Both summands carry its full cotangent, so both slots read the
    // same directional derivative: the analytic must agree with the numeric on both.
    const int64_t total = 5 * 128;
    std::vector<double> dir((size_t)total);
    for (double& t : dir) t = urand(&cs);
    const double eps = 1e-6;
    auto loss_at = [&](double e) {
        std::vector<double> vp = c.v;
        for (size_t i = 0; i < vp.size(); ++i) vp[i] += e * dir[i];
        std::vector<AdValue> iv = in;
        iv[1] = values("bf16", c.rec.operands[1].shape, vp);
        std::vector<AdValue> y;
        twin(c.rec, iv, &y);
        double loss = 0.0;
        for (size_t o = 0; o < y.size() && o < cot.size(); ++o)
            for (size_t i = 0; i < y[o].value.size() && i < cot[o].value.size(); ++i)
                loss += y[o].value[i] * cot[o].value[i];
        return loss;
    };
    const double numeric = (loss_at(eps) - loss_at(-eps)) / (2.0 * eps);
    const double analytic0 = std::inner_product(dc[0].value.begin(), dc[0].value.end(), dir.begin(),
                                                0.0);
    const double analytic1 = std::inner_product(dc[1].value.begin(), dc[1].value.end(), dir.begin(),
                                                0.0);
    const double denom =
        std::max(std::max(std::fabs(numeric), std::fabs(analytic0)), std::fabs(analytic1)) + 1e-30;
    const double err0 = std::fabs(analytic0 - numeric) / denom;
    const double err1 = std::fabs(analytic1 - numeric) / denom;
    std::printf("rmsnorm_residual gradient: dc[0] rel %.3g, dc[1] rel %.3g (numeric %.6g)\n", err0,
                err1, numeric);
    if (err0 > 1e-5)
        fail("dc[0] (the x slot) differs from the decomposition's finite difference: " +
             std::to_string(err0));
    if (err1 > 1e-5)
        fail("dc[1] (the residual slot) differs from the decomposition's finite difference: " +
             std::to_string(err1));
}

}  // namespace
