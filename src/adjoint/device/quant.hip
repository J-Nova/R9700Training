// src/adjoint/device/quant.hip -- the device straight-through adjoint of the quant family.
//
// The same ops as src/adjoint/host/quant.cpp, on the card: had_quant_act_i8g and the fused
// producers the card issues (rmsnorm_had_quant_i8g, gated_had_quant_i8g, gate_had_quant_i8g).
// Each rotates an activation, quantises it per group of `group` values, and stores the int8 codes
// and one f32 scale per group; a fused producer also applies an elementwise op (an RMS norm, the
// gated activation's silu(gate) * up, the gate's sigmoid(gate) * x) before the rotation.
//
// The adjoint of one group is docs/adjoints/quant.md's straight-through rule: the cotangent of the
// codes is divided by the group's scale, dropped where the forward's value saturated the int8 grid,
// and rotated back with the transposed Hadamard transform -- the same transform, since H is its own
// transpose and the forward does not normalise. A fused producer's cotangent with respect to the
// rotated activation then continues through the elementwise part, whose cotangent lands on the
// record's read operands. One kernel does the mask and the rotation; a second does the elementwise
// backprop.
//
// Where the mask comes from. docs/adjoints/quant.md's rule is |v * (1/scale)| > 127, v the group's
// rotated activation, and this kernel applies exactly that: the codes the replayed forward wrote
// are read, and a code at the top of the grid (|code| == 127) keeps its cotangent only when the
// rotated value was inside the grid. Both halves are needed. A code of 127 is not by itself
// evidence of a clamp -- every value in (126.5, 127] rounds to it, and the group's amax reaches it
// by construction in either direction of the scale's rounding -- so the reference's mask and a mask
// read from the codes alone would disagree on the amax of roughly half the groups, which is what
// the test would catch. Reading the codes and confirming with v is the reference's rule exactly,
// and the codes are what tell a replay which side of the grid the forward's rounding landed on.
//
// v is formed in f64, stage by stage, the way the host adjoint's rotated() forms it, so a group
// whose amax sits an ulp from the grid's edge decides as the reference decides. The butterflies of
// one stage are independent, so threads may share a stage without changing a bit of the result.
//
// The contract gives a device adjoint f64 operands, not device buffers, so this call owns every
// device buffer it touches. They are scratch, freed before it returns; the harness reports the
// high-water mark. The forward values are uploaded at f64 where a decision depends on them (the
// rotated activation, the scale) and at f32 where only a smooth derivative does (the elementwise
// part's operands); gradients are read at f32, summed in f32, and rounded once to bf16 on the way
// out, as add.hip does. A fused producer's intermediate is not on the tape, so it is rebuilt
// through the parts' registered twin -- the same definition the host reference composes -- and a
// record whose activation this row does not serve is refused at the row lookup rather than served
// approximately.
#include "r97/adjoint.h"

#include "scratch.h"

#include "rad_device.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

namespace r97 {
namespace {

// bf16 is stored as the high half of an f32, with round-to-nearest-even on the way down (add.hip's
// conversion, so both kernels round the tape's cotangents the same way).
__host__ __device__ inline float bf16_to_f32(unsigned short b) {
    const unsigned int u = (unsigned int)b << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

__host__ __device__ inline unsigned short f32_to_bf16(float f) {
    unsigned int u;
    std::memcpy(&u, &f, sizeof u);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x0040u);  // NaN
    const unsigned int lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (unsigned short)(u >> 16);
}

/* ---------------------------------------------------------------- the family's parts */

// Which elementwise op a fused producer applies before the rotation. `quant` is the plain
// quantiser: the whole of had_quant_act_i8g.
enum class Part { quant, rmsnorm, gated, gate };

// The host twin and host adjoint of the elementwise part, by op name.
const char* part_op(Part part) {
    switch (part) {
        case Part::rmsnorm: return "rmsnorm";
        case Part::gated: return "silu_mul";
        case Part::gate: return "act_mul";
        case Part::quant: break;
    }
    return nullptr;
}

// How many operands of the record the adjoint answers for (every operand but the codes and scale).
size_t part_reads(Part part) {
    return part == Part::rmsnorm || part == Part::gate ? 2 : 1;
}

/* ---------------------------------------------------------------- parameters */

// A numeric parameter, as the host adjoint reads it: an int or a decimal string. Missing or
// non-numeric leaves false, because an adjoint must not throw.
bool num_param(const std::vector<OpParam>& params, const char* name, double* out) {
    for (const OpParam& p : params) {
        if (p.first != name) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) {
            *out = (double)*i;
            return true;
        }
        const std::string* s = std::get_if<std::string>(&p.second);
        if (!s || s->empty()) return false;
        char* end = nullptr;
        const double v = std::strtod(s->c_str(), &end);
        if (!end || *end != '\0') return false;
        *out = v;
        return true;
    }
    return false;
}

double param_num(const OpRecord& rec, const char* name, double fallback) {
    double v = 0.0;
    return num_param(rec.params, name, &v) ? v : fallback;
}

// The rotation and the per-group scale are the record's `n` and `group`: a power of two that
// divides the width, no wider than one block's threads. A record outside that is refused at the
// row lookup rather than served approximately.
bool rotated(const std::vector<OpParam>& params) {
    double n = 0.0, group = 0.0;
    if (!num_param(params, "n", &n) || !num_param(params, "group", &group)) return false;
    if (n <= 0.0 || n != std::floor(n)) return false;
    if (group <= 0.0 || group != std::floor(group) || group > 1024.0) return false;
    const int64_t g = (int64_t)group;
    if ((g & (g - 1)) != 0 || (int64_t)n % g != 0) return false;
    return true;
}

// The gated producer's activation: silu, its documented first mode and the one the card's swiglu
// sites issue. The sigmoid form would take a different second half and is not served yet.
bool serves_gated(const std::vector<OpParam>& params) {
    if (!rotated(params)) return false;
    for (const OpParam& p : params) {
        if (p.first != "act") continue;
        const std::string* act = std::get_if<std::string>(&p.second);
        return act && *act == "silu";
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
        return act && *act == "sigmoid";
    }
    return false;
}

/* ---------------------------------------------------------------- kernels */

// One block per (row, group), one thread per element of the group: form the group's rotated
// activation in f64, decide the mask against the replayed forward's codes and scale, then rotate
// the masked cotangent back in f32. The forward transform's stages are the host's rotated()'s, so
// a value one ulp inside or outside the grid lands on the side the reference puts it; the
// butterflies of a stage are independent, so sharing a stage across threads changes no bit.
__global__ void mask_rotate_kernel(const signed char* codes, const double* y, const double* scale,
                                   const float* cot, float* rotated_cot, int n, int group,
                                   int ngrp, unsigned int* bad_count, unsigned int* bad_worst) {
    extern __shared__ char smem[];
    double* v = reinterpret_cast<double*>(smem);
    float* d = reinterpret_cast<float*>(smem + (size_t)group * sizeof(double));

    // The quotient's guard: `cot / (float)s` is exact only while it stays inside f32, and an Inf
    // here is not a per-group event -- it enters the inverse rotation and then the elementwise
    // backprop, so one overflowing element poisons the whole row and the run's first non-finite
    // value names a row that only did its own arithmetic. Counted where the division happens, so
    // an element the mask zeroed is not counted: this is exact, not a bound.
    __shared__ unsigned int block_bad;
    __shared__ unsigned int block_worst;

    const int tid = (int)threadIdx.x;
    const int64_t row = (int64_t)blockIdx.x / ngrp;
    const int gi = (int)((int64_t)blockIdx.x - row * ngrp);
    const int64_t base = row * (int64_t)n + (int64_t)gi * (int64_t)group;

    if (tid == 0) {
        block_bad = 0u;
        block_worst = 0u;
    }
    v[tid] = y[base + tid];
    __syncthreads();
    for (int h = 1; h < group; h <<= 1) {
        if (tid < group / 2) {
            const int j = (tid / h) * (2 * h) + (tid % h);
            const double a = v[j], b = v[j + h];
            v[j] = a + b;
            v[j + h] = a - b;
        }
        __syncthreads();
    }

    const double s = scale[row * (int64_t)ngrp + gi];
    const double u = std::fabs(v[tid] * (1.0 / s));
    const signed char code = codes[base + tid];
    const bool top = code == 127 || code == -127;
    const bool live = top ? u <= 127.0 : true;
    const float q = live ? cot[base + tid] / (float)s : 0.0f;
    d[tid] = q;
    if (!isfinite(q)) {
        atomicAdd(&block_bad, 1u);
        atomicMax(&block_worst, __float_as_uint(fabsf(q)));
    }
    __syncthreads();
    if (tid == 0 && block_bad != 0u) {
        atomicAdd(bad_count, block_bad);
        atomicMax(bad_worst, block_worst);
    }
    for (int h = 1; h < group; h <<= 1) {
        if (tid < group / 2) {
            const int j = (tid / h) * (2 * h) + (tid % h);
            const float a = d[j], b = d[j + h];
            d[j] = a + b;
            d[j + h] = a - b;
        }
        __syncthreads();
    }
    rotated_cot[base + tid] = d[tid];
}

// The plain quantiser: the rotated cotangent is the activation's cotangent.
__global__ void quant_back_kernel(const float* rotated_cot, unsigned short* dx, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dx[i] = f32_to_bf16(rotated_cot[i]);
}

// silu_mul: y = silu(g) * u over gate_up's two halves, so g's cotangent carries the slope and the
// second half's the activation alone.
__global__ void gated_back_kernel(const float* rotated_cot, const float* gate_up,
                                  unsigned short* d, int64_t total, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t row = i / n;
    const int64_t gb = row * 2 * n + (i - row * n);
    const float g = gate_up[gb], u = gate_up[gb + n];
    const float s = 1.0f / (1.0f + expf(-g));
    d[gb] = f32_to_bf16(rotated_cot[i] * u * (s * (1.0f + g * (1.0f - s))));
    d[gb + n] = f32_to_bf16(rotated_cot[i] * (g * s));
}

// The gate: y = sigmoid(g) * x.
__global__ void gate_back_kernel(const float* rotated_cot, const float* gate, const float* x,
                                 unsigned short* dgate, unsigned short* dx, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float s = 1.0f / (1.0f + expf(-gate[i]));
    dgate[i] = f32_to_bf16(rotated_cot[i] * s * (1.0f - s) * x[i]);
    dx[i] = f32_to_bf16(rotated_cot[i] * s);
}

// The RMS norm, one block per row: ss = sum x^2 and t = sum r x c share the row's reduction, c =
// w + wadd. dx_j = mult * (rs c_j r_j - rs^3 x_j t / n), the row's shared term carried by every
// element. ss and t accumulate in f64 -- the note below says why; the gain is frozen (the host
// rmsnorm adjoint leaves it), so only the norm input's cotangent comes back. On a record with a
// residual the caller hands the sum's snapshot as `x` (the row), so this body is unchanged.
__global__ void rmsnorm_back_kernel(const float* rotated_cot, const float* x, const float* w,
                                    unsigned short* dx, int64_t n, float eps, float wadd,
                                    float mult) {
    // The row reduction is carried in f64, matching the host twin's `double drs`
    // (src/adjoint/host/quant.cpp:628-632). A cotangent of order 1e34 -- which the real container
    // reaches by block 51 and the fixture never does -- makes the 5120-term sum of
    // `rotated_cot * x * (w + wadd)` cross f32's 3.4e38 and the f64 one not, so an f32 accumulator
    // here is a device-versus-reference divergence, not a tolerance. The operands are still the
    // f32 ones the record carries (rounded to its declared dtype, as the host rounds them), so the
    // sum differs from the host's only in its width. `ss` rides the same reduction because the
    // shared term uses `rs^3`, which amplifies any error in it.
    __shared__ double sh_ss[256];
    __shared__ double sh_t[256];
    const int tid = (int)threadIdx.x;
    const int64_t base = (int64_t)blockIdx.x * n;

    double ss = 0.0, t = 0.0;
    for (int64_t i = tid; i < n; i += blockDim.x) {
        const double xi = (double)x[base + i];
        ss += xi * xi;
        t += (double)rotated_cot[base + i] * xi * ((double)w[i] + (double)wadd);
    }
    sh_ss[tid] = ss;
    sh_t[tid] = t;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sh_ss[tid] += sh_ss[tid + s];
            sh_t[tid] += sh_t[tid + s];
        }
        __syncthreads();
    }

    const double inv_n = 1.0 / (double)n;
    const double rs = 1.0 / sqrt(sh_ss[0] * inv_n + (double)eps);
    const double shared = (double)mult * rs * rs * rs * sh_t[0] * inv_n;
    for (int64_t i = tid; i < n; i += blockDim.x) {
        const double c = (double)w[i] + (double)wadd;
        const double v = (double)mult * (rs * c * (double)rotated_cot[base + i]) -
                         shared * (double)x[base + i];
        dx[base + i] = f32_to_bf16((float)v);
    }
}

/* ---------------------------------------------------------------- the call */

// One stream for every device adjoint call in this process, as add.hip has it: upload, launch and
// download are one queue's worth of work, which is what the harness times.
RadStream adjoint_stream() {
    static RadStream s = [] {
        RadStream created = nullptr;
        if (rad_stream_create(&created, 0) != 0) return (RadStream) nullptr;
        return created;
    }();
    return s;
}

int64_t blocks(int64_t n) { return (n + 255) / 256; }

// The value the quantiser rotated: the record's own activation for the plain quantiser, and for a
// fused producer the intermediate the elementwise part wrote, rebuilt through its twin so this
// call and the host reference rotate the same numbers.
bool intermediate(const AdjointCall& call, Part part, std::vector<AdValue>* y) {
    const char* op = part_op(part);
    if (!op) {
        y->assign(1, call.fwd[0]);
        return true;
    }
    TwinFn twin = twin_find(op, call.rec.params, nullptr);
    if (!twin) return false;
    // The parts' twins reject a record they cannot serve by throwing; an adjoint must not throw.
    try {
        twin(call.rec, call.fwd, y);
    } catch (...) {
        return false;
    }
    return !y->empty();
}

// The name of the last HIP error, so a refusal says which copy failed and what the runtime called
// it. The radiance calls return their own code; this is the name behind the failure.
std::string hip_error_name() {
    return hipGetErrorString(hipGetLastError());
}

// The mask and the rotation: uploads what the decision reads, launches the one kernel, and keeps
// nothing. `d_rotated` is the caller's `rows * n` f32 buffer.
// The operands are the roles the caller found, not positions: the live tape's fused records carry a
// phantom write ahead of the codes, so the codes' cotangent is cot[1] there and cot[0] on the
// schema's two-write shape.
void mask_and_rotate(const AdValue& codes, const AdValue& scale, const AdValue& cot, int64_t rows,
                     int64_t n, int64_t group, int64_t ngrp, const std::vector<double>& y,
                     RadStream s, void* d_rotated, const char* op, int64_t position) {
    const int64_t total = rows * n;

    std::vector<double> h_y((size_t)total), h_scale((size_t)(rows * ngrp));
    std::vector<signed char> h_codes((size_t)total);
    std::vector<float> h_cot((size_t)total);
    for (int64_t i = 0; i < total; ++i) {
        h_y[(size_t)i] = y[(size_t)i];
        h_codes[(size_t)i] = (signed char)codes.value[(size_t)i];
        h_cot[(size_t)i] = (float)cot.value[(size_t)i];
    }
    for (int64_t i = 0; i < rows * ngrp; ++i) h_scale[(size_t)i] = scale.value[(size_t)i];
    // The kernel divides the codes' cotangent by the group's scale (`d[tid] = cot / (float)s`).
    // That is exact only while the quotient stays inside f32: the serving quantiser stores `m / 127`
    // for ANY `m > 0` and never floors it, so a group whose rotated amax is tiny stores a tiny
    // scale, and `cot / scale` overflows to Inf in one step -- and that Inf then enters the
    // inverse rotation and the elementwise backprop, which is a whole-row event, not a per-group
    // one. Report the smallest scale and where it sits, and how many are below the range where a
    // quotient of order 1 already reaches f32's ceiling (3.4e38).
    if (const char* dq = std::getenv("R97_DUMP_QUANT"); dq != nullptr && dq[0] == '1') {
        double min_s = 0.0;
        int64_t min_r = -1, min_g = -1, extreme = 0;
        for (int64_t i = 0; i < rows * ngrp; ++i) {
            const double sv = scale.value[(size_t)i];
            if (i == 0 || sv < min_s) {
                min_s = sv;
                min_r = i / ngrp;
                min_g = i % ngrp;
            }
            if (sv < 1e-32) ++extreme;
        }
        std::fprintf(stderr,
                     "SCALERANGE rows=%lld ngrp=%lld groups=%lld min=%.6g at=%lld,%lld "
                     "below1e-32=%lld\n",
                     (long long)rows, (long long)ngrp, (long long)(rows * ngrp), min_s,
                     (long long)min_r, (long long)min_g, (long long)extreme);
        std::fflush(stderr);
    }

    device::ScratchBlock d_y(total * (int64_t)sizeof(double));
    device::ScratchBlock d_scale(rows * ngrp * (int64_t)sizeof(double));
    device::ScratchBlock d_codes(total);
    device::ScratchBlock d_cot(total * (int64_t)sizeof(float));
    if (!d_y || !d_scale || !d_codes || !d_cot)
        throw std::runtime_error("r97: quant adjoint: the mask pass could not take its scratch (" +
                                 std::to_string(total * 8) + " + " +
                                 std::to_string(rows * ngrp * 8) + " + " + std::to_string(total) +
                                 " + " + std::to_string(total * 4) + " bytes)");
    if (rad_memcpy_async(d_y.p, h_y.data(), total * 8, s) != 0)
        throw std::runtime_error("r97: quant adjoint: the mask pass could not upload the rotated "
                                 "activation (" + std::to_string(total * 8) + " bytes, " +
                                 hip_error_name() + ")");
    if (rad_memcpy_async(d_scale.p, h_scale.data(), rows * ngrp * 8, s) != 0)
        throw std::runtime_error("r97: quant adjoint: the mask pass could not upload the scale (" +
                                 std::to_string(rows * ngrp * 8) + " bytes, " + hip_error_name() +
                                 ")");
    if (rad_memcpy_async(d_codes.p, h_codes.data(), total, s) != 0)
        throw std::runtime_error("r97: quant adjoint: the mask pass could not upload the codes (" +
                                 std::to_string(total) + " bytes, " + hip_error_name() + ")");
    if (rad_memcpy_async(d_cot.p, h_cot.data(), total * 4, s) != 0)
        throw std::runtime_error("r97: quant adjoint: the mask pass could not upload the "
                                 "cotangent (" + std::to_string(total * 4) + " bytes, " +
                                 hip_error_name() + ")");

    // Two counters the kernel fills where it divides: how many quotients left f32, and the largest
    // one. Eight bytes the host zeroes before the launch and reads back after it, on the same
    // stream, so the read costs one copy and no extra sync.
    device::ScratchBlock d_bad(2 * (int64_t)sizeof(unsigned int));
    if (!d_bad)
        throw std::runtime_error("r97: quant adjoint: op '" + std::string(op) + "' at " +
                                 std::to_string(position) +
                                 ": the quotient guard's scratch was refused");
    const unsigned int bad_zero[2] = {0u, 0u};
    if (rad_memcpy_async(d_bad.p, bad_zero, sizeof bad_zero, s) != 0)
        throw std::runtime_error("r97: quant adjoint: op '" + std::string(op) + "' at " +
                                 std::to_string(position) +
                                 ": the quotient guard's counters would not clear (" +
                                 hip_error_name() + ")");

    const size_t shared = (size_t)(group * ((int64_t)sizeof(double) + (int64_t)sizeof(float)));
    hipLaunchKernelGGL(mask_rotate_kernel, dim3((unsigned int)(rows * ngrp)),
                       dim3((unsigned int)group), shared, (hipStream_t)s,
                       (const signed char*)d_codes.p, (const double*)d_y.p,
                       (const double*)d_scale.p, (const float*)d_cot.p, (float*)d_rotated,
                       (int)n, (int)group, (int)ngrp,
                       (unsigned int*)d_bad.p,
                       (unsigned int*)d_bad.p + 1);
    if (rad_stream_sync(s) != 0)
        throw std::runtime_error("r97: quant adjoint: the mask pass's kernel did not finish (" +
                                 hip_error_name() + ")");
    unsigned int bad[2] = {0u, 0u};
    if (rad_memcpy_async(bad, d_bad.p, sizeof bad, s) != 0)
        throw std::runtime_error("r97: quant adjoint: op '" + std::string(op) + "' at " +
                                 std::to_string(position) +
                                 ": the quotient guard's counters would not read back (" +
                                 hip_error_name() + ")");
    if (rad_stream_sync(s) != 0)
        throw std::runtime_error("r97: quant adjoint: the quotient guard's read did not finish (" +
                                 hip_error_name() + ")");
    if (bad[0] != 0) {
        // The smallest scale in the launch, so the report says what made the quotient large.
        double min_scale = 0.0;
        for (int64_t i = 0; i < rows * ngrp; ++i)
            if (i == 0 || scale.value[(size_t)i] < min_scale) min_scale = scale.value[(size_t)i];
        float worst_f = 0.0f;
        std::memcpy(&worst_f, &bad[1], sizeof worst_f);
        const double worst = (double)worst_f;
        std::fprintf(stderr,
                     "QUANTGUARD op=%s pos=%lld nonfinite=%u of %lld worst=%.6g min_scale=%.6g\n",
                     op, (long long)position, bad[0], (long long)(rows * n), worst, min_scale);
        std::fflush(stderr);
        // The refusal is the caller's to ask for. A container whose cotangents already divide past
        // f32 -- this one does, 2.66e36 over a 5.96e-5 scale -- would otherwise abort the step
        // before the A/B that exists to find out why, so the default is to report the count and
        // carry on, and R97_QUANT_GUARD=1 is what turns the same count into the refusal
        // tests/gpu/quant_guard.cpp asserts. Read per call, not cached: one test exercises both
        // arms in one process, and the knob is a getenv against one launch per record.
        const char* knob = std::getenv("R97_QUANT_GUARD");
        if (knob != nullptr && knob[0] == '1')
            throw std::runtime_error(
                "r97: quant adjoint: op '" + std::string(op) + "' at " + std::to_string(position) +
                ": " + std::to_string(bad[0]) + " of " + std::to_string(rows * n) +
                " codes' cotangents divide to something that is not finite -- an Inf past f32's "
                "range, or a NaN the cotangent already carried (worst |cot / scale| = " +
                std::to_string(worst) + ", the smallest group scale in this record is " +
                std::to_string(min_scale) + "); the quotient is exact only while it stays inside "
                "f32, and a non-finite value there enters the inverse rotation and the elementwise "
                "backprop as a whole-row event, so this refuses rather than emitting it");
    }
}

// The tape's cotangents are f64 and are added to, never overwritten: other ops share them.
// `rows_per` and `read_rows`, when given, are the two-way rule (r97/adjoint.h's
// AdjointCall::read_rows): a row no consumer reads is provably dead, so its values are skipped
// rather than summed -- a non-finite value there is an artefact of the formula over a row nothing
// reads, not a gradient.
void add_back(unsigned short* d_out, int64_t count, AdValue& into, RadStream s, int64_t rows_per = 0,
              const std::vector<int64_t>* read_rows = nullptr) {
    std::vector<unsigned short> h((size_t)count);
    if (rad_memcpy_async(h.data(), d_out, count * 2, s) != 0)
        throw std::runtime_error("r97: quant adjoint: could not read back the gradient (" +
                                 std::to_string(count) + " bf16 values, " + hip_error_name() + ")");
    if (rad_stream_sync(s) != 0)
        throw std::runtime_error("r97: quant adjoint: the back kernel did not finish (" +
                                 hip_error_name() + ")");
    for (int64_t i = 0; i < count; ++i) {
        if (rows_per > 0 && !row_readable(read_rows, i / rows_per)) continue;
        into.value[(size_t)i] += bf16_to_f32(h[(size_t)i]);
    }
}

// The elementwise output's own cotangent, summed into the one the codes' path produced: the fused
// producer writes that output beside the codes, so its consumers' cotangent is a second term of the
// same gradient. `+=`, never an assignment -- a slot the codes' path already wrote keeps its term.
__global__ void add_into_kernel(float* dst, const float* src, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

const char* part_name(Part part) {
    switch (part) {
        case Part::quant: return "quant";
        case Part::rmsnorm: return "rmsnorm";
        case Part::gated: return "gated";
        case Part::gate: return "gate";
    }
    return "unknown";
}

// The cotangent slot of a written operand: the walk's `cot` holds one entry per write, in the
// record's write order.
int64_t cot_slot(const OpRecord& rec, int64_t operand) {
    int64_t at = 0;
    for (int64_t j = 0; j < (int64_t)rec.operands.size(); ++j) {
        if (!rec.operands[j].written) continue;
        if (j == operand) return at;
        ++at;
    }
    return -1;
}

void quant_adjoint(const AdjointCall& call, Part part) {
    if (!call.dc)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op +
                                 "': the caller handed no gradient slots");
    if (call.cot.size() < 2)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op + "': " +
                                 std::to_string(call.cot.size()) +
                                 " cotangents for a record that writes at least two operands");
    const int64_t operands = (int64_t)call.rec.operands.size();
    int64_t reads = 0;
    for (const Operand& o : call.rec.operands)
        if (!o.written) ++reads;
    // Every operand, reads included: `dc` is indexed by the record's schema position, and a slot
    // shorter than that would be read out of bounds once the writes' roles are used.
    if (call.fwd.size() != (size_t)operands || (int64_t)call.dc->size() < operands)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op + "': " +
                                 std::to_string(call.fwd.size()) + " forward values and " +
                                 std::to_string(call.dc->size()) + " gradient slots for a record of " +
                                 std::to_string(operands) + " operands");

    // The row width, the group count and the group size come from the operands, never from the
    // record's `n`/`group` params: the live `gate_had_quant_i8g` record declares n=256 group=128
    // while its codes are 6144 wide with a 48-group scale (group 128). The codes' own shape and the
    // scale's own shape are the truth; the params are only a cross-check.
    //
    // The written operands' roles, by dtype and shape, never by position. `phantom` is the write a
    // fused record carries with neither a dtype nor a shape.
    int64_t codes_at = -1, scale_at = -1, writes = 0, phantom = 0;
    for (int64_t j = 0; j < operands; ++j) {
        const Operand& o = call.rec.operands[(size_t)j];
        if (!o.written) continue;
        ++writes;
        if (o.dtype.empty() || o.dtype == "invalid" || o.shape.empty()) {
            ++phantom;
            continue;
        }
        if (o.dtype == "i8" || o.dtype == "fp8_e4m3" || o.dtype == "fp8_e5m2") {
            if (codes_at < 0 && (int64_t)o.shape.size() == 2) codes_at = j;
        }
    }
    if (codes_at < 0)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op + "': the " +
                                 part_name(part) + " record's " + std::to_string(writes) +
                                 " writes carry no codes (i8/fp8) operand");
    const int64_t rows = call.rec.operands[(size_t)codes_at].shape[0];
    const int64_t codes_n = call.rec.operands[(size_t)codes_at].shape[1];
    // The elementwise output's width is the row width `n`. The gated producer's codes span gate_up,
    // both halves, while its output is one half -- so the output's own shape decides, and the two
    // shapes the tape really carries are both served: codes == n (schema's gated record) and
    // codes == 2n (the live one).
    int64_t n = codes_n;
    for (int64_t j = 0; j < operands; ++j) {
        const Operand& o = call.rec.operands[(size_t)j];
        if (!o.written || j == codes_at) continue;
        if ((int64_t)o.shape.size() != 2 || o.shape[0] != rows) continue;
        if (o.shape[1] == codes_n || 2 * o.shape[1] == codes_n) {
            n = o.shape[1];
            break;
        }
    }
    if (n <= 0 || codes_n % n != 0)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op + "': the " +
                                 part_name(part) + " record's codes are " +
                                 std::to_string(codes_n) +
                                 " wide and no write of theirs names a row width that divides it "
                                 "(n=" + std::to_string(n) + ")");
    // The group scale is the one f32 write of `[rows, k]` whose k divides the codes' width.
    for (int64_t j = 0; j < operands; ++j) {
        const Operand& o = call.rec.operands[(size_t)j];
        if (!o.written || j == codes_at) continue;
        if (o.dtype != "f32" && o.dtype != "f16" && o.dtype != "bf16") continue;
        if ((int64_t)o.shape.size() != 2 || o.shape[0] != rows || o.shape[1] <= 0) continue;
        // A scale is narrower than the row: without this, the in-place elementwise output (the same
        // width as the codes) matched first and the real `h.s8` was left looking like an output.
        if (o.shape[1] >= codes_n || codes_n % o.shape[1] != 0) continue;
        if (scale_at < 0) scale_at = j;
    }
    const int64_t ngrp = scale_at < 0 ? 0 : call.rec.operands[(size_t)scale_at].shape[1];
    const int64_t group = ngrp > 0 ? codes_n / ngrp : 0;
    if (group <= 0 || group > 1024 || (group & (group - 1)) != 0)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op + "': the " +
                                 part_name(part) + " record's codes are " +
                                 std::to_string(codes_n) + " wide with a " +
                                 std::to_string(ngrp) + "-group scale, which is not a group size");
    // The layouts this row serves, named per part and enumerated -- never a relaxed count, which
    // would let the row index operands on a layout it never verified. The schema's elementwise part
    // plus quantiser is two writes; the live tape's fused producers write the elementwise output
    // beside the codes as well (three), and `rmsnorm_had_quant_i8g` writes the normed value twice --
    // in place over its input and again as `h` (four). Anything else refuses by name, with the
    // writes listed, instead of returning a zero gradient.
    // Every write, as `[name dtype dims]`: both refusals below carry it, because a count that does
    // not match and a role that is not found are the same question -- what did the tape write?
    const auto writes_list = [&]() {
        std::string list;
        for (int64_t j = 0; j < operands; ++j) {
            const Operand& o = call.rec.operands[(size_t)j];
            if (!o.written) continue;
            list += " [";
            list += o.buffer + " " + o.dtype;
            for (int64_t d : o.shape) list += " " + std::to_string(d);
            list += "]";
        }
        return list;
    };
    const int64_t served = part == Part::quant ? 2 : 3;
    const bool layout = writes == 2 || writes == served ||
                        (part != Part::quant && writes == served + 1 && phantom <= 1);
    if (reads != (int64_t)part_reads(part) || !layout)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op + "': a " +
                                 part_name(part) + " record with " + std::to_string(operands) +
                                 " operands (" + std::to_string(reads) + " reads, " +
                                 std::to_string(writes) + " writes, " + std::to_string(phantom) +
                                 " phantom) is not a layout this row serves; its writes:" +
                                 writes_list());
    if (codes_at < 0 || scale_at < 0)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op + "': the " +
                                 part_name(part) + " record's " + std::to_string(writes) +
                                 " writes carry no " +
                                 (codes_at < 0 ? "codes (i8/fp8)" : "group scale (n=" +
                                                                        std::to_string(n) +
                                                                        " group=" +
                                                                        std::to_string(group) +
                                                                        " ngrp=" +
                                                                        std::to_string(ngrp) + ")") +
                                 " operand; its writes:" + writes_list());

    // Every refusal that names an operand uses this: the operand, what it has, what the row needs.
    // Declared here because the value counts are checked before the slots are sized.
    const auto short_operand = [&](const char* what, size_t have, int64_t need) {
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op + "' at " +
                                 std::to_string(call.rec.position) + ": operand '" + what +
                                 "' has " + std::to_string(have) + " values, the row needs " +
                                 std::to_string(need));
    };

    const AdValue& codes = call.fwd[(size_t)codes_at];
    const AdValue& scale = call.fwd[(size_t)scale_at];
    const int64_t cot_at = cot_slot(call.rec, codes_at);
    if (cot_at < 0 || (size_t)cot_at >= call.cot.size())
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op +
                                 "': the codes' write has no cotangent slot");
    const AdValue& cot = call.cot[(size_t)cot_at];
    const int64_t total = rows * n;
    if (rows <= 0)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op +
                                 "': the codes' first dimension is " + std::to_string(rows));
    if ((int64_t)codes.value.size() < rows * codes_n)
        short_operand("codes", codes.value.size(), rows * codes_n);
    if ((int64_t)scale.value.size() < rows * ngrp)
        short_operand("scale", scale.value.size(), rows * ngrp);
    // The codes' cotangent is the codes' own width (2n for the gated producer's gate_up).
    if ((int64_t)cot.value.size() < rows * codes_n)
        short_operand("codes cotangent", cot.value.size(), rows * codes_n);

    // The residual operand: the schema's `residual`, an INOUT write the record also reads -- and
    // the one whose logical shape is the norm's input. The shape test is not decoration. The live
    // tape carries a PHANTOM in-place write (buffer `none:1`, one value) ahead of the residual, so
    // `written && in_place` alone selects a tensor that is not the residual at all; the row then
    // refused on that operand's size and the whole walk died:
    //   "quant adjoint: op 'rmsnorm_had_quant_i8g' at 1: operand 'residual' has 1 values, the row
    //    needs 2621440"
    // The declared shape separates them. The size check below still refuses rather than falling
    // back when the operand that HAS the residual's shape carries no values.
    //
    // The tape snapshots an in-out operand after the issue, so that operand's snapshot already
    // holds the narrowed sum v = x + residual_old -- the sum the reference normalises and reads
    // back (ref_quant.cpp:184-186), and the sum whose cotangent the row splits onto both summands.
    // The row used operand 0 instead, and the fused norm's own activation then disagreed with it at
    // every site. A record without one behaves exactly as before.
    int64_t res_at = -1;
    if (part == Part::rmsnorm) {
        for (int64_t k = 0; k < operands; ++k) {
            const Operand& o = call.rec.operands[(size_t)k];
            if (!o.written || !o.in_place || k == codes_at || k == scale_at) continue;
            if (o.shape != call.rec.operands[0].shape) continue;
            res_at = k;
            break;
        }
        if (res_at >= 0 && (int64_t)call.fwd[(size_t)res_at].value.size() !=
                               (int64_t)call.fwd[0].value.size())
            short_operand("residual", call.fwd[(size_t)res_at].value.size(),
                          call.fwd[0].value.size());
    }

    // What the backprop reads: the elementwise operands' widths, and the slots it answers for.
    const int64_t arg_count = part == Part::quant ? 0
                              : part == Part::rmsnorm ? total + n
                                                      : 2 * total;
    // The gated producer answers for gate_up's two halves; the gate answers for two operands.
    const int64_t out_slot = part == Part::gated ? 2 * total : total;
    const int64_t out_count = part == Part::gated || part == Part::gate ? 2 * total : total;
    // The elementwise op's second operand (`rmsnorm`'s weight, `gate`'s input). For the gate it is
    // the record's second READ -- never `fwd[1]`, which on the live tape's fused records is the
    // phantom `none:1` write with no values. For the norm it is the gain, and the second read is
    // NOT it on the live record: that record is `h, x, w`, so the second read is the norm's own
    // input and using it made the kernel's `w` a per-row activation. The gain is the operand whose
    // rows are all the same one (`gain_row`), which the input never is.
    int64_t second_at = -1;
    for (int64_t k = 1; k < operands && second_at < 0; ++k)
        if (!call.rec.operands[(size_t)k].written) second_at = k;
    // The norm's `x` is an in-place write on the live record (`h, x:inout, w`), so the first read
    // after 0 is the gain and the rule above already finds it. The shape test `gain_row` would not
    // separate them -- both are [rows, n] -- and the tape's gain rows are not even identical
    // (src/adjoint/host/elementwise.cpp's `note_gain_rows` says so once), so position is what is
    // left to go on and the host row's scan lands on the same operand.
    if ((part == Part::rmsnorm || part == Part::gate) && second_at < 0)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op + "': a " +
                                 part_name(part) + " record with no second read");
    // A short operand is a refusal by name, never a return: a silent return here is a zero
    // gradient that agrees with nothing, which is the signature this row exists to remove.
    if (part == Part::rmsnorm && (int64_t)call.fwd[0].value.size() != total)
        short_operand("fwd[0]", call.fwd[0].value.size(), total);
    if (part == Part::rmsnorm && (int64_t)call.fwd[(size_t)second_at].value.size() < n)
        short_operand("fwd[1]", call.fwd[(size_t)second_at].value.size(), n);
    if (part != Part::rmsnorm && (int64_t)call.fwd[0].value.size() !=
                                     (part == Part::gated ? 2 * total : total))
        short_operand("fwd[0]", call.fwd[0].value.size(),
                      part == Part::gated ? 2 * total : total);
    if (part == Part::gate && (int64_t)call.fwd[(size_t)second_at].value.size() != total)
        short_operand("fwd[1]", call.fwd[(size_t)second_at].value.size(), total);
    if ((int64_t)call.dc->at(0).value.size() < out_slot)
        short_operand("dc[0]", call.dc->at(0).value.size(), out_slot);
    if (part == Part::gate && (int64_t)call.dc->at(1).value.size() < total)
        short_operand("dc[1]", call.dc->at(1).value.size(), total);

    std::vector<AdValue> y;
    // The elementwise part's twin is what makes this row's rotation the same numbers as the host
    // reference. A twin that is missing or refuses the record used to return here -- a zero gradient
    // that agreed with nothing. It is a named refusal now.
    if (!intermediate(call, part, &y))
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op + "': no " +
                                 std::string(part_op(part) ? part_op(part) : "elementwise") +
                                 " twin could serve this record, so the " + part_name(part) +
                                 " value the quantiser rotated cannot be rebuilt");
    if ((int64_t)y[0].value.size() != total)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op + "': its " +
                                 part_name(part) + " intermediate carries " +
                                 std::to_string(y[0].value.size()) + " values, not " +
                                 std::to_string(total));

    // The two-way rule (r97/adjoint.h's AdjointCall::read_rows): a row a consumer reads is
    // differentiated, and a non-finite activation in it refuses -- the analytic `0 * inf` over a
    // live row is a bug, not a gradient. A row no consumer reads is provably dead, so it is left
    // zero and counted: the zero is the correct cotangent for a value nothing reads. When the
    // caller could not prove the read set every row is readable, and the refusal is what it was
    // before the read set existed. The codes' path above is untouched: a record whose codes are
    // read by a weight has every code read, so this rule only ever fires for a record whose every
    // write is gathered.
    int64_t zeroed = 0, admitted = 0, refused = 0;
    if (part == Part::rmsnorm) {
        for (int64_t r = 0; r < rows; ++r) {
            if (!row_readable(call, r)) {
                ++zeroed;
                continue;
            }
            ++admitted;
            const int64_t b = r * n;
            double ss = 0.0;
            for (int64_t i = 0; i < n && std::isfinite(ss); ++i) {
                // The live row's activation is v = x + residual (operand 1's snapshot, when the
                // record carries one), so a non-finite residual refuses the same way x would.
                const double v = res_at >= 0
                                     ? call.fwd[(size_t)res_at].value[(size_t)(b + i)]
                                     : call.fwd[0].value[(size_t)(b + i)];
                ss += v * v;
            }
            if (!std::isfinite(ss)) ++refused;
        }
        count_read_rows(call.row_count, call.read_rows, zeroed, admitted, refused);
        if (refused > 0)
            throw std::runtime_error(
                "r97: rmsnorm adjoint: op '" + call.rec.op + "' at " +
                std::to_string(call.rec.position) + ": " + std::to_string(refused) + " of " +
                std::to_string(admitted) +
                " rows a consumer reads have a non-finite activation; refusing rather than zeroing "
                "a live row's gradient");
    }

    RadStream s = adjoint_stream();
    if (!s)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op +
                                 "': the device stream could not be created (" + hip_error_name() +
                                 ")");

    // The rotation covers the codes' whole width -- gate_up's both halves for the gated producer --
    // so the buffer is sized by the codes, not by the elementwise output's `total`.
    device::ScratchBlock d_rotated(rows * codes_n * (int64_t)sizeof(float));
    if (!d_rotated)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op +
                                 "': the rotated cotangent's scratch (" +
                                 std::to_string(rows * codes_n * 4) + " bytes) was refused");
    mask_and_rotate(codes, scale, cot, rows, codes_n, group, ngrp, y[0].value, s, d_rotated.p,
                    call.rec.op.c_str(), call.rec.position);

    // R97_DUMP_QUANT=1: what the row's stages produced, so a zero names its stage instead of the
    // whole row. `cot` and `y` are the host inputs; `rotated` is read back off the card.
    if (const char* dq = std::getenv("R97_DUMP_QUANT"); dq != nullptr && dq[0] == '1') {
        double sc = 0.0, sy = 0.0;
        for (double v : cot.value) sc += std::fabs(v);
        for (double v : y[0].value) sy += std::fabs(v);
        std::vector<float> h_rot((size_t)total);
        if (rad_memcpy_async(h_rot.data(), d_rotated.p, total * 4, s) == 0 &&
            rad_stream_sync(s) == 0) {
            double sr = 0.0;
            for (float v : h_rot) sr += std::fabs((double)v);
            std::fprintf(stderr, "QUANT %s cot=%.6g y=%.6g rotated=%.6g\n", call.rec.op.c_str(),
                         sc, sy, sr);
            std::fflush(stderr);
        }
    }

    // The elementwise output's own cotangent, summed in before the back kernel. Every write that is
    // not the codes or the scale is the elementwise output: `total` values join the rotated
    // cotangent on the card; a write the size of the elementwise input over a buffer the record also
    // reads is the direct path and is added to that read's slot. Both are `+=`, never an assignment
    // -- a slot the codes' path already wrote keeps its term.
    //
    // One scratch block for the fused output's cotangent, alive across the whole loop: a launch is
    // asynchronous, so a block declared inside the loop is freed while the kernel still reads it --
    // that faulted the card inside add_into_kernel (src a recycled 10 MiB range, queue stopped).
    // Reusing one block is safe: every memcpy and launch below is on `s`, in program order.
    device::ScratchBlock d_out_cot(total * (int64_t)sizeof(float));
    for (int64_t j = 0; j < operands; ++j) {
        // The residual is a write too, but its cotangent is the in-place buffer's own dL/dv (its
        // consumers'), not the elementwise output's: it belongs on BOTH summands of v = x +
        // residual, which the split below does. Leaving it out of this loop keeps it off the
        // rotation, where it would enter the norm's Jacobian as if it were dy.
        if (!call.rec.operands[(size_t)j].written || j == codes_at || j == scale_at ||
            j == res_at)
            continue;
        const int64_t at = cot_slot(call.rec, j);
        if (at < 0 || (size_t)at >= call.cot.size())
            throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op +
                                     "': the fused output write '" +
                                     call.rec.operands[(size_t)j].buffer +
                                     "' has no cotangent slot");
        const AdValue& oc = call.cot[(size_t)at];
        const int64_t out_values = (int64_t)oc.value.size();
        if (out_values == 0) continue;  // a write with no consumers: nothing to add
        // The engine's no-value placeholder (`none:3@1453` in the live gated record) is written as
        // one scalar. It cannot be the elementwise output -- that is millions of values -- so it is
        // skipped rather than refused.
        if (out_values == 1) continue;
        if (out_values == total) {
            std::vector<float> h_out((size_t)total);
            for (int64_t i = 0; i < total; ++i) h_out[(size_t)i] = (float)oc.value[(size_t)i];
            if (!d_out_cot)
                throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op +
                                         "': the output cotangent's scratch (" +
                                         std::to_string(total * 4) + " bytes) was refused");
            if (rad_memcpy_async(d_out_cot.p, h_out.data(), total * 4, s) != 0)
                throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op +
                                         "': could not upload the output cotangent (" +
                                         std::to_string(total * 4) + " bytes, " +
                                         hip_error_name() + ")");
            hipLaunchKernelGGL(add_into_kernel, dim3((unsigned int)blocks(total)), dim3(256), 0,
                               (hipStream_t)s, (float*)d_rotated.p,
                               (const float*)d_out_cot.p, total);
            continue;
        }
        int64_t read_at = -1;
        for (int64_t k = 0; k < operands && read_at < 0; ++k)
            if (!call.rec.operands[(size_t)k].written &&
                call.rec.operands[(size_t)k].buffer == call.rec.operands[(size_t)j].buffer)
                read_at = k;
        if (read_at < 0 || out_values != out_slot)
            throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op +
                                     "': the fused output write '" +
                                     call.rec.operands[(size_t)j].buffer + "' carries " +
                                     std::to_string(out_values) + " values, not " +
                                     std::to_string(total) + " or the read's " +
                                     std::to_string(out_slot));
        AdValue& into = call.dc->at((size_t)read_at);
        if ((int64_t)into.value.size() != out_values)
            throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op +
                                     "': the direct path's slot for '" +
                                     call.rec.operands[(size_t)read_at].buffer +
                                     "' is not the size of its cotangent");
        for (int64_t i = 0; i < out_values; ++i)
            into.value[(size_t)i] += oc.value[(size_t)i];
    }

    // The elementwise part's own operands, at f32: they carry a smooth derivative, not a decision.
    device::ScratchBlock d_arg(arg_count * (int64_t)sizeof(float));
    if (arg_count > 0 && !d_arg)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op +
                                 "': the elementwise operands' scratch (" +
                                 std::to_string(arg_count * 4) + " bytes) was refused");
    if (arg_count > 0) {
        std::vector<float> h_arg((size_t)arg_count);
        // The gated producer's first operand is gate_up, both halves of it; the others' is one
        // row of the elementwise op's input. With a residual the input is operand 1's snapshot --
        // the narrowed sum, already written back by the forward -- not operand 0; without one it
        // is operand 0, exactly as before.
        const AdValue& norm_in = res_at >= 0 ? call.fwd[(size_t)res_at] : call.fwd[0];
        const int64_t first = part == Part::gated ? 2 * total : total;
        for (int64_t i = 0; i < first; ++i)
            h_arg[(size_t)i] = (float)norm_in.value[(size_t)i];
        if (part == Part::rmsnorm || part == Part::gate) {
            const AdValue& second = call.fwd[(size_t)second_at];
            const int64_t count = part == Part::rmsnorm ? n : total;
            for (int64_t i = 0; i < count; ++i)
                h_arg[(size_t)(total + i)] = (float)second.value[(size_t)i];
        }
        if (rad_memcpy_async(d_arg.p, h_arg.data(), arg_count * 4, s) != 0)
            throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op +
                                     "': could not upload the elementwise operands (" +
                                     std::to_string(arg_count * 4) + " bytes, " + hip_error_name() +
                                     ")");
    }

    device::ScratchBlock d_out(out_count * (int64_t)sizeof(unsigned short));
    if (!d_out)
        throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op +
                                 "': the gradient's scratch (" + std::to_string(out_count * 2) +
                                 " bytes) was refused");

    const float* rotated = (const float*)d_rotated.p;
    const float* arg = (const float*)d_arg.p;
    switch (part) {
        case Part::quant:
            hipLaunchKernelGGL(quant_back_kernel, dim3((unsigned int)blocks(total)), dim3(256), 0,
                               (hipStream_t)s, rotated, (unsigned short*)d_out.p, total);
            break;
        case Part::gated:
            hipLaunchKernelGGL(gated_back_kernel, dim3((unsigned int)blocks(total)), dim3(256), 0,
                               (hipStream_t)s, rotated, arg, (unsigned short*)d_out.p, total, n);
            break;
        case Part::gate:
            hipLaunchKernelGGL(gate_back_kernel, dim3((unsigned int)blocks(total)), dim3(256), 0,
                               (hipStream_t)s, rotated, arg, arg + total,
                               (unsigned short*)d_out.p, (unsigned short*)d_out.p + total, total);
            break;
        case Part::rmsnorm:
            hipLaunchKernelGGL(rmsnorm_back_kernel, dim3((unsigned int)rows), dim3(256), 0,
                               (hipStream_t)s, rotated, arg, arg + total,
                               (unsigned short*)d_out.p, n,
                               (float)param_num(call.rec, "eps", 1e-6),
                               (float)param_num(call.rec, "wadd", 0.0),
                               (float)param_num(call.rec, "mult", 1.0));
            break;
    }

    add_back((unsigned short*)d_out.p, out_slot, call.dc->at(0), s,
             part == Part::rmsnorm ? n : 0, part == Part::rmsnorm ? call.read_rows : nullptr);
    if (part == Part::gate) add_back((unsigned short*)d_out.p + total, total, call.dc->at(1), s);
    // v = x + residual_old is a two-input sum, so the cotangent of v -- the norm's `dv` plus the
    // incoming cotangent g_r of the new residual value (its external consumers' dL/dv, which the
    // direct-path loop above kept off the rotation) -- lands on BOTH summands at full weight, as
    // `add` does. Each slot gets one copy of dv and one of g_r; with dv = 0 the row reduces to
    // `add` (both slots get g_r once). The residual is an INOUT write: the walk would otherwise
    // pass its own cotangent through into its read slot (src/backward.cpp's in-place passthrough),
    // so the row writes that slot's two terms explicitly here and the passthrough -- which fires
    // only on an all-zero slot -- leaves it alone.
    if (part == Part::rmsnorm && res_at >= 0) {
        AdValue& dres = call.dc->at((size_t)res_at);
        if ((int64_t)dres.value.size() != total)
            throw std::runtime_error("r97: quant adjoint: op '" + call.rec.op + "' at " +
                                     std::to_string(call.rec.position) +
                                     ": the residual's gradient slot has " +
                                     std::to_string(dres.value.size()) + " values, the row needs " +
                                     std::to_string(total));
        add_back((unsigned short*)d_out.p, total, dres, s, n, call.read_rows);
        const int64_t res_cot_at = cot_slot(call.rec, res_at);
        if (res_cot_at >= 0 && (size_t)res_cot_at < call.cot.size()) {
            const AdValue& gr = call.cot[(size_t)res_cot_at];
            if ((int64_t)gr.value.size() == total)
                for (int64_t i = 0; i < total; ++i) {
                    call.dc->at(0).value[(size_t)i] += gr.value[(size_t)i];
                    dres.value[(size_t)i] += gr.value[(size_t)i];
                }
        }
    }
    if (const char* dq = std::getenv("R97_DUMP_QUANT"); dq != nullptr && dq[0] == '1') {
        double sd = 0.0;
        for (double v : call.dc->at(0).value) sd += std::fabs(v);
        std::fprintf(stderr, "QUANT %s dc0=%.6g\n", call.rec.op.c_str(), sd);
        std::fflush(stderr);
    }
}

void had_device_adjoint(const AdjointCall& call) { quant_adjoint(call, Part::quant); }
void rmsnorm_device_adjoint(const AdjointCall& call) { quant_adjoint(call, Part::rmsnorm); }
void gated_device_adjoint(const AdjointCall& call) { quant_adjoint(call, Part::gated); }
void gate_device_adjoint(const AdjointCall& call) { quant_adjoint(call, Part::gate); }

}  // namespace

R97_REGISTER_ADJOINT_IF("had_quant_act_i8g", device, had_device_adjoint, rotated)
R97_REGISTER_ADJOINT_IF_F("rmsnorm_had_quant_i8g", device, rmsnorm_device_adjoint, rotated, 1)
R97_REGISTER_ADJOINT_IF("gated_had_quant_i8g", device, gated_device_adjoint, serves_gated)
R97_REGISTER_ADJOINT_IF("gate_had_quant_i8g", device, gate_device_adjoint, serves_gate)

}  // namespace r97
