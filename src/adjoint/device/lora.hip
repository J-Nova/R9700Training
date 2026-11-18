// src/adjoint/device/lora.hip -- the adapter's adjoints on the card, for the rotated int8 trunk.
//
// The delta's trunk is the model's own weights in int8 with a group scale per 128 columns, and the
// adapter reads its input as the value those codes stand for, x = code * scale, in the order the
// card stores the codes: a row's columns in column order, one scale per group of them
// (docs/ADAPTER.md 1; radiance reads the same operand that way in libr4d/r4d_lora_bf16.hip).
//
// Two ops make the adapter, and their host rows are the reference this file answers to
// (docs/adjoints/lora.md):
//
//   lora_shrink   t[m, j] = sum_k x[m, k] * A[s, j, k]
//   lora_expand   y[m, n] = y[m, n] + scale[s, n] * sum_j t[m, j] * B[s, n, j]
//
// The adjoint of the shrink, in f64 on the host:
//
//   g[m, k]        = sum_j cot t[m, j] * A[s, j, k]
//   dx[m, k]       = g[m, k]                                   (a float activation)
//   dcode[m, k]    = g[m, k] * scale[m, k/group]               (the coded form)
//   dscale[m, grp] += g[m, k] * code[m, k]                     (over k in the group)
//   dA[j, k]       += cot t[m, j] * x[m, k]                    (over the step's valid rows)
//
// and of the expand:
//
//   dt[m, j]  += sum_n cot y[m, n] * scale[s, n] * B[s, n, j]
//   dB[n, j]  += sum_{m: slot[m] = s} cot y[m, n] * scale[s, n] * t[m, j]
//
// A base row -- a slot below 0, or one past the pool -- takes no cotangent and gives no gradient,
// as on the host. The gradients are leaves of the walk's ParamSet, not operands of the record: this
// kernel adds dA and dB into the step's f32 accumulators on the card (grad_stage.h), over every
// token of every call of the step, and touches no leaf of the ParamSet. The step's total leaves the
// card when the walk ends it, rounded once to bf16 -- the dtype a gradient has on the card -- by
// grad_stage_flush. The leaves are named and shaped the way the host rows name them for the call's
// naming of the pool's factors (grad_dest below, docs/BACKWARD.md "Slot names"): one factor the
// pool rows' blocks share under the prefix, one leaf per named pool row, or the one leaf the
// record's rows sum into.
//
// What the caller hands each call (the record's operand order, reads first):
//
//   lora_shrink   [0] a [tokens, K] codes        [1] a_scale [tokens, K/group]   [2] slot [tokens]
//                 [3] lora_a [S, R, K]           [4] t [tokens, R] written
//   lora_expand   [0] t [tokens, R]              [1] slot [tokens]   [2] lora_b [S, N, R]
//                 [3] scale [S, N]               [4] y [tokens, N] written
//
// What this row serves, and what it does not:
//
//  * the coded trunk, i.e. a record whose `dtype` names a coded activation and which carries an
//    `a_scale`; a float activation's shrink is left to the host row, which needs no scale at all.
//    The code cotangent is not a code -- a code's cotangent is a real number -- so the kernel stages
//    it in f32 and hands it back as it is instead of rounding it to the operand's i8 dtype.
//  * `lora_expand`, not `lora_expand_mm`: the two differ only in the forward, where the mm form
//    rounds t to bf16 before the product, and that rounding is not differentiated.
//  * a pool of R columns a row can hold in shared memory (1024): the real packing is 16 or 32.
//
// The memory: every buffer comes from the scratch allocator, a row of a token's cotangent is
// reduced in f32, and a token's R values live in shared memory so that a row of A is read once per
// token rather than once per (token, j).

#include "r97/adjoint.h"

#include "grad_stage.h"
#include "scratch.h"

#include "rad_device.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace r97 {
namespace {

constexpr int kBlock = 256;            // threads in a block; one column each where a kernel walks K
constexpr int kMaxJ = 32;              // dA rows a thread accumulates in registers at once
constexpr int64_t kRowsChunk = 64;     // token rows one dA block sums before adding its partials
constexpr int64_t kMaxRShared = 1024;  // pool columns a token row can hold in shared memory

int64_t param_int(const std::vector<OpParam>& params, const char* name) {
    for (const OpParam& p : params) {
        if (p.first != name) continue;
        if (const auto* i = std::get_if<int64_t>(&p.second)) return *i;
    }
    return 0;
}

// Whether the record's activation is a code stream: the `dtype` parameter names one of the coded
// dtypes radiance reads `a` as, so the record has an `a_scale` and the schema is one wider.
bool coded_act(const std::vector<OpParam>& params) {
    for (const OpParam& p : params) {
        if (p.first != "dtype") continue;
        const auto* d = std::get_if<std::string>(&p.second);
        if (!d) return false;
        return *d == "i8" || *d == "fp8_e4m3" || *d == "fp8_e5m2";
    }
    return false;
}

// The width of one scale's group, which is not the record's `group`: the stream's column grid is
// what make_scales turns back into a width, and radiance refuses a record whose `group` divides K
// into a different number of groups. The kernel reads the partial last group the same way.
int64_t group_width(int64_t k, int64_t groups) {
    if (k <= 0) return 1;
    if (groups <= 1) return k;
    const int64_t lo = (k + groups - 1) / groups;
    int64_t b = 1;
    while (b < lo) b <<= 1;
    return (k + b - 1) / b == groups ? b : lo;
}

std::string grad_name(const char* prefix, const char* leaf) {
    if (!prefix || !*prefix) return leaf;
    return std::string(prefix) + "." + leaf;
}

// The slot's trainable factor goes under the record's fused name when the caller splits a fused
// factor afterwards, under the slot's module when the walk names modules, and under the record's
// prefix otherwise. The host rows of docs/adjoints/lora.md name it the same way.
std::string factor_grad_name(const AdjointCall& call, const char* leaf, int64_t slot) {
    if (call.fused_slots) return grad_name(call.prefix, leaf);
    if (call.slot_prefix) return grad_name(call.slot_prefix(slot).c_str(), leaf);
    return grad_name(call.prefix, leaf);
}

// The step's accumulator a pool row's rows land in, one per pool row: the row's block of the
// fused factor, the row's own leaf, or the one leaf the record's rows sum into. A pool row the
// record's rows never name gets no accumulator -- the host rows make no leaf for it, and the
// step's flush would make one out of the stage. `stride` is the distance in elements between two
// pool rows' blocks of the fused factor (R*K for A, R for B); 0 outside the fused naming. A slot
// the step's stage refused comes out nullptr, and the row's launch skips it.
std::vector<float*> grad_dest(const AdjointCall& call, const char* leaf, int64_t S,
                              const std::vector<char>& present, const std::vector<int64_t>& shape,
                              int64_t stride) {
    std::vector<float*> dest((size_t)S, nullptr);
    if (!call.grads) return dest;
    if (call.fused_slots) {
        float* base = device::grad_stage_slot(factor_grad_name(call, leaf, 0), shape);
        for (int64_t s = 0; base && s < S; ++s)
            if (present[(size_t)s]) dest[(size_t)s] = base + (size_t)s * (size_t)stride;
        return dest;
    }
    if (call.slot_prefix) {
        for (int64_t s = 0; s < S; ++s) {
            if (!present[(size_t)s]) continue;
            dest[(size_t)s] = device::grad_stage_slot(factor_grad_name(call, leaf, s), shape);
        }
        return dest;
    }
    float* base = device::grad_stage_slot(factor_grad_name(call, leaf, 0), shape);
    for (int64_t s = 0; base && s < S; ++s)
        if (present[(size_t)s]) dest[(size_t)s] = base;
    return dest;
}

// One stream for every device adjoint call in this process: upload, launch and download are one
// queue's worth of work, which is what the bench times.
RadStream adjoint_stream() {
    static RadStream s = [] {
        RadStream created = nullptr;
        if (rad_stream_create(&created, 0) != 0) return (RadStream) nullptr;
        return created;
    }();
    return s;
}

int64_t blocks(int64_t n) { return (n + kBlock - 1) / kBlock; }

int64_t f32_bytes(int64_t n) { return n * (int64_t)sizeof(float); }
int64_t f64_bytes(int64_t n) { return n * (int64_t)sizeof(double); }

// The record's operands, in the schema's order: a, a_scale, slot, lora_a, t. Only the coded form is
// this row's: a float activation leaves `a_scale` absent, in place, and has no device row.
struct Shrink {
    const AdValue* a = nullptr;
    const AdValue* as = nullptr;
    const AdValue* slot = nullptr;
    const AdValue* la = nullptr;
    int64_t M = 0, K = 0, R = 0, S = 0;
    int64_t groups = 0, group = 0;
};

bool shrink_layout(const OpRecord& rec, const AdValue* v, size_t n, Shrink* sh) {
    if (!coded_act(rec.params)) return false;
    if (n != 4u && n != 5u) return false;
    const AdValue& a = v[0];
    const AdValue& held = v[1];  // a_scale
    const AdValue& slot = v[2];
    const AdValue& la = v[3];
    if (a.shape.size() != 2 || slot.shape.size() != 1 || la.shape.size() != 3) return false;
    if (held.dtype == "invalid") return false;  // a float activation: the host row's
    if (held.shape.size() != 2 || held.shape[0] <= 0 || held.shape[1] <= 0) return false;
    const int64_t M = a.shape[0], K = a.shape[1], S = la.shape[0], R = la.shape[1];
    if (M <= 0 || K <= 0 || R <= 0 || S <= 0) return false;
    if (la.shape[2] != K || slot.shape[0] != M) return false;
    if ((int64_t)a.value.size() != M * K || (int64_t)slot.value.size() != M ||
        (int64_t)la.value.size() != S * R * K)
        return false;
    if (held.shape[0] != M || (int64_t)held.value.size() != M * held.shape[1]) return false;
    sh->a = &a;
    sh->as = &held;
    sh->slot = &slot;
    sh->la = &la;
    sh->M = M;
    sh->K = K;
    sh->R = R;
    sh->S = S;
    sh->groups = held.shape[1];
    sh->group = group_width(K, sh->groups);
    // The record's `group` is a consistency statement, not the width the kernel reads with.
    const int64_t stated = param_int(rec.params, "group");
    if (stated > 0 && (K + stated - 1) / stated != (K + sh->group - 1) / sh->group) return false;
    return true;
}

// The record's operands, in the schema's order: t, slot, lora_b, scale, y.
struct Expand {
    const AdValue* t = nullptr;
    const AdValue* slot = nullptr;
    const AdValue* lb = nullptr;
    const AdValue* sc = nullptr;
    int64_t M = 0, N = 0, R = 0, S = 0;
};

bool expand_layout(const AdValue* v, size_t n, Expand* ex) {
    if (n != 4u && n != 5u) return false;
    const AdValue& t = v[0];
    const AdValue& slot = v[1];
    const AdValue& lb = v[2];
    const AdValue& sc = v[3];
    if (t.shape.size() != 2 || slot.shape.size() != 1 || lb.shape.size() != 3 ||
        sc.shape.size() != 2)
        return false;
    const int64_t M = t.shape[0], R = t.shape[1], S = lb.shape[0], N = lb.shape[1];
    if (M <= 0 || N <= 0 || R <= 0 || S <= 0) return false;
    if (lb.shape[2] != R || slot.shape[0] != M || sc.shape[0] != S || sc.shape[1] != N) return false;
    if ((int64_t)t.value.size() != M * R || (int64_t)slot.value.size() != M ||
        (int64_t)lb.value.size() != S * N * R || (int64_t)sc.value.size() != S * N)
        return false;
    ex->t = &t;
    ex->slot = &slot;
    ex->lb = &lb;
    ex->sc = &sc;
    ex->M = M;
    ex->N = N;
    ex->R = R;
    ex->S = S;
    return true;
}

// A slot as the card wants it: the pool row 0..S-1, or -1 for a base row. The whole array is built
// once on the host and read by every kernel of the call.
void fill_slots(const AdValue& slot, int64_t S, std::vector<int>* out) {
    out->resize(slot.value.size());
    for (size_t m = 0; m < slot.value.size(); ++m) {
        const double v = slot.value[m];
        const int64_t s = (v >= 0.0) ? (int64_t)v : -1;
        (*out)[m] = (s >= 0 && s < S) ? (int)s : -1;
    }
}

// -- the kernels -------------------------------------------------------------

// The value-space cotangent of a token's row, and the split of it the coded form takes: one block
// per token, one thread per column. The row of cot t goes to shared memory when it fits, so a row
// of A is read once per (token, column) instead of once per (token, column, j); a pool too wide for
// the shared row reads it from global, which costs the reads and keeps the row served.
__global__ void shrink_dx_kernel(const float* __restrict__ a, const float* __restrict__ as,
                                 const float* __restrict__ ct, const float* __restrict__ la,
                                 const int* __restrict__ slot, double* __restrict__ dx,
                                 double* __restrict__ sct, int64_t M, int64_t K, int64_t R,
                                 int64_t groups, int64_t group, int64_t row_in_shared) {
    const int64_t m = blockIdx.x;
    if (m >= M) return;
    const int64_t s = slot[m];
    if (s < 0) return;  // a base row: no cotangent
    const int t = threadIdx.x;
    extern __shared__ float sh[];   // R, the token's cotangent, when it fits
    __shared__ double sh_p[kBlock]; // one column's product with its code
    const float* ct_row = ct + m * R;
    if (row_in_shared) {
        for (int64_t j = t; j < R; j += blockDim.x) sh[j] = ct_row[j];
        __syncthreads();
        ct_row = sh;
    }
    const float* scale_row = as + m * groups;
    const float* A = la + s * R * K;
    const float* a_row = a + m * K;
    double* dx_row = dx + m * K;
    double* sct_row = sct + m * groups;
    // f64 throughout, as the host row is (`host/reference_lora.cpp:292`): the cotangents this row
    // is handed reach 1e36 at block 50, where a group's sum of `g * code` over 128 columns lands
    // on f32's 3.4e38 ceiling and saturates -- measured as the first non-finite in the whole walk
    // (`lora_shrink`, operand `h.s8`, position 1150, maxabs 3.40178e+38). `g` is a reduction over
    // R and is the same argument one level down; the codes' own output `dx` sat at 1.6e36.
    for (int64_t base = 0; base < K; base += blockDim.x) {
        const int64_t k = base + t;
        double p = 0.0;
        if (k < K) {
            double g = 0.0;
            for (int64_t j = 0; j < R; ++j) g += (double)ct_row[j] * (double)A[j * K + k];
            dx_row[k] = g * (double)scale_row[k / group];
            p = g * (double)a_row[k];
        }
        sh_p[t] = p;
        __syncthreads();
        // One thread per group this tile reaches, summing the tile's part of it: the group is a
        // run of columns, so a tile of columns holds the runs it overlaps whole or in part.
        const int64_t k0 = base;
        const int64_t k1 = min(base + (int64_t)blockDim.x, K);
        const int64_t g_first = k0 / group;
        const int64_t g_last = (k1 - 1) / group;
        for (int64_t g = g_first + t; g <= g_last; g += blockDim.x) {
            const int64_t lo = (g * group > k0 ? g * group : k0) - k0;
            const int64_t hi = min(k1, min((g + 1) * group, K)) - k0;
            double sum = 0.0;
            for (int64_t i = lo; i < hi; ++i) sum += sh_p[i];
            atomicAdd(&sct_row[g], sum);
        }
        __syncthreads();
    }
}

// dA = sum over the step's valid rows of cot t[m, j] * x[m, k]: one thread per column, its rows'
// sums in registers, its partials added to the f32 staging once per row chunk. The row launches
// it once per pool row it names, filtering the tokens to that row's, so a launch writes only the
// rows its slot's own: the buffer the row asked the step's stage for, at the slot's block of the
// fused factor when the call has one.
__global__ void shrink_da_kernel(const float* __restrict__ a, const float* __restrict__ as,
                                 const float* __restrict__ ct, const int* __restrict__ slot,
                                 float* __restrict__ dA, int64_t M, int64_t K, int64_t R,
                                 int64_t groups, int64_t group, int64_t rows_chunk, int s) {
    const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const int64_t m0 = (int64_t)blockIdx.y * rows_chunk;
    const int64_t m1 = min(m0 + rows_chunk, M);
    const int64_t grp = k / group;
    for (int64_t j0 = 0; j0 < R; j0 += kMaxJ) {
        const int64_t jn = min((int64_t)kMaxJ, R - j0);
        float acc[kMaxJ];
        for (int i = 0; i < kMaxJ; ++i) acc[i] = 0.0f;
        for (int64_t m = m0; m < m1; ++m) {
            if (slot[m] != s) continue;
            const float xv = a[m * K + k] * as[m * groups + grp];
            const float* ctm = ct + m * R + j0;
            for (int64_t i = 0; i < jn; ++i) acc[i] += ctm[i] * xv;
        }
        for (int64_t i = 0; i < jn; ++i) {
            if (acc[i] != 0.0f) atomicAdd(&dA[(j0 + i) * K + k], acc[i]);
        }
    }
}

// dt[m, j] = sum_n cot y[m, n] * scale[s, n] * B[s, n, j]: one thread per (token, j), walking the
// projection's columns in order, so a warp reads B's rows as runs.
__global__ void expand_dt_kernel(const float* __restrict__ cy, const float* __restrict__ lb,
                                 const float* __restrict__ sc, const int* __restrict__ slot,
                                 double* __restrict__ dt, int64_t M, int64_t N, int64_t R) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * R) return;
    const int64_t m = i / R, j = i - m * R;
    const int64_t s = slot[m];
    if (s < 0) {
        dt[i] = 0.0;  // a base row: no cotangent
        return;
    }
    const float* cy_row = cy + m * N;
    const float* sc_row = sc + s * N;
    const float* B = lb + s * N * R + j;
    // f64: this is a reduction over the projection's N columns -- 17408 of them on the live trunk
    // -- where the host accumulates in double (`host/reference_lora.cpp:435`). At block 50's
    // magnitudes, a y-cotangent of 1e36 against a scale of 0.19, the f32 sum lands on the ceiling:
    // the whole of `lora.t.r32.lane0` came out non-finite at position 1138 (16384 of 16384)
    // before this was widened.
    double acc = 0.0;
    for (int64_t n = 0; n < N; ++n)
        acc += (double)cy_row[n] * (double)sc_row[n] * (double)B[n * R];
    dt[i] = acc;
}

// dB[n, j] = sum over the step's valid rows of cot y[m, n] * scale[s, n] * t[m, j]: one thread per
// (row of the projection, j), each its own output, the tokens walked in order. The row launches
// it once per pool row it names, into that slot's columns of the buffer the row asked the step's
// stage for (`cols` of them per row, the slot's block of the fused factor when the call has one).
__global__ void expand_db_kernel(const float* __restrict__ cy, const float* __restrict__ t,
                                 const float* __restrict__ sc, const int* __restrict__ slot,
                                 float* __restrict__ db, int64_t M, int64_t N, int64_t R,
                                 int64_t cols, int s) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * R) return;
    const int64_t n = i / R, j = i - n * R;
    const float* sc_row = sc + s * N;
    float acc = 0.0f;
    for (int64_t m = 0; m < M; ++m) {
        if (slot[m] != s) continue;
        acc += cy[m * N + n] * sc_row[n] * t[m * R + j];
    }
    db[n * cols + j] += acc;  // the step's accumulator: this call's sum adds to the calls before it
}

// -- the plumbing ------------------------------------------------------------

bool upload_f32(const AdValue& v, void* dst, int64_t n, std::vector<float>* stage, RadStream stream) {
    stage->resize((size_t)n);
    for (int64_t i = 0; i < n; ++i) (*stage)[(size_t)i] = (float)v.value[(size_t)i];
    return rad_memcpy_async(dst, stage->data(), f32_bytes(n), stream) == 0;
}

bool upload_slots(const std::vector<int>& slots, void* dst, RadStream stream) {
    return rad_memcpy_async(dst, slots.data(), (int64_t)(slots.size() * sizeof(int)), stream) == 0;
}

// Adds an f32 buffer the kernel filled to a slot of the cotangent, in the value space the host row
// works in.
bool download_add_f32(AdValue* slot, void* src, std::vector<float>* stage, RadStream stream) {
    if (!slot) return true;
    const int64_t n = (int64_t)slot->value.size();
    stage->resize((size_t)n);
    if (rad_memcpy_async(stage->data(), src, f32_bytes(n), stream) != 0) return false;
    if (rad_stream_sync(stream) != 0) return false;
    for (int64_t i = 0; i < n; ++i) slot->value[(size_t)i] += (double)(*stage)[(size_t)i];
    return true;
}

// The same for an f64 buffer: the rows that can carry a cotangent past f32's ceiling stage in
// double, so a sum that would overflow f32 is taken before anything is narrowed.
bool download_add_f64(AdValue* slot, void* src, std::vector<double>* stage, RadStream stream) {
    if (!slot) return true;
    const int64_t n = (int64_t)slot->value.size();
    stage->resize((size_t)n);
    if (rad_memcpy_async(stage->data(), src, f64_bytes(n), stream) != 0) return false;
    if (rad_stream_sync(stream) != 0) return false;
    for (int64_t i = 0; i < n; ++i) slot->value[(size_t)i] += (*stage)[(size_t)i];
    return true;
}

// The shrink's adjoint: the coded form only. The row of cot t in shared memory when the pool's
// width fits it, and one block per token for the value-space cotangent.
void lora_shrink_device_adjoint(const AdjointCall& call) {
    Shrink sh;
    if (!shrink_layout(call.rec, call.fwd.data(), call.fwd.size(), &sh)) return;
    if (call.cot.empty() || !call.dc || call.dc->size() < 2) return;
    const AdValue& t_cot = call.cot[0];
    if ((int64_t)t_cot.value.size() != sh.M * sh.R) return;
    AdValue& a_cot = call.dc->at(0);
    AdValue& s_cot = call.dc->at(1);
    if ((int64_t)a_cot.value.size() != sh.M * sh.K) return;
    if ((int64_t)s_cot.value.size() != sh.M * sh.groups) return;

    RadStream stream = adjoint_stream();
    if (!stream) return;

    // dA belongs to the step, not to the call: this row adds the call's sum into the f32
    // accumulator the step is holding on the card, and the walk copies the step's total out when
    // the step ends (grad_stage.h). Nothing of the leaf is touched here. The accumulator is the
    // one the naming gives the slot (grad_dest), one per pool row the record's rows name, and
    // each launch below writes the rows its slot's own.
    std::vector<int> slots;
    fill_slots(*sh.slot, sh.S, &slots);
    std::vector<char> present((size_t)sh.S, 0);
    for (int v : slots)
        if (v >= 0) present[(size_t)v] = 1;
    const std::vector<int64_t> a_shape = call.fused_slots
                                             ? std::vector<int64_t>{sh.S * sh.R, sh.K}
                                             : std::vector<int64_t>{sh.R, sh.K};
    std::vector<float*> d_da = grad_dest(call, "lora_A", sh.S, present, a_shape, sh.R * sh.K);
    for (int64_t s = 0; call.grads && s < sh.S; ++s)
        if (present[(size_t)s] && !d_da[(size_t)s]) return;  // the step's stage refused the leaf

    device::ScratchBlock d_a(f32_bytes(sh.M * sh.K));
    device::ScratchBlock d_as(f32_bytes(sh.M * sh.groups));
    device::ScratchBlock d_ct(f32_bytes(sh.M * sh.R));
    device::ScratchBlock d_la(f32_bytes(sh.S * sh.R * sh.K));
    device::ScratchBlock d_slot((int64_t)(sh.M * (int64_t)sizeof(int)));
    device::ScratchBlock d_dx(f64_bytes(sh.M * sh.K));
    device::ScratchBlock d_sct(f64_bytes(sh.M * sh.groups));
    if (!d_a || !d_as || !d_ct || !d_la || !d_slot || !d_dx || !d_sct) return;

    std::vector<float> h_a, h_as, h_ct, h_la;
    if (!upload_f32(*sh.a, d_a.p, sh.M * sh.K, &h_a, stream)) return;
    if (!upload_f32(*sh.as, d_as.p, sh.M * sh.groups, &h_as, stream)) return;
    if (!upload_f32(t_cot, d_ct.p, sh.M * sh.R, &h_ct, stream)) return;
    if (!upload_f32(*sh.la, d_la.p, sh.S * sh.R * sh.K, &h_la, stream)) return;
    if (!upload_slots(slots, d_slot.p, stream)) return;
    if (rad_memset_async(d_dx.p, 0, f64_bytes(sh.M * sh.K), stream) != 0) return;
    if (rad_memset_async(d_sct.p, 0, f64_bytes(sh.M * sh.groups), stream) != 0) return;

    const int64_t row_in_shared = sh.R <= kMaxRShared ? 1 : 0;
    const int64_t shared = row_in_shared ? sh.R * (int64_t)sizeof(float) : 0;
    hipLaunchKernelGGL(shrink_dx_kernel, dim3((unsigned int)sh.M), dim3(kBlock),
                       (size_t)shared, (hipStream_t)stream, (const float*)d_a.p,
                       (const float*)d_as.p, (const float*)d_ct.p, (const float*)d_la.p,
                       (const int*)d_slot.p, (double*)d_dx.p, (double*)d_sct.p, sh.M, sh.K, sh.R,
                       sh.groups, sh.group, row_in_shared);
    const int64_t chunks = (sh.M + kRowsChunk - 1) / kRowsChunk;
    for (int64_t s = 0; s < sh.S; ++s) {
        if (!d_da[(size_t)s]) continue;
        hipLaunchKernelGGL(shrink_da_kernel, dim3((unsigned int)blocks(sh.K), (unsigned int)chunks),
                           dim3(kBlock), 0, (hipStream_t)stream, (const float*)d_a.p,
                           (const float*)d_as.p, (const float*)d_ct.p, (const int*)d_slot.p,
                           d_da[(size_t)s], sh.M, sh.K, sh.R, sh.groups, sh.group, kRowsChunk,
                           (int)s);
    }
    if (rad_stream_sync(stream) != 0) return;

    std::vector<double> h_dx, h_sct;
    if (!download_add_f64(&a_cot, d_dx.p, &h_dx, stream)) return;
    if (!download_add_f64(&s_cot, d_sct.p, &h_sct, stream)) return;
}

// The expand's adjoint: the cotangent of t over the projection's columns, and dB over the tokens.
void lora_expand_device_adjoint(const AdjointCall& call) {
    Expand ex;
    if (!expand_layout(call.fwd.data(), call.fwd.size(), &ex)) return;
    if (call.cot.empty() || !call.dc || call.dc->empty()) return;
    const AdValue& y_cot = call.cot[0];
    if ((int64_t)y_cot.value.size() != ex.M * ex.N) return;
    AdValue& t_cot = call.dc->at(0);
    if ((int64_t)t_cot.value.size() != ex.M * ex.R) return;

    // The direct path of an in-place write, as the host row fills it (src/adjoint/host/lora.cpp):
    // the buffer the write replaces reads its own cotangent back, unchanged. The walk's passthrough
    // fills an empty slot with the same value, so this changes nothing in a walk -- it makes the
    // row say the same thing when a check calls it without one, which is how the aliased records
    // are held to a finite difference.
    for (size_t i = 0; i < call.rec.operands.size(); ++i) {
        const Operand& o = call.rec.operands[i];
        if (!o.written || !o.in_place) continue;
        if (i >= call.dc->size()) continue;
        AdValue& d_base = call.dc->at(i);
        if (d_base.value.size() == y_cot.value.size())
            for (size_t e = 0; e < y_cot.value.size(); ++e) d_base.value[e] += y_cot.value[e];
    }

    RadStream stream = adjoint_stream();
    if (!stream) return;

    // dB belongs to the step, not to the call, the way dA does (grad_stage.h): the accumulator
    // the naming gives the slot (grad_dest), one per pool row the record's rows name, and each
    // launch below writes the columns its slot's own.
    std::vector<int> slots;
    fill_slots(*ex.slot, ex.S, &slots);
    std::vector<char> present((size_t)ex.S, 0);
    for (int v : slots)
        if (v >= 0) present[(size_t)v] = 1;
    const std::vector<int64_t> b_shape = call.fused_slots
                                             ? std::vector<int64_t>{ex.N, ex.S * ex.R}
                                             : std::vector<int64_t>{ex.N, ex.R};
    std::vector<float*> d_db = grad_dest(call, "lora_B", ex.S, present, b_shape, ex.R);
    for (int64_t s = 0; call.grads && s < ex.S; ++s)
        if (present[(size_t)s] && !d_db[(size_t)s]) return;  // the step's stage refused the leaf

    device::ScratchBlock d_cy(f32_bytes(ex.M * ex.N));
    device::ScratchBlock d_t(f32_bytes(ex.M * ex.R));
    device::ScratchBlock d_lb(f32_bytes(ex.S * ex.N * ex.R));
    device::ScratchBlock d_sc(f32_bytes(ex.S * ex.N));
    device::ScratchBlock d_slot((int64_t)(ex.M * (int64_t)sizeof(int)));
    device::ScratchBlock d_dt(f64_bytes(ex.M * ex.R));
    if (!d_cy || !d_t || !d_lb || !d_sc || !d_slot || !d_dt) return;

    std::vector<float> h_cy, h_t, h_lb, h_sc;
    if (!upload_f32(y_cot, d_cy.p, ex.M * ex.N, &h_cy, stream)) return;
    if (!upload_f32(*ex.t, d_t.p, ex.M * ex.R, &h_t, stream)) return;
    if (!upload_f32(*ex.lb, d_lb.p, ex.S * ex.N * ex.R, &h_lb, stream)) return;
    if (!upload_f32(*ex.sc, d_sc.p, ex.S * ex.N, &h_sc, stream)) return;
    if (!upload_slots(slots, d_slot.p, stream)) return;
    if (rad_memset_async(d_dt.p, 0, f64_bytes(ex.M * ex.R), stream) != 0) return;

    hipLaunchKernelGGL(expand_dt_kernel, dim3((unsigned int)blocks(ex.M * ex.R)),
                       dim3(kBlock), 0, (hipStream_t)stream, (const float*)d_cy.p,
                       (const float*)d_lb.p, (const float*)d_sc.p, (const int*)d_slot.p,
                       (double*)d_dt.p, ex.M, ex.N, ex.R);
    const int64_t b_cols = call.fused_slots ? ex.S * ex.R : ex.R;
    for (int64_t s = 0; s < ex.S; ++s) {
        if (!d_db[(size_t)s]) continue;
        hipLaunchKernelGGL(expand_db_kernel, dim3((unsigned int)blocks(ex.N * ex.R)),
                           dim3(kBlock), 0, (hipStream_t)stream, (const float*)d_cy.p,
                           (const float*)d_t.p, (const float*)d_sc.p, (const int*)d_slot.p,
                           d_db[(size_t)s], ex.M, ex.N, ex.R, b_cols, (int)s);
    }
    if (rad_stream_sync(stream) != 0) return;

    std::vector<double> h_dt;
    if (!download_add_f64(&t_cot, d_dt.p, &h_dt, stream)) return;
}

}  // namespace

// The coded trunk's form of the shrink, whose two operands are the code and its group scale; the
// float activation's shrink is the host row's. `slot` and `lora_a` are frozen, as on the host.
R97_REGISTER_ADJOINT_IF_F("lora_shrink", device, lora_shrink_device_adjoint, coded_act, 2, 3)
// The slot, the pool's B and the per-row scale are frozen, as on the host.
R97_REGISTER_ADJOINT_FF("lora_expand", device, lora_expand_device_adjoint, fills_in_place_slots, 1, 2, 3)

}  // namespace r97
