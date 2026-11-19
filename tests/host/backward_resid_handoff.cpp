// tests/host/backward_resid_handoff.cpp -- the residual handoff across a block boundary (issue 55).
//
// A fused norm (`rmsnorm_had_quant_i8g`) reads the branch, reads the residual stream in place,
// writes the sum v = branch + residual_old back into the residual, and normalises the sum
// (third_party/radiance/libref/ref_quant.cpp:149-206). The last such write of a block IS the
// block's residual output; the next block's first fused norm reads it. On the live container the
// walk's `SPLIT` print (src/adjoint/host/elementwise.cpp:712-716, R97_GAIN_POS=1) shows the
// ffn norm's residual write receives an incoming cotangent of exactly zero at every block
// boundary (docs/reports/2026-11-18-device-backward-55.md:320-336), while the attn norm's does
// not: the accumulated residual cotangent is discarded as the chain crosses each block, the x0.3
// per-block decay.
//
// The mechanism, and what this tape reproduces of it. An in-place write's cotangent travels to
// the version its read segments name (src/backward.cpp:1030-1048): `seg.version >= 0` adds to the
// version, otherwise the cotangent leaves the block as one of its inputs (`add_input_full`). The
// next block's attn norm reads the residual with no earlier writer in its own block -- the replay
// serves that read from the block-boundary store (DEVRESTORE) because the engine reuses its
// arena -- so its read segment names the block input, and the cotangent leaves the block through
// the input path. The join then hands the block below ONE input cotangent
// (src/step_device.cpp:1593-1599, the pick at src/join_common.h:343-364): a boundary-sized
// branch output and the boundary-sized residual input both qualify, and the pick keeps the first
// it finds, so the incoming lands on the block's branch write
// (src/backward.cpp:381-410's `enter` adds to the version live at the block's end) instead of
// the ffn norm's residual write. That write has no reader left in the version graph, its slot is
// zero, and the row's output is the norm's own term alone.
//
// The constraint on the fix (docs/reports/2026-11-18-device-backward-55.md:348-357): the
// boundary store stays -- the in-place read is served from it correctly -- and the fix credits
// the writer's version IN ADDITION to `add_input_full`. A fixed walk hands the ffn norm's
// residual write the next block's residual input cotangent (here, block B's input "x", printed
// below as the want); this test pins the DEFECT's observable -- the write's slot is exactly
// zero -- and therefore fails once the fix lands. Flipping it is the fix's success criterion:
// the zero assertion becomes an equality against block B's "x" input.
//
// The rows are stand-ins for the fused norm's slot rule (src/adjoint/host/elementwise.cpp:
// 674-698, gain 1): v is the in-place operand's snapshot, which the tape takes after the issue
// and so already IS the sum, and both slots receive s*dy - s^3*v*t/n plus the residual write's
// own incoming cotangent. Only the walk is under test.

#include "r97/backward.h"

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "support/test.h"

namespace {

using r97::AdValue;
using r97::OpRecord;
using r97::Operand;
using r97::Seed;
using r97::IncomingCotangent;

const std::vector<int64_t> kShape{3, 2};
const double kEps = 1e-6;

// ------------------------------------------------------------------ the rows

// What SPLIT names l1_cot_res, captured per record position: the residual write's incoming
// cotangent exactly as the walk hands it to the row.
std::map<int64_t, std::vector<double>> g_cot_res;

void fused_norm_adjoint(const r97::AdjointCall& call) {
    if (call.cot.empty() || call.dc->empty()) return;
    int64_t res_at = -1, res_cot = -1, writes = 0;
    for (size_t j = 0; j < call.rec.operands.size(); ++j)
        if (call.rec.operands[j].written) {
            if (call.rec.operands[j].in_place) {
                res_at = static_cast<int64_t>(j);
                res_cot = writes;
            }
            ++writes;
        }
    if (res_at < 0 || res_cot < 0) return;
    const AdValue& v = call.fwd[static_cast<size_t>(res_at)];  // the post-issue snapshot: the sum
    const AdValue& dy = call.cot.back();                       // the normed output's write
    const AdValue& cot_res = call.cot[static_cast<size_t>(res_cot)];
    g_cot_res[call.rec.position] = cot_res.value;
    const int64_t n = v.shape.back();
    const int64_t rows = static_cast<int64_t>(v.value.size()) / n;
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t b = r * n;
        double ss = 0.0, t = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            ss += v.value[static_cast<size_t>(b + i)] * v.value[static_cast<size_t>(b + i)];
            t += dy.value[static_cast<size_t>(b + i)] * v.value[static_cast<size_t>(b + i)];
        }
        const double s = 1.0 / std::sqrt(ss / (double)n + kEps);
        for (int64_t i = 0; i < n; ++i) {
            const double d = s * dy.value[static_cast<size_t>(b + i)] -
                             s * s * s * v.value[static_cast<size_t>(b + i)] * t / (double)n +
                             cot_res.value[static_cast<size_t>(b + i)];
            (*call.dc)[0].value[static_cast<size_t>(b + i)] += d;
            (*call.dc)[static_cast<size_t>(res_at)].value[static_cast<size_t>(b + i)] += d;
        }
    }
}
R97_REGISTER_ADJOINT("bwd_fused_norm", host, fused_norm_adjoint)

// a -> m, the block's mlp stand-in: its write is the block's branch output.
void copy_adjoint(const r97::AdjointCall& call) {
    if (call.cot.empty() || call.dc->empty()) return;
    for (size_t i = 0; i < (*call.dc)[0].value.size() && i < call.cot[0].value.size(); ++i)
        (*call.dc)[0].value[i] += call.cot[0].value[i];
}
R97_REGISTER_ADJOINT("bwd_copy", host, copy_adjoint)

// ------------------------------------------------------------------ the tape

Operand read(const std::string& buffer) {
    Operand o;
    o.buffer = buffer;
    o.dtype = "f32";
    o.shape = kShape;
    o.written = false;
    return o;
}

Operand in_out(const std::string& buffer) {
    Operand o = read(buffer);
    o.written = true;
    o.in_place = true;  // the engine's RAD_OPD_INOUT
    return o;
}

Operand written(const std::string& buffer) {
    Operand o = read(buffer);
    o.written = true;
    return o;
}

OpRecord op(const std::string& name, std::vector<Operand> operands, int64_t position) {
    OpRecord rec;
    rec.op = name;
    rec.operands = std::move(operands);
    rec.position = position;
    return rec;
}

double l1(const std::vector<double>& v) {
    double s = 0.0;
    for (double t : v) s += std::fabs(t);
    return s;
}

// The join's rule, copied from src/join_common.h:343-364 and driven as in
// src/step_device.cpp:1593-1599: the largest boundary-shaped input the block left, its first
// rows, becomes the cotangent the block below enters.
bool residual_shape(const r97::InputCotangent& in, int64_t tokens, int64_t hidden) {
    return in.value.size() >= static_cast<size_t>(tokens * hidden) && !in.shape.empty() &&
           in.shape.back() == hidden && in.row_hi >= tokens;
}

const r97::InputCotangent* pick_boundary(const std::vector<r97::InputCotangent>& inputs,
                                         int64_t tokens, int64_t hidden) {
    const r97::InputCotangent* best = nullptr;
    for (const auto& in : inputs)
        if (residual_shape(in, tokens, hidden) &&
            (best == nullptr || in.value.size() > best->value.size()))
            best = &in;
    return best;
}

std::vector<double> first_rows(const r97::InputCotangent& in, int64_t tokens, int64_t hidden) {
    const size_t n = static_cast<size_t>(tokens * hidden);
    if (in.value.size() < n) return {};
    return std::vector<double>(in.value.begin(), in.value.begin() + static_cast<ptrdiff_t>(n));
}

const r97::InputCotangent* input_named(const r97::BackwardResult& got, const std::string& name) {
    for (const auto& in : got.inputs)
        if (in.buffer == name) return &in;
    return nullptr;
}

// ------------------------------------------------------------------ the case

// One synthetic step, two blocks. Block A (the lower block, the walk that must receive the
// handoff): the attn norm (pos 0) and the ffn norm (pos 1) both add their branch to the
// residual in place, and the mlp stand-in (pos 2) writes the branch m. The ffn norm's write is
// the last write of the residual -- the block output. Block B (the block above): its attn norm
// (pos 100) reads m and the same residual buffer in place (the first residual touch of the
// block, so its read segment names the block input), and a copy (pos 101) seeds the walk on z.
R97_TEST(backward_resid_handoff_ffn_write_receives_no_boundary_cot) {
    g_cot_res.clear();
    const int64_t rows = kShape[0];
    const int64_t hidden = kShape[1];

    std::vector<OpRecord> block_a{
        op("bwd_fused_norm", {read("hprev"), in_out("x"), written("yA")}, 0),
        op("bwd_fused_norm", {read("yA"), in_out("x"), written("yB")}, 1),
        op("bwd_copy", {read("yB"), written("m")}, 2),
    };
    std::vector<OpRecord> block_b{
        op("bwd_fused_norm", {read("m"), in_out("x"), written("yC")}, 100),
        op("bwd_copy", {read("yC"), written("z")}, 101),
    };

    // The forward: v = branch + residual, residual := v, out = v * (1/sqrt(mean(v^2)+eps)) per
    // row. The in-place operand's snapshot is taken after the issue, so it carries the sum.
    std::map<std::string, std::vector<double>> env;
    env["hprev"] = {1.0, -2.0, 0.5, 3.0, -1.5, 0.25};
    env["x"] = {0.25, 0.5, -1.0, 0.25, 2.0, -0.75};
    r97::StoredSnapshots snaps;
    const auto snap = [&](int64_t pos, size_t j, const std::string& buffer) {
        AdValue a;
        a.dtype = "f32";
        a.shape = kShape;
        a.value = env[buffer];
        snaps.set(pos, j, std::move(a));
    };
    const auto fused = [&](int64_t pos, const std::string& branch, const std::string& out) {
        snap(pos, 0, branch);
        std::vector<double>& res = env["x"];
        for (size_t i = 0; i < res.size(); ++i) res[i] += env[branch][i];
        std::vector<double> nrm(res.size());
        for (int64_t r = 0; r < rows; ++r) {
            double ss = 0.0;
            for (int64_t i = 0; i < hidden; ++i) {
                const double v = res[static_cast<size_t>(r * hidden + i)];
                ss += v * v;
            }
            const double s = 1.0 / std::sqrt(ss / (double)hidden + kEps);
            for (int64_t i = 0; i < hidden; ++i)
                nrm[static_cast<size_t>(r * hidden + i)] = res[static_cast<size_t>(r * hidden + i)] * s;
        }
        env[out] = std::move(nrm);
        snap(pos, 1, "x");
        snap(pos, 2, out);
    };
    fused(0, "hprev", "yA");
    fused(1, "yA", "yB");
    snap(2, 0, "yB");
    env["m"] = env["yB"];
    snap(2, 1, "m");
    fused(100, "m", "yC");
    snap(101, 0, "yC");
    env["z"] = env["yC"];
    snap(101, 1, "z");

    // Block B's walk, seeded on the step's output z.
    r97::BoundaryNames bnd_b;
    bnd_b.add("z", 0, rows);
    const std::vector<Seed> seeds{{"z", 0, rows, {1.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f}}};
    r97::BackwardOptions bo;
    bo.in_place_passthrough = true;  // both step paths set it (src/step_device.cpp:1135)
    const r97::BackwardResult above =
        r97::backward_block(block_b, snaps, bnd_b, seeds, {}, bo);

    // The boundary path fired: the attn norm's residual read named the block input and its
    // cotangent left the block through `add_input_full`. This holds on this tree and must still
    // hold after the fix -- the boundary store is the reason the read cannot come from the live
    // buffer, and the fix keeps it.
    const r97::InputCotangent* x_in = input_named(above, "x");
    R97_CHECK(x_in != nullptr);
    R97_CHECK(l1(x_in->value) > 0.0);

    // The join, as the device step drives it.
    const r97::InputCotangent* below = pick_boundary(above.inputs, rows, hidden);
    R97_CHECK(below != nullptr);
    const IncomingCotangent incoming{below->buffer, 0, rows, first_rows(*below, rows, hidden)};
    std::printf("JOIN below=%s l1=%.6g | the residual input x left block B with l1=%.6g\n",
                below->buffer.c_str(), l1(below->value), l1(x_in->value));

    // Block A's walk with the joined cotangent.
    r97::BoundaryNames bnd_a;
    bnd_a.add(incoming.buffer, 0, rows);
    r97::backward_block(block_a, snaps, bnd_a, {}, {incoming}, bo);

    const auto it_ffn = g_cot_res.find(1);
    const auto it_attn = g_cot_res.find(0);
    R97_CHECK(it_ffn != g_cot_res.end() && it_ffn->second.size() == static_cast<size_t>(rows * hidden));
    R97_CHECK(it_attn != g_cot_res.end() && it_attn->second.size() == static_cast<size_t>(rows * hidden));
    std::printf(
        "SPLIT pos=1 role=ffn l1_cot_res=%.6g (want the next block's residual cot l1=%.6g)\n"
        "SPLIT pos=0 role=attn l1_cot_res=%.6g (the in-block reader; control)\n",
        l1(it_ffn->second), l1(x_in->value), l1(it_attn->second));

    // The defect, the live container's observable (docs/reports/2026-11-18-device-backward-55.md:
    // 320-336): the ffn norm's residual write receives EXACTLY no cotangent. A fixed walk hands it
    // the next block's residual input, so this assertion is the one that must fail once the fix
    // lands; it then becomes the equality against x_in->value.
    for (double t : it_ffn->second) R97_CHECK(t == 0.0);
    // The in-block handoff works and stays working: the attn norm's write carries its in-block
    // reader's cotangent, which is why the live SPLIT is non-zero at every attn norm.
    R97_CHECK(l1(it_attn->second) > 0.0);
}

}  // namespace
