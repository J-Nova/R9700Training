// tests/gpu/adjoint_quant.cpp -- the device straight-through adjoint of the quant family against
// the host adjoints of the families quant and elementwise.
//
// The worked example docs/kernels/README.md points a new kernel at:
//   * on the fixture's shape (hidden 128, docs/FIXTURES.md) and on the real container's (hidden
//     5120, docs/DESIGN.md), for one sequence and for a batch of unequal lengths, the device
//     adjoint of had_quant_act_i8g and of the fused producers rmsnorm_had_quant_i8g,
//     gated_had_quant_i8g and gate_had_quant_i8g matches the harness's reference within its bf16
//     tolerance;
//   * the harness rejects a deliberately wrong kernel rather than accepting anything plausible;
//   * the scratch allocator reports its high-water mark, which the harness prints for a kernel.
//
// The reference is the host row itself: the composition of the family's parts' host adjoints
// (docs/adjoints/quant.md), which src/adjoint/host/quant.cpp registers under each producer's
// name and the harness looks up. A reference that is not the row would compare the device
// against a function the walk never runs, and the two could disagree where the walk runs.
//
// Needs one R9700 (gfx1201, 32 GB) or similar; runs under the GPU test guard. No container is
// read: the real shapes are numbers from the design, so this test does not carry the `model` label.
#include "r97/adjoint.h"

#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace r97;
using device_check::numel;

// -- the parts a fused producer is made of -----------------------------------

// The elementwise op the fused producer applies before the rotation. `none` is the plain quantiser
// of had_quant_act_i8g, which is the whole of that op.
enum class Part { none, rmsnorm, silu_mul, sigmoid_mul };

Operand read_operand(const char* buffer, int64_t rows, int64_t width) {
    return Operand{buffer, "bf16", {rows, width}, 0, rows, false};
}

// -- the cases --------------------------------------------------------------

struct Shape {
    int64_t rows;
    int64_t n;
    int64_t group;
    const char* label;
};

// The fixture's hidden and the container's (docs/FIXTURES.md, docs/DESIGN.md), one sequence and a
// batch of unequal lengths (300 + 700 + 1024 tokens). A row-wise op sees the batch as its token
// total, so the ragged case is its own row count.
const Shape kShapes[] = {
    {4, 128, 128, "4 x 128 tiny"},
    {4, 128, 64, "4 x 128 tiny, 2 groups"},
    {1024, 5120, 128, "1024 x 5120 one sequence"},
    {2024, 5120, 128, "2024 x 5120 batch 300+700+1024"},
    {4096, 5120, 128, "4096 x 5120 one sequence"},
};

device_check::Case quant_case(const Shape& s, Part part, uint64_t seed) {
    const int64_t rows = s.rows, n = s.n, group = s.group;
    const int64_t ngrp = n / group;

    device_check::Case c;
    c.rec.position = 0;
    c.rec.params = {{"M", rows}, {"n", n}, {"group", group}, {"dtype", std::string("bf16")}};

    if (part == Part::none) {
        c.rec.op = "had_quant_act_i8g";
        c.rec.operands = {read_operand("x", rows, n),
                          Operand{"q", "i8", {rows, n}, 0, rows, true},
                          Operand{"scale", "f32", {rows, ngrp}, 0, rows, true}};
        c.in = {device_check::random_value("bf16", {rows, n}, &seed)};
    } else if (part == Part::rmsnorm) {
        c.rec.op = "rmsnorm_had_quant_i8g";
        c.rec.params.push_back({"eps", std::string("1e-6")});
        c.rec.params.push_back({"wadd", std::string("0.0")});
        c.rec.operands = {read_operand("x", rows, n),
                          Operand{"w", "bf16", {n}, 0, 1, false},
                          Operand{"q", "i8", {rows, n}, 0, rows, true},
                          Operand{"scale", "f32", {rows, ngrp}, 0, rows, true}};
        c.in = {device_check::random_value("bf16", {rows, n}, &seed),
                device_check::random_value("bf16", {n}, &seed)};
    } else if (part == Part::silu_mul) {
        c.rec.op = "gated_had_quant_i8g";
        c.rec.operands = {read_operand("gate_up", rows, 2 * n),
                          Operand{"q", "i8", {rows, n}, 0, rows, true},
                          Operand{"scale", "f32", {rows, ngrp}, 0, rows, true}};
        c.in = {device_check::random_value("bf16", {rows, 2 * n}, &seed)};
    } else {
        c.rec.op = "gate_had_quant_i8g";
        c.rec.params.push_back({"act", std::string("sigmoid")});
        c.rec.operands = {read_operand("gate", rows, n), read_operand("x", rows, n),
                          Operand{"q", "i8", {rows, n}, 0, rows, true},
                          Operand{"scale", "f32", {rows, ngrp}, 0, rows, true}};
        c.in = {device_check::random_value("bf16", {rows, n}, &seed),
                device_check::random_value("bf16", {rows, n}, &seed)};
    }
    // The fused producer writes the part's output beside the codes, and a consumer's
    // cotangent for it is what the row's output term differentiates. Without it the term is
    // zero, and a row that drops it agrees with the other side anyway.
    if (part != Part::none)
        c.rec.operands.push_back(Operand{"out_bf16", "bf16", {rows, n}, 0, rows, true});
    c.cot = {device_check::random_value("f32", {rows, n}, &seed),
             device_check::random_value("f32", {rows, ngrp}, &seed)};
    if (part != Part::none) c.cot.push_back(device_check::random_value("f32", {rows, n}, &seed));
    c.label = std::string(s.label);
    return c;
}

// Deliberately wrong: hands the cotangent back unrotated and unmasked, which the harness must
// catch. A device adjoint is an AdjointFn, so a test can hand one in place of the registered row.
void wrong_quant_adjoint(const AdjointCall& call) {
    AdValue& dx = call.dc->at(0);
    for (size_t i = 0; i < dx.value.size(); ++i) dx.value[i] += call.cot.at(0).value[i];
}

// The live tape's fused producers: the elementwise output is a third write beside the codes and the
// scale, and `rmsnorm_had_quant_i8g`'s record carries a phantom write ahead of them. The row must
// find the roles by dtype and shape -- the codes' cotangent is cot[1] there, not cot[0] -- sum the
// output's own cotangent in, and refuse a count it does not serve by name. Each of those was a
// silent zero on the real container (399 of 400 leaves), so they are pinned here on the tape's
// shapes, not the schema's.
void fused_tape_shapes() {
    const int64_t rows = 4, n = 128, group = 128, ngrp = n / group, total = rows * n;
    uint64_t seed = 0x5eed1234ull;
    struct Spec {
        Part part;
        const char* op;
        bool phantom;
    };
    const Spec specs[] = {{Part::rmsnorm, "rmsnorm_had_quant_i8g", true},
                          {Part::silu_mul, "gated_had_quant_i8g", false},
                          {Part::sigmoid_mul, "gate_had_quant_i8g", false}};
    for (const Spec& sp : specs) {
        OpRecord rec;
        rec.op = sp.op;
        rec.params = {{"M", rows}, {"n", n}, {"group", group}, {"dtype", std::string("bf16")}};
        if (sp.part == Part::rmsnorm) {
            rec.params.push_back({"eps", std::string("1e-6")});
            rec.params.push_back({"wadd", std::string("0.0")});
        }
        if (sp.part == Part::sigmoid_mul) rec.params.push_back({"act", std::string("sigmoid")});
        std::vector<AdValue> fwd;
        if (sp.part == Part::rmsnorm) {
            rec.operands.push_back(read_operand("x", rows, n));
            // The live tape's phantom write is IN PLACE as well as written (`Operand`'s order is
            // `... rows, written, in_place`), and that is the whole point: a by-role scan for the
            // residual that tests only `written && in_place` picks this operand -- one value where
            // the norm needs rows*n -- and the walk dies on it. This case carried `written` alone,
            // so it could not see that; the declared shape is what must separate them.
            rec.operands.push_back(Operand{"none:1", "invalid", {}, 0, 0, true, true});
            rec.operands.push_back(Operand{"w", "bf16", {n}, 0, 1, false});
            fwd.push_back(device_check::random_value("bf16", {rows, n}, &seed));
            fwd.emplace_back();
            fwd.push_back(device_check::random_value("bf16", {n}, &seed));
        } else if (sp.part == Part::silu_mul) {
            rec.operands.push_back(read_operand("gate_up", rows, 2 * n));
            fwd.push_back(device_check::random_value("bf16", {rows, 2 * n}, &seed));
        } else {
            rec.operands.push_back(read_operand("gate", rows, n));
            rec.operands.push_back(read_operand("x", rows, n));
            fwd.push_back(device_check::random_value("bf16", {rows, n}, &seed));
            fwd.push_back(device_check::random_value("bf16", {rows, n}, &seed));
        }
        rec.operands.push_back(Operand{"q", "i8", {rows, n}, 0, rows, true});
        AdValue codes;
        codes.dtype = "i8";
        codes.shape = {rows, n};
        codes.value.resize((size_t)total);
        for (int64_t i = 0; i < total; ++i)
            codes.value[(size_t)i] = double((i * 37) % 255) - 127.0;
        fwd.push_back(std::move(codes));
        rec.operands.push_back(Operand{"scale", "f32", {rows, ngrp}, 0, rows, true});
        AdValue scale;
        scale.dtype = "f32";
        scale.shape = {rows, ngrp};
        scale.value.resize((size_t)(rows * ngrp));
        for (int64_t i = 0; i < rows * ngrp; ++i) scale.value[(size_t)i] = 1.0 + 0.01 * (i % 7);
        fwd.push_back(std::move(scale));
        // The elementwise output the fused producer also writes, and its consumers' cotangent.
        rec.operands.push_back(Operand{"out_bf16", "bf16", {rows, n}, 0, rows, true});
        fwd.push_back(device_check::random_value("bf16", {rows, n}, &seed));
        std::vector<AdValue> cot;
        // One cotangent per write, in the record's write order: the walk hands the row exactly
        // that, so the phantom write's slot is a hole the codes' cotangent must not land in.
        for (const Operand& o : rec.operands) {
            if (!o.written) continue;
            if (o.buffer == "q")
                cot.push_back(device_check::random_value("f32", {rows, n}, &seed));
            else if (o.buffer == "scale")
                cot.push_back(device_check::random_value("f32", {rows, ngrp}, &seed));
            else if (o.buffer == "out_bf16")
                cot.push_back(device_check::random_value("f32", {rows, n}, &seed));
            else
                cot.emplace_back();
        }
        std::vector<AdValue> dc;
        for (const Operand& o : rec.operands) {
            AdValue v;
            v.dtype = o.dtype;
            v.shape = o.shape;
            v.value.assign((size_t)numel(o.shape), 0.0);
            dc.push_back(std::move(v));
        }
        AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
        AdjointFn fn = adjoint_find(rec.op, Side::device, rec.params, nullptr);
        R97_CHECK(fn != nullptr);
        fn(call);
        double sum = 0.0;
        for (double v : dc.at(0).value) sum += std::fabs(v);
        R97_CHECK(sum > 0.0);
        std::printf("%s fused shape (%lld operands, phantom %d): dc[0] non-zero, |sum| %.6g\n",
                    rec.op.c_str(), (long long)rec.operands.size(), (int)sp.phantom, sum);

        // A count the row does not serve refuses by name instead of returning a zero gradient.
        // Two extras, not one: one more write than the schema's shape is the fused producer's own
        // layout (codes, scale, output), so only a count past every served layout is a wrong one.
        OpRecord wrong = rec;
        wrong.operands.push_back(Operand{"extra", "bf16", {rows, n}, 0, rows, true});
        wrong.operands.push_back(Operand{"extra2", "bf16", {rows, n}, 0, rows, true});
        std::vector<AdValue> wfwd = fwd, wcot = cot, wdc = dc;
        wfwd.emplace_back();
        wfwd.emplace_back();
        wcot.emplace_back();
        wcot.emplace_back();
        wdc.emplace_back();
        wdc.emplace_back();
        AdjointCall wcall{wrong, wfwd, wcot, &wdc, nullptr, ""};
        std::string what;
        try {
            fn(wcall);
        } catch (const std::exception& e) {
            what = e.what();
        }
        R97_CHECK(!what.empty());
        R97_CHECK(what.find(rec.op) != std::string::npos);
        std::printf("%s wrong count: %s\n", rec.op.c_str(), what.c_str());
    }
}


// The live `gate_had_quant_i8g` record, as the tape carries it: the params declare n=256
// group=128 while the codes are 6144 wide with a 48-group scale (group 128), and the elementwise
// output is a third write beside them. Every role is the shapes', not the params' -- pinning this
// is what the schema's four-operand shape could not do, and the short-operand refusals below are
// the ones that used to be silent zeros.
void live_gate_shape() {
    const int64_t rows = 4, codes_n = 6144, ngrp = 48;  // group 128, the live record's
    const int64_t total = rows * codes_n;
    uint64_t seed = 0x9a7f0badull;

    OpRecord rec;
    rec.position = 1453;
    rec.op = "gate_had_quant_i8g";
    // `n` names something else entirely (256): the row must read the widths off the operands, and
    // only the activation and the group's power-of-two come from the params.
    rec.params = {{"M", rows},
                  {"n", int64_t(256)},
                  {"group", int64_t(128)},
                  {"dtype", std::string("bf16")},
                  {"act", std::string("sigmoid")}};
    rec.operands = {read_operand("gate", rows, codes_n), read_operand("x", rows, codes_n),
                    Operand{"q8", "i8", {rows, codes_n}, 0, rows, true},
                    Operand{"s8", "f32", {rows, ngrp}, 0, rows, true},
                    Operand{"out", "bf16", {rows, codes_n}, 0, rows, true}};
    std::vector<AdValue> fwd;
    fwd.push_back(device_check::random_value("bf16", {rows, codes_n}, &seed));
    fwd.push_back(device_check::random_value("bf16", {rows, codes_n}, &seed));
    AdValue codes;
    codes.dtype = "i8";
    codes.shape = {rows, codes_n};
    codes.value.resize((size_t)total);
    for (int64_t i = 0; i < total; ++i) codes.value[(size_t)i] = double((i * 37) % 255) - 127.0;
    fwd.push_back(std::move(codes));
    AdValue scale;
    scale.dtype = "f32";
    scale.shape = {rows, ngrp};
    scale.value.resize((size_t)(rows * ngrp));
    for (int64_t i = 0; i < rows * ngrp; ++i) scale.value[(size_t)i] = 1.0 + 0.01 * (i % 7);
    fwd.push_back(std::move(scale));
    fwd.push_back(device_check::random_value("bf16", {rows, codes_n}, &seed));

    // One cotangent per write, in the record's write order: the codes' is cot[0] here.
    const std::vector<AdValue> cot = {device_check::random_value("f32", {rows, codes_n}, &seed),
                                      device_check::random_value("f32", {rows, ngrp}, &seed),
                                      device_check::random_value("f32", {rows, codes_n}, &seed)};
    std::vector<AdValue> dc;
    for (const Operand& o : rec.operands) {
        AdValue v;
        v.dtype = o.dtype;
        v.shape = o.shape;
        v.value.assign((size_t)numel(o.shape), 0.0);
        dc.push_back(std::move(v));
    }
    AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    AdjointFn fn = adjoint_find(rec.op, Side::device, rec.params, nullptr);
    R97_CHECK(fn != nullptr);
    fn(call);
    double sg = 0.0, sx = 0.0;
    for (double v : dc.at(0).value) sg += std::fabs(v);
    for (double v : dc.at(1).value) sx += std::fabs(v);
    R97_CHECK(sg > 0.0);
    R97_CHECK(sx > 0.0);
    std::printf("gate_had_quant_i8g live shape (codes %lld wide, %lld-group scale): roles found, "
                "|dc| %.6g + %.6g\n",
                (long long)codes_n, (long long)ngrp, sg, sx);

    // A slot the walk hands back short refuses by name: `dc[1]` is x's cotangent, `fwd[0]` the
    // gate's activation. A return here would be the zero this row was fixed never to produce.
    {
        std::vector<AdValue> short_dc = dc;
        short_dc.at(1).value.resize((size_t)(total - 1));
        AdjointCall sc{rec, fwd, cot, &short_dc, nullptr, ""};
        std::string what;
        try {
            fn(sc);
        } catch (const std::exception& e) {
            what = e.what();
        }
        R97_CHECK(what.find("dc[1]") != std::string::npos);
        R97_CHECK(what.find(rec.op) != std::string::npos);
        std::printf("gate_had_quant_i8g short dc[1]: %s\n", what.c_str());
    }
    {
        std::vector<AdValue> short_fwd = fwd;
        short_fwd.at(0).value.resize((size_t)(total - 1));
        AdjointCall sf{rec, short_fwd, cot, &dc, nullptr, ""};
        std::string what;
        try {
            fn(sf);
        } catch (const std::exception& e) {
            what = e.what();
        }
        R97_CHECK(what.find("fwd[0]") != std::string::npos);
        std::printf("gate_had_quant_i8g short fwd[0]: %s\n", what.c_str());
    }
    // The tape's own values short, not the walk's slots: a scale snapshot one value short is what
    // produced 399 zeros on the real container, and the codes' cotangent short is the other half of
    // the same silence. Both refuse by name, before any scratch is taken.
    {
        std::vector<AdValue> short_fwd = fwd;
        short_fwd.at(3).value.resize((size_t)(rows * ngrp - 1));
        AdjointCall ss{rec, short_fwd, cot, &dc, nullptr, ""};
        std::string what;
        try {
            fn(ss);
        } catch (const std::exception& e) {
            what = e.what();
        }
        R97_CHECK(what.find("operand 'scale'") != std::string::npos);
        R97_CHECK(what.find(rec.op) != std::string::npos);
        std::printf("gate_had_quant_i8g short scale: %s\n", what.c_str());
    }
    {
        std::vector<AdValue> short_cot = cot;
        short_cot.at(0).value.resize((size_t)(total - 1));
        AdjointCall sc2{rec, fwd, short_cot, &dc, nullptr, ""};
        std::string what;
        try {
            fn(sc2);
        } catch (const std::exception& e) {
            what = e.what();
        }
        R97_CHECK(what.find("codes cotangent") != std::string::npos);
        R97_CHECK(what.find(rec.op) != std::string::npos);
        std::printf("gate_had_quant_i8g short codes cotangent: %s\n", what.c_str());
    }
}

// The direct path: a write over a buffer the record also reads. The engine writes the
// elementwise input back, so the write's cotangent is the read's, and both rows must land it on
// the read's slot at the identity. The schema's cases carry no such write, so the branch is
// pinned here, where device and host are the registered rows -- the same function on both sides.
void direct_path_case() {
    const int64_t rows = 4, n = 128, group = 128, ngrp = n / group;
    uint64_t seed = 0xd17ec5ull;
    device_check::Case c;
    c.rec.position = 0;
    c.rec.op = "gated_had_quant_i8g";
    c.rec.params = {{"M", rows}, {"n", n}, {"group", group}, {"dtype", std::string("bf16")}};
    c.rec.operands = {read_operand("gate_up", rows, 2 * n),
                      Operand{"q", "i8", {rows, n}, 0, rows, true},
                      Operand{"scale", "f32", {rows, ngrp}, 0, rows, true},
                      Operand{"gate_up", "bf16", {rows, 2 * n}, 0, rows, true}};
    c.in = {device_check::random_value("bf16", {rows, 2 * n}, &seed)};
    c.cot = {device_check::random_value("f32", {rows, n}, &seed),
             device_check::random_value("f32", {rows, ngrp}, &seed),
             device_check::random_value("f32", {rows, 2 * n}, &seed)};
    c.label = "direct path, the write over the read's buffer";
    const device_check::Report r = device_check::check(c);
    device_check::require(r);
    R97_CHECK(r.scratch_peak > 0);
    std::printf("%s direct path: device == host, the write's cotangent on the read's slot, "
                "scratch peak %lld KiB\n",
                c.rec.op.c_str(), (long long)(r.scratch_peak / 1024));
}

void run() {
    const auto options = device_check::guard_options(int64_t(1) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    const Part parts[] = {Part::none, Part::rmsnorm, Part::silu_mul, Part::sigmoid_mul};
    for (Part part : parts) {
        const std::string op = part == Part::none ? "had_quant_act_i8g"
                               : part == Part::rmsnorm ? "rmsnorm_had_quant_i8g"
                               : part == Part::silu_mul ? "gated_had_quant_i8g"
                                                        : "gate_had_quant_i8g";
        R97_CHECK(r97::op_rows(op).device_adjoint);
        // The RMS norm's gain is frozen (src/adjoint/host/elementwise.cpp registers it so), and the
        // fused producer inherits that: the walk asks for x's cotangent only.
        if (part == Part::rmsnorm) {
            const OpRecord tiny = quant_case(kShapes[0], part, 0).rec;
            R97_CHECK(r97::adjoint_frozen(op, Side::device, tiny.params) ==
                      std::vector<int>{1});
        }
    }

    // The row serves the activations the card issues, and refuses the ones it does not yet: a
    // gated record built for gelu has no device adjoint rather than a silently wrong one.
    {
        std::string err;
        const OpRecord gated = quant_case(kShapes[0], Part::silu_mul, 0).rec;
        std::vector<OpParam> gelu = gated.params;
        gelu.push_back({"act", std::string("gelu")});
        R97_CHECK(adjoint_find("gated_had_quant_i8g", Side::device, gelu, &err) == nullptr);
        R97_CHECK(!err.empty());
        R97_CHECK(adjoint_find("gated_had_quant_i8g", Side::device, gated.params, &err) != nullptr);
    }

    uint64_t seed = 0x9a7f0001ull;
    for (Part part : parts) {
        for (const Shape& s : kShapes) {
            const device_check::Case c = quant_case(s, part, seed++);
            // No explicit reference: the harness looks up the registered host row, the
            // composition src/adjoint/host/quant.cpp registers under the producer's name, so
            // device and host are the same function.
            const device_check::Report r = device_check::check(c);
            device_check::require(r);
            R97_CHECK(r.scratch_peak > 0);  // the kernel took its buffers from the allocator
            std::printf("%s %s: device == the parts' host adjoints, scratch peak %lld KiB\n",
                        c.rec.op.c_str(), c.label.c_str(), (long long)(r.scratch_peak / 1024));
            if (r97test::bench()) device_check::print_bench(c);
        }
    }

    // A kernel that skips the rotation and the mask must be rejected, in x's slot.
    {
        const device_check::Case c = quant_case(kShapes[0], Part::rmsnorm, 0xa11ceull);
        const device_check::Report wrong = device_check::check(c, wrong_quant_adjoint);
        R97_CHECK(!wrong.ok);
        R97_CHECK(wrong.first_slot == 0);
        std::printf("rejected the wrong kernel: %s\n", wrong.note.c_str());
    }

    // The live tape's shapes, which the schema's count check used to refuse in silence, and the
    // gate record's own widths, which its params do not name.
    fused_tape_shapes();
    live_gate_shape();
    direct_path_case();
}

}  // namespace

R97_TEST(adjoint_quant_device) { run(); }
