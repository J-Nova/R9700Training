// tests/host/lora_w8a8.cpp -- a LoRA adapter on the rotated int8 trunk, through the host kernels.
//
// patches/serving/0210-lora-int8-trunk-host.patch lifts, for an adapter that declares
// `"radiance_input_basis": "fwht128"` (the key docs/ADAPTER.md defines), both refusals in
// core/lora/lora.cpp: the weight "stored in the 'fwht128' transform basis" one and the step-order
// dtype one (w4a8/w8a8/w2a8), because the host lora_shrink form now unexchanges quant_act_i8's
// codes before it reads them -- the declaration hands it its GEMM's encoding as `gemm_dtype`.
// The trunk applies the 128-wide Hadamard before quantising, so the codes had_quant_act_i8g
// writes are already in the activation's column order and need no unexchange; no site of this
// trunk has a step-order dtype. What is left to prove is the arithmetic: the trunk's int8 GEMM
// already multiplies the rotated activation, so an adapter whose A factors are in that basis adds
// its delta to the same product, and the sum has to be the model's output at that site.
//
// These tests run the engine's prefill on the int8 fixture with an adapter over every module it
// can ride, and read the step's tape:
//
//   the sites       each one is the trunk's GEMM plus (alpha / r) * B * A * x, computed in f64
//                   from the tape's own operands (the int8 codes with their scales, the bf16
//                   factors, the f32 scale stream) -- not from the container's weights
//   B = 0           the logits are the base model's, bit for bit
//   the refusals    an adapter with no key is turned away with the message it got before, at
//                   startup and again when it is loaded at runtime, and two that stay: the
//                   vocabulary weight is not a site, and a weight the recipe left unrotated
//                   cannot be ridden in the rotated basis
//   the encoding    every declared shrink carries the encoding of the GEMM whose activation it
//                   reads (`gemm_dtype`), the only thing that says how its codes are stored
//
// The prefill, the observer and the tape are patches/training/'s; a serving build applies none of
// that set, so this file registers no test there. The include sits above the guard, the way
// tests/host/fixture_w8a8.cpp includes the fixture: it is what defines R97_FIXTURE_PREFILL (through
// tests/support/fixture.h), and the header's own body compiles to nothing when the set is absent.
#include "support/lora_w8a8.h"

#if R97_FIXTURE_PREFILL && defined(RAD_ISSUE_OBSERVER)

#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using r97::Prefill;
using r97::Tape;

namespace {

// The host tier runs anywhere and must not take the shared card: radiance's device layer reads the
// visible devices once, when something first brings it up, so the card is hidden while this
// translation unit registers.
struct HideTheCard {
    HideTheCard() { setenv("ROCR_VISIBLE_DEVICES", "99", 1); }
};
const HideTheCard g_hide_the_card;

const std::vector<int32_t>& prompt() {
    static const std::vector<int32_t> ids = { 5, 9, 13, 17 };
    return ids;
}

std::string temp_dir(const char* name) {
    return (std::filesystem::path(radtest::tmpdir()) / name).string();
}

// The fixture is 3 delta net layers with an ssm_inz/gate_up/down and one attention layer with a
// qg/k/v/output/gate_up/down: 3 * 3 + 6 sites, every weight an adapter in the rotated basis can
// ride, once per block that reads it.
constexpr size_t kSites = 3 * 3 + 1 * 6;

// bf16 keeps 8 mantissa bits, so a sum of 128 products in f32 and its store to bf16 lands within
// 2^-8 of the f64 value; the +1 in the denominator keeps the check meaningful where the value
// cancels.
constexpr double kTolerance = 2e-2;

void fail(const std::string& what, const std::string& weight, double got, double want) {
    throw std::runtime_error("lora_w8a8: " + weight + " " + what + ": " + std::to_string(got) +
                             " against " + std::to_string(want));
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size())
        throw std::runtime_error("lora_w8a8: two runs returned " + std::to_string(a.size()) +
                                 " and " + std::to_string(b.size()) + " logits");
    double out = 0;
    for (size_t i = 0; i < a.size(); ++i) out = std::max(out, std::fabs((double)a[i] - b[i]));
    return out;
}

}  // namespace

// The trunk serves the adapter: every site gets its shrink and expand pair, and the model's output
// moves. Before the patch the start is refused ("stored in the 'fwht128' transform basis").
R97_TEST(lora_w8a8_accepts_the_adapter_at_every_site) {
    const r97::Fixture& f = r97::w8a8_fixture();
    const std::string   dir =
        r97::lora_w8a8::write_adapter(temp_dir("lora_w8a8_accepted"));

    const Prefill base = r97::lora_w8a8::run(f, prompt(), std::string());
    Tape          tape;
    const Prefill with = r97::lora_w8a8::run(f, prompt(), dir, &tape);

    const std::vector<r97::lora_w8a8::Site> found = r97::lora_w8a8::sites(tape);
    R97_CHECK(found.size() == kSites);
    for (const r97::lora_w8a8::Site& s : found)
        // A fused weight (ffn_gate_up) pools the ranks of the modules that share it: 2 * 4.
        R97_CHECK(s.rank >= 4 && s.rank % 4 == 0 && s.rows == (int64_t)prompt().size() &&
                  s.k > 0 && s.n > 0);

    const double moved = max_abs_diff(base.logits, with.logits);
    R97_CHECK(moved > 1.0);
    std::printf("  lora_w8a8: %zu sites carry the adapter, the logits move by %.3f ok\n",
                found.size(), moved);
}

// The shrink is told the encoding of the GEMM whose activation it reads, not the activation's
// element dtype: `gemm_dtype` is the GEMM's own `dtype` string. At a step-order site the two
// differ, and only the first says whether the codes need unexchanging, so the declaration must
// carry it -- `dtype` alone reads "i8" at every coded site and cannot.
R97_TEST(lora_w8a8_hands_the_shrink_the_site_encoding) {
    const r97::Fixture& f = r97::w8a8_fixture();
    const std::string   dir = r97::lora_w8a8::write_adapter(temp_dir("lora_w8a8_encoding"));

    Tape tape;
    R97_CHECK(!r97::lora_w8a8::run(f, prompt(), dir, &tape).logits.empty());

    const std::vector<r97::lora_w8a8::Site> found = r97::lora_w8a8::sites(tape);
    R97_CHECK(!found.empty());
    bool coded = false;
    for (const r97::lora_w8a8::Site& s : found) {
        const r97::OpRecord& gemm   = tape.ops()[s.gemm];
        const r97::OpRecord& shrink = tape.ops()[s.shrink];
        const std::string    site   = r97::lora_w8a8::str_param(gemm, "dtype");
        R97_CHECK(!site.empty());
        coded = coded || site == "i8a8h";
        // The element dtype stays what it was: these are int8 codes.
        R97_CHECK(r97::lora_w8a8::str_param(shrink, "dtype") == "i8");
        // And the site's encoding rides along, equal to the GEMM's own, on every site.
        R97_CHECK(r97::lora_w8a8::str_param(shrink, "gemm_dtype") == site);
    }
    R97_CHECK(coded);  // the fixture's sites are the rotated int8 ones this patch is about
    std::printf("  lora_w8a8: every shrink carries its GEMM's encoding (%zu sites, coded)\n",
                found.size());
}

// Each site computed the trunk's own GEMM plus (alpha / r) * B * A * x. The tape's operands are
// the int8 codes and their scales, so this recomputes both halves of the sum in f64; the delta is
// the adapter's, and a site that dropped it would miss by far more than the tolerance.
R97_TEST(lora_w8a8_sites_compute_base_plus_the_delta) {
    const r97::Fixture& f = r97::w8a8_fixture();
    const std::string   dir = r97::lora_w8a8::write_adapter(temp_dir("lora_w8a8_sites"));

    Tape          tape;
    const Prefill with = r97::lora_w8a8::run(f, prompt(), dir, &tape);
    R97_CHECK(!with.logits.empty());
    R97_CHECK(tape.keeps_contents());

    double worst = 0, weakest = 1e9;
    for (const r97::lora_w8a8::Site& s : r97::lora_w8a8::sites(tape)) {
        R97_CHECK(s.has_output());
        const std::vector<r97::OpRecord>& ops = tape.ops();
        const int64_t group = r97::lora_w8a8::int_param(ops[s.gemm], "group");
        R97_CHECK(group > 0 && s.k % group == 0);
        const int64_t groups = s.k / group;

        const r97::lora_w8a8::Matrix x  = r97::lora_w8a8::matrix(tape, s.gemm, 0, s.k);
        const r97::lora_w8a8::Matrix xs = r97::lora_w8a8::matrix(tape, s.gemm, 1, groups);
        const r97::lora_w8a8::Matrix w  = r97::lora_w8a8::matrix(tape, s.gemm, 2, s.k);
        const r97::lora_w8a8::Matrix ws = r97::lora_w8a8::matrix(tape, s.gemm, 3, groups);
        const r97::lora_w8a8::Matrix a  = r97::lora_w8a8::matrix(tape, s.shrink, 3, s.k);
        const r97::lora_w8a8::Matrix b  = r97::lora_w8a8::matrix(tape, s.expand, 2, s.rank);
        const r97::lora_w8a8::Matrix sc = r97::lora_w8a8::matrix(tape, s.expand, 3, 1);
        const r97::lora_w8a8::Matrix y =
            r97::lora_w8a8::matrix(tape, s.out_op, s.out_operand, s.n);

        double site_worst = 0, site_base = 0;
        for (int64_t m = 0; m < s.rows; ++m) {
            // The shrink's low-rank picture of the activation: A * x, in the rotated basis.
            std::vector<double> t((size_t)s.rank, 0.0);
            for (int64_t j = 0; j < s.rank; ++j)
                for (int64_t k = 0; k < s.k; ++k)
                    t[(size_t)j] += a.at(j, k) * x.at(m, k) * xs.at(m, k / group);
            for (int64_t n = 0; n < s.n; ++n) {
                double base = 0;
                for (int64_t k = 0; k < s.k; ++k)
                    base += w.at(n, k) * ws.at(n, k / group) * x.at(m, k) * xs.at(m, k / group);
                double delta = 0;
                for (int64_t j = 0; j < s.rank; ++j) delta += b.at(n, j) * t[(size_t)j];
                delta *= sc.at(n, 0);
                const double got = y.at(m, n);
                const double want = base + delta;
                site_worst =
                    std::max(site_worst, std::fabs(got - want) / (std::fabs(want) + 1));
                site_base = std::max(site_base, std::fabs(got - base) / (std::fabs(base) + 1));
            }
        }
        if (!(site_worst <= kTolerance))
            fail("is base plus the delta", s.weight, site_worst, kTolerance);
        // The delta is what this test is about: a site the adapter reached only partly, or a
        // recomputation that forgot it, would look like the base model here.
        if (!(site_base > 10 * kTolerance))
            fail("base only misses by", s.weight, site_base, 10 * kTolerance);
        worst = std::max(worst, site_worst);
        weakest = std::min(weakest, site_base);
    }
    std::printf("  lora_w8a8: every site is base + (alpha/r) * B * A * x, worst f64 gap %.2e, "
                "the base alone is off by %.2e ok\n",
                worst, weakest);
}

// B = 0: the expand adds nothing, and the model's logits are the base model's, bit for bit.
R97_TEST(lora_w8a8_zero_b_leaves_the_logits_alone) {
    const r97::Fixture& f = r97::w8a8_fixture();
    r97::lora_w8a8::Spec spec;
    spec.zero_b = true;
    const std::string dir = r97::lora_w8a8::write_adapter(temp_dir("lora_w8a8_zero_b"), spec);

    const Prefill base = r97::run_prefill(f, prompt());
    const Prefill zero = r97::lora_w8a8::run(f, prompt(), dir);
    R97_CHECK(base.logits.size() == zero.logits.size());
    R97_CHECK(std::memcmp(base.logits.data(), zero.logits.data(),
                          base.logits.size() * sizeof(float)) == 0);
    std::printf("  lora_w8a8: B = 0 leaves the logits bit-identical to the base model ok\n");
}

// An adapter that does not say which basis its factors are in is refused, with the message it got
// before this patch.
R97_TEST(lora_w8a8_refuses_an_adapter_without_the_key) {
    const r97::Fixture& f = r97::w8a8_fixture();
    r97::lora_w8a8::Spec spec;
    spec.key = false;
    const std::string dir = r97::lora_w8a8::write_adapter(temp_dir("lora_w8a8_no_key"), spec);

    const r97::lora_w8a8::Refusal r = r97::lora_w8a8::refuse(f, dir);
    R97_CHECK(r.code < 0);
    R97_CHECK(r.mentions("is stored in the 'fwht128' transform basis"));
    R97_CHECK(r.mentions("is fed a transformed activation"));
    // This fixture has no step-order site, so the missing key reaches the basis refusal first.
    R97_CHECK(!r.mentions("step order"));
    std::printf("  lora_w8a8: an adapter with no key is refused with today's message ok\n");
}

// The basis is the adapter's, so a load at runtime is checked against the same sites with that
// adapter's own: the sites were built from the startup adapter's "fwht128", and a keyless one
// loaded later would be placed on the rotated trunk and multiply a transformed activation with
// factors in the natural basis, silently. It is refused with the message startup gives, and an
// adapter in the trunk's basis still loads.
R97_TEST(lora_w8a8_refuses_a_runtime_load_without_the_key) {
    const r97::Fixture& f       = r97::w8a8_fixture();
    const std::string   rotated = r97::lora_w8a8::write_adapter(temp_dir("lora_w8a8_runtime"));
    r97::lora_w8a8::Spec keyless_spec;
    keyless_spec.key = false;
    const std::string keyless =
        r97::lora_w8a8::write_adapter(temp_dir("lora_w8a8_runtime_no_key"), keyless_spec);

    rad::Config cfg = r97::lora_w8a8::config(f, { prompt() }, rotated, 4);
    cfg.lora.max_loras = 3;  // the startup adapter takes one; two loads at runtime need two more
    rad::Engine engine(cfg);
    R97_CHECK(engine.start() >= 0);

    // The same adapter again, under a name of its own: it is in the trunk's basis and is taken.
    std::string err;
    R97_CHECK(engine.lora_load("second", rotated, &err) >= 0);

    err.clear();
    const int loaded = engine.lora_load("keyless", keyless, &err);
    R97_CHECK(loaded < 0);
    R97_CHECK(err.find("is stored in the 'fwht128' transform basis") != std::string::npos);
    R97_CHECK(err.find("is fed a transformed activation") != std::string::npos);
    std::printf("  lora_w8a8: a keyless adapter is refused again when it is loaded at runtime ok\n");
}

// The refusals that stay. The vocabulary weight is not a site, at any basis. And a targeted weight
// whose own GEMM is fed a natural activation cannot carry A factors that are in the rotated basis:
// the delta would be applied in the wrong one, silently, so it is refused instead. The tiny
// recipe leaves blk.*.ssm_out bf16 -- its K is 64, which does not divide into 128-column groups --
// although docs/ADAPTER.md's 27b container rotates it at a vdim of 128.
R97_TEST(lora_w8a8_refuses_what_it_cannot_serve) {
    const r97::Fixture& f = r97::w8a8_fixture();

    r97::lora_w8a8::Spec vocabulary;
    vocabulary.lm_head = true;
    const std::string head =
        r97::lora_w8a8::write_adapter(temp_dir("lora_w8a8_lm_head"), vocabulary);
    const r97::lora_w8a8::Refusal r1 = r97::lora_w8a8::refuse(f, head);
    R97_CHECK(r1.code < 0);
    R97_CHECK(r1.mentions("no GEMM of this model multiplies a weight of the targeted modules "
                          "(lm_head)"));

    r97::lora_w8a8::Spec unrotated;
    unrotated.leaves = { "linear_attn.out_proj" };
    const std::string out =
        r97::lora_w8a8::write_adapter(temp_dir("lora_w8a8_out_proj"), unrotated);
    const r97::lora_w8a8::Refusal r2 = r97::lora_w8a8::refuse(f, out);
    R97_CHECK(r2.code < 0);
    R97_CHECK(r2.mentions("weight 'blk.0.ssm_out.weight'"));
    R97_CHECK(r2.mentions("is not stored in a transform basis"));
    R97_CHECK(r2.mentions("radiance_input_basis: fwht128"));

    // A basis radiance does not serve is refused by name rather than read as the natural one.
    r97::lora_w8a8::Spec unknown;
    unknown.basis = "hadamard256";
    const std::string dir = r97::lora_w8a8::write_adapter(temp_dir("lora_w8a8_unknown"), unknown);
    const r97::lora_w8a8::Refusal r3 = r97::lora_w8a8::refuse(f, dir);
    R97_CHECK(r3.code < 0);
    R97_CHECK(r3.mentions("radiance_input_basis is 'hadamard256'"));
    std::printf("  lora_w8a8: refusals that stay: the vocabulary, an unrotated weight, an "
                "unknown basis ok\n");
}

#else  // R97_FIXTURE_PREFILL && RAD_ISSUE_OBSERVER

// A serving build applies neither the tape nor the prefill driver, so there is no step to record
// and this file registers no test there.

#endif
