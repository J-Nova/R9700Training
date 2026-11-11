// tests/host/step_host_grad.cpp -- the host join's adapter gradient is the gradient of the step's
// own loss. The cases here use the tiny bf16 fixture, whose host kernels compute in f32 without
// rounding their operands, and compare central differences of the driver's own forward with the
// `StepFn` the join returns. The step size is printed and is `h = 1e-2`: the adapter travels
// through the wire as bf16, so a smaller step rounds away and a larger one leaves the linear
// region; the check is therefore a secant over representable values, not an infinitesimal.
//
// The oracle comparison lives in step_host_oracle.cpp; this file does not read the reference.
#include "r97/adapter_io.h"
#include "r97/batch.h"
#include "r97/config.h"
#include "r97/forward.h"
#include "r97/model.h"
#include "r97/step_host.h"
#include "support/fixture.h"
#include "support/late_fixture.h"
#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#if R97_FORWARD

namespace {

// Copy the paths before the atexit runner; the check reads them after the binary's statics are
// gone.
const r97test::LatePaths& kLate = r97test::late_paths();

constexpr double kStep = 1e-2;
constexpr double kRelativeTolerance = 3e-1;

r97::Batch batch_of(const std::vector<std::vector<int32_t>>& sequences) {
    r97::Batch batch;
    for (const std::vector<int32_t>& ids : sequences) {
        r97::TokenSeq s;
        s.ids = ids;
        s.score.assign(ids.size(), 1);
        s.score[0] = 0;
        batch.push_back(std::move(s));
    }
    return batch;
}

std::vector<int32_t> prompt(int32_t seed, int32_t n) {
    std::vector<int32_t> ids(n);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (seed + 7 * int32_t(i)) % 256;
    return ids;
}

std::unique_ptr<r97::ForwardDriver> open_forward(const std::string& container) {
    r97::ForwardConfig cfg;
    cfg.model = container;
    cfg.radiance_home = r97::detail::radiance_home();
    cfg.window_rows = 1 << 20;
    std::string err;
    std::unique_ptr<r97::ForwardDriver> driver = r97::ForwardDriver::open(cfg, &err);
    if (!driver) throw std::runtime_error(err);
    return driver;
}

r97::Config lora_config(const std::string& container) {
    r97::Config cfg;
    cfg.model = container;
    cfg.objective = "ce";
    cfg.delta.kind = "lora";
    cfg.delta.rank = 4;
    cfg.delta.alpha = 8.0;
    return cfg;
}

// A random but repeatable value in [-1, 1].
double jitter(uint64_t* state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return double((*state >> 11) & ((1ULL << 53) - 1)) / double(1ULL << 52) * 2.0 - 1.0;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

std::string module_of(const std::string& name) {
    for (const char* suffix : {".lora_A", ".lora_B"}) {
        const size_t at = name.rfind(suffix);
        if (at != std::string::npos) return name.substr(0, at);
    }
    return name;
}

// One direction on one module: +d or -d on every element of both factors, exactly the way a
// finite difference perturbs the adapter. The modules excluded by the caller are the q/k/v sites
// whose pool state the family's attention join does not restore yet; this case is about the
// residual and MLP path and says so in its name.
using Direction = std::vector<std::vector<float>>;

Direction direction_for(const r97::ParamSet& params, const std::string& module, uint64_t* state,
                        bool every_element) {
    Direction dir(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        dir[i].assign(params[i].value.size(), 0.0f);
        if (module_of(params[i].name) != module) continue;
        if (every_element) {
            for (float& v : dir[i]) v = float(jitter(state));
        } else if (!dir[i].empty()) {
            dir[i][0] = float(jitter(state));
        }
    }
    return dir;
}

double dot(const Direction& dir, const r97::ParamSet& params, const r97::ParamSet& grads,
           const std::string& module) {
    double sum = 0.0;
    for (size_t i = 0; i < params.size(); ++i) {
        if (module_of(params[i].name) != module) continue;
        const r97::Param* g = nullptr;
        for (const r97::Param& p : grads)
            if (p.name == params[i].name) g = &p;
        if (!g || g->value.size() != params[i].value.size())
            throw std::runtime_error("host grad: no gradient for " + params[i].name);
        for (size_t j = 0; j < params[i].value.size(); ++j)
            sum += double(dir[i][j]) * double(g->value[j]);
    }
    return sum;
}

r97::ParamSet perturb(const r97::ParamSet& params, const Direction& dir, double h) {
    r97::ParamSet out = params;
    for (size_t i = 0; i < out.size(); ++i)
        for (size_t j = 0; j < out[i].value.size(); ++j)
            out[i].value[j] = float(double(params[i].value[j]) + h * double(dir[i][j]));
    return out;
}

void check_direction(const r97::StepFn& step, const r97::Batch& batch,
                     const r97::ParamSet& params, const r97::ParamSet& grads,
                     const std::string& module, int64_t scored, uint64_t* state) {
    const Direction dir = direction_for(params, module, state, true);
    const double analytic = dot(dir, params, grads, module);
    const double plus = step(batch, perturb(params, dir, kStep), nullptr).loss / double(scored);
    const double minus = step(batch, perturb(params, dir, -kStep), nullptr).loss / double(scored);
    const double fd = (plus - minus) / (2.0 * kStep);
    const double rel = std::fabs(fd - analytic) / (1.0 + std::fabs(fd));
    std::fprintf(stderr, "  host grad: %s  analytic %.9g  fd %.9g  relative %.3g (h=%.0e)\n", module.c_str(),
                analytic, fd, rel, kStep);
    if (!(rel <= kRelativeTolerance))
        throw std::runtime_error("host grad: " + module + " differs from finite differences: " +
                                 std::to_string(rel));
}

}  // namespace

R97_TEST(host_step_adapter_gradient_matches_finite_differences) {
    const std::string container = r97test::late_container();
    const r97::Batch batch = batch_of({prompt(21, 5), prompt(13, 7)});
    const r97::Config cfg = lora_config(container);
    const r97::Model model = r97::make_model(cfg);
    const r97::ModelMeta meta = r97::read_model_meta(container);
    r97::ParamSet params = model.delta->init(meta, cfg.delta, 1234);
    // The issue asks for non-zero A and B; LoraDelta's init leaves B at zero (the usual start),
    // and a zero B skips the expand path in the live forward, so make both factors non-zero.
    {
        uint64_t fill = 0x94d049bb133111ebULL;
        for (r97::Param& p : params)
            for (float& v : p.value) v = float(0.05 * jitter(&fill));
    }

    std::unique_ptr<r97::ForwardDriver> driver = open_forward(container);
    r97::HostStepReport report;
    const r97::StepFn step = r97::host_step(*driver, model, {}, &report);

    r97::ParamSet grads;
    const r97::StepResult r = step(batch, params, &grads);
    R97_CHECK(r.scored > 0);
    R97_CHECK(!grads.empty());
    std::printf("  host grad: loss %.9g over %lld scored, %zu params, peak snapshots %lld, "
                "peak cotangents %lld\n",
                r.loss, (long long)r.scored, grads.size(), (long long)report.peak_snapshot_bytes,
                (long long)report.peak_cotangent_bytes);
    R97_CHECK(report.peak_cotangent_bytes > 0);
    R97_CHECK(report.peak_snapshot_bytes > 0);

    uint64_t state = 0x9e3779b97f4a7c15ULL;
    // The two adapted linears on the residual/MLP path. The q/k/v and GDN qkv sites are the
    // family's live state and are checked by their own card, not by this one.
    for (int k = 0; k < 10; ++k)
        check_direction(step, batch, params, grads, "model.layers.3.mlp.down_proj", r.scored,
                        &state);
}

namespace {

// The alternate objective: the sum of one logit column. Its gradient enters at the logits buffer;
// the join must not know which objective it is walking. `column` is in the batch's row space, and
// the seed is one for every row the window covers.
r97::Objective column_objective(int64_t column) {
    return [column](const r97::ForwardTape& fwd, const r97::ParamSet&) {
        int64_t lo = 0, hi = 0;
        fwd.window("logits", &lo, &hi);
        const std::vector<int64_t> shape = fwd.shape("logits");
        R97_CHECK(shape.size() == 2);
        const int64_t vocab = shape[1];
        R97_CHECK(column >= 0 && column < vocab);
        const std::vector<float> logits = fwd.read("logits", lo, hi, true);
        R97_CHECK(int64_t(logits.size()) == (hi - lo) * vocab);
        r97::ObjectiveResult out;
        out.loss = 0.0;
        out.scored = hi - lo;
        out.seeds.push_back(r97::Seed{"logits", lo, hi, std::vector<float>(logits.size(), 0.0f)});
        for (int64_t r = 0; r < hi - lo; ++r) {
            out.loss += logits[size_t(r * vocab + column)];
            out.seeds[0].cotangent[size_t(r * vocab + column)] = 1.0f;
        }
        return out;
    };
}

}  // namespace

R97_TEST(host_step_walks_an_objective_that_is_not_the_ce) {
    const std::string container = r97test::late_container();
    const r97::Batch batch = batch_of({prompt(5, 6), prompt(29, 4)});
    const r97::Config cfg = lora_config(container);
    r97::Model model = r97::make_model(cfg);
    model.objective = column_objective(7);
    const r97::ModelMeta meta = r97::read_model_meta(container);
    r97::ParamSet params = model.delta->init(meta, cfg.delta, 99);
    {
        uint64_t fill = 0x2545f4914f6cdd1dULL;
        for (r97::Param& p : params)
            for (float& v : p.value) v = float(0.05 * jitter(&fill));
    }

    std::unique_ptr<r97::ForwardDriver> driver = open_forward(container);
    const r97::StepFn step = r97::host_step(*driver, model);
    r97::ParamSet grads;
    const r97::StepResult r = step(batch, params, &grads);
    R97_CHECK(r.scored == (6 - 1) + (4 - 1));

    uint64_t state = 0x243f6a8885a308d3ULL;
    const std::string module = "model.layers.3.mlp.down_proj";
    const Direction dir = direction_for(params, module, &state, false);
    const double analytic = dot(dir, params, grads, module);
    const double plus = step(batch, perturb(params, dir, kStep), nullptr).loss;
    const double minus = step(batch, perturb(params, dir, -kStep), nullptr).loss;
    const double fd = (plus - minus) / (2.0 * kStep);
    const double rel = std::fabs(fd - analytic) / (1.0 + std::fabs(fd));
    std::fprintf(stderr, "  host grad column: analytic %.9g  fd %.9g  relative %.3g (h=%.0e)\n", analytic,
                fd, rel, kStep);
    constexpr double kColumnTolerance = 1e-1;
    if (!(rel <= kColumnTolerance))
        throw std::runtime_error("host grad column: finite difference relative " +
                                 std::to_string(rel));
}

#endif  // R97_FORWARD
