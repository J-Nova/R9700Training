// tests/host/optim.cpp -- host tests for r97::Optimiser (include/r97/optim.h).
// Convex-quadratic convergence; the update over three steps matches a hand-computed AdamW
// to 1e-6; schedule values; clipping; gradient accumulation over k steps; serialise-and-
// continue bit-identity; and thread-count invariance of both the clip and the update. The
// update-time measurement behind docs/OPTIM.md lives here too,
// but opt-in: at 10^8 parameters it allocates a few GB, and the host label runs anywhere, so
// R97_OPTIM_BENCH=<parameter count> asks for it and the always-run suite never pays for it.
#include "r97/optim.h"
#include "r97/params.h"
#include "support/test.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

using r97::AdamWConfig;
using r97::Optimiser;
using r97::Param;
using r97::ParamSet;

namespace {

Param make(const char* name, const std::vector<int64_t>& shape, const std::vector<float>& v) {
    return Param{name, shape, v};
}

// The hand computation, in double, straight from the formulas in include/r97/optim.h.
// `decay` is only so the no-decay test can flip one flag; cfg.no_decay does the selecting
// inside the optimiser.
void reference_step(std::vector<double>& w, std::vector<double>& m, std::vector<double>& v,
                    const std::vector<double>& g, double lr, const AdamWConfig& cfg, int t,
                    bool decay) {
    const double b1 = cfg.beta1, b2 = cfg.beta2, eps = cfg.eps;
    for (size_t i = 0; i < w.size(); ++i) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        const double mh = m[i] / (1 - std::pow(b1, t));
        const double vh = v[i] / (1 - std::pow(b2, t));
        const double wd = decay ? cfg.weight_decay : 0.0;
        w[i] -= lr * (mh / (std::sqrt(vh) + eps) + wd * w[i]);
    }
}

double max_abs(const std::vector<float>& a, const std::vector<double>& b) {
    double e = 0.0;
    for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs(double(a[i]) - b[i]));
    return e;
}

std::vector<float> flat_of(const ParamSet& params) {
    std::vector<float> out;
    for (const auto& p : params) out.insert(out.end(), p.value.begin(), p.value.end());
    return out;
}

// The shared layout for the hand-computed tests: two params of two elements each, so the
// flat weight/gradient vector has four slots in parameter order.
ParamSet two_params(double a0, double a1, double b0, double b1) {
    ParamSet params;
    params.push_back(make("w.lora_A", {2}, {float(a0), float(a1)}));
    params.push_back(make("b.lora_B", {2}, {float(b0), float(b1)}));
    return params;
}

ParamSet grads_for(const ParamSet& params, const double g[4]) {
    ParamSet grads;
    size_t i = 0;
    for (const auto& p : params) {
        std::vector<float> gv(p.value.size());
        for (auto& x : gv) x = float(g[i++]);
        grads.push_back(Param{p.name, p.shape, gv});
    }
    return grads;
}

void test_quadratic_converges() {
    // f(x) = 1/2 * sum(q_i * x_i^2), diagonal curvature; the loss must fall to the minimum.
    const double q[4] = {1.0, 4.0, 0.25, 9.0};
    ParamSet params = two_params(2.0, -1.0, 3.0, 0.5);
    AdamWConfig cfg;
    cfg.lr = 0.1f;
    Optimiser opt(params, cfg);
    double first_loss = 0.0, final_loss = 0.0;
    for (int it = 0; it < 400; ++it) {
        double loss = 0.0;
        ParamSet grads;
        size_t i = 0;  // flat index, for q and the loss
        for (const auto& p : params) {
            std::vector<float> g(p.value.size());
            for (size_t j = 0; j < p.value.size(); ++j) {
                g[j] = float(q[i] * p.value[j]);
                loss += 0.5 * q[i] * double(p.value[j]) * double(p.value[j]);
                ++i;
            }
            grads.push_back(Param{p.name, p.shape, g});
        }
        if (it == 0) first_loss = loss;
        if (it == 399) final_loss = loss;
        R97_CHECK(opt.step(grads));
    }
    R97_CHECK(final_loss < first_loss);
    R97_CHECK(final_loss < 1e-8);
    std::printf("  quadratic: loss %.3e -> %.3e\n", first_loss, final_loss);
}

void test_three_steps_match_hand_computation() {
    ParamSet params = two_params(1.0, -2.0, 0.5, 4.0);
    AdamWConfig cf;  // decoupled decay on A and B alike, per the issue's default
    cf.lr = 0.01f;
    cf.weight_decay = 0.1f;
    const double G[3][4] = {{0.3, -1.2, 2.0, -0.7}, {-0.5, 0.8, 1.1, 0.9}, {0.2, 0.1, -0.3, 1.4}};
    Optimiser opt(params, cf);
    std::vector<double> w{1.0, -2.0, 0.5, 4.0}, m(4, 0.0), v(4, 0.0);
    for (int t = 0; t < 3; ++t) {
        const ParamSet grads = grads_for(params, G[t]);
        R97_CHECK(grads.size() == 2 && grads[0].value.size() == 2 && grads[1].value.size() == 2);
        R97_CHECK(opt.step(grads));
        const double lr = r97::scheduled_lr(cf, t + 1);
        reference_step(w, m, v, std::vector<double>(G[t], G[t] + 4), lr, cf, t + 1, true);
        const double err = max_abs(flat_of(params), w);
        R97_CHECK(err < 1e-6);
        if (t == 2)
            std::printf("  three AdamW steps match the hand computation (max err %.2e)\n", err);
    }
    R97_CHECK(opt.updates_done() == 3);
}

void test_schedule_values() {
    AdamWConfig cfg;
    cfg.lr = 2e-3f;
    cfg.warmup_steps = 10;
    cfg.total_steps = 100;
    cfg.lr_floor_frac = 0.1f;
    R97_CHECK(r97::scheduled_lr(cfg, 0) == 0.0f);
    R97_CHECK(std::abs(r97::scheduled_lr(cfg, 1) - 2e-3f * 0.1) < 1e-6);   // step 1: 1/10 of lr
    R97_CHECK(std::abs(r97::scheduled_lr(cfg, 5) - 2e-3f * 0.5) < 1e-6);   // linear warm-up
    R97_CHECK(std::abs(r97::scheduled_lr(cfg, 10) - 2e-3f) < 1e-6);        // warm-up ends at lr
    const double floor = 2e-3 * 0.1;
    R97_CHECK(std::abs(r97::scheduled_lr(cfg, 55) - 0.5 * (floor + 2e-3)) < 1e-6);  // mid cosine
    R97_CHECK(std::abs(r97::scheduled_lr(cfg, 100) - floor) < 1e-6);  // the floor at the horizon
    R97_CHECK(std::abs(r97::scheduled_lr(cfg, 500) - floor) < 1e-6);       // flat past the horizon
    AdamWConfig nocos;
    nocos.lr = 3e-3f;
    R97_CHECK(r97::scheduled_lr(nocos, 1) == 3e-3f);
    R97_CHECK(r97::scheduled_lr(nocos, 77) == 3e-3f);
    std::printf("  schedule values ok\n");
}

void test_clipping() {
    ParamSet empty;
    R97_CHECK(r97::clip_global_norm(empty, 1.0f) == 0.0f);
    ParamSet g2;
    g2.push_back(make("g", {4}, {3.0f, 4.0f, 0.0f, 0.0f}));  // global norm 5
    const float norm = r97::clip_global_norm(g2, 2.5f);
    R97_CHECK(std::abs(norm - 5.0f) < 1e-5);
    R97_CHECK(std::abs(g2[0].value[0] - 1.5f) < 1e-5);          // 3 * 2.5/5
    R97_CHECK(std::abs(g2[0].value[1] - 2.0f) < 1e-5);
    r97::clip_global_norm(g2, 100.0f);                       // norm below the cap: untouched
    R97_CHECK(std::abs(g2[0].value[0] - 1.5f) < 1e-5);

    // A norm past f32's 3.4e38: two leaves of 3e38 give a norm of 4.24e38. The compare and the
    // scale are double, so the gradient is rescaled to unit norm; a float norm would be Inf and
    // the scale 1/Inf = 0, zeroing the whole gradient. The returned norm still converts to Inf.
    ParamSet big;
    big.push_back(make("g", {2}, {3e38f, 3e38f}));
    const float big_norm = r97::clip_global_norm(big, 1.0f);
    R97_CHECK(std::isinf(big_norm));
    R97_CHECK(std::abs(big[0].value[0] - 0.70710678f) < 1e-3f);
    R97_CHECK(std::abs(big[0].value[1] - 0.70710678f) < 1e-3f);

    // Through the optimiser: clipping rescales the gradient before m and v see it, so the
    // hand computation must run on the rescaled gradient.
    const double G[2][4] = {{3.0, 0.0, -4.0, 0.0}, {6.0, 8.0, 0.0, 0.0}};  // norms 5 and 10
    ParamSet params = two_params(1.0, -2.0, 0.5, 4.0);
    AdamWConfig cf;
    cf.lr = 0.01f;
    cf.clip_norm = 1.0f;
    Optimiser opt(params, cf);
    std::vector<double> w{1.0, -2.0, 0.5, 4.0}, m(4, 0.0), v(4, 0.0);
    for (int t = 0; t < 2; ++t) {
        R97_CHECK(opt.step(grads_for(params, G[t])));
        std::vector<double> gs(4);
        for (size_t i = 0; i < 4; ++i) gs[i] = G[t][i] * (1.0 / (t == 0 ? 5.0 : 10.0));
        reference_step(w, m, v, gs, cf.lr, cf, t + 1, true);
        R97_CHECK(max_abs(flat_of(params), w) < 1e-6);
    }
    std::printf("  clipping ok\n");
}

// A parameter set large enough that the clip's thread split is used (more than one
// 4096-element chunk), with odd sizes so chunks cross parameter boundaries, and values
// from a fixed LCG so every run starts from the same bits.
ParamSet big_set() {
    const size_t sizes[3] = {70000, 90001, 41000};
    unsigned state = 12345u;
    ParamSet set;
    for (int p = 0; p < 3; ++p) {
        std::vector<float> v(sizes[p]);
        for (auto& x : v) {
            state = state * 1664525u + 1013904223u;
            x = float(int(state >> 8) % 2001 - 1000) / 500.0f;  // [-2, 2)
        }
        set.push_back(Param{"p" + std::to_string(p), {int64_t(sizes[p])}, v});
    }
    return set;
}

double flat_norm_sq(const std::vector<float>& x) {
    double sq = 0.0;
    for (size_t lo = 0; lo < x.size(); lo += 4096) {
        double part = 0.0;
        const size_t hi = std::min(x.size(), lo + 4096);
        for (size_t i = lo; i < hi; ++i) part += double(x[i]) * double(x[i]);
        sq += part;
    }
    return sq;
}

void test_clip_threading() {
    // The clip is a parallel reduction and must still be deterministic: the same fixed
    // 4096-element chunks, one double partial each, summed in chunk order, on any thread
    // count. The norm and the clipped values must be bit-identical across thread counts.
    const ParamSet base = big_set();
    const std::vector<float> flat = flat_of(base);
    R97_CHECK(flat.size() > 4096);
    const float ref_norm = float(std::sqrt(flat_norm_sq(flat)));

    ParamSet one = base, four = base, many = base;
    const float n1 = r97::clip_global_norm(one, 1.0f, 1);
    const float n4 = r97::clip_global_norm(four, 1.0f, 4);
    const float n48 = r97::clip_global_norm(many, 1.0f, 48);
    R97_CHECK(std::abs(n1 - ref_norm) <= 1e-6f * ref_norm);
    R97_CHECK(n4 == n1);   // bit for bit, not just close
    R97_CHECK(n48 == n1);

    const std::vector<float> f1 = flat_of(one);
    R97_CHECK(std::memcmp(f1.data(), flat_of(four).data(), sizeof(float) * f1.size()) == 0);
    R97_CHECK(std::memcmp(f1.data(), flat_of(many).data(), sizeof(float) * f1.size()) == 0);
    const double sq = flat_norm_sq(f1);
    R97_CHECK(std::abs(std::sqrt(sq) - 1.0) < 1e-5);  // rescaled onto the cap
    // A second call on the same (already clipped) set is a no-op with norm 1.
    R97_CHECK(std::abs(r97::clip_global_norm(one, 1.0f, 8) - 1.0f) < 1e-5f);
    R97_CHECK(std::memcmp(f1.data(), flat_of(one).data(), sizeof(float) * f1.size()) == 0);
    std::printf("  clipping is thread-count invariant (%zu elements, norm %.6f)\n",
                flat.size(), double(ref_norm));
}

void test_thread_count_invariance() {
    // Through the optimiser, both the clip and the element passes: one thread and sixteen
    // threads must agree bit for bit at the end of three updates.
    ParamSet a = big_set(), b = big_set();
    AdamWConfig one;
    one.lr = 1e-3f;
    one.weight_decay = 0.01f;
    one.clip_norm = 1.0f;
    one.threads = 1;
    AdamWConfig many = one;
    many.threads = 16;
    Optimiser oa(a, one), ob(b, many);
    for (int t = 0; t < 3; ++t) {
        ParamSet ga, gb;
        for (size_t i = 0; i < a.size(); ++i) {
            std::vector<float> x = a[i].value;
            for (auto& v : x) v = 0.5f * v + 0.1f * float(t + 1);
            ga.push_back(Param{a[i].name, a[i].shape, x});
            gb.push_back(Param{b[i].name, b[i].shape, x});
        }
        R97_CHECK(oa.step(ga));
        R97_CHECK(ob.step(gb));
    }
    for (size_t i = 0; i < a.size(); ++i) {
        R97_CHECK(a[i].value.size() == b[i].value.size());
        R97_CHECK(std::memcmp(a[i].value.data(), b[i].value.data(),
                              sizeof(float) * a[i].value.size()) == 0);
    }
    std::printf("  update is thread-count invariant\n");
}

void test_accumulation() {
    // Three gradients over k=3 calls: the applied mean must equal a direct step on the mean.
    AdamWConfig cf;
    cf.lr = 0.01f;
    cf.accum_steps = 3;
    cf.weight_decay = 0.05f;
    // Chosen so every flat slot's triple sums to a float-exact, nonzero value: a zero
    // double mean with float noise would be amplified by the first-step normalisation.
    const double GG[3][4] = {{0.25, -1.5, 2.0, -0.75}, {-0.5, 0.75, 1.25, 1.0},
                             {1.0, 0.25, -0.5, 1.5}};
    ParamSet acc_params = two_params(1.0, -2.0, 0.5, 4.0);
    Optimiser acc(acc_params, cf);
    std::vector<double> mean(4, 0.0);
    for (int s = 0; s < 3; ++s) {
        const ParamSet grads = grads_for(acc_params, GG[s]);
        R97_CHECK(s < 2 ? !acc.step(grads) : acc.step(grads));
        R97_CHECK(acc.pending() == (s < 2 ? s + 1 : 0));
        for (size_t i = 0; i < 4; ++i) mean[i] += GG[s][i] / 3.0;
    }
    std::vector<double> w{1.0, -2.0, 0.5, 4.0}, m(4, 0.0), v(4, 0.0);
    reference_step(w, m, v, mean, cf.lr, cf, 1, true);
    R97_CHECK(max_abs(flat_of(acc_params), w) < 1e-6);
    R97_CHECK(acc.updates_done() == 1);
    std::printf("  accumulation over k steps ok\n");
}

void test_no_decay_list() {
    const double g0[2] = {0.4, -0.9};
    AdamWConfig cf;
    cf.lr = 0.01f;
    cf.weight_decay = 0.5f;
    cf.no_decay = {"lora_B"};
    ParamSet both;
    both.push_back(make("m.lora_A", {2}, {1.0f, -2.0f}));
    both.push_back(make("m.lora_B", {2}, {1.0f, -2.0f}));
    Optimiser opt(both, cf);
    ParamSet grads;
    grads.push_back(make("m.lora_A", {2}, {float(g0[0]), float(g0[1])}));
    grads.push_back(make("m.lora_B", {2}, {float(g0[0]), float(g0[1])}));
    R97_CHECK(opt.step(grads));
    std::vector<double> g(g0, g0 + 2);
    std::vector<double> w{1.0, -2.0}, m(2, 0.0), v(2, 0.0);
    reference_step(w, m, v, g, cf.lr, cf, 1, true);
    R97_CHECK(max_abs(both[0].value, w) < 1e-6);  // decayed
    std::vector<double> w2{1.0, -2.0}, m2(2, 0.0), v2(2, 0.0);
    reference_step(w2, m2, v2, g, cf.lr, cf, 1, false);
    R97_CHECK(max_abs(both[1].value, w2) < 1e-6);  // excluded
    std::printf("  no-decay list ok\n");
}

void test_bad_inputs_throw() {
    bool threw = false;
    try {
        ParamSet empty;
        Optimiser o(empty, AdamWConfig{});
        (void)o;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    R97_CHECK(threw);

    ParamSet params = two_params(1.0, 2.0, 3.0, 4.0);
    Optimiser o(params, AdamWConfig{});
    threw = false;
    try {
        ParamSet g;
        g.push_back(make("w.lora_A", {3}, {1.0f, 2.0f, 3.0f}));
        g.push_back(Param{"b.lora_B", {2}, {1.0f, 2.0f}});
        o.step(g);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    R97_CHECK(threw);
    threw = false;
    try {
        ParamSet g;
        g.push_back(make("other", {2}, {1.0f, 2.0f}));
        g.push_back(Param{"b.lora_B", {2}, {1.0f, 2.0f}});
        o.step(g);  // one wrong name among the right count
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    R97_CHECK(threw);
    std::printf("  bad inputs throw\n");
}

void test_serialise_and_continue() {
    // Stop after three steps; the restored optimiser and the never-stopped one then run
    // the same fourth step on identical copies of the start, and must agree bit for bit.
    const double G[4][4] = {{0.3, -1.2, 2.0, -0.7}, {-0.5, 0.8, 1.1, 0.9},
                            {0.2, 0.1, -0.3, 1.4},  {0.9, -0.4, 0.5, -1.1}};
    AdamWConfig cf;
    cf.lr = 0.01f;
    cf.weight_decay = 0.1f;
    cf.clip_norm = 0.7f;

    ParamSet a = two_params(1.0, -2.0, 0.5, 4.0);  // never stops
    Optimiser opt_a(a, cf);
    for (int t = 0; t < 3; ++t) R97_CHECK(opt_a.step(grads_for(a, G[t])));
    const std::vector<uint8_t> bytes = opt_a.serialise();
    R97_CHECK(opt_a.step(grads_for(a, G[3])));
    R97_CHECK(opt_a.updates_done() == 4);

    ParamSet b = two_params(1.0, -2.0, 0.5, 4.0);  // stops and restores
    Optimiser opt_b(b, cf);
    opt_b.restore(bytes);
    R97_CHECK(opt_b.updates_done() == 3);
    R97_CHECK(opt_b.step(grads_for(b, G[3])));
    for (size_t i = 0; i < a.size(); ++i) {
        R97_CHECK(a[i].shape == b[i].shape);
        R97_CHECK(std::memcmp(a[i].value.data(), b[i].value.data(),
                           sizeof(float) * a[i].value.size()) == 0);
    }
    // Same property mid-window, accumulation half done (two of three gradients in).
    AdamWConfig cf2 = cf;
    cf2.accum_steps = 3;
    ParamSet c = two_params(1.0, -2.0, 0.5, 4.0);  // never stops, k=3
    Optimiser opt_c(c, cf2);
    for (int t = 0; t < 2; ++t) R97_CHECK(!opt_c.step(grads_for(c, G[t])));  // pending: 2
    const std::vector<uint8_t> bytes_c = opt_c.serialise();
    R97_CHECK(opt_c.step(grads_for(c, G[2])));  // third call closes the window
    R97_CHECK(opt_c.updates_done() == 1);

    ParamSet d = two_params(1.0, -2.0, 0.5, 4.0);
    Optimiser opt_d(d, cf2);
    opt_d.restore(bytes_c);
    R97_CHECK(opt_d.pending() == 2 && opt_d.updates_done() == 0);
    R97_CHECK(opt_d.step(grads_for(d, G[2])));  // window closes
    R97_CHECK(opt_d.updates_done() == 1);
    for (size_t i = 0; i < c.size(); ++i)
        R97_CHECK(std::memcmp(c[i].value.data(), d[i].value.data(),
                           sizeof(float) * c[i].value.size()) == 0);

    // Bad bytes refuse, leaving the optimiser unchanged.
    std::vector<uint8_t> bad = bytes;
    bad[0] = 'X';
    bool threw = false;
    try {
        opt_d.restore(bad);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    R97_CHECK(threw);
    bad = bytes;
    bad.pop_back();  // truncated
    threw = false;
    try {
        opt_d.restore(bad);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    R97_CHECK(threw);
    threw = false;
    try {
        opt_d.restore(std::vector<uint8_t>(4, 0));  // wrong parameter count and magic
    } catch (const std::runtime_error&) {
        threw = true;
    }
    R97_CHECK(threw);
    std::printf("  serialise-and-continue ok (%zu bytes)\n", bytes.size());
}

// The measurement docs/OPTIM.md records: one update over `R97_OPTIM_BENCH` parameters, the
// adapter's scale at 10^8. It reports the time and passes no judgement on it -- the number
// belongs in the doc, and a wall-clock assertion in the always-run suite would fail under
// load or on a machine that cannot hold the state (about 2 GB at 10^8, weights + moments +
// accumulation buffer + gradients).
void bench_one_update() {
    const char* asked = std::getenv("R97_OPTIM_BENCH");
    if (asked == nullptr || *asked == '\0') return;
    const size_t n = size_t(std::strtoull(asked, nullptr, 10));
    if (n == 0) {
        std::fprintf(stderr, "  bench: R97_OPTIM_BENCH is not a parameter count: '%s'\n", asked);
        return;
    }
    ParamSet params;
    params.push_back(make("model.layers.3.mlp.down_proj.lora_A", {int64_t(n)},
                          std::vector<float>(n, 0.01f)));
    AdamWConfig cfg;
    cfg.lr = 1e-3f;
    cfg.weight_decay = 0.01f;
    cfg.clip_norm = 1.0f;  // the step the trainer takes: clip, then AdamW
    cfg.threads = 0;       // every core the machine has
    Optimiser opt(params, cfg);
    ParamSet grads;
    grads.push_back(Param{"model.layers.3.mlp.down_proj.lora_A", {int64_t(n)},
                          std::vector<float>(n, 1e-4f)});
    opt.step(grads);  // warm: first-touch the pages and the thread pool
    const auto t0 = std::chrono::steady_clock::now();
    opt.step(grads);
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("  bench: one AdamW update at %zu parameters: %.3f s on %d threads\n", n, secs,
                opt.threads());
}

}  // namespace

R97_TEST(optim) {
    test_quadratic_converges();
    test_three_steps_match_hand_computation();
    test_schedule_values();
    test_clipping();
    test_clip_threading();
    test_thread_count_invariance();
    test_accumulation();
    test_no_decay_list();
    test_bad_inputs_throw();
    test_serialise_and_continue();
    bench_one_update();
}

