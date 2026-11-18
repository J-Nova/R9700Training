// src/optim.cpp -- AdamW over a ParamSet; see include/r97/optim.h.
//
// Threading: the element range across all parameters is cut into fixed chunks and each
// chunk takes a global element index, decoded through the prefix-sum table into a
// (parameter, local index) pair. Every pass (accumulate, scale, update) runs the same
// way. The only cross-element reduction, the global norm, runs on the same threads over
// fixed 4096-element chunks and sums one partial per chunk in chunk order, so the result
// depends neither on thread count nor on scheduling and a restart reproduces it bit for
// bit.
#include "r97/optim.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace r97 {
namespace {

// One element view: which parameter and where in it.
struct Loc {
    size_t p;     // parameter index
    size_t off;   // element offset inside the parameter
};

struct Decoded {
    size_t p;     // parameter index of the first element in [lo, hi)
    size_t off;   // element offset inside it
    size_t len;   // elements inside that same parameter, starting at (p, off)
};

Decoded decode(const size_t* offsets, size_t lo, size_t hi) {
    // offsets is the prefix-sum table of length nparams+1; find the parameter of lo.
    size_t pi = 0;
    while (offsets[pi + 1] <= lo) ++pi;
    const size_t cap = offsets[pi + 1] - lo;
    Decoded d{pi, lo - offsets[pi], std::min(cap, hi - lo)};
    return d;
}

// Run body(lo, hi) over [0, total) split into a fixed number of ranges, on nthreads
// threads. The body must be safe to call concurrently on disjoint ranges.
template <class Body>
void parallel_split(size_t total, int nthreads, const Body& body) {
    if (nthreads <= 1 || total <= 1) {
        body(size_t(0), total);
        return;
    }
    const size_t chunks = size_t(nthreads) * 4;
    const size_t per = (total + chunks - 1) / chunks;
    std::atomic<size_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const size_t c = next.fetch_add(1);
            const size_t lo = c * per;
            if (lo >= total) return;
            body(lo, std::min(total, lo + per));
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nthreads - 1);
    for (int i = 0; i < nthreads - 1; ++i) pool.emplace_back(worker);
    worker();  // the calling thread joins in
    for (auto& t : pool) t.join();
}

// The element passes: below this many elements the threads cost more than they save.
template <class Body>
void parallel_ranges(size_t total, int nthreads, const Body& body) {
    if (total < 4096) {
        body(size_t(0), total);
        return;
    }
    parallel_split(total, nthreads, body);
}

constexpr char kMagic[] = "R97OPT";
constexpr uint16_t kVersion = 1;

template <class T>
void put(std::vector<uint8_t>& out, const T& v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

template <class T>
const uint8_t* get(const uint8_t* p, const uint8_t* end, T& v) {
    if (size_t(end - p) < sizeof(T))
        throw std::runtime_error("r97::Optimiser::restore: truncated buffer");
    std::memcpy(&v, p, sizeof(T));
    return p + sizeof(T);
}

// Bulk forms of the same bytes, so a checkpoint of 10^8 parameters is a memcpy per tensor
// rather than 4*10^8 one-element appends.
void put_floats(std::vector<uint8_t>& out, const float* src, size_t n) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
    out.insert(out.end(), p, p + n * sizeof(float));
}

const uint8_t* get_floats(const uint8_t* p, const uint8_t* end, float* dst, size_t n) {
    if (size_t(end - p) < n * sizeof(float))
        throw std::runtime_error("r97::Optimiser::restore: truncated buffer");
    std::memcpy(dst, p, n * sizeof(float));
    return p + n * sizeof(float);
}

}  // namespace

float scheduled_lr(const AdamWConfig& cfg, int step) {
    if (step <= 0) return 0.0f;
    const double base = cfg.lr > 0.0f ? double(cfg.lr) : 0.0;
    if (cfg.warmup_steps > 0 && step <= cfg.warmup_steps) {
        const double r = double(step) / double(cfg.warmup_steps);
        return float(base * r);
    }
    if (cfg.total_steps <= 0 || cfg.total_steps <= cfg.warmup_steps) return cfg.lr;
    const double floor = base * (cfg.lr_floor_frac >= 0.0f ? double(cfg.lr_floor_frac) : 0.0);
    const double horizon = double(cfg.total_steps - cfg.warmup_steps);
    const double t = std::min(double(step - cfg.warmup_steps), horizon);
    const double angle = 3.14159265358979323846 * t / horizon;
    return float(floor + (base - floor) * 0.5 * (1.0 + std::cos(angle)));
}

namespace {

// The norm and the rescale, with the thread count in hand: the optimiser passes its own,
// the public clip_global_norm() asks the machine.
float clip_impl(ParamSet& grads, float max_norm, int nthreads) {
    std::vector<size_t> offs(grads.size() + 1, 0);
    for (size_t i = 0; i < grads.size(); ++i) offs[i + 1] = offs[i] + grads[i].value.size();
    const size_t total = offs.back();
    if (total == 0) return 0.0f;
    const size_t* off = offs.data();
    auto* gd = grads.data();

    // A fixed chunk size, so the reduction is the same on any thread count: one double
    // partial per chunk, then the partials in chunk order. Small gradient sets do not pay
    // for the threads, and the bits are the same either way.
    constexpr size_t kChunk = 4096;
    const size_t nchunks = (total + kChunk - 1) / kChunk;
    if (total < 65536) nthreads = 1;
    std::vector<double> parts(nchunks, 0.0);
    parallel_split(nchunks, nthreads, [&](size_t c0, size_t c1) {
        for (size_t c = c0; c < c1; ++c) {
            const size_t lo = c * kChunk;
            const size_t hi = std::min(total, lo + kChunk);
            double part = 0.0;
            size_t i = lo;
            while (i < hi) {
                const Decoded d = decode(off, i, hi);
                const float* x = gd[d.p].value.data() + d.off;
                for (size_t k = 0; k < d.len; ++k) part += double(x[k]) * double(x[k]);
                i += d.len;
            }
            parts[c] = part;
        }
    });
    double sq = 0.0;
    for (size_t c = 0; c < nchunks; ++c) sq += parts[c];
    // The norm is compared and the scale computed in double: a gradient whose norm passes f32's
    // 3.4e38 (the device walk's is ~7.5e38 at the 512-token step) would otherwise convert to Inf,
    // make the scale 1/Inf = 0, and zero the whole gradient instead of clipping it. The scale
    // itself is a representable f32 subnormal (1/7.5e38 = 1.3e-39).
    const double norm_d = std::sqrt(sq);
    const float norm = float(norm_d);
    if (max_norm > 0.0f && norm_d > double(max_norm) && norm_d > 0.0) {
        const float s = float(double(max_norm) / norm_d);
        parallel_split(nchunks, nthreads, [&](size_t c0, size_t c1) {
            for (size_t c = c0; c < c1; ++c) {
                const size_t lo = c * kChunk;
                const size_t hi = std::min(total, lo + kChunk);
                size_t i = lo;
                while (i < hi) {
                    const Decoded d = decode(off, i, hi);
                    float* x = gd[d.p].value.data() + d.off;
                    for (size_t k = 0; k < d.len; ++k) x[k] *= s;
                    i += d.len;
                }
            }
        });
    }
    return norm;
}

}  // namespace

float clip_global_norm(ParamSet& grads, float max_norm, int threads) {
    const int nthreads = threads > 0 ? threads : int(std::thread::hardware_concurrency());
    return clip_impl(grads, max_norm, nthreads > 0 ? nthreads : 1);
}

Optimiser::Optimiser(ParamSet& params, AdamWConfig cfg) : params_(params), cfg_(std::move(cfg)) {
    if (params_.empty()) throw std::invalid_argument("r97: Optimiser: empty parameter set");
    if (cfg_.accum_steps <= 0)
        throw std::invalid_argument("r97: Optimiser: accum_steps must be >= 1");
    for (const auto& p : params_) {
        size_t prod = 1;
        for (int64_t s : p.shape) prod *= size_t(s);
        if (prod != p.value.size())
            throw std::invalid_argument("r97: Optimiser: shape and value size disagree for '" +
                                        p.name + "'");
    }
    for (const auto& p : params_) {
        m_.push_back(Param{p.name, p.shape, std::vector<float>(p.value.size(), 0.0f)});
        v_.push_back(Param{p.name, p.shape, std::vector<float>(p.value.size(), 0.0f)});
        g_.push_back(Param{p.name, p.shape, std::vector<float>(p.value.size(), 0.0f)});
    }
    offsets_.resize(params_.size() + 1);
    for (size_t i = 0; i < params_.size(); ++i)
        offsets_[i + 1] = offsets_[i] + params_[i].value.size();
    decay_.resize(params_.size(), 1);
    for (size_t i = 0; i < params_.size(); ++i)
        for (const auto& nd : cfg_.no_decay)
            if (!nd.empty() && params_[i].name.find(nd) != std::string::npos) {
                decay_[i] = 0;
                break;
            }
    nthreads_ = cfg_.threads > 0 ? cfg_.threads : int(std::thread::hardware_concurrency());
    if (nthreads_ <= 0) nthreads_ = 1;
}

Optimiser::~Optimiser() = default;

bool Optimiser::step(const ParamSet& grads) {
    if (grads.size() != params_.size())
        throw std::invalid_argument("r97: Optimiser step: grads do not match the parameter set");
    for (size_t i = 0; i < params_.size(); ++i)
        if (grads[i].name != params_[i].name || grads[i].shape != params_[i].shape ||
            grads[i].value.size() != params_[i].value.size())
            throw std::invalid_argument(
                "r97: Optimiser step: grads do not match the parameter set");
    const size_t total = offsets_.back();
    const auto* off = offsets_.data();
    const auto* gp = grads.data();
    auto* gp0 = g_.data();
    parallel_ranges(total, nthreads_, [&](size_t lo, size_t hi) {
        size_t i = lo;
        while (i < hi) {
            const Decoded d = decode(off, i, hi);
            const float* gd = gp[d.p].value.data();
            float* acc = gp0[d.p].value.data();
            for (size_t k = 0; k < d.len; ++k) acc[d.off + k] += gd[d.off + k];
            i += d.len;
        }
    });
    ++accum_count_;
    if (accum_count_ < cfg_.accum_steps) return false;

    // The mean of the accumulated gradients, then the global-norm clip in place.
    const float inv_k = 1.0f / float(accum_count_);
    parallel_ranges(total, nthreads_, [&](size_t lo, size_t hi) {
        size_t i = lo;
        while (i < hi) {
            const Decoded d = decode(off, i, hi);
            float* acc = gp0[d.p].value.data();
            for (size_t k = 0; k < d.len; ++k) acc[d.off + k] *= inv_k;
            i += d.len;
        }
    });
    if (cfg_.clip_norm > 0.0f) clip_impl(g_, cfg_.clip_norm, nthreads_);

    const int t = step_ + 1;
    const float lr = scheduled_lr(cfg_, t);
    const float b1 = cfg_.beta1, b2 = cfg_.beta2, eps = cfg_.eps;
    const float bc1 = 1.0f - std::pow(b1, float(t));
    const float bc2 = 1.0f - std::pow(b2, float(t));
    const size_t* offd = offsets_.data();
    auto* wp = params_.data();
    auto* mp = m_.data();
    auto* vp = v_.data();
    auto* acc0 = g_.data();
    const uint8_t* dec = decay_.data();
    parallel_ranges(total, nthreads_, [&](size_t lo, size_t hi) {
        size_t i = lo;
        while (i < hi) {
            const Decoded d = decode(offd, i, hi);
            const float wd = dec[d.p] && cfg_.weight_decay != 0.0f ? cfg_.weight_decay : 0.0f;
            const float* gd = acc0[d.p].value.data();
            float* md = mp[d.p].value.data();
            float* vd = vp[d.p].value.data();
            float* w = wp[d.p].value.data();
            for (size_t k = 0; k < d.len; ++k) {
                const size_t j = d.off + k;
                const float g = gd[j];
                const float m = md[j] = b1 * md[j] + (1.0f - b1) * g;
                const float v = vd[j] = b2 * vd[j] + (1.0f - b2) * g * g;
                const float mh = m / bc1;
                const float vh = v / bc2;
                w[j] -= lr * (mh / (std::sqrt(vh) + eps) + wd * w[j]);
            }
            i += d.len;
        }
    });

    // Clear the buffer for the next accumulation window.
    parallel_ranges(total, nthreads_, [&](size_t lo, size_t hi) {
        size_t i = lo;
        while (i < hi) {
            const Decoded d = decode(offd, i, hi);
            float* acc = acc0[d.p].value.data();
            for (size_t k = 0; k < d.len; ++k) acc[d.off + k] = 0.0f;
            i += d.len;
        }
    });
    ++step_;
    accum_count_ = 0;
    return true;
}

std::vector<uint8_t> Optimiser::serialise() const {
    std::vector<uint8_t> out;
    out.reserve(sizeof(float) * 4 * offsets_.back() + 1024);
    for (int i = 0; i < 6; ++i) out.push_back(uint8_t(kMagic[i]));
    put(out, kVersion);
    out.push_back(0);
    out.push_back(0);
    put(out, int64_t(step_));
    put(out, int32_t(accum_count_));
    put(out, int32_t(params_.size()));
    for (size_t i = 0; i < params_.size(); ++i) {
        const size_t n = params_[i].value.size();
        put(out, uint64_t(n));
        const float* src[4] = {params_[i].value.data(), m_[i].value.data(), v_[i].value.data(),
                               g_[i].value.data()};
        for (int a = 0; a < 4; ++a) put_floats(out, src[a], n);
    }
    return out;
}

void Optimiser::restore(const std::vector<uint8_t>& bytes) {
    const uint8_t* p = bytes.data();
    const uint8_t* end = p + bytes.size();
    try {
        for (int i = 0; i < 6; ++i) {
            uint8_t c;
            p = get(p, end, c);
            if (c != uint8_t(kMagic[i]))
                throw std::runtime_error("r97::Optimiser::restore: bad magic");
        }
        uint16_t version, pad;
        p = get(p, end, version);
        p = get(p, end, pad);
        if (version != kVersion)
            throw std::runtime_error("r97::Optimiser::restore: unsupported version " +
                                     std::to_string(version));
        int64_t step;
        int32_t accum_count, nparams;
        p = get(p, end, step);
        p = get(p, end, accum_count);
        p = get(p, end, nparams);
        if (step < 0 || accum_count < 0 || accum_count >= cfg_.accum_steps)
            throw std::runtime_error("r97::Optimiser::restore: bad step or accumulation count");
        if (size_t(nparams) != params_.size())
            throw std::runtime_error("r97::Optimiser::restore: wrong parameter count");
        ParamSet w = params_, m = m_, v = v_, g = g_;  // staged, so a throw leaves us whole
        for (size_t i = 0; i < params_.size(); ++i) {
            uint64_t n;
            p = get(p, end, n);
            if (n != params_[i].value.size())
                throw std::runtime_error("r97::Optimiser::restore: wrong size for '" +
                                         params_[i].name + "'");
            float* dst[4] = {w[i].value.data(), m[i].value.data(), v[i].value.data(),
                             g[i].value.data()};
            for (int a = 0; a < 4; ++a) p = get_floats(p, end, dst[a], n);
        }
        if (p != end) throw std::runtime_error("r97::Optimiser::restore: trailing bytes");
        params_ = std::move(w);
        m_ = std::move(m);
        v_ = std::move(v);
        g_ = std::move(g);
        step_ = int(step);
        accum_count_ = accum_count;
    } catch (const std::runtime_error&) {
        throw;
    }
}

}  // namespace r97
