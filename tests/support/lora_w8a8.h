// tests/support/lora_w8a8.h -- a PEFT adapter on the int8 fixture, and the tape of a run with it.
//
// The int8 fixture's trunk stores its weights i8 with a bf16 scale per 128 columns and a /fwht128
// rotation (tests/fixtures/w8a8_fwht128.recipe), so an adapter that rides it says which basis its
// A factors are in: `"radiance_input_basis": "fwht128"` in adapter_config.json (docs/ADAPTER.md).
// This header writes such an adapter -- random A and B over every module the fixture declares, on
// every layer that declares it, one rank and one alpha -- and runs the engine's prefill with it as
// the prefill's adapter, handing back the tape of that step with every operand's bytes copied.
// Copied, because a block reuses the activation buffers the previous one filled: by the end of the
// step an address holds the last writer's values, and a test that compares what each site computed
// needs the values the site's own ops saw.
//
// Header-only, like the rest of tests/support/: one includer, no object file to collect.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "fixture_w8a8.h"  // the int8 fixture, and through it fixture.h (run_prefill, tiny)
#include "r97/tape.h"

// The prefill is patches/training/'s: 0103-prefill-logits.patch adds run_prefill(),
// 0104-prefill-lora.patch lets that prefill run a named adapter, 0101-issue-observer.patch the
// tape, and 0119-issue-observer-hook.patch the Engine accessor it attaches through. A serving
// build applies none of the set, so the body compiles to nothing there (docs/FIXTURES.md) and an
// includer needs no guard of its own.
#if R97_FIXTURE_PREFILL && defined(RAD_ISSUE_OBSERVER)

namespace r97 {
namespace lora_w8a8 {

// The name the adapter is loaded under and the name the prefill runs: radiance takes adapters as
// NAME=PATH (--lora), and cfg.prefill_lora names one of those.
inline constexpr const char* kAdapterName = "mine";

// ============================================================================ the modules
// A linear the fixture's trunk declares, as PEFT names it: the module's rows and columns in the
// weight it reads (a fused weight carries two or three modules' rows), and the layers that have it.
// The fixture is 4 layers, every fourth one full attention: layers 0..2 are gated delta net layers
// with an in_proj_qkv/in_proj_z/out_proj, layer 3 a self_attn layer with q/k/v/o_proj, and all four
// have the MLP.
enum Layers { kAllLayers = 0, kAttentionLayers = 1, kDeltaNetLayers = 2 };

struct Module {
    const char* leaf;
    int64_t     rows;
    int64_t     cols;
    int         layers;
};

inline const Module kModules[] = {
    { "self_attn.q_proj", 256, 128, kAttentionLayers },   // q and its gate, one weight
    { "self_attn.k_proj", 64, 128, kAttentionLayers },
    { "self_attn.v_proj", 64, 128, kAttentionLayers },
    { "self_attn.o_proj", 128, 128, kAttentionLayers },
    { "linear_attn.in_proj_qkv", 128, 128, kDeltaNetLayers },
    { "linear_attn.in_proj_z", 64, 128, kDeltaNetLayers },
    { "linear_attn.out_proj", 128, 64, kDeltaNetLayers },
    { "mlp.gate_proj", 256, 128, kAllLayers },
    { "mlp.up_proj", 256, 128, kAllLayers },
    { "mlp.down_proj", 128, 256, kAllLayers },
};

inline bool declared_on(const Module& m, int64_t layer) {
    if (m.layers == kAllLayers) return true;
    const bool attention = (layer + 1) % 4 == 0;
    return (m.layers == kAttentionLayers) == attention;
}

// Every module an adapter in the rotated basis can ride. `linear_attn.out_proj` is not among them
// at this checkpoint: its GEMM's K is the delta net's vdim, 64, and the recipe rotates a weight
// only when its K divides into 128-column groups, so that weight stays bf16 with a natural
// activation. docs/ADAPTER.md's 27b container has a vdim of 128 and rotates it.
inline std::vector<std::string> targetable() {
    std::vector<std::string> out;
    for (const Module& m : kModules)
        if (std::string(m.leaf) != "linear_attn.out_proj") out.push_back(m.leaf);
    return out;
}

// ============================================================================ the adapter
struct Spec {
    std::vector<std::string> leaves;                  // empty: targetable()
    int64_t                  rank  = 4;
    int64_t                  alpha = 8;
    bool                     key   = true;            // write radiance_input_basis
    const char*              basis = "fwht128";       // its value when `key`
    bool                     zero_b = false;          // B = 0: the adapter must change nothing
    // A module outside the trunk's layers: the vocabulary weight, which is never a site.
    bool lm_head = false;
};

// Write a PEFT directory to `dir` (replacing what was there) and return it. The A and B factors are
// `tiny::noise` over a fixed seed, so two calls write the same adapter.
inline std::string write_adapter(const std::string& dir, const Spec& spec = Spec()) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    if (ec) throw std::runtime_error("lora_w8a8: cannot make " + dir + ": " + ec.message());

    const std::vector<std::string> leaves = spec.leaves.empty() ? targetable() : spec.leaves;
    std::vector<tiny::Tensor> ts;
    tiny::Rng rng(17);
    if (!spec.lm_head) {
        for (const Module& m : kModules) {
            if (std::find(leaves.begin(), leaves.end(), m.leaf) == leaves.end()) continue;
            for (int64_t l = 0; l < 4; ++l) {
                if (!declared_on(m, l)) continue;
                const std::string p = "base_model.model.model.language_model.layers." +
                                      std::to_string(l) + "." + m.leaf;
                ts.push_back(tiny::noise(p + ".lora_A.weight", { spec.rank, m.cols }, rng, 0.1f));
                ts.push_back(tiny::noise(p + ".lora_B.weight", { m.rows, spec.rank }, rng,
                                         spec.zero_b ? 0.f : 0.1f));
            }
        }
    } else {
        // The vocabulary weight is [n_vocab, hidden], the fixture's 320 x 128.
        ts.push_back(
            tiny::noise("base_model.model.lm_head.lora_A.weight", { spec.rank, 128 }, rng, 0.1f));
        ts.push_back(
            tiny::noise("base_model.model.lm_head.lora_B.weight", { 320, spec.rank }, rng, 0.1f));
    }
    if (!tiny::write_safetensors(dir + "/adapter_model.safetensors", ts))
        throw std::runtime_error("lora_w8a8: cannot write " + dir + "/adapter_model.safetensors");

    // target_modules names the leaf, and the fixture declares every one of them.
    std::vector<std::string> names;
    if (spec.lm_head) {
        names.push_back("lm_head");
    } else {
        for (const std::string& leaf : leaves) {
            const size_t dot = leaf.rfind('.');
            const std::string name = dot == std::string::npos ? leaf : leaf.substr(dot + 1);
            if (std::find(names.begin(), names.end(), name) == names.end()) names.push_back(name);
        }
    }
    std::string json = "{\"peft_type\":\"LORA\",\"r\":" + std::to_string(spec.rank) +
                       ",\"lora_alpha\":" + std::to_string(spec.alpha) + ",\"target_modules\":[";
    for (size_t i = 0; i < names.size(); ++i) json += (i ? ",\"" : "\"") + names[i] + "\"";
    json += "],\"base_model_name_or_path\":\"tiny\"";
    if (spec.key)
        json += std::string(",\"radiance_input_basis\":\"") + spec.basis + "\"";
    json += "}";
    tiny::write_text(dir + "/adapter_config.json", json);
    return dir;
}

// ============================================================================ the engine
// The config a prefill of the fixture needs, with `adapter` (empty for none) loaded and run.
inline rad::Config config(const Fixture& f, const std::vector<std::vector<int32_t>>& prompts,
                          const std::string& adapter, int64_t rank) {
    rad::Config cfg;
    cfg.model            = f.container;
    cfg.radiance_home    = detail::radiance_home();
    cfg.kernel_hierarchy = { "libavx", "libref" };
    cfg.n_spec           = 0;
    cfg.max_tok          = 512;
    cfg.max_seqs         = 8;
    cfg.max_ctx          = 512;
    cfg.prefill_prompts  = prompts;
    if (!adapter.empty()) {
        cfg.prefill_lora = kAdapterName;
        cfg.lora.adapters.push_back({ kAdapterName, adapter });
        cfg.lora.max_rank = rank;
    }
    return cfg;
}

// Run one prompt through the engine's prefill and return its logits, with the step recorded into
// `tape` (contents copied) when one is given. Throws std::runtime_error when the engine cannot
// start on the container or the prefill fails.
inline Prefill run(const Fixture& f, const std::vector<int32_t>& ids, const std::string& adapter,
                   Tape* tape = nullptr, int64_t rank = 4) {
    rad::Engine engine(config(f, { ids }, adapter, rank));
    const int s = engine.start();
    if (s < 0)
        throw std::runtime_error(std::string("lora_w8a8: the engine could not start on ") +
                                 f.container + ": " + rad_strerror(s));
    if (tape != nullptr) {
        tape->keep_contents(true);
        tape->attach(engine.issue_ctx());
    }
    const int served = engine.serve(std::unique_ptr<rad::StallWatch>());
    if (tape != nullptr) tape->detach();
    if (served < 0)
        throw std::runtime_error(std::string("lora_w8a8: the prefill failed: ") +
                                 rad_strerror(served));

    const std::vector<std::vector<float>>& rows = engine.config().prefill_logits;
    if (rows.empty() || rows[0].size() != ids.size() * (size_t)f.n_vocab)
        throw std::runtime_error("lora_w8a8: the engine returned no logits for the prompt");
    Prefill out;
    out.n_tokens = (int32_t)ids.size();
    out.n_vocab  = f.n_vocab;
    out.logits   = rows[0];
    return out;
}

// What radiance said when it turned an adapter away: the status start() returned, and the `E `
// lines it printed. The engine prints them to stderr, so this captures them for the length of the
// start; a test that reads them is testing the message the operator sees.
struct Refusal {
    int         code = 0;
    std::string said;

    bool mentions(const char* text) const { return said.find(text) != std::string::npos; }
};

inline Refusal refuse(const Fixture& f, const std::string& adapter, int64_t rank = 4) {
    namespace fs = std::filesystem;
    const std::string   name = "lora_w8a8_refusal_" + std::to_string(::getpid()) + ".err";
    const std::string   path = (fs::path(radtest::tmpdir()) / name).string();
    std::fflush(stderr);
    const int saved = dup(STDERR_FILENO);
    FILE*     out   = std::fopen(path.c_str(), "w");
    if (out == nullptr || saved < 0)
        throw std::runtime_error("lora_w8a8: cannot capture stderr");
    dup2(fileno(out), STDERR_FILENO);

    Refusal r;
    {
        rad::Engine engine(config(f, {}, adapter, rank));
        r.code = engine.start();
    }

    std::fflush(stderr);
    dup2(saved, STDERR_FILENO);
    close(saved);
    std::fclose(out);
    std::ifstream in(path);
    std::string   line;
    while (std::getline(in, line))
        if (line.rfind("E ", 0) == 0) r.said += line + "\n";
    std::error_code ec;
    fs::remove(path, ec);
    return r;
}

// ============================================================================ the tape
inline int64_t int_param(const OpRecord& rec, const char* key) {
    for (const auto& [k, v] : rec.params)
        if (k == key && std::holds_alternative<int64_t>(v)) return std::get<int64_t>(v);
    return -1;
}

// A string parameter, or "" when the op does not carry it.
inline std::string str_param(const OpRecord& rec, const char* key) {
    for (const auto& [k, v] : rec.params)
        if (k == key && std::holds_alternative<std::string>(v)) return std::get<std::string>(v);
    return std::string();
}

// One operand as numbers: `cols` is the stride to read it with, which for a pool of [slots, rows,
// cols] or a scale of [1, n] is not shape[1].
struct Matrix {
    const uint8_t* bytes = nullptr;
    const char*    dtype = "";
    int64_t        cols  = 0;

    int64_t width() const {
        if (std::strcmp(dtype, "i8") == 0) return 1;
        if (std::strcmp(dtype, "bf16") == 0) return 2;
        if (std::strcmp(dtype, "f32") == 0 || std::strcmp(dtype, "i32") == 0) return 4;
        throw std::runtime_error(std::string("lora_w8a8: no reading for dtype ") + dtype);
    }

    double at(int64_t r, int64_t c) const {
        const uint8_t* p = bytes + (r * cols + c) * width();
        if (std::strcmp(dtype, "i8") == 0) return (double)(int8_t)*p;
        if (std::strcmp(dtype, "f32") == 0) {
            float v;
            std::memcpy(&v, p, 4);
            return v;
        }
        if (std::strcmp(dtype, "i32") == 0) {
            int32_t v;
            std::memcpy(&v, p, 4);
            return v;
        }
        uint16_t u;
        std::memcpy(&u, p, 2);
        const uint32_t f = (uint32_t)u << 16;   // bf16 to f32 is the top half of the bits
        float          v;
        std::memcpy(&v, &f, 4);
        return v;
    }
};

inline Matrix matrix(const Tape& t, size_t op, size_t operand, int64_t cols) {
    const OperandContent& c = t.content(op, operand);
    Matrix                m;
    m.bytes = static_cast<const uint8_t*>(c.data);
    m.dtype = t.ops()[op].operands[operand].dtype.c_str();
    m.cols  = cols;
    if (m.bytes == nullptr) throw std::runtime_error("lora_w8a8: an operand has no contents");
    return m;
}

// One place an adapter lands: the trunk weight, the GEMM that reads it, the shrink radiance puts
// before it and the expand after it, and the copy that holds the site's output. The observer runs
// after its op (patches/training/0101-issue-observer.patch), so the expand's own `y` (:inout)
// operand is copied once the delta is in it: that copy is the site's output, base plus delta. A
// site whose expand did not name the whole output is not readable after the step and
// `has_output()` is false.
struct Site {
    std::string weight;                  // "blk.3.ffn_down.weight"
    size_t      gemm = 0, shrink = 0, expand = 0;
    int64_t     rows = 0;                // the tokens the GEMM was issued for
    int64_t     n = 0, k = 0, rank = 0;  // the weight's rows, its columns, the adapter's rank
    size_t      out_op = 0, out_operand = 0;

    bool has_output() const { return out_op != 0 || out_operand != 0; }
};

// The sites of one recorded step, in issue order. Throws when the tape's ops do not carry the
// shape radiance's adapter support gives them (a shrink over a pool named "lora.<weight>.a" whose
// next op is its expand), because a test that silently found no site would pass.
inline std::vector<Site> sites(const Tape& t) {
    const std::vector<OpRecord>& ops = t.ops();
    std::vector<Site>            out;
    for (size_t i = 1; i + 1 < ops.size(); ++i) {
        if (ops[i].op != "lora_shrink" || ops[i].operands.size() < 4) continue;
        const std::string& pool = ops[i].operands[3].buffer;
        if (pool.rfind("lora.", 0) != 0 || pool.size() < 8 ||
            pool.compare(pool.size() - 2, 2, ".a"))
            throw std::runtime_error(
                "lora_w8a8: a lora_shrink's A pool is not 'lora.<weight>.a': " + pool);
        Site s;
        s.weight = pool.substr(5, pool.size() - 7);
        s.shrink = i;
        s.expand = i + 1;
        if (ops[s.expand].op != "lora_expand")
            throw std::runtime_error("lora_w8a8: the op after the shrink on " + s.weight + " is " +
                                     ops[s.expand].op + ", not its expand");
        s.gemm = s.shrink - 1;
        while (s.gemm > 0 && !(ops[s.gemm].op.rfind("gemm", 0) == 0 &&
                               ops[s.gemm].operands.size() > 2 &&
                               ops[s.gemm].operands[2].buffer == s.weight))
            --s.gemm;
        if (ops[s.gemm].operands[2].buffer != s.weight)
            throw std::runtime_error("lora_w8a8: no GEMM feeding the shrink on " + s.weight);
        s.rows = ops[s.gemm].n;
        s.n    = int_param(ops[s.gemm], "N");
        s.k    = int_param(ops[s.gemm], "K");
        s.rank = int_param(ops[s.expand], "R");

        // The expand's `y` is its last operand (:inout), and the copy of it is taken after the
        // expand ran. The GEMM's output is the same buffer, so this copy is base plus delta.
        const size_t want = (size_t)(s.rows * s.n * 2);   // bf16 rows of the site's output
        if (ops[s.expand].operands.size() > 4 && ops[s.expand].operands[4].rows == s.rows &&
            t.content(s.expand, 4).bytes == want) {
            s.out_op      = s.expand;
            s.out_operand = 4;
        }
        out.push_back(std::move(s));
    }
    return out;
}

}  // namespace lora_w8a8
}  // namespace r97

#endif  // R97_FIXTURE_PREFILL && RAD_ISSUE_OBSERVER
