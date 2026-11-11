// tools/grad.cpp -- the `grad` subcommand: build the configured Model, run one host-joined step
// over a JSONL batch, and print the loss with the gradient norm of every adapted linear. The
// model and the delta come from the configuration registry; the join is the same `host_step` the
// training loop uses, so this is a direct look at what a step would hand the optimiser.
#include "commands.h"

#include "r97/adapter_io.h"
#include "r97/chatdata.h"
#include "r97/forward.h"
#include "r97/model.h"
#include "r97/step_host.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <system_error>

#if R97_FORWARD

#include "engine.h"  // the complete rad::Engine behind ForwardDriver's unique_ptr

namespace r97tool {
namespace {

struct Args {
    std::string model;
    std::string data;
    std::string delta = "lora";
    int64_t rank = 16;
    double alpha = 32.0;
    int64_t limit = -1;
};

bool parse(int argc, char** argv, Args* a) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](std::string* out) {
            if (i + 1 >= argc) return false;
            *out = argv[++i];
            return true;
        };
        if (arg == "--model") {
            if (!next(&a->model)) return false;
        } else if (arg == "--data") {
            if (!next(&a->data)) return false;
        } else if (arg == "--delta") {
            if (!next(&a->delta)) return false;
        } else if (arg == "--rank") {
            std::string n;
            if (!next(&n)) return false;
            try {
                a->rank = std::stoll(n);
            } catch (...) {
                return false;
            }
            if (a->rank <= 0) return false;
        } else if (arg == "--alpha") {
            std::string n;
            if (!next(&n)) return false;
            try {
                a->alpha = std::stod(n);
            } catch (...) {
                return false;
            }
            if (!(a->alpha > 0.0)) return false;
        } else if (arg == "--limit") {
            std::string n;
            if (!next(&n)) return false;
            try {
                a->limit = std::stoll(n);
            } catch (...) {
                return false;
            }
            if (a->limit < 0) return false;
        } else {
            std::cerr << "r97train grad: unknown argument " << arg << "\n";
            return false;
        }
    }
    return !a->model.empty() && !a->data.empty();
}

std::string radiance_home() {
    if (const char* env = std::getenv("RADIANCE_HOME"); env && *env) return env;
    std::error_code ec;
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) return {};
    const std::filesystem::path beside = exe.parent_path();
    for (const std::filesystem::path& p :
         {beside / "radiance-build" / "radiance_home", beside / "radiance_home"}) {
        if (std::filesystem::exists(p / "architectures", ec)) return p.string();
    }
    return {};
}

// "model.layers.3.mlp.down_proj.lora_A" -> "model.layers.3.mlp.down_proj"; both factors of a
// module share a line.
std::string module_of(const std::string& name) {
    for (const char* suffix : {".lora_A", ".lora_B"}) {
        const size_t at = name.rfind(suffix);
        if (at != std::string::npos) return name.substr(0, at);
    }
    return name;
}

int grad_main(int argc, char** argv) {
    Args args;
    if (!parse(argc, argv, &args)) {
        std::cerr << "usage: r97train grad --model <file.rad> --data <file.jsonl> "
                     "[--delta lora] [--rank N] [--alpha A] [--limit N]\n";
        return 2;
    }

    r97::ChatData chat;
    if (!chat.open(args.model)) {
        std::cerr << "r97train grad: " << chat.error() << "\n";
        return 3;
    }
    r97::Batch batch;
    r97::ChatDataStats stats;
    if (!chat.read_file(args.data, r97::ChatData::kDefaultMaxTokens, &batch, &stats)) {
        std::cerr << "r97train grad: " << chat.error() << "\n";
        return 3;
    }
    if (args.limit >= 0 && size_t(args.limit) < batch.size()) batch.resize(size_t(args.limit));
    if (batch.empty()) {
        std::cerr << "r97train grad: no conversations to score\n";
        return 3;
    }
    int64_t longest = 0;
    for (const r97::TokenSeq& s : batch) longest = std::max(longest, int64_t(s.ids.size()));

    r97::Config cfg;
    cfg.model = args.model;
    cfg.data = args.data;
    cfg.objective = "ce";
    cfg.delta.kind = args.delta;
    cfg.delta.rank = args.rank;
    cfg.delta.alpha = args.alpha;

    try {
        r97::Model model = r97::make_model(cfg);
        const r97::ModelMeta meta = r97::read_model_meta(args.model);
        r97::ParamSet params = model.delta->init(meta, cfg.delta, 0);

        const std::string home = radiance_home();
        if (home.empty()) {
            std::cerr << "r97train grad: no radiance home: set RADIANCE_HOME to the directory "
                         "holding architectures/, kernels/ and quantizers/\n";
            return 3;
        }
        r97::ForwardConfig fwd;
        fwd.model = args.model;
        fwd.radiance_home = home;
        fwd.max_tok = std::max<int64_t>(512, longest);
        fwd.max_ctx = fwd.max_tok;
        fwd.max_seqs = 8;
        std::string err;
        std::unique_ptr<r97::ForwardDriver> driver = r97::ForwardDriver::open(fwd, &err);
        if (!driver) {
            std::cerr << "r97train grad: " << err << "\n";
            return 3;
        }

        r97::HostStepReport report;
        const r97::StepFn step = r97::host_step(*driver, model, {}, &report);
        r97::ParamSet grads;
        step(batch, params, &grads);
        if (report.scored <= 0) {
            std::cerr << "r97train grad: the file has no scored tokens\n";
            return 4;
        }

        std::map<std::string, double> norms;
        for (const r97::Param& p : grads) {
            double sum = 0.0;
            for (float v : p.value) sum += double(v) * double(v);
            norms[module_of(p.name)] += sum;
        }
        std::cout << "loss: " << std::setprecision(12)
                  << report.loss / double(report.scored) << "\n"
                  << "scored: " << report.scored << "\n";
        for (const auto& [module, sum] : norms)
            std::cout << "grad " << module << ": " << std::setprecision(9) << std::sqrt(sum)
                      << "\n";
    } catch (const std::exception& e) {
        std::cerr << "r97train grad: " << e.what() << "\n";
        return 3;
    }
    return 0;
}

R97_REGISTER_COMMAND("grad", "--model <file.rad> --data <file.jsonl> [--delta lora] "
                              "[--rank N] [--alpha A] [--limit N]",
                     grad_main);

}  // namespace
}  // namespace r97tool

#endif  // R97_FORWARD
