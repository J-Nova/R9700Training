// src/step_host.cpp -- the host join. See include/r97/step_host.h and docs/STEPS-HOST.md.
#include "r97/step_host.h"

#if defined(RAD_ISSUE_OBSERVER) && defined(RAD_REPLAY_BLOCK)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "r97/adapter_io.h"
#include "r97/adjoint.h"
#include "r97/backward.h"
#include "r97/live_adapter.h"
#include "r97/replay.h"
#include "r97/tape.h"

namespace r97 {
namespace {

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

int64_t elem_size(const std::string& dtype) {
    if (dtype == "bf16" || dtype == "f16") return 2;
    if (dtype == "f32" || dtype == "i32" || dtype == "u32") return 4;
    if (dtype == "f64" || dtype == "i64" || dtype == "u64") return 8;
    if (dtype == "i8" || dtype == "u8" || dtype == "f8") return 1;
    return 0;
}

double read_elem(const std::string& dtype, const void* data, int64_t i) {
    if (dtype == "f32") return static_cast<const float*>(data)[i];
    if (dtype == "f64") return static_cast<const double*>(data)[i];
    if (dtype == "i32") return static_cast<const int32_t*>(data)[i];
    if (dtype == "u32") return static_cast<const uint32_t*>(data)[i];
    if (dtype == "i64") return static_cast<double>(static_cast<const int64_t*>(data)[i]);
    if (dtype == "u64") return static_cast<double>(static_cast<const uint64_t*>(data)[i]);
    if (dtype == "i8") return static_cast<const int8_t*>(data)[i];
    if (dtype == "u8") return static_cast<const uint8_t*>(data)[i];
    if (dtype == "bf16") {
        const uint32_t u = static_cast<uint32_t>(static_cast<const uint16_t*>(data)[i]) << 16;
        float f;
        std::memcpy(&f, &u, sizeof f);
        return f;
    }
    if (dtype == "f16") {
        const uint16_t h = static_cast<const uint16_t*>(data)[i];
        const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1fu;
        uint32_t mant = h & 0x3ffu;
        uint32_t bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign;
            } else {
                exp = 127 - 15 + 1;
                while ((mant & 0x400u) == 0) {
                    mant <<= 1;
                    --exp;
                }
                bits = sign | (exp << 23) | ((mant & 0x3ffu) << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7f800000u | (mant << 13);
        } else {
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &bits, sizeof f);
        return f;
    }
    throw std::runtime_error("host step: cannot snapshot operand dtype '" + dtype + "'");
}

// One operand as the op saw it, dense over the operand's full declared shape. The tape keeps the
// slice the kernel saw, which for a ranged operand starts at row_lo; the walk's adjoints expect
// the full buffer with the untouched rows zero.
AdValue snapshot(const Operand& op, const OperandContent& content) {
    AdValue v;
    v.shape = op.shape;
    const int64_t total = numel(op.shape);
    v.value.assign(static_cast<size_t>(std::max<int64_t>(total, 0)), 0.0);
    if (!content.data || content.bytes == 0 || total == 0) return v;

    const int64_t es = elem_size(op.dtype);
    if (es == 0)
        throw std::runtime_error("host step: cannot snapshot operand dtype '" + op.dtype + "'");
    const int64_t count = static_cast<int64_t>(content.bytes) / es;
    if (count <= 0) return v;

    int64_t cols = total;
    int64_t declared_rows = 1;
    if (!op.shape.empty()) {
        declared_rows = op.shape[0];
        cols = declared_rows > 0 ? total / declared_rows : 0;
    }
    for (int64_t i = 0; i < count; ++i) {
        int64_t at;
        if (op.rows == 0 || count == total) {
            at = i;
        } else {
            const int64_t r = i / (cols > 0 ? cols : 1);
            const int64_t c = cols > 0 ? i % cols : 0;
            at = (op.row_lo + r) * cols + c;
        }
        if (at < 0 || at >= total) continue;
        v.value[static_cast<size_t>(at)] = read_elem(op.dtype, content.data, i);
    }
    return v;
}

bool residual_shape(const InputCotangent& in, int64_t tokens, int64_t hidden) {
    return in.value.size() == static_cast<size_t>(tokens * hidden) && !in.shape.empty() &&
           in.shape.back() == hidden;
}

const InputCotangent* pick_boundary(const std::vector<InputCotangent>& inputs, int64_t tokens,
                                    int64_t hidden) {
    const InputCotangent* best = nullptr;
    for (const InputCotangent& in : inputs) {
        if (!residual_shape(in, tokens, hidden)) continue;
        if (best == nullptr || in.value.size() > best->value.size()) best = &in;
    }
    return best;
}

std::string module_leaf(const std::string& module) {
    const size_t dot = module.find_last_of('.');
    return dot == std::string::npos ? module : module.substr(dot + 1);
}

// The weight a lora record belongs to, from the adapter factor it names: lora.<weight>.a or .b.
std::string record_weight(const OpRecord& rec) {
    for (const Operand& o : rec.operands) {
        if (o.buffer.rfind("lora.", 0) != 0) continue;
        const size_t dot = o.buffer.find_last_of('.');
        if (dot == std::string::npos) continue;
        const std::string leaf = o.buffer.substr(dot + 1);
        if (leaf == "a" || leaf == "b") return o.buffer.substr(5, dot - 5);
    }
    return {};
}

void set_error(std::string* err, const char* what) {
    if (err) *err = what;
}

}  // namespace

StepFn host_step(ForwardDriver& driver, const Model& model, const HostStepOptions& options,
                 HostStepReport* report) {
    const ModelMeta meta = read_model_meta(driver.config().model);
    if (meta.hidden <= 0)
        throw std::runtime_error("host step: the model's metadata gives no hidden size");

    return [&driver, model, options, report, meta](const Batch& batch, const ParamSet& params,
                                                   ParamSet* grads) -> StepResult {
        int64_t tokens = 0;
        for (const TokenSeq& s : batch) tokens += static_cast<int64_t>(s.ids.size());
        if (tokens <= 0) throw std::runtime_error("host step: the batch has no token rows");

        WireBuffers wire;
        if (model.delta) model.delta->to_wire(params, &wire);

        std::set<std::string> target_set;
        std::map<std::string, std::map<int64_t, std::string>> site_module;
        int64_t max_rank = 0;
        for (const WireBuffers::Site& site : wire.sites) {
            site_module[site.weight_name][site.slot] = site.module;
            target_set.insert(module_leaf(site.module));
            max_rank = std::max(max_rank, site.rank);
        }
        std::vector<std::string> targets(target_set.begin(), target_set.end());
        const bool live = model.delta && !wire.sites.empty();
        driver.set_lora(live ? 1 : 0, live ? std::max<int64_t>(max_rank, 1) : 0, targets,
                        live ? "train" : std::string());

        std::shared_ptr<BoundaryRecorder> recorder;
        driver.set_engine_hook([&driver, &recorder, &wire, live, targets, max_rank, tokens,
                               meta](rad::Engine& engine) {
            if (live) {
                LiveAdapterConfig cfg{"train", targets, std::max<int64_t>(max_rank, 1), 1.0};
                std::string err;
                std::unique_ptr<LiveAdapter> adapter = LiveAdapter::open(engine, cfg, &err);
                if (!adapter)
                    throw std::runtime_error("host step: cannot open the live adapter: " + err);
                if (adapter->update(wire, &err) != 0)
                    throw std::runtime_error("host step: cannot fill the live adapter: " + err);
            }
            recorder = std::make_shared<BoundaryRecorder>(
                driver.tape(), tokens, meta.hidden);
        });

        const StepResult result = driver.step(model)(batch, params, nullptr);
        if (!recorder)
            throw std::runtime_error("host step: the engine hook did not install the recorder");
        recorder->detach();
        const ResidualStore store = recorder->store();

        const Tape& tape = driver.last().tape();
        const std::vector<OpRecord>& ops = tape.ops();
        const std::vector<Tape::Block>& blocks = tape.blocks();
        if (blocks.empty())
            throw std::runtime_error("host step: the step has no block brackets to replay");
        if (store.boundaries() != static_cast<int64_t>(blocks.size()) + 1)
            throw std::runtime_error("host step: the boundary store does not match the blocks");

        // The head and the final norm are the ops after the last block. They are the step's last
        // ops, so their operand contents are still live and can be copied now. Everything inside a
        // block is copied by its replay instead, one layer at a time.
        std::vector<OpRecord> head_records(ops.begin() + static_cast<std::ptrdiff_t>(blocks.back().end),
                                           ops.end());
        StoredSnapshots head_snaps;
        for (const OpRecord& rec : head_records) {
            const std::vector<OperandContent>& content = tape.contents(static_cast<size_t>(rec.position));
            for (size_t j = 0; j < rec.operands.size(); ++j)
                head_snaps.set(rec.position, j, snapshot(rec.operands[j], content[j]));
        }

        BackwardOptions bo;
        bo.grads = grads;
        bo.prefix = model.delta ? model.delta->grad_prefix() : "";
        if (!site_module.empty()) {
            bo.slot_prefix = [site_module](const OpRecord& rec, int64_t slot) -> std::string {
                const std::string weight = record_weight(rec);
                const auto w = site_module.find(weight);
                if (w == site_module.end()) return {};
                const auto s = w->second.find(slot);
                return s == w->second.end() ? std::string() : s->second;
            };
        }

        HostStepReport measured;
        measured.loss = result.loss;
        measured.scored = result.scored;
        measured.tokens = tokens;
        measured.hidden = meta.hidden;
        measured.layers = static_cast<int64_t>(blocks.size());
        measured.residual_store_bytes = store.bytes();
        if (options.keep_boundaries) {
            measured.boundary_name.assign(static_cast<size_t>(measured.layers + 1), {});
            measured.boundary_value.assign(static_cast<size_t>(measured.layers + 1), {});
            measured.boundary_cotangent.assign(static_cast<size_t>(measured.layers + 1), {});
        }

        const BoundaryNames names = BoundaryNames::from(driver.last());
        const BackwardResult head = backward_block(head_records, head_snaps, names,
                                                   driver.last().seeds(), {}, bo);
        measured.peak_snapshot_bytes = head.report.peak_snapshot_bytes;
        measured.peak_cotangent_bytes = head.report.peak_cotangent_bytes;

        const InputCotangent* above = pick_boundary(head.inputs, tokens, meta.hidden);
        if (!above)
            throw std::runtime_error("host step: the head walk left no cotangent on the final hidden");

        auto store_boundary = [&](int64_t boundary, const InputCotangent* in) {
            if (!options.keep_boundaries || !in) return;
            measured.boundary_name[static_cast<size_t>(boundary)] = in->buffer;
            measured.boundary_cotangent[static_cast<size_t>(boundary)] = in->value;
            const uint16_t* value = store.at(boundary);
            if (!value) return;
            const int64_t row = meta.hidden;
            std::vector<double>& out = measured.boundary_value[static_cast<size_t>(boundary)];
            out.resize(static_cast<size_t>(tokens * row));
            for (int64_t i = 0; i < tokens * row; ++i) {
                const uint32_t u = static_cast<uint32_t>(value[i]) << 16;
                float f;
                std::memcpy(&f, &u, sizeof f);
                out[static_cast<size_t>(i)] = f;
            }
        };

        store_boundary(measured.layers, above);
        IncomingCotangent incoming{above->buffer, above->row_lo, above->row_hi, above->value};

        LayerReplay replay(*driver.engine(), driver.tape(), store);
        for (int64_t layer = measured.layers - 1; layer >= 0; --layer) {
            StoredSnapshots snaps;
            replay.replay(layer, [&snaps](const OpRecord& op,
                                          const std::vector<OperandContent>& content) {
                for (size_t j = 0; j < op.operands.size(); ++j)
                    snaps.set(op.position, j, snapshot(op.operands[j], content[j]));
            });

            const Tape::Block& range = blocks[static_cast<size_t>(layer)];
            std::vector<OpRecord> records(ops.begin() + static_cast<std::ptrdiff_t>(range.begin),
                                          ops.begin() + static_cast<std::ptrdiff_t>(range.end));
            const BackwardResult walked = backward_block(records, snaps, BoundaryNames{}, {},
                                                         {incoming}, bo);
            measured.peak_snapshot_bytes =
                std::max(measured.peak_snapshot_bytes, walked.report.peak_snapshot_bytes);
            measured.peak_cotangent_bytes =
                std::max(measured.peak_cotangent_bytes, walked.report.peak_cotangent_bytes);

            const InputCotangent* below = pick_boundary(walked.inputs, tokens, meta.hidden);
            if (!below)
                throw std::runtime_error("host step: layer " + std::to_string(layer) +
                                         " left no cotangent on its input boundary");
            store_boundary(layer, below);
            incoming = IncomingCotangent{below->buffer, below->row_lo, below->row_hi, below->value};
        }

        if (report) *report = std::move(measured);
        return result;
    };
}

}  // namespace r97

#else

#include <stdexcept>

namespace r97 {
StepFn host_step(ForwardDriver&, const Model&, const HostStepOptions&, HostStepReport*) {
    throw std::runtime_error("host step: this build has no training patches");
}
}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER && RAD_REPLAY_BLOCK
