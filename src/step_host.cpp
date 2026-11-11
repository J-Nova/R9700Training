// src/step_host.cpp -- the host join. See include/r97/step_host.h and docs/STEPS-HOST.md.
#include "r97/step_host.h"

#if defined(RAD_ISSUE_OBSERVER) && defined(RAD_REPLAY_BLOCK)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine.h"  // the complete rad::Engine behind ForwardDriver's unique_ptr
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
    // The head's ops see the engine's whole buffer, which is longer than the step's rows; its
    // first `tokens` rows are the step's. A block boundary is already the step's row count.
    return in.value.size() >= static_cast<size_t>(tokens * hidden) && !in.shape.empty() &&
           in.shape.back() == hidden && in.row_hi >= tokens;
}

std::vector<double> first_rows(const InputCotangent& in, int64_t tokens, int64_t hidden) {
    const size_t n = static_cast<size_t>(tokens * hidden);
    if (in.value.size() < n) return {};
    return std::vector<double>(in.value.begin(), in.value.begin() + static_cast<ptrdiff_t>(n));
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

// A record names a whole engine buffer, but this step's op only touched `rows` of them, starting
// at `row_lo`. The block walk is over the step's rows, not the engine's capacity: hand every
// adjoint the slice it actually saw, with the row window at zero, so read and write buffers of
// different capacities line up and the snapshots hold no inactive rows. Parameters and other
// whole-buffer operands have no row window and are left as they are.
void tighten_operand(Operand& o) {
    if (o.rows <= 0 || o.shape.empty() || o.rows > o.shape[0]) return;
    o.shape[0] = o.rows;
    o.row_lo = 0;
}

void tighten_rows(OpRecord& rec) {
    for (Operand& o : rec.operands) tighten_operand(o);
}

// The objective reads the logits boundary in the batch's row space, L-1 rows a sequence. The
// head writes the step's row space, all L rows a sequence, after the sampler's rows: its logits
// buffer row for the batch's row r is `hidden_rows()[i] + 1`, because row i of hidden() is the
// head's input row that predicts that scored token. A seed has to move to that space before the
// walk, or it lands on the wrong token (or on a sampler row that predicts nothing).
std::vector<Seed> head_seeds(const ForwardStep& step, const Batch& batch, const std::vector<OpRecord>& head,
                             int64_t tokens) {
    int64_t rows = 0, vocab = 0;
    const std::vector<int64_t> lshape = step.shape("logits");
    if (lshape.size() != 2) throw std::runtime_error("host step: the logits boundary is not 2-D");
    rows = lshape[0];
    vocab = lshape[1];

    int64_t head_rows = 0;
    for (const OpRecord& rec : head)
        for (const Operand& o : rec.operands)
            if (o.written && o.buffer == "logits") head_rows = o.rows;

    // One entry a batch row, -1 where the row predicts nothing. hidden_rows() lists the scored
    // rows in the batch's order, so walking the scores in that order pairs them up.
    std::vector<int64_t> at(static_cast<size_t>(rows), -1);
    const std::vector<int64_t>& hidden_rows = step.hidden_rows();
    size_t scored = 0;
    int64_t off = 0;
    for (const TokenSeq& seq : batch) {
        const int64_t len = static_cast<int64_t>(seq.ids.size());
        for (int64_t p = 1; p < len; ++p) {
            if (p < static_cast<int64_t>(seq.score.size()) && !seq.score[static_cast<size_t>(p)]) continue;
            if (scored >= hidden_rows.size())
                throw std::runtime_error("host step: hidden_rows() is shorter than the scored rows");
            const int64_t batch_row = off + p - 1;
            if (batch_row < 0 || batch_row >= rows)
                throw std::runtime_error("host step: a scored row is outside the logits boundary");
            at[static_cast<size_t>(batch_row)] = hidden_rows[scored];
            ++scored;
        }
        off += len > 0 ? len - 1 : 0;
    }
    if (head_rows <= 0)
        throw std::runtime_error("host step: the head wrote no logits rows to seed");
    if (head_rows < tokens)
        throw std::runtime_error("host step: the head wrote fewer logits rows than tokens");

    std::vector<Seed> out;
    for (const Seed& seed : step.seeds()) {
        if (seed.buffer != "logits") {
            out.push_back(seed);
            continue;
        }
        const int64_t span = seed.row_hi - seed.row_lo;
        if (span <= 0) continue;
        if (static_cast<int64_t>(seed.cotangent.size()) != span * vocab)
            throw std::runtime_error("host step: a logits seed's cotangent does not match its rows");
        Seed moved;
        moved.buffer = seed.buffer;
        moved.row_lo = 0;
        moved.row_hi = head_rows;
        moved.cotangent.assign(static_cast<size_t>(head_rows * vocab), 0.0f);
        for (int64_t r = seed.row_lo; r < seed.row_hi; ++r) {
            const int64_t to = at[static_cast<size_t>(r)];
            if (to < 0) continue;
            if (to >= head_rows)
                throw std::runtime_error("host step: a seed maps past the head's logits rows");
            std::memcpy(moved.cotangent.data() + static_cast<size_t>(to * vocab),
                        seed.cotangent.data() + static_cast<size_t>((r - seed.row_lo) * vocab),
                        static_cast<size_t>(vocab) * sizeof(float));
        }
        out.push_back(std::move(moved));
    }
    return out;
}

// The observer names an unnamed operand after its position in the schema ("raw:2"), so two ops
// can give the same name to buffers of different shapes. A raw or absent operand is per-op by
// definition (a position table, a slot list, a token range), so give each its own name; nothing
// reads an unnamed operand by name.
OpRecord own_raws(OpRecord rec) {
    for (Operand& o : rec.operands)
        if (o.buffer.rfind("raw:", 0) == 0 || o.buffer.rfind("none:", 0) == 0)
            o.buffer += "@" + std::to_string(rec.position);
    return rec;
}

// A weight that several delta sites land in is one fused adapter on the engine's side: the
// shrink concatenates their A factors along the rank and the expand makes their B factors a
// block-diagonal matrix. The walk sees one gradient for the fused pair; split it back into the
// per-module pairs the delta's ParamSet names, using the order and shapes the sites gave.
struct SiteSlice {
    std::string module;
    int64_t rank = 0;
    int64_t out = 0;
    int64_t rank_off = 0;
    int64_t out_off = 0;
};

using SiteGroups = std::map<std::string, std::vector<SiteSlice>>;

SiteGroups group_sites(const WireBuffers& wire) {
    SiteGroups groups;
    for (const WireBuffers::Site& site : wire.sites) {
        if (site.rank < 1) throw std::runtime_error("host step: a delta site has no rank");
        if (site.b.size() % static_cast<size_t>(site.rank) != 0)
            throw std::runtime_error("host step: delta site '" + site.module +
                                     "' has a B whose size is not its rank times its rows");
        SiteSlice slice;
        slice.module = site.module;
        slice.rank = site.rank;
        slice.out = static_cast<int64_t>(site.b.size()) / site.rank;
        std::vector<SiteSlice>& group = groups[site.weight_name];
        if (!group.empty()) {
            const SiteSlice& prev = group.back();
            slice.rank_off = prev.rank_off + prev.rank;
            slice.out_off = prev.out_off + prev.out;
        }
        group.push_back(std::move(slice));
    }
    return groups;
}

std::string fused_prefix(const std::string& weight) { return "@fused:" + weight; }

void add_grad(ParamSet* grads, const std::string& name, const std::vector<int64_t>& shape,
              const std::vector<float>& values) {
    for (Param& p : *grads) {
        if (p.name != name) continue;
        if (p.shape != shape || p.value.size() != values.size())
            throw std::runtime_error("host step: gradient '" + name + "' arrived twice with shapes");
        for (size_t i = 0; i < values.size(); ++i) p.value[i] += values[i];
        return;
    }
    Param p;
    p.name = name;
    p.shape = shape;
    p.value = values;
    grads->push_back(std::move(p));
}

const Param* find_grad(const ParamSet& grads, const std::string& name) {
    for (const Param& p : grads)
        if (p.name == name) return &p;
    return nullptr;
}

// The live forward may skip an adapter path (a zero B is the usual init), so the walk sees no
// record for that factor. Every delta parameter still has a gradient: zero where the path did not
// run. Fill those before the split, which needs both factors' shapes.
void fill_missing_grads(ParamSet* grads, const ParamSet& params) {
    if (!grads) return;
    for (const Param& p : params) {
        if (find_grad(*grads, p.name)) continue;
        Param zero;
        zero.name = p.name;
        zero.shape = p.shape;
        zero.value.assign(p.value.size(), 0.0f);
        grads->push_back(std::move(zero));
    }
}

void split_adapter_grads(ParamSet* grads, const SiteGroups& groups) {
    if (!grads) return;
    ParamSet split;
    std::vector<std::string> replaced;
    for (const auto& [weight, slices] : groups) {
        if (slices.empty()) continue;
        const bool fused = slices.size() > 1;
        const std::string prefix = fused ? fused_prefix(weight) : slices.front().module;
        const Param* a = find_grad(*grads, prefix + ".lora_A");
        const Param* b = find_grad(*grads, prefix + ".lora_B");
        if (!a && !b) continue;
        // A live record can carry only one factor (the other path was skipped); the split then
        // has nothing to do for that weight and leaves the factor it does have in place.
        if (!a || !b) continue;
        int64_t rank_total = 0, out_total = 0;
        for (const SiteSlice& slice : slices) {
            rank_total += slice.rank;
            out_total += slice.out;
        }
        if (rank_total <= 0 || static_cast<int64_t>(a->value.size()) % rank_total != 0 ||
            static_cast<int64_t>(b->value.size()) % rank_total != 0)
            throw std::runtime_error("host step: the gradient of '" + weight +
                                     "' does not match its delta sites");
        const int64_t in = static_cast<int64_t>(a->value.size()) / rank_total;
        const int64_t b_rows = static_cast<int64_t>(b->value.size()) / rank_total;
        // The kernel weight can have unadapted output rows after the adapter's own (a fused qkv
        // with a gate tail, say); the site's B is shorter than the weight and those rows are not
        // part of any module's gradient.
        if (b_rows < out_total)
            throw std::runtime_error("host step: the gradient of '" + weight +
                                     "' is shorter than its delta sites' B factors");
        for (const SiteSlice& slice : slices) {
            std::vector<float> av(static_cast<size_t>(slice.rank * in));
            for (int64_t j = 0; j < slice.rank; ++j)
                for (int64_t k = 0; k < in; ++k)
                    av[static_cast<size_t>(j * in + k)] =
                        a->value[static_cast<size_t>((slice.rank_off + j) * in + k)];
            add_grad(&split, slice.module + ".lora_A", {slice.rank, in}, av);

            std::vector<float> bv(static_cast<size_t>(slice.out * slice.rank));
            for (int64_t n = 0; n < slice.out; ++n)
                for (int64_t j = 0; j < slice.rank; ++j)
                    bv[static_cast<size_t>(n * slice.rank + j)] =
                        b->value[static_cast<size_t>((slice.out_off + n) * rank_total +
                                                     slice.rank_off + j)];
            add_grad(&split, slice.module + ".lora_B", {slice.out, slice.rank}, bv);
        }
        replaced.push_back(prefix + ".lora_A");
        replaced.push_back(prefix + ".lora_B");
    }
    if (replaced.empty()) return;
    grads->erase(std::remove_if(grads->begin(), grads->end(), [&](const Param& p) {
                     return std::find(replaced.begin(), replaced.end(), p.name) !=
                            replaced.end();
                 }),
                 grads->end());
    for (Param& p : split) grads->push_back(std::move(p));
}

std::string module_leaf(const std::string& module) {
    const size_t dot = module.find_last_of('.');
    return dot == std::string::npos ? module : module.substr(dot + 1);
}

// The weight a lora record belongs to, from the adapter factor it names: lora.<weight>.a or .b.
// The record names the kernel weight (`blk.3.ffn_down.weight`), while the delta's site names the
// parameter without the container's `.weight` leaf (`blk.3.ffn_down`); drop that leaf so both
// name the same site.
std::string record_weight(const OpRecord& rec) {
    for (const Operand& o : rec.operands) {
        if (o.buffer.rfind("lora.", 0) != 0) continue;
        const size_t dot = o.buffer.find_last_of('.');
        if (dot == std::string::npos) continue;
        const std::string leaf = o.buffer.substr(dot + 1);
        if (leaf != "a" && leaf != "b") continue;
        std::string weight = o.buffer.substr(5, dot - 5);
        const std::string suffix = ".weight";
        if (weight.size() > suffix.size() &&
            weight.compare(weight.size() - suffix.size(), suffix.size(), suffix) == 0)
            weight.resize(weight.size() - suffix.size());
        return weight;
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
        const SiteGroups groups = group_sites(wire);
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
        for (OpRecord& rec : head_records) {
            rec = own_raws(std::move(rec));
            tighten_rows(rec);
        }
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
            bo.slot_prefix = [site_module, groups](const OpRecord& rec, int64_t slot) -> std::string {
                const std::string weight = record_weight(rec);
                const auto g = groups.find(weight);
                if (g == groups.end())
                    throw std::runtime_error("host step: no delta site for lora weight '" + weight + "'");
                if (g->second.size() > 1) return fused_prefix(weight);
                const auto w = site_module.find(weight);
                const auto s = w->second.find(slot);
                if (s == w->second.end())
                    throw std::runtime_error("host step: lora weight '" + weight +
                                             "' has no delta site at slot " + std::to_string(slot));
                return s->second;
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

        BoundaryNames names = BoundaryNames::from(driver.last());
        int64_t head_logits_rows = 0;
        for (const OpRecord& rec : head_records)
            for (const Operand& o : rec.operands)
                if (o.written && o.buffer == "logits") head_logits_rows = o.rows;
        if (head_logits_rows <= 0)
            throw std::runtime_error("host step: the head wrote no logits rows");
        // The objective's window is L-1 rows a sequence; the head's buffer is all L after the
        // sampler's rows, and the moved seeds live in that space.
        names.add("logits", 0, head_logits_rows);
        const std::vector<Seed> seeds = head_seeds(driver.last(), batch, head_records, tokens);
        const BackwardResult head = backward_block(head_records, head_snaps, names, seeds, {}, bo);
        measured.peak_snapshot_bytes = head.report.peak_snapshot_bytes;
        measured.peak_cotangent_bytes = head.report.peak_cotangent_bytes;

        const InputCotangent* above = pick_boundary(head.inputs, tokens, meta.hidden);
        if (!above) {
            std::string dbg = "host step: the head walk left no cotangent on the final hidden; blocks=" +
                              std::to_string(blocks.size()) + " head_records=" +
                              std::to_string(head_records.size()) + " inputs=" +
                              std::to_string(head.inputs.size());
            for (const auto& in : head.inputs) {
                dbg += " [";
                dbg += in.buffer;
                dbg += " rows=" + std::to_string(in.row_lo) + ".." + std::to_string(in.row_hi) + " shape=";
                for (int64_t d : in.shape) dbg += std::to_string(d) + ",";
                dbg += "]";
            }
            for (const auto& rec : head_records) dbg += " op=" + rec.op;
            throw std::runtime_error(dbg);
        }

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
        IncomingCotangent incoming{above->buffer, 0, tokens,
                                   first_rows(*above, tokens, meta.hidden)};

        LayerReplay replay(*driver.engine(), driver.tape(), store);
        for (int64_t layer = measured.layers - 1; layer >= 0; --layer) {
            StoredSnapshots snaps;
            replay.replay(layer, [&snaps](const OpRecord& op,
                                          const std::vector<OperandContent>& content) {
                for (size_t j = 0; j < op.operands.size(); ++j) {
                    Operand operand = op.operands[j];
                    tighten_operand(operand);
                    snaps.set(op.position, j, snapshot(operand, content[j]));
                }
            });

            const Tape::Block& range = blocks[static_cast<size_t>(layer)];
            std::vector<OpRecord> records(ops.begin() + static_cast<std::ptrdiff_t>(range.begin),
                                          ops.begin() + static_cast<std::ptrdiff_t>(range.end));
            for (OpRecord& rec : records) {
                rec = own_raws(std::move(rec));
                tighten_rows(rec);
            }
            BoundaryNames block_names;
            block_names.add(incoming.buffer, 0, tokens);
            const BackwardResult walked = backward_block(records, snaps, block_names, {},
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
            incoming = IncomingCotangent{below->buffer, 0, tokens,
                                         first_rows(*below, tokens, meta.hidden)};
        }

        fill_missing_grads(grads, params);
        split_adapter_grads(grads, groups);
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
