// src/step_host.cpp -- the host join. See include/r97/step_host.h and docs/STEPS-HOST.md.
#include "rad_runtime.h"  // RAD_ISSUE_OBSERVER / RAD_REPLAY_BLOCK, before the guard

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

#include "engine.h"  // the complete rad::Engine behind ForwardDriver's unique_ptr
#include "rad_planes.h"  // the joined form a quantised lm_head reaches the tape in
#include "rad_plugin.h"  // rad_fp8e4m3_to_f32, for the int8 trunk's fp8 cache operands
#include "r97/step_host.h"
#include "r97/adapter_io.h"
#include "r97/adjoint.h"
#include "r97/backward.h"
#include "r97/live_adapter.h"
#include "r97/replay.h"
#include "r97/tape.h"
#include "join_common.h"  // the join helpers both joins share

namespace r97 {

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
        int64_t max_rank = 0;
        for (const WireBuffers::Site& site : wire.sites) {
            target_set.insert(module_leaf(site.module));
            max_rank = std::max(max_rank, site.rank);
        }
        const SiteGroups groups = group_sites(wire);
        std::vector<std::string> targets(target_set.begin(), target_set.end());
        const bool live = model.delta && !wire.sites.empty();
        driver.set_lora(live ? 1 : 0, live ? std::max<int64_t>(max_rank, 1) : 0, targets,
                        live ? "train" : std::string(), live ? meta.input_basis : std::string());

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
            // The boundary store's plan. In the int8 trunk every block after the first reads the
            // previous block's output beside the residual (its first norm is a fused add), so the
            // plan holds a second copy at every boundary after the first -- without it the
            // reverse replay below would feed those blocks a stale buffer. The natural trunk
            // reads the residual alone. The recorder cross-checks the plan against the tape and
            // throws when the two disagree.
            ResidualPlan plan;
            plan.boundaries = meta.layers + 1;
            plan.tokens     = tokens;
            plan.hidden     = meta.hidden;
            if (meta.input_basis == "fwht128")
                for (int64_t b = 0; b < plan.boundaries; ++b)
                    plan.extra.push_back(b >= 1 && b < meta.layers ? 1 : 0);
            recorder = std::make_shared<BoundaryRecorder>(driver.tape(), ResidualStore(plan));
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
            if (options.reference_head_split) mark_join_lora(rec);
        }
        StoredSnapshots head_snaps;
        for (const OpRecord& rec : head_records) {
            const std::vector<OperandContent>& content = tape.contents(static_cast<size_t>(rec.position));
            for (size_t j = 0; j < rec.operands.size(); ++j)
                head_snaps.set(rec.position, j, snapshot(rec.operands[j], content[j]));
        }

        BackwardOptions bo;
        bo.grads = grads;
        const std::string grad_prefix = model.delta ? model.delta->grad_prefix() : "";
        bo.prefix = grad_prefix.c_str();
        bo.in_place_passthrough = true;
        // The engine pools a fused weight's modules into one lora record; write one fused factor
        // and let split_adapter_grads put each module's rows under its own name.
        bo.fused_slots = true;
        if (!groups.empty()) {
            bo.record_prefix = [groups, grad_prefix](const OpRecord& rec) -> std::string {
                const std::string weight = record_weight(rec);
                if (weight.empty()) return grad_prefix;
                const auto g = groups.find(weight);
                if (g == groups.end()) return grad_prefix;
                // A fused pool is one record on the engine, so its gradient comes back once; the
                // split below puts each module's factor under its own name.
                return grad_prefix + (g->second.size() > 1 ? fused_prefix(weight)
                                                           : g->second.front().module);
            };
        }
        // A fused weight's slots are its delta sites in order; name each slot's factor after the
        // module it belongs to, so an adapter row that knows how to read a head-split activation
        // writes one gradient per module rather than one fused one.
        bo.slot_prefix = [groups, grad_prefix](const OpRecord& rec, int64_t slot) -> std::string {
            const std::string weight = record_weight(rec);
            const auto g = groups.find(weight);
            if (g == groups.end() || slot < 0 ||
                slot >= static_cast<int64_t>(g->second.size()))
                return grad_prefix;
            return grad_prefix + g->second[static_cast<size_t>(slot)].module;
        };
        // The live tape names the engine's buffers; a family's adjoint finds a few operands by the
        // schema's own names. The walk keeps the tape names for versions, boundaries and snapshots.
        bo.prepare_record = prepare_adjoint_record;

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
        const OpRecord* head_logits_rec = nullptr;
        for (const OpRecord& rec : head_records)
            for (const Operand& o : rec.operands)
                if (o.written && o.buffer == "logits") {
                    head_logits_rows = o.rows;
                    head_logits_rec = &rec;
                }
        if (head_logits_rows <= 0)
            throw std::runtime_error("host step: the head wrote no logits rows");
        // The objective's window is L-1 rows a sequence; the head's buffer is all L after the
        // sampler's rows, and the moved seeds live in that space.
        names.add("logits", 0, head_logits_rows);
        const std::vector<Seed> seeds = head_seeds(driver.last(), batch, head_records, tokens);
        // The rows a consumer reads (src/read_set.h's derive_read_sets): a record whose every
        // written output is read by a gather_rows and nothing else takes the gather's index list as
        // its read set, and every other record is left unproven and keeps the loud refusal. The
        // buffers read outside the walk are the objective's -- the logits -- and `head_external_reads`
        // is the one place that says so, shared with the device step: the two paths marking
        // different buffers is exactly how the device step came to mark its own input and never
        // prove the head's norm.
        std::set<std::string> head_external = head_external_reads(*head_logits_rec);
        for (const Seed& s : seeds) head_external.insert(s.buffer);
        const std::map<int64_t, std::vector<int64_t>> head_reads =
            derive_read_sets(head_records, ops, head_snaps, head_external);
        bo.read_rows = &head_reads;
        const BackwardResult head = backward_block(head_records, head_snaps, names, seeds, {}, bo);
        // What the join held while the head walked: its whole store, which is held up front, plus
        // the copies of the one op being adjointed. The walk's own figure counts the latter only.
        measured.peak_snapshot_store_bytes = head_snaps.bytes();
        measured.peak_snapshot_op_bytes = head.report.peak_snapshot_bytes;
        measured.peak_snapshot_bytes =
            measured.peak_snapshot_store_bytes + measured.peak_snapshot_op_bytes;
        measured.all_snapshot_store_bytes = measured.peak_snapshot_store_bytes;
        measured.peak_cotangent_bytes = head.report.peak_cotangent_bytes;
        // The head's snapshots are spent once its walk has run and the layer walks build their own,
        // so free them: the peak stays one chunk's worth rather than the head's store added to a
        // block's.
        head_snaps = StoredSnapshots{};
        // The head's input cotangents stay alive with `head` for the whole layer loop, so they are
        // part of what the join holds during every block walk.
        int64_t head_input_bytes = 0;
        for (const InputCotangent& in : head.inputs) head_input_bytes += value_bytes(in.value);

        // Every buffer the head left on the boundary, as its own entry: the top block reads the
        // residual in place and the branch as a plain read, and both cross the boundary, so both
        // enter the top block at the version its last write produced.
        std::vector<IncomingCotangent> incoming =
            boundary_cotangents(head.inputs, ops,
                                &blocks[static_cast<size_t>(measured.layers - 1)], tokens,
                                meta.hidden);
        if (incoming.empty()) {
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

        auto store_boundary = [&](int64_t boundary, const std::vector<IncomingCotangent>& cot,
                                  const std::string& want) {
            if (!options.keep_boundaries) return;
            // The store holds the residual, and the report's boundary cotangent is that buffer's:
            // the host tier FD-checks it against residual perturbations and the oracle matches it
            // to the residual's gradient. Select the entry the block below reads in place, not
            // whichever entry the walk listed first -- that is the branch as often as not.
            const IncomingCotangent* in = nullptr;
            if (!want.empty()) {
                for (const IncomingCotangent& c : cot)
                    if (c.buffer == want) in = &c;
                if (!in)
                    throw std::runtime_error("r97: the join: the boundary store holds '" + want +
                                             "' but the walk left no cotangent for it");
            } else if (!cot.empty()) {
                in = &cot.front();
            }
            if (!in) return;
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

        store_boundary(measured.layers, incoming, store.name(measured.layers));

        LayerReplay replay(*driver.engine(), driver.tape(), store);
        for (int64_t layer = measured.layers - 1; layer >= 0; --layer) {
            StoredSnapshots snaps;
            // The post-issue observer runs after the op, so an operand an op replaces in place
            // holds the value the op wrote. A read of that same buffer is the value before the
            // op: the last write it saw, which is the previous op's post-issue content. Keep the
            // latter per buffer and give the in-place read that, or the adjoint differentiates
            // the op at its output instead of its input (the gated norm's gdn_o, above all).
            std::map<std::string, AdValue> before;
            // A boundary's residual is the block's [tokens, hidden] activation. The buffer an op
            // reads it from is not always the one the store copied out of -- the int8 path's
            // `rmsnorm_quant_fp8` reads an `h` of its own -- so the test is the operand's byte count
            // and its declared shape, not its pointer. When the pointer *is* inside the residual,
            // the copy keeps the operand's row offset.
            const int64_t boundary_bytes = replay.store().boundary_bytes();
            const int64_t boundary_tokens = replay.store().tokens();
            const int64_t boundary_hidden = replay.store().hidden();
            const uintptr_t residual_lo = reinterpret_cast<uintptr_t>(replay.residual(layer));
            const auto in_boundary = [&](const OperandContent& c, const Operand& o) {
                // `named` is the same test `replay.cpp`'s residual_bytes makes: a KV write's tape
                // identity is synthetic, so it must never be read as the residual.
                if (!c.named || c.data == nullptr || c.bytes != boundary_bytes) return false;
                return o.shape.size() == 2 && o.shape[0] == boundary_tokens &&
                       o.shape[1] == boundary_hidden;
            };
            replay.replay(layer, [&snaps, &before, &options, &replay, layer, boundary_bytes,
                                  residual_lo, &in_boundary](const OpRecord& op,
                                                   const std::vector<OperandContent>& content) {
                // The adjoint sees the step's rows and, for attention, the q view its family
                // reads; the snapshots must have the same shapes the walk will ask for.
                OpRecord shaped = op;
                tighten_rows(shaped);
                if (options.reference_head_split) mark_join_lora(shaped);
                std::vector<AdValue> now(shaped.operands.size());
                for (size_t j = 0; j < shaped.operands.size(); ++j)
                    now[j] = snapshot(shaped.operands[j], content[j], shaped.op.c_str(), shaped.position);
                for (size_t j = 0; j < shaped.operands.size(); ++j) {
                    const Operand& read = shaped.operands[j];
                    if (read.written) continue;
                    bool in_place = false;
                    for (size_t k = 0; k < shaped.operands.size(); ++k) {
                        const Operand& write = shaped.operands[k];
                        if (write.written && write.buffer == read.buffer) in_place = true;
                    }
                    if (!in_place) continue;
                    const auto prev = before.find(read.buffer);
                    if (prev != before.end() && prev->second.shape == now[j].shape) {
                        now[j] = prev->second;
                    } else if (read.buffer.rfind("kv:", 0) == 0) {
                        // The engine cleared the recurrent state before the replay, so a
                        // carried state read that no op in the block wrote yet enters zero.
                        now[j].value.assign(now[j].value.size(), 0.0);
                    } else if (const void* extra = replay.store().source_extra(layer);
                               replay.store().extra_at(layer) && extra != nullptr &&
                               content[j].data == extra) {
                        // A block that reads a second boundary-sized buffer -- the fwht fixture's
                        // `x` -- gets that buffer's own copy back, not the residual's. The pointer
                        // is exact here, so this test is stronger than the shape one below.
                        std::vector<uint8_t> boundary(static_cast<size_t>(boundary_bytes));
                        replay.store().read_extra(layer, boundary.data());
                        OperandContent restored = content[j];
                        restored.data = boundary.data();
                        now[j] = snapshot(shaped.operands[j], restored, shaped.op.c_str(),
                                          shaped.position);
                    } else if (in_boundary(content[j], shaped.operands[j])) {
                        // The operand's span lies inside this boundary's residual, and `replay()`
                        // restored boundary `layer` into that buffer before issuing the block. The
                        // tape snapshots after the op issued, so `now[j]` holds the op's own output;
                        // the store still holds what the buffer carried on entry, which is what the
                        // read consumed. Reading it back is the whole correction -- no refusal, and
                        // the op differentiates at its input. Position 1429 on the live tape is this
                        // case.
                        std::vector<uint8_t> boundary(static_cast<size_t>(boundary_bytes));
                        replay.store().read(layer, boundary.data());
                        if (std::getenv("R97_DUMP_RESTORE") != nullptr)
                            std::fprintf(stderr,
                                         "RESTORE boundary=%lld pos=%lld op=%s buffer=%s bytes=%lld\n",
                                         static_cast<long long>(layer),
                                         static_cast<long long>(shaped.position), shaped.op.c_str(),
                                         read.buffer.c_str(),
                                         static_cast<long long>(boundary_bytes));
                        size_t offset = 0;
                        const uintptr_t p = reinterpret_cast<uintptr_t>(content[j].data);
                        if (residual_lo != 0 && p >= residual_lo &&
                            p < residual_lo + static_cast<uintptr_t>(boundary_bytes))
                            offset = static_cast<size_t>(p - residual_lo);
                        OperandContent restored = content[j];
                        restored.data = boundary.data() + offset;
                        now[j] = snapshot(shaped.operands[j], restored, shaped.op.c_str(),
                                          shaped.position);
                    } else {
                        // Nothing above applies, so `now[j]` still holds what the op wrote: the
                        // tape snapshots an operand after the op issued, nothing in this block wrote
                        // this buffer, and the buffer is not the boundary residual, so no stored
                        // copy exists to restore. The read's cotangent would be differentiated at
                        // the op's own output, and refusing is the only honest answer.
                        throw std::runtime_error(
                            std::string("step: op '") + shaped.op + "' at position " +
                            std::to_string(shaped.position) + ": operand '" + read.buffer +
                            "' is read in place and nothing earlier in this block wrote it, so the "
                            "snapshot holds the op's own output -- the read is what the op consumed. "
                            "This buffer is not the boundary residual either, so there is no stored "
                            "copy to restore; refusing rather than differentiating at the output.");
                    }
                }
                for (size_t j = 0; j < shaped.operands.size(); ++j) {
                    const Operand& operand = shaped.operands[j];
                    if (operand.written && operand.buffer.rfind("kv:", 0) == 0 &&
                        before.find(operand.buffer) == before.end()) {
                        // A carried state this block has not written yet enters cold; the op's
                        // adjoint wants the state before the op, not the one it leaves.
                        now[j].value.assign(now[j].value.size(), 0.0);
                    }
                    snaps.set(op.position, j, now[j]);
                    if (operand.written && operand.view_shape.empty() && operand.rows > 0 &&
                        operand.rows == operand.shape[0]) {
                        before[operand.buffer] = std::move(now[j]);
                    }
                }
            });

            const Tape::Block& range = blocks[static_cast<size_t>(layer)];
            std::vector<OpRecord> records(ops.begin() + static_cast<std::ptrdiff_t>(range.begin),
                                          ops.begin() + static_cast<std::ptrdiff_t>(range.end));
            for (OpRecord& rec : records) {
                rec = own_raws(std::move(rec));
                tighten_rows(rec);
                if (options.reference_head_split) mark_join_lora(rec);
            }
            BoundaryNames block_names;
            std::set<std::string> external;
            for (const IncomingCotangent& in : incoming) {
                block_names.add(in.buffer, 0, tokens);
                external.insert(in.buffer);
            }
            // This block's own read sets, from the whole tape (src/join_common.h). Every block
            // record's consumer is a GEMM or a norm, so none is proven and every one keeps the
            // loud refusal; the map is built anyway so the rule lives in one place.
            const std::map<int64_t, std::vector<int64_t>> block_reads =
                derive_read_sets(records, ops, snaps, external);
            bo.read_rows = &block_reads;
            const BackwardResult walked = backward_block(records, snaps, block_names, {},
                                                         incoming, bo);
            // The join holds the whole block's snapshots while the walk copies one op out of them,
            // so what it holds at once is the store plus that one op, not the walk's figure alone.
            // `before` holds the last write of every buffer an op replaced in place, a second copy
            // of what the store already keeps, and is part of what the replay holds.
            int64_t before_bytes = 0;
            for (const auto& entry : before) before_bytes += value_bytes(entry.second.value);
            const int64_t store_bytes = snaps.bytes() + before_bytes;
            const int64_t op_bytes = walked.report.peak_snapshot_bytes;
            if (store_bytes + op_bytes > measured.peak_snapshot_bytes) {
                measured.peak_snapshot_bytes = store_bytes + op_bytes;
                measured.peak_snapshot_store_bytes = store_bytes;
                measured.peak_snapshot_op_bytes = op_bytes;
            }
            measured.all_snapshot_store_bytes += store_bytes;
            int64_t incoming_bytes = 0;
            for (const IncomingCotangent& in : incoming) incoming_bytes += value_bytes(in.value);
            measured.peak_cotangent_bytes = std::max(
                measured.peak_cotangent_bytes,
                head_input_bytes + incoming_bytes + walked.report.peak_cotangent_bytes);

            // Every buffer the walk left on the boundary, as its own entry for the block below;
            // null at the bottom block, which has no consumer below it.
            incoming = boundary_cotangents(
                walked.inputs, ops,
                layer > 0 ? &blocks[static_cast<size_t>(layer - 1)] : nullptr, tokens,
                meta.hidden);
            if (incoming.empty())
                throw std::runtime_error("host step: layer " + std::to_string(layer) +
                                         " left no cotangent on its input boundary");
            store_boundary(layer, incoming, store.name(layer));
        }

        // The split runs first: a fused weight's engine gradient is named after the container's
        // weight ("@fused:..."), so filling the missing ones before splitting would add a zero
        // under every module name the split then adds the real gradient for.
        split_adapter_grads(grads, params, groups, grad_prefix);
        fill_missing_grads(grads, params);
        order_grads_like(grads, params);
        // The report's own boundary copies grow with the number of layers; name them rather than
        // fold them into the peaks, which describe the join's working set.
        for (const std::vector<double>& v : measured.boundary_value)
            measured.kept_boundary_bytes += value_bytes(v);
        for (const std::vector<double>& v : measured.boundary_cotangent)
            measured.kept_boundary_bytes += value_bytes(v);
        if (report) *report = std::move(measured);
        return result;
    };
}

}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER && RAD_REPLAY_BLOCK
