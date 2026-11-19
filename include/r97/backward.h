// r97/backward.h -- the backward core: one recorded block, walked in reverse.
//
// The core knows nothing about the model. Its caller gives it the OpRecords of one block, a
// provider for the forward values the ops saw, and the cotangents the walk starts from: the
// objective's Seeds and what the block above handed down at the boundary. It answers with the
// cotangents of the block's inputs and with additions to a gradient ParamSet.
//
// Versions. A block writes the same buffer several times -- a transient reused, an op in place,
// a row range written by one op and another by the next. Every write therefore creates a version
// of the buffer, and a cotangent belongs to (buffer, rows, version): it is created on first use,
// summed over every reader that saw that version, and released once the writer's adjoint has run.
// A seed, and the cotangent handed down from the block above, enter at the version that is live
// when the block ends; two contributions on the same rows add.
//
// A record that gives no row count (`rows == 0`) touches all the buffer's rows. Within one record
// the reads are taken before the writes, so an op that writes the buffer it reads (in place) reads
// the version that was live before it, and its adjoint receives that version's cotangent.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "r97/adjoint.h"
#include "r97/model.h"
#include "r97/objective.h"
#include "r97/op.h"
#include "r97/params.h"

namespace r97 {
// A written operand that is also the op's input, though no separate read names it: rope's
// in-place write and the recurrent/conv state buffers. The prior version is the read. The issue
// record marks such an operand (`Operand::in_place`, a RAD_OPD_INOUT operand); a record built from
// a program instead of the live tape may not carry the mark, so the two shapes the ops always
// have -- rope's operand 0, and a `kv:` buffer no separate read names -- are recognised here too.
// The walk's version graph and the join's boundary pick both derive their in-place reads from
// this one rule.
inline bool in_place_read(const OpRecord& rec, size_t j) {
    const Operand& o = rec.operands[j];
    if (!o.written) return false;
    if (o.in_place) return true;
    if (rec.op == "rope" && j == 0) return true;
    if (o.buffer.rfind("kv:", 0) != 0) return false;
    for (size_t k = 0; k < rec.operands.size(); ++k)
        if (k != j && !rec.operands[k].written && rec.operands[k].buffer == o.buffer) return false;
    return true;
}

// The forward value of one operand of one op, as the op saw it. A buffer the block writes twice
// has two values, so the provider answers by the op's position in the step and the operand's index
// in schema order. A block handed to the walk keeps that position on every record.
class SnapshotProvider {
public:
    virtual ~SnapshotProvider() = default;
    // Null when there is no value: the walk refuses with the op's name, because an adjoint cannot
    // run without the forward values its op read.
    virtual const AdValue* at(int64_t position, size_t operand) const = 0;
};

// A SnapshotProvider over values held in memory, keyed by (position, operand). Tests and fixtures
// assemble one; a replay provider answers the same two keys without holding the whole block.
class StoredSnapshots final : public SnapshotProvider {
public:
    void set(int64_t position, size_t operand, AdValue value);
    const AdValue* at(int64_t position, size_t operand) const override;
    size_t size() const { return values_.size(); }
    // The payload the store holds, in bytes of values (not shapes or names), so a caller that
    // keeps a whole block's snapshots can say what it holds.
    int64_t bytes() const;

private:
    std::map<std::pair<int64_t, size_t>, AdValue> values_;
};

// The cotangent the block above hands down at the boundary: the same thing a Seed is, but already
// f64 and already carrying every row's values (a Seed carries f32, cast to the writer's dtype).
// It enters the walk the same way a Seed does and adds to what a Seed on the same rows adds.
struct IncomingCotangent {
    std::string buffer;             // a name the forward recorded on the tape
    int64_t row_lo = 0, row_hi = 0; // the rows it covers
    std::vector<double> value;      // (row_hi-row_lo) * cols values, row-major
};

// What the walk needs besides its input.
struct BackwardOptions {
    Side side = Side::host;          // which adjoint rows the walk asks the registry for
    ParamSet* grads = nullptr;       // added to, or null when the walk adds no gradients
    const char* prefix = "";         // the name prefix the grads entries carry (Delta::grad_prefix)
    // Optional: when set, a fused lora record's per-slot leaf names come from this callback instead
    // of `prefix` (slot -> module path). It is given the record, because a fused weight's lora
    // record carries several modules in one pool and the caller needs the record's weight to say
    // which pool the slot belongs to. Empty for a delta with no fused slots.
    std::function<std::string(const OpRecord&, int64_t)> slot_prefix;
    // Optional: the name a record's parameter gradients take, when one walk covers records whose
    // parameters live in different places (a fused pool's modules, or a record that names none and
    // keeps `prefix`). Empty keeps `prefix` for every record.
    std::function<std::string(const OpRecord&)> record_prefix;
    // Optional: the record the adjoint sees. A live tape names the engine's buffers and its op
    // parameters can give the engine's capacity where the schema names the step's own rows; the
    // walk's versions, boundaries and snapshots still use the record as recorded. The callback
    // fills `out` from the record when it wants a different view.
    std::function<void(const OpRecord&, OpRecord*)> prepare_record;
    // Op names whose path ends here: their adjoint never runs and no cotangent reaches their
    // reads. The record still consumes the cotangent of what it wrote.
    std::vector<std::string> stops;
    // A write that replaces the value it read also carries that value's cotangent on the direct
    // path. Set when the caller's adjoint row leaves the in-place read's slot empty; the walk then
    // copies the write's cotangent into it, and the read travels to the version it saw.
    bool in_place_passthrough = false;
    // A record's slots share one fused factor: the lora rows write a slot's rows into one
    // [S*R, K] (A) / [N, S*R] (B) gradient under the record's prefix, for a caller that splits it
    // afterwards. False keeps each slot under `prefix` or `slot_prefix`.
    bool fused_slots = false;
    // When the requested side has no row for an op, take the host side's row for it instead of
    // refusing. The device join sets this: a device row is an optimisation, the host row is the
    // same arithmetic in f64, and the join already pulls every operand back for the walk. The
    // step's coverage names the ops that took the host path.
    bool allow_host_fallback = false;
    // The rows of a record's output that any consumer reads, when the caller proved them
    // (r97/adjoint.h's AdjointCall::read_rows): record position -> sorted row indices. A position
    // absent from the map is unproven, and the walk hands that record null -- unknown is never
    // unreadable. The map outlives the walk; the join derives it from the whole tape
    // (src/read_set.h's derive_read_sets) and points this at the walk's own list.
    const std::map<int64_t, std::vector<int64_t>>* read_rows = nullptr;
};

// The cotangent of one of the block's inputs: a buffer the block read but never wrote. It is dense
// over the buffer -- the rows the block read carry their cotangent, the others are zero -- so the
// block above can take it as an IncomingCotangent without re-indexing.
struct InputCotangent {
    std::string buffer;
    int64_t row_lo = 0, row_hi = 0;  // the buffer's rows, the whole buffer
    std::vector<int64_t> shape;      // the buffer's full logical shape
    std::vector<double> value;       // row_hi * cols values, row-major
};

// One op's share of one block's walk: the adjoint row calls the walk made for it and the wall
// time around them. A device row owns the bus inside the call -- its upload, launch and
// download -- so the split across op names says where the block's backward seconds went.
struct OpTiming {
    std::string op;       // the op name the row was resolved under
    int64_t calls = 0;    // the row calls the walk made for the op
    double seconds = 0.0; // wall time around them
    // Which side served them. Without it a slow op cannot be attributed: the host row is f64
    // arithmetic that `parallel_for` spreads over the cores, the device row is a launch and a sync
    // with the card mostly idle, and the two want opposite fixes.
    bool host = false;
    // The op's slots whose cotangent held a non-finite value (a NaN or an inf) when the walk was
    // about to hand it on: one per slot, the scan stopping at the slot's first such value. A
    // non-zero count names the op a NaN was born at instead of bisecting the step blind; the
    // report's first_non_finite_* says which slot and whether the row produced it.
    int64_t non_finite = 0;
    // The two-way rule's outcome, summed over the calls (r97/adjoint.h's AdjointCall::RowCount): the
    // rows the row left zero because the read set proves nothing reads them, the rows it
    // differentiated, and the rows it refused because they are read and their value was not finite.
    // A step that zeroes a live row or serves a dead one is wrong, and these are the numbers that
    // say which happened -- 511 zeroed and 1 readable for the head's norm at 512 tokens.
    int64_t rows_zeroed_unreadable = 0;
    int64_t rows_readable = 0;
    int64_t rows_refused_non_finite = 0;
};

// What the walk held at once, in bytes of tensor payload (values, not shapes or names): the
// forward values of one op (the only snapshots the core holds itself, beside the provider the
// caller holds), and the cotangents -- the live version store, the block-input accumulators, and
// the cot/dc vectors of the op being adjointed. `ops` splits the walk's time across the rows it
// called, sorted by op name; a row the walk never reached (off the path, behind a stop) names
// nothing.
struct BackwardReport {
    int64_t peak_snapshot_bytes = 0;
    int64_t peak_cotangent_bytes = 0;
    std::vector<OpTiming> ops;

    // The first non-finite cotangent the walk saw in a value it was about to hand on, named so a
    // single run attributes a NaN instead of bisecting the step blind. Empty `first_non_finite_op`
    // means the walk saw none. The op is the name its row was resolved under (the prepared
    // record's, when the caller prepares one); the operand is the tape's buffer name and its index
    // in the record's schema; `first_non_finite_position` is the record's position in the step and
    // `first_non_finite_index` the value's offset in that operand's payload. `from_row` says the
    // row produced it; false says the walk handed it in already non-finite -- a seed, the
    // cotangent from above, or a record later in the forward that the walk has already walked.
    std::string first_non_finite_op;
    std::string first_non_finite_operand;
    int64_t first_non_finite_operand_index = -1;
    int64_t first_non_finite_position = -1;
    int64_t first_non_finite_index = -1;
    bool first_non_finite_from_row = false;

    // The same question asked of the forward: whether every operand value the walk read was
    // finite. A NaN born in the forward is a different bug from one born in the walk, and a
    // gradient is only as good as the values it differentiates. `forward_first_*` name the first
    // operand the walk met that was not, in the walk's order (the block's last record first).
    bool forward_all_finite = true;
    std::string forward_first_non_finite_operand;
    int64_t forward_first_non_finite_operand_index = -1;
    int64_t forward_first_non_finite_position = -1;
    int64_t forward_first_non_finite_index = -1;
};

struct BackwardResult {
    std::vector<InputCotangent> inputs;  // the block's inputs, sorted by buffer name
    BackwardReport report;
};

// Walk one block in reverse. `boundaries` resolves a Seed's buffer to its record and rows, and
// `records` are the block's ops in append order. `seeds` and `incoming` may each be empty. A seed
// on a buffer that is not a boundary, or on rows the block never wrote, is an error that names it;
// so is an op on the path that no adjoint serves, and an operand with no snapshot.
BackwardResult backward_block(const std::vector<OpRecord>& records,
                              const SnapshotProvider& snapshots,
                              const BoundaryNames& boundaries,
                              const std::vector<Seed>& seeds,
                              const std::vector<IncomingCotangent>& incoming,
                              const BackwardOptions& options = {});

}  // namespace r97
