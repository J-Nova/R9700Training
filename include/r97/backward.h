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
    // Optional: the name prefix for one slot of a record that covers several sites of a fused
    // weight (a lora record's slot operand). Empty means use `prefix` for every slot.
    std::function<std::string(const OpRecord&, int64_t)> slot_prefix;
    // Op names whose path ends here: their adjoint never runs and no cotangent reaches their
    // reads. The record still consumes the cotangent of what it wrote.
    std::vector<std::string> stops;
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

// What the walk held at once, in bytes of tensor payload (values, not shapes or names): the
// forward values of one op (the only snapshots the core holds itself), and the cotangents -- the
// live version store, the block-input accumulators, and the cot/dc vectors of the op being
// adjointed.
struct BackwardReport {
    int64_t peak_snapshot_bytes = 0;
    int64_t peak_cotangent_bytes = 0;
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
