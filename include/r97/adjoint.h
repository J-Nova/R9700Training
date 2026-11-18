// r97/adjoint.h -- the adjoint contract: twins, adjoints and their registry.
//
// A twin is the forward of one op in f64: given the op's record and the forward values of its
// operands, it produces what the op would have written had it computed in f64. An adjoint walks
// one op's record backwards: it takes the op's OpRecord, the forward values of its operands and
// the cotangents of the operands it wrote, and adds to the cotangents of the operands it read. An
// adjoint of an op with trainable leaves also adds to a gradient ParamSet, under a name derived
// from the prefix the walk supplies. Host adjoints compute in f64.
//
// Registration is a static object per op, in the op's own file, so a new op needs no edit to a
// shared file:
//
//     #include "r97/adjoint.h"
//
//     R97_REGISTER_TWIN("add", add_twin)
//     R97_REGISTER_ADJOINT("add", host, add_host_adjoint)
//     R97_REGISTER_ADJOINT_IF("mul", device, mul_gpu, mul_is_bf16)
//     R97_REGISTER_ADJOINT_F("rmsnorm", host, rmsnorm_adj, 2)  // operand index 2 is frozen
//
// Variants: R97_REGISTER_TWIN_IF(op, fn, pred); R97_REGISTER_ADJOINT takes the side `host` or
// `device`; the `_F` forms take the frozen operand positions (a weight, a scale -- an operand
// whose cotangent the walk does not ask for), counted from 0 in schema order; the `_IF` forms take
// a predicate on the op's parameters, so one op can register a row per dtype or layout. Rows are
// keyed by op name and lookup applies the first row whose predicate accepts the record's
// parameters.
//
// Where the registry ends: an Objective (r97/objective.h) gives the seeds the walk starts from,
// the cotangents of the output the loss scores; the parameters of a Delta (r97/delta.h) are the
// trainable leaves an adjoint adds gradients for, under the name prefix Delta::grad_prefix gives.
// Neither is registered here: the op registry is keyed by op and does not know them.
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "r97/op.h"
#include "r97/params.h"

namespace r97 {

// One op parameter as OpRecord carries it: its name and its value.
using OpParam = std::pair<std::string, std::variant<int64_t, std::string>>;

// An op can carry a twin and an adjoint on each side.
enum class Side { host, device };

// The forward value of one operand, or one cotangent. f64 everywhere on the host; a device adjoint
// is handed the same f64 contract and rounds on its own way in.
struct AdValue {
    std::string dtype;          // the forward dtype, as the record carries it
    std::vector<int64_t> shape; // full logical shape, row-major
    std::vector<double> value;  // numel(shape) values, row-major
    // The operand's bytes where the tape recorded them -- a device pointer under the device join,
    // the caller's pointer under the host join. Set only for an operand whose dtype has no f64
    // form (a plugin-private quantised tensor the engine decodes in its own stored layout); null
    // for every operand the tape gave values for. `value` is empty for such an operand, and the
    // shape is still the op's, so a row that decodes the bytes owns the layout and one that needs
    // f64 must decline. A row never adds to a gradient through this pointer.
    const void* raw = nullptr;
    // The stored encoding the tape carried beside `raw` (radiance's RadEncoding, patch 0124): the
    // planes the engine's layout hook itself derived the stored form from. A row that decodes the
    // bytes derives the form from this, never from a second classification of its own, so the form
    // travels with the operand exactly as the pointer does and a wrong one is visible in the dump.
    const void* enc = nullptr;
};

// One call of an adjoint.
struct AdjointCall {
    const OpRecord& rec;             // the op's record: name, parameters, dtypes
    const std::vector<AdValue>& fwd; // every operand, in schema order, with the forward's values
    const std::vector<AdValue>& cot; // the cotangent of every operand the op wrote, writes' order
    std::vector<AdValue>* dc;        // every operand, in schema order, added to, never overwritten
    ParamSet* grads;                 // the trainable leaves, or null when the walk runs without
    const char* prefix;              // the name prefix grads entries carry (Delta::grad_prefix)
    // Optional: when set, a trainable leaf's name is this callback's answer for its slot plus the
    // leaf's suffix, instead of `prefix`. A site's name prefix for one record that covers several
    // sites of a fused weight: the lora rows ask it for a slot, and only the record's `slot`
    // operand tells the modules in its pool apart. Empty means every slot uses `prefix`.
    std::function<std::string(int64_t slot)> slot_prefix;
    // A record whose slots share one factor: the lora rows write their slot's rows into one
    // `[S*R, K]` (A) / `[N, S*R]` (B) gradient under `prefix`, for a caller that splits the fused
    // factor afterwards. False keeps each slot's factor under its own name.
    bool fused_slots = false;
    // The rows of the op's output that a consumer of the whole tape reads, sorted and unique, when
    // the caller can prove them; null when it cannot, which is the default and means every row is
    // readable. A row outside a proven set is provably unread, so a non-finite forward value there
    // is zeroed rather than refused: the zero is the correct cotangent for a value nothing reads.
    // A row inside it that is not finite refuses, as every row did before this field existed.
    //
    // Proven means derived from the tape, not inferred: today only a record whose every written
    // output is read by gather_rows and nothing else takes the gather's index operand as its read
    // set (docs/adjoints/read-set.md). Any other consumer -- an op, a boundary the next block
    // reads, a seed, a buffer rewritten before it is read -- leaves the set null, because unknown
    // is never unreadable.
    const std::vector<int64_t>* read_rows = nullptr;
    // Where the row reports the two-way rule's outcome, or null when the caller keeps no count.
    // A row that zeroes a readable row, or refuses a dead one, is wrong, and these three numbers
    // are what a step's report says happened instead.
    //
    // The counts answer "which rows did the proved set admit", so a row fills them only when the
    // caller proved a set for it (read_rows != null). A row handed no set reports nothing: a run
    // that proved no set at all reads 0/0/0, and a run whose rule fired reports the rows it
    // admitted even when it zeroed none -- a set covering every row reads N/0/0. Without that, a
    // proven-and-nothing-to-zero run is indistinguishable from a rule that never fired.
    struct RowCount {
        int64_t zeroed_unreadable = 0;  // rows left zero because the proved set excludes them
        int64_t readable = 0;           // rows the proved set admits
        int64_t refused_non_finite = 0; // readable rows refused because their value was not finite
    };
    RowCount* row_count = nullptr;
};

// Whether a row is read by a consumer, from a read set: a null set means the caller could not
// prove the rows, and every row counts as read. So does an EMPTY set: a proof that nothing reads a
// value is vacuous, not evidence of death -- a derivation that finds no consumer may simply not
// have been handed the reader -- and acting on one zeroes a live value's whole gradient. The
// invariant therefore holds at both ends: the derivation never produces an empty proof
// (src/read_set.h's saw_consumer), and no row acts on one if it appears. `rows` is sorted
// (AdjointCall::read_rows).
inline bool row_readable(const std::vector<int64_t>* rows, int64_t row) {
    if (rows == nullptr || rows->empty()) return true;  // unproven, not dead
    return std::binary_search(rows->begin(), rows->end(), row);
}
inline bool row_readable(const AdjointCall& call, int64_t row) {
    return row_readable(call.read_rows, row);
}

// Count what one row did with the two-way rule, when the caller keeps a count AND proved a set. A
// row handed no set reports nothing (AdjointCall::RowCount): 0/0/0 then says the rule was never
// asked, while a set covering every row still reports N/0/0, so a fired rule is never mistaken for
// one that never ran. A row that does not consult the set at all leaves row_count untouched, which
// is the same outcome by a different road.
inline void count_read_rows(AdjointCall::RowCount* c, const std::vector<int64_t>* rows,
                            int64_t zeroed, int64_t admitted, int64_t refused) {
    if (c == nullptr || rows == nullptr) return;
    c->zeroed_unreadable += zeroed;
    c->readable += admitted;
    c->refused_non_finite += refused;
}

// A twin: the forward in f64. `operands` are the op's read operands in schema order; `out` receives
// one AdValue per written operand, in the writes' schema order.
using TwinFn = void (*)(const OpRecord& rec, const std::vector<AdValue>& operands,
                        std::vector<AdValue>* out);
using AdjointFn = void (*)(const AdjointCall& call);
// A predicate on the op's parameters, for example the dtype; null accepts every record.
using ParamPred = bool (*)(const std::vector<OpParam>& params);
// Whether the adjoint row fills one operand's dc slot itself, for the records the walk marks in
// place. A row that answers an in-place write -- the value it replaced carries the write's
// cotangent -- registers one: the walk then knows a zero slot is the row's answer, not an empty
// slot to pass the cotangent through. Null (the default) is a row that leaves the slot to the
// passthrough.
using FillsFn = bool (*)(const OpRecord& rec, size_t operand);

// The fills predicate a row registers when it fills every slot the record marks in place: the
// direct path, where the value a write replaced carries the write's cotangent.
inline bool fills_in_place_slots(const OpRecord& rec, size_t operand) {
    return operand < rec.operands.size() && rec.operands[operand].in_place;
}

// -- registry ----------------------------------------------------------------

void twin_register(const char* op, TwinFn fn, ParamPred pred);
void adjoint_register(const char* op, Side side, AdjointFn fn, const std::vector<int>& frozen,
                      ParamPred pred, FillsFn fills = nullptr);

// Lookup applies the first row of that side whose predicate accepts the parameters. Rows for one
// op are tried in registration order, which across translation units the link order decides, so
// keep one op's predicates disjoint and register a catch-all row last. When nothing serves the op,
// `error` (when given) names the op and its parameters, and the lookup returns null.
TwinFn twin_find(const std::string& op, const std::vector<OpParam>& params, std::string* error);
AdjointFn adjoint_find(const std::string& op, Side side, const std::vector<OpParam>& params,
                       std::string* error);

// The frozen operand positions of the adjoint row that serves the op; empty when nothing does.
std::vector<int> adjoint_frozen(const std::string& op, Side side, const std::vector<OpParam>& params);

// The fills predicate of the adjoint row that serves the op; null when nothing does or the row
// leaves its in-place slots to the walk's passthrough.
FillsFn adjoint_fills(const std::string& op, Side side, const std::vector<OpParam>& params);

// Every op with at least one row, sorted.
std::vector<std::string> ops_registered();

// What the rows under one op name say, without applying their predicates: a listing, not a lookup.
struct OpRows {
    bool twin = false;
    bool host_adjoint = false;
    bool device_adjoint = false;
    std::vector<int> host_frozen;
    std::vector<int> device_frozen;
};
OpRows op_rows(const std::string& op);

// -- registration macros -----------------------------------------------------
// A static object per op, in the op's own file. `op` is the name the record's op field carries;
// `pred` takes the record's parameters; the frozen positions are plain ints.

struct TwinRow {
    const char* op;
    TwinFn fn;
    ParamPred pred;
};
struct AdjointRow {
    const char* op;
    Side side;
    AdjointFn fn;
    std::vector<int> frozen;
    ParamPred pred;
    FillsFn fills = nullptr;
};
struct Registrar {
    explicit Registrar(const TwinRow& r) { twin_register(r.op, r.fn, r.pred); }
    explicit Registrar(const AdjointRow& r) {
        adjoint_register(r.op, r.side, r.fn, r.frozen, r.pred, r.fills);
    }
};

// The registrar object's name comes from the line, so one function may serve several ops (a twin
// reused under another name, as the failure-mode tests do).
#define R97_REG_NAME2(a, b) a##b
#define R97_REG_NAME(a, b) R97_REG_NAME2(a, b)

#define R97_REGISTER_TWIN(op, fn) \
    static ::r97::Registrar R97_REG_NAME(r97_reg_twin_, __LINE__) { \
        ::r97::TwinRow{op, fn, nullptr} \
    };
#define R97_REGISTER_TWIN_IF(op, fn, pred) \
    static ::r97::Registrar R97_REG_NAME(r97_reg_twin_, __LINE__) { \
        ::r97::TwinRow{op, fn, pred} \
    };
#define R97_REGISTER_ADJOINT(op, side, fn) \
    static ::r97::Registrar R97_REG_NAME(r97_reg_adj_, __LINE__) { \
        ::r97::AdjointRow{op, ::r97::Side::side, fn, {}, nullptr} \
    };
#define R97_REGISTER_ADJOINT_F(op, side, fn, ...) \
    static ::r97::Registrar R97_REG_NAME(r97_reg_adj_, __LINE__) { \
        ::r97::AdjointRow{op, ::r97::Side::side, fn, {__VA_ARGS__}, nullptr} \
    };
#define R97_REGISTER_ADJOINT_IF(op, side, fn, pred) \
    static ::r97::Registrar R97_REG_NAME(r97_reg_adj_, __LINE__) { \
        ::r97::AdjointRow{op, ::r97::Side::side, fn, {}, pred} \
    };
#define R97_REGISTER_ADJOINT_IF_F(op, side, fn, pred, ...) \
    static ::r97::Registrar R97_REG_NAME(r97_reg_adj_, __LINE__) { \
        ::r97::AdjointRow{op, ::r97::Side::side, fn, {__VA_ARGS__}, pred} \
    };
// A row that fills the in-place slots its `fills` predicate names. The walk then treats a zero
// slot as the row's answer instead of passing the incoming cotangent through it.
#define R97_REGISTER_ADJOINT_FF(op, side, fn, fills, ...) \
    static ::r97::Registrar R97_REG_NAME(r97_reg_adj_, __LINE__) { \
        ::r97::AdjointRow{op, ::r97::Side::side, fn, {__VA_ARGS__}, nullptr, fills} \
    };

}  // namespace r97
