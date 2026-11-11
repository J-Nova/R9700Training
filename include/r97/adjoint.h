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
};

// One call of an adjoint.
struct AdjointCall {
    const OpRecord& rec;             // the op's record: name, parameters, dtypes
    const std::vector<AdValue>& fwd; // every operand, in schema order, with the forward's values
    const std::vector<AdValue>& cot; // the cotangent of every operand the op wrote, writes' order
    std::vector<AdValue>* dc;        // every operand, in schema order, added to, never overwritten
    ParamSet* grads;                 // the trainable leaves, or null when the walk runs without
    const char* prefix;              // the name prefix grads entries carry (Delta::grad_prefix)
    // Optional: a site's name prefix when one record covers several sites of a fused weight. The
    // lora rows ask it for a slot; an empty callback means every slot uses `prefix`.
    std::function<std::string(int64_t)> slot_prefix;
};

// A twin: the forward in f64. `operands` are the op's read operands in schema order; `out` receives
// one AdValue per written operand, in the writes' schema order.
using TwinFn = void (*)(const OpRecord& rec, const std::vector<AdValue>& operands,
                        std::vector<AdValue>* out);
using AdjointFn = void (*)(const AdjointCall& call);
// A predicate on the op's parameters, for example the dtype; null accepts every record.
using ParamPred = bool (*)(const std::vector<OpParam>& params);

// -- registry ----------------------------------------------------------------

void twin_register(const char* op, TwinFn fn, ParamPred pred);
void adjoint_register(const char* op, Side side, AdjointFn fn, const std::vector<int>& frozen,
                      ParamPred pred);

// Lookup applies the first row of that side whose predicate accepts the parameters. Rows for one
// op are tried in registration order, which across translation units the link order decides, so
// keep one op's predicates disjoint and register a catch-all row last. When nothing serves the op,
// `error` (when given) names the op and its parameters, and the lookup returns null.
TwinFn twin_find(const std::string& op, const std::vector<OpParam>& params, std::string* error);
AdjointFn adjoint_find(const std::string& op, Side side, const std::vector<OpParam>& params,
                       std::string* error);

// The frozen operand positions of the adjoint row that serves the op; empty when nothing does.
std::vector<int> adjoint_frozen(const std::string& op, Side side, const std::vector<OpParam>& params);

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
};
struct Registrar {
    explicit Registrar(const TwinRow& r) { twin_register(r.op, r.fn, r.pred); }
    explicit Registrar(const AdjointRow& r) {
        adjoint_register(r.op, r.side, r.fn, r.frozen, r.pred);
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

}  // namespace r97
