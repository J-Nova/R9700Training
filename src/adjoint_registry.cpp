// src/adjoint_registry.cpp -- the op registry behind r97/adjoint.h.
//
// Rows register from the op's own file (static objects, run at load); lookup applies the first row
// whose predicate accepts the record's parameters. An op that nothing serves gets an error naming
// the op and its parameters, so a missing adjoint reads as one in a failed run.
#include "r97/adjoint.h"

#include <map>

namespace r97 {

namespace {

struct TwinEntry {
    TwinFn fn;
    ParamPred pred;
};
struct AdjointEntry {
    Side side;
    AdjointFn fn;
    std::vector<int> frozen;
    ParamPred pred;
    FillsFn fills;
};

struct Table {
    std::multimap<std::string, TwinEntry> twins;
    std::multimap<std::string, AdjointEntry> adjoints;

    // A function-local static: rows register from other files' static initialisers, whose order
    // against this one the standard leaves open.
    static Table& get() {
        static Table t;
        return t;
    }
};

std::string params_text(const std::vector<OpParam>& params) {
    std::string s;
    for (const OpParam& p : params) {
        if (!s.empty()) s += ", ";
        s += p.first + "=";
        if (auto* i = std::get_if<int64_t>(&p.second)) s += std::to_string(*i);
        else s += std::get<std::string>(p.second);
    }
    return s;
}

const char* side_text(Side side) { return side == Side::host ? "host" : "device"; }

// A row with a predicate is the more specific one, so it wins over a catch-all whatever order
// their files registered in: a file added later for one shape cannot be shadowed by the family's
// unconditional row. Rows with the same specificity keep registration order.
template <typename Row>
const Row* first_match(const std::multimap<std::string, Row>& table, const std::string& op,
                       const std::vector<OpParam>& params) {
    const Row* catch_all = nullptr;
    for (auto it = table.find(op); it != table.end() && it->first == op; ++it) {
        const Row& r = it->second;
        if (!r.pred) {
            if (!catch_all) catch_all = &r;
            continue;
        }
        if (r.pred(params)) return &r;
    }
    return catch_all;
}

// The adjoint row that serves one op on one side: the first specific row of that side whose
// predicate accepts the parameters, else its catch-all. Both adjoint_find and adjoint_frozen read
// the row through here, so the frozen positions a walk skips are always the ones of the adjoint it
// calls.
const AdjointEntry* adjoint_match(const std::string& op, Side side,
                                 const std::vector<OpParam>& params) {
    const auto& table = Table::get().adjoints;
    const AdjointEntry* catch_all = nullptr;
    for (auto it = table.find(op); it != table.end() && it->first == op; ++it) {
        const AdjointEntry& r = it->second;
        if (r.side != side) continue;
        if (!r.pred) {
            if (!catch_all) catch_all = &r;
            continue;
        }
        if (r.pred(params)) return &r;
    }
    return catch_all;
}

}  // namespace

void twin_register(const char* op, TwinFn fn, ParamPred pred) {
    Table::get().twins.emplace(op, TwinEntry{fn, pred});
}

void adjoint_register(const char* op, Side side, AdjointFn fn, const std::vector<int>& frozen,
                      ParamPred pred, FillsFn fills) {
    Table::get().adjoints.emplace(op, AdjointEntry{side, fn, frozen, pred, fills});
}

TwinFn twin_find(const std::string& op, const std::vector<OpParam>& params, std::string* error) {
    if (const TwinEntry* r = first_match(Table::get().twins, op, params)) return r->fn;
    if (error)
        *error = "no twin registered for op '" + op + "' (params: " + params_text(params) + ")";
    return nullptr;
}

AdjointFn adjoint_find(const std::string& op, Side side, const std::vector<OpParam>& params,
                       std::string* error) {
    if (const AdjointEntry* r = adjoint_match(op, side, params)) return r->fn;
    if (error)
        *error = "no " + std::string(side_text(side)) + " adjoint for op '" + op + "' (params: " +
                 params_text(params) + ")";
    return nullptr;
}

std::vector<int> adjoint_frozen(const std::string& op, Side side,
                                const std::vector<OpParam>& params) {
    if (const AdjointEntry* r = adjoint_match(op, side, params)) return r->frozen;
    return {};
}

FillsFn adjoint_fills(const std::string& op, Side side, const std::vector<OpParam>& params) {
    if (const AdjointEntry* r = adjoint_match(op, side, params)) return r->fills;
    return nullptr;
}

std::vector<std::string> ops_registered() {
    std::map<std::string, bool> names;
    for (const auto& kv : Table::get().twins) names[kv.first] = true;
    for (const auto& kv : Table::get().adjoints) names[kv.first] = true;
    std::vector<std::string> out;
    out.reserve(names.size());
    for (const auto& kv : names) out.push_back(kv.first);
    return out;
}

OpRows op_rows(const std::string& op) {
    OpRows r;
    const Table& t = Table::get();
    for (auto it = t.twins.find(op); it != t.twins.end() && it->first == op; ++it) {
        r.twin = true;
        break;
    }
    for (auto it = t.adjoints.find(op); it != t.adjoints.end() && it->first == op; ++it) {
        if (it->second.side == Side::host) {
            if (r.host_adjoint) continue;
            r.host_adjoint = true;
            r.host_frozen = it->second.frozen;
        } else {
            if (r.device_adjoint) continue;
            r.device_adjoint = true;
            r.device_frozen = it->second.frozen;
        }
    }
    return r;
}

}  // namespace r97
