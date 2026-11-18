// tests/host/adjoint_registry_precedence.cpp -- a specific row wins over a catch-all whichever
// file registered first.
//
// The reference step adds its shape-aware rows for a few families in files of their own rather
// than editing the family files. It tags its records, and the registry must reach the tagged row
// even though the family's unconditional row may have been registered earlier in link order.
// Label: host.
#include "support/test.h"

#include <cstdint>
#include <string>
#include <vector>

#include "r97/adjoint.h"

using namespace r97;

namespace {

bool tagged(const std::vector<OpParam>& params) {
    for (const auto& p : params)
        if (const int64_t* v = std::get_if<int64_t>(&p.second))
            if (p.first == "registry_probe_tag" && *v == 1) return true;
    return false;
}

void catch_all_twin(const OpRecord&, const std::vector<AdValue>&, std::vector<AdValue>*) {}
void specific_twin(const OpRecord&, const std::vector<AdValue>&, std::vector<AdValue>*) {}
void catch_all_adjoint(const AdjointCall&) {}
void specific_adjoint(const AdjointCall&) {}

}  // namespace

R97_TEST(adjoint_registry_specific_row_beats_catch_all) {
    const char* op = "registry_precedence_probe";
    /* Catch-all first, as a family file linked earlier would register it. */
    twin_register(op, catch_all_twin, nullptr);
    twin_register(op, specific_twin, tagged);
    adjoint_register(op, Side::host, catch_all_adjoint, {}, nullptr);
    adjoint_register(op, Side::host, specific_adjoint, {}, tagged);

    const std::vector<OpParam> untagged = {{"M", int64_t{4}}};
    const std::vector<OpParam> marked = {{"M", int64_t{4}}, {"registry_probe_tag", int64_t{1}}};
    std::string err;
    R97_CHECK(twin_find(op, untagged, &err) == catch_all_twin);
    R97_CHECK(twin_find(op, marked, &err) == specific_twin);
    R97_CHECK(adjoint_find(op, Side::host, untagged, &err) == catch_all_adjoint);
    R97_CHECK(adjoint_find(op, Side::host, marked, &err) == specific_adjoint);
}

// The fills predicate a walk consults is the one of the row it calls: a specific row's predicate
// is what a record with its parameters gets, and a row that registered none leaves the walk's
// in-place passthrough on.
R97_TEST(adjoint_registry_fills_follow_the_served_row) {
    const char* op = "registry_fills_probe";
    adjoint_register(op, Side::host, catch_all_adjoint, {}, nullptr);
    adjoint_register(op, Side::host, specific_adjoint, {}, tagged, fills_in_place_slots);

    const std::vector<OpParam> untagged = {{"M", int64_t{4}}};
    const std::vector<OpParam> marked = {{"M", int64_t{4}}, {"registry_probe_tag", int64_t{1}}};
    R97_CHECK(adjoint_fills(op, Side::host, untagged) == nullptr);
    R97_CHECK(adjoint_fills(op, Side::host, marked) == fills_in_place_slots);
    R97_CHECK(adjoint_fills(op, Side::device, marked) == nullptr);  // no device row
}
