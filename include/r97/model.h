// r97/model.h -- the registry of objectives and trainable kinds by name, and the seed check.
//
// A new loss or a new trainable kind is one new file that registers itself here, like the op
// registry does: nothing in the forward driver or the backward core ever names one. The
// registry is filled by static registration, so both lists are complete before main runs.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "r97/config.h"
#include "r97/delta.h"
#include "r97/objective.h"
#include "r97/step.h"

namespace r97 {

// ------------------------------------------------------------------ the registries by name

using ObjectiveFactory = Objective (*)(const Config&);                 // gives an Objective
using DeltaFactory     = std::shared_ptr<Delta> (*)(const Config&);    // the trainable kind

// What the registries hold, so listing (r97train objectives) and lookups (make_model) see the
// same tables. Static registration happens before main, in any order. A table lives until the
// process exits, so a reader that runs at exit (the host test runner) always sees it whole.
std::map<std::string, ObjectiveFactory>& objective_registry();
std::map<std::string, DeltaFactory>& delta_registry();

// The registered names, sorted. A test that registers only in its own binary sees its
// registrations here.
std::vector<std::string> objective_names();
std::vector<std::string> delta_names();

// Registration happens at static-init time from each objective's (or delta's) own file. The
// `name` must be a bare identifier; the same name twice is refused, since a duplicate would make
// it nondeterministic which of them resolves a name.
class ObjectiveRegistration {
public:
    ObjectiveRegistration(const char* name, ObjectiveFactory factory);
};
class DeltaRegistration {
public:
    DeltaRegistration(const char* kind, DeltaFactory factory);
};

// One line at file scope, with a semicolon, is the whole change to the trainer: the file sits
// where the library's glob collects it (src/, src/adjoint/host/, src/adjoint/device/) and nothing
// else is edited. A registration is a static initialiser, so the binary that lists the name must
// actually link the file -- a static archive drops a member nothing references (see
// docs/OBJECTIVES.md, "Where a registration runs").
#define R97_REGISTER_OBJECTIVE(name, factory)                                              \
    static r97::ObjectiveRegistration r97_reg_objective_##name(#name, factory)

#define R97_REGISTER_DELTA(kind, factory)                                                  \
    static r97::DeltaRegistration r97_reg_delta_##kind(#kind, factory)

// ------------------------------------------------------------------ Model by name

// The Model a run trains: its loss and its trainable kind, both resolved by name. An unknown
// name is refused with the list of the registered ones.
Model make_model(const Config& cfg);

// ------------------------------------------------------------------ what the forward recorded

// The table from a boundary's name to the buffer id and the rows the forward driver recorded for
// it, from which the backward core resolves a Seed and check_seeds enforces the window.
struct BoundaryRecord {
    int64_t id = 0;       // the boundary's position in ForwardTape::boundaries()
    int64_t row_lo = 0;   // the rows recorded for it, the window the backward walks
    int64_t row_hi = 0;
};

class BoundaryNames {
public:
    // Built from what the tape holds: the boundary order is the id, the window is the rows.
    static BoundaryNames from(const ForwardTape& fwd);

    // A boundary the caller knows about but the tape does not, such as a block's residual
    // entry during a replay walk. Its id is its insertion order.
    void add(const std::string& buffer, int64_t row_lo, int64_t row_hi);

    bool has(const std::string& buffer) const;
    const BoundaryRecord& at(const std::string& buffer) const;   // throws when absent
    std::vector<std::string> names() const;                      // sorted

private:
    std::map<std::string, BoundaryRecord> by_name_;
};

// One read the objective made: (buffer, row range, differentiated or not), like VectorTape
// records them. A tape check_seeds can audit exposes its record through RecordedTape; the
// forward driver's tape records the same way, so check_seeds runs unchanged there.
struct TapeRead {
    std::string buffer;
    int64_t row_lo = 0;
    int64_t row_hi = 0;
    bool differentiated = false;
};

class RecordedTape : public virtual ForwardTape {
public:
    virtual const std::vector<TapeRead>& reads() const = 0;
};

// The seed check, called by the forward driver before any walk. It refuses, with the buffer's
// name (see docs/OBJECTIVES.md for the rules):
//   - a seed on a buffer that is not one of ForwardTape::boundaries;
//   - a seed whose row range lies outside the window, or whose cotangent does not have the size
//     of its rows;
//   - two seeds of one result that overlap on a buffer (a Seed here and the cotangent handed down
//     from the block above do not overlap, they add);
//   - a differentiated read of a boundary that no seed answers;
//   - a seed on a boundary the objective never read as differentiated.
// A result with no seed at all is valid: it is a forward-only objective.
void check_seeds(const ForwardTape& fwd, const ObjectiveResult& result);

}  // namespace r97
