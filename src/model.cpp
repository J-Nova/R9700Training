// src/model.cpp -- the registries by name, make_model, BoundaryNames, and the seed check.
#include "r97/model.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace r97 {

// ------------------------------------------------------------------ the registries by name

std::map<std::string, ObjectiveFactory>& objective_registry() {
    // The table is deliberately never destroyed. Static registration fills it, and the host tests
    // run from an exit handler: exit destroys each translation unit's statics in the reverse of
    // the order they initialised in, and a test file whose translation unit happens to initialise
    // before this one would then find the table torn down when the handler runs. A table that
    // outlives the process is a table no reader can see destroyed.
    static auto* table = new std::map<std::string, ObjectiveFactory>();
    return *table;
}

std::map<std::string, DeltaFactory>& delta_registry() {
    static auto* table = new std::map<std::string, DeltaFactory>();   // never destroyed: see above
    return *table;
}

std::vector<std::string> objective_names() {
    std::vector<std::string> names;
    for (const auto& [name, _] : objective_registry()) names.push_back(name);
    return names;
}

std::vector<std::string> delta_names() {
    std::vector<std::string> names;
    for (const auto& [name, _] : delta_registry()) names.push_back(name);
    return names;
}

ObjectiveRegistration::ObjectiveRegistration(const char* name, ObjectiveFactory factory) {
    auto& table = objective_registry();
    if (table.count(name))
        throw std::runtime_error(std::string("r97: objective '") + name +
                                 "' is registered twice");
    table[name] = factory;
}

DeltaRegistration::DeltaRegistration(const char* kind, DeltaFactory factory) {
    auto& table = delta_registry();
    if (table.count(kind))
        throw std::runtime_error(std::string("r97: delta kind '") + kind +
                                 "' is registered twice");
    table[kind] = factory;
}

// ------------------------------------------------------------------ Model by name

static std::string join(const std::vector<std::string>& names) {
    std::ostringstream out;
    bool first = true;
    for (const std::string& n : names) {
        if (!first) out << ", ";
        out << n;
        first = false;
    }
    return out.str();
}

Model make_model(const Config& cfg) {
    Model model;

    const auto& objectives = objective_registry();
    auto obj = objectives.find(cfg.objective);
    if (obj == objectives.end())
        throw std::runtime_error("r97: unknown objective '" + cfg.objective +
                                 "' -- registered: " + join(objective_names()));
    model.objective = obj->second(cfg);

    const auto& deltas = delta_registry();
    auto del = deltas.find(cfg.delta.kind);
    if (del == deltas.end())
        throw std::runtime_error("r97: unknown delta kind '" + cfg.delta.kind +
                                 "' -- registered: " + join(delta_names()));
    model.delta = del->second(cfg);

    return model;
}

// ------------------------------------------------------------------ what the forward recorded

BoundaryNames BoundaryNames::from(const ForwardTape& fwd) {
    BoundaryNames names;
    std::vector<std::string> boundaries = fwd.boundaries();
    for (int64_t id = 0; id < static_cast<int64_t>(boundaries.size()); ++id) {
        BoundaryRecord record;
        record.id = id;
        fwd.window(boundaries[id], &record.row_lo, &record.row_hi);
        names.by_name_[boundaries[id]] = record;
    }
    return names;
}

void BoundaryNames::add(const std::string& buffer, int64_t row_lo, int64_t row_hi) {
    BoundaryRecord record;
    record.id = static_cast<int64_t>(by_name_.size());
    record.row_lo = row_lo;
    record.row_hi = row_hi;
    by_name_[buffer] = record;
}

bool BoundaryNames::has(const std::string& buffer) const {
    return by_name_.count(buffer) != 0;
}

const BoundaryRecord& BoundaryNames::at(const std::string& buffer) const {
    return by_name_.at(buffer);
}

std::vector<std::string> BoundaryNames::names() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : by_name_) names.push_back(name);
    return names;
}

// ------------------------------------------------------------------ the seed check

static int64_t cols_of(const std::vector<int64_t>& shape) {
    int64_t cols = 1;
    for (size_t i = 1; i < shape.size(); ++i) cols *= shape[i];
    return cols;
}

void check_seeds(const ForwardTape& fwd, const ObjectiveResult& result) {
    const auto* record = dynamic_cast<const RecordedTape*>(&fwd);
    if (record == nullptr)
        throw std::runtime_error(
            "r97: check_seeds: the tape does not record its reads "
            "(derive from r97::RecordedTape so the seed check can audit them)");
    const std::vector<TapeRead>& reads = record->reads();

    const BoundaryNames boundaries = BoundaryNames::from(fwd);

    // The five refusals, each naming the buffer.
    for (const Seed& seed : result.seeds) {
        if (!boundaries.has(seed.buffer))
            throw std::runtime_error("r97: check_seeds: seed on '" + seed.buffer +
                                     "' is not a boundary of the tape -- boundaries: " +
                                     join(boundaries.names()));
        const BoundaryRecord& boundary = boundaries.at(seed.buffer);
        if (seed.row_lo < boundary.row_lo || seed.row_hi > boundary.row_hi)
            throw std::runtime_error("r97: check_seeds: seed on '" + seed.buffer +
                                     "' rows [" + std::to_string(seed.row_lo) + ", " +
                                     std::to_string(seed.row_hi) +
                                     ") lies outside the window [" +
                                     std::to_string(boundary.row_lo) + ", " +
                                     std::to_string(boundary.row_hi) + ")");
        const int64_t need =
            (seed.row_hi - seed.row_lo) * cols_of(fwd.shape(seed.buffer));
        if (static_cast<int64_t>(seed.cotangent.size()) != need)
            throw std::runtime_error(
                "r97: check_seeds: seed on '" + seed.buffer + "' rows [" +
                std::to_string(seed.row_lo) + ", " + std::to_string(seed.row_hi) +
                ") has a cotangent of " + std::to_string(seed.cotangent.size()) +
                " values, its rows " + std::to_string(need));
    }

    // Two seeds of one result must not overlap on a buffer. (A seed here and the cotangent
    // handed down from the block above do not overlap, they add.)
    std::map<std::string, std::vector<const Seed*>> per_buffer;
    for (const Seed& seed : result.seeds) per_buffer[seed.buffer].push_back(&seed);
    for (auto& [buffer, seeds] : per_buffer) {
        std::sort(seeds.begin(), seeds.end(),
                  [](const Seed* a, const Seed* b) { return a->row_lo < b->row_lo; });
        for (size_t i = 1; i < seeds.size(); ++i) {
            if (seeds[i - 1]->row_hi > seeds[i]->row_lo)
                throw std::runtime_error(
                    "r97: check_seeds: seeds on '" + buffer + "' overlap: rows [" +
                    std::to_string(seeds[i - 1]->row_lo) + ", " +
                    std::to_string(seeds[i - 1]->row_hi) + ") and [" +
                    std::to_string(seeds[i]->row_lo) + ", " +
                    std::to_string(seeds[i]->row_hi) + ")");
        }
    }

    // The reciprocity between what the result seeds and what it read as differentiated.
    for (const TapeRead& read : reads) {
        if (!read.differentiated || !boundaries.has(read.buffer)) continue;
        bool answered = false;
        for (const Seed& seed : result.seeds) {
            if (seed.buffer != read.buffer) continue;
            // A seed answers a read when the two share a row. It need not cover the read: the
            // loss reads its whole window but seeds only the rows it scores, so a window whose
            // ends are unscored is answered by the seed over its scored middle.
            if (seed.row_lo < read.row_hi && read.row_lo < seed.row_hi) {
                answered = true;
                break;
            }
        }
        if (!answered)
            throw std::runtime_error(
                "r97: check_seeds: differentiated read of '" + read.buffer + "' rows [" +
                std::to_string(read.row_lo) + ", " + std::to_string(read.row_hi) +
                ") has no seed");
    }
    for (const Seed& seed : result.seeds) {
        bool read = false;
        for (const TapeRead& r : reads)
            if (r.differentiated && r.buffer == seed.buffer) {
                read = true;
                break;
            }
        if (!read)
            throw std::runtime_error(
                "r97: check_seeds: seed on '" + seed.buffer +
                "' but the objective never read it as differentiated");
    }
}

}  // namespace r97
