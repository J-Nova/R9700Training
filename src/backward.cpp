// src/backward.cpp -- the backward core: one recorded block walked in reverse.
//
// The walk is two passes over the block's OpRecords. The first, in append order, turns every write
// into a version of its buffer and remembers, per row, which version is live at each read -- reads
// before writes within one record, so an in-place op reads what it is about to overwrite. The
// second, in reverse, asks the adjoint registry for each op that has a live cotangent on what it
// wrote, hands it the forward values the snapshot provider answers, releases the versions it
// consumed and adds the cotangents it produced to the versions its reads saw (or to the block's
// input when the read goes back past the block).
#include "r97/backward.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace r97 {
namespace {

[[noreturn]] void refuse(const std::string& what) {
    throw std::runtime_error("r97: backward: " + what);
}

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

// The values of one row; 0 when the tensor holds nothing, so an empty buffer adds no bytes.
int64_t cols_of(const std::vector<int64_t>& shape) {
    if (shape.size() <= 1) return 1;
    if (shape[0] <= 0) return 0;
    return numel(shape) / shape[0];
}

std::string rows_text(int64_t lo, int64_t hi) {
    return "[" + std::to_string(lo) + ", " + std::to_string(hi) + ")";
}

int64_t payload_bytes(const AdValue& v) {
    return static_cast<int64_t>(v.value.size()) * static_cast<int64_t>(sizeof(double));
}

// One write's claim on a buffer: the rows it produced and the cotangent they accumulate. The
// storage belongs to the walk only while the version is live.
struct Version {
    int64_t row_lo = 0;
    int64_t rows = 0;
    int64_t cols = 1;
    std::vector<int64_t> shape;
    std::string dtype;
    std::vector<double> cot;  // rows*cols, row-major over the written rows
    bool live = false;
};

// A buffer the block touched. `row_version` is filled in append order; it holds, after the block,
// the version a seed or the cotangent from above enters.
struct Buffer {
    std::string name;
    std::vector<int64_t> shape;
    int64_t rows = 0;
    int64_t cols = 1;
    std::vector<Version> versions;
    std::vector<int64_t> row_version;  // per row: the live version, -1 before any write
};

// One contiguous run of rows a read took from one version; `version == -1` means the block never
// wrote those rows, so the cotangent is one of the block's inputs.
struct Segment {
    int64_t version = -1;
    int64_t row_lo = 0;
    int64_t row_hi = 0;
};

struct Plan {
    std::vector<Buffer> buffers;
    std::map<std::string, int64_t> by_name;
    std::vector<std::vector<int64_t>> write_version;               // [record][operand]
    std::vector<std::vector<std::vector<Segment>>> read_segments;  // [record][operand][segment]
    // The versions an in-place write's previous value lived in, per row. The walk passes the
    // write's cotangent there after the adjoint has had it.
    std::vector<std::vector<std::vector<Segment>>> base_segments;   // [record][operand][segment]
};

// The rows one operand touches: its own range, or every row of the buffer when it gives none.
void operand_rows(const Operand& o, const Buffer& b, int64_t* lo, int64_t* hi) {
    *lo = o.row_lo;
    *hi = o.row_lo + (o.rows != 0 ? o.rows : b.rows);
    if (*lo < 0 || *hi < *lo || *hi > b.rows)
        refuse("operand on buffer '" + b.name + "' rows " + rows_text(*lo, *hi) +
               " lie outside the buffer's " + std::to_string(b.rows) + " rows");
}

int64_t buffer_index(const Operand& o, Plan* plan) {
    const auto it = plan->by_name.find(o.buffer);
    if (it != plan->by_name.end()) {
        const Buffer& b = plan->buffers[static_cast<size_t>(it->second)];
        if (b.shape != o.shape)
            refuse("buffer '" + o.buffer + "' is recorded with two shapes");
        return it->second;
    }
    Buffer b;
    b.name = o.buffer;
    b.shape = o.shape;
    b.rows = o.shape.empty() ? 1 : o.shape[0];
    b.cols = cols_of(o.shape);
    if (b.rows < 0) refuse("buffer '" + o.buffer + "' has a negative row count");
    b.row_version.assign(static_cast<size_t>(b.rows), -1);
    plan->buffers.push_back(std::move(b));
    const int64_t index = static_cast<int64_t>(plan->buffers.size()) - 1;
    plan->by_name[o.buffer] = index;
    return index;
}

std::vector<Segment> split_rows(const Buffer& b, int64_t lo, int64_t hi) {
    std::vector<Segment> out;
    for (int64_t r = lo; r < hi;) {
        const int64_t v = b.row_version[static_cast<size_t>(r)];
        int64_t e = r + 1;
        while (e < hi && b.row_version[static_cast<size_t>(e)] == v) ++e;
        out.push_back(Segment{v, r, e});
        r = e;
    }
    return out;
}

Plan make_plan(const std::vector<OpRecord>& records) {
    Plan plan;
    plan.write_version.resize(records.size());
    plan.read_segments.resize(records.size());
    plan.base_segments.resize(records.size());
    for (size_t i = 0; i < records.size(); ++i) {
        const OpRecord& rec = records[i];
        plan.write_version[i].assign(rec.operands.size(), -1);
        plan.read_segments[i].resize(rec.operands.size());
        plan.base_segments[i].resize(rec.operands.size());
        // Reads first: an op that writes the buffer it reads sees the version before its own.
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            if (rec.operands[j].written) continue;
            const int64_t bi = buffer_index(rec.operands[j], &plan);
            Buffer& b = plan.buffers[static_cast<size_t>(bi)];
            int64_t lo = 0, hi = 0;
            operand_rows(rec.operands[j], b, &lo, &hi);
            plan.read_segments[i][j] = split_rows(b, lo, hi);
        }
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            if (!rec.operands[j].written) continue;
            const int64_t bi = buffer_index(rec.operands[j], &plan);
            Buffer& b = plan.buffers[static_cast<size_t>(bi)];
            int64_t lo = 0, hi = 0;
            operand_rows(rec.operands[j], b, &lo, &hi);
            if (rec.operands[j].in_place) plan.base_segments[i][j] = split_rows(b, lo, hi);
            Version v;
            v.row_lo = lo;
            v.rows = hi - lo;
            v.cols = b.cols;
            v.shape = b.shape;
            v.dtype = rec.operands[j].dtype;
            const int64_t vi = static_cast<int64_t>(b.versions.size());
            b.versions.push_back(std::move(v));
            for (int64_t r = lo; r < hi; ++r) b.row_version[static_cast<size_t>(r)] = vi;
            plan.write_version[i][j] = vi;
        }
    }
    return plan;
}

}  // namespace

// ---------------------------------------------------------------- StoredSnapshots

void StoredSnapshots::set(int64_t position, size_t operand, AdValue value) {
    values_[{position, operand}] = std::move(value);
}

const AdValue* StoredSnapshots::at(int64_t position, size_t operand) const {
    const auto it = values_.find({position, operand});
    return it == values_.end() ? nullptr : &it->second;
}

// ---------------------------------------------------------------- the walk

BackwardResult backward_block(const std::vector<OpRecord>& records,
                              const SnapshotProvider& snapshots,
                              const BoundaryNames& boundaries,
                              const std::vector<Seed>& seeds,
                              const std::vector<IncomingCotangent>& incoming,
                              const BackwardOptions& options) {
    Plan plan = make_plan(records);
    const std::set<std::string> stops(options.stops.begin(), options.stops.end());

    BackwardResult result;
    std::map<std::string, InputCotangent> inputs;  // the block's inputs, sorted by name
    int64_t live_bytes = 0;                        // version storage plus input accumulators

    const auto touch = [&]() {
        result.report.peak_cotangent_bytes =
            std::max(result.report.peak_cotangent_bytes, live_bytes);
    };

    // The cotangent of one version, created on first use.
    const auto version_cot = [&](Version& v) -> std::vector<double>& {
        if (!v.live) {
            v.cot.assign(static_cast<size_t>(v.rows * v.cols), 0.0);
            v.live = true;
            live_bytes +=
                static_cast<int64_t>(v.cot.size()) * static_cast<int64_t>(sizeof(double));
            touch();
        }
        return v.cot;
    };

    const auto release = [&](Version& v) {
        if (!v.live) return;
        live_bytes -= static_cast<int64_t>(v.cot.size()) * static_cast<int64_t>(sizeof(double));
        v.cot.clear();
        v.cot.shrink_to_fit();
        v.live = false;
    };

    // A view operand is snapped packed over its own shape (see snapshot in step_host.cpp); its
    // cotangent comes back the same way. Scatter it into the layout the buffer's other readers and
    // writers add up in, so the projection that wrote the buffer sees what it contributed.
    const auto dense_view = [](const Operand& op, const std::vector<double>& view,
                               int64_t rows, int64_t cols) {
        if (op.view_shape.empty() || cols <= 0) return std::vector<double>();
        std::vector<double> full(static_cast<size_t>(std::max<int64_t>(rows * cols, 0)), 0.0);
        const size_t rank = op.view_shape.size();
        std::vector<int64_t> idx(rank, 0);
        for (size_t e = 0; e < view.size(); ++e) {
            int64_t flat = op.view_off;
            for (size_t d = 0; d < rank; ++d) flat += idx[d] * op.view_stride[d];
            if (flat >= 0 && flat < static_cast<int64_t>(full.size()))
                full[static_cast<size_t>(flat)] += view[e];
            for (size_t d = rank; d-- > 0;) {
                if (++idx[d] < op.view_shape[d]) break;
                idx[d] = 0;
            }
        }
        return full;
    };

    // Add the rows [lo, hi) of a full-buffer, row-major tensor into a version's written rows.
    const auto add_version_full = [&](Version& v, int64_t lo, int64_t hi,
                                      const std::vector<double>& full) {
        if (lo >= hi || v.cols <= 0) return;
        std::vector<double>& dst = version_cot(v);
        const int64_t cols = v.cols;
        for (int64_t r = lo; r < hi; ++r) {
            const double* src = &full[static_cast<size_t>(r * cols)];
            double* at = dst.data() + static_cast<size_t>((r - v.row_lo) * cols);
            for (int64_t k = 0; k < cols; ++k) at[k] += src[k];
        }
    };

    // The same, from a window that starts at `win_lo`.
    const auto add_version_window = [&](Version& v, int64_t lo, int64_t hi, int64_t win_lo,
                                        const std::vector<double>& win) {
        if (lo >= hi || v.cols <= 0) return;
        std::vector<double>& dst = version_cot(v);
        const int64_t cols = v.cols;
        for (int64_t r = lo; r < hi; ++r) {
            const double* src = &win[static_cast<size_t>((r - win_lo) * cols)];
            double* at = dst.data() + static_cast<size_t>((r - v.row_lo) * cols);
            for (int64_t k = 0; k < cols; ++k) at[k] += src[k];
        }
    };

    // A cotangent that leaves the block: dense over the buffer, created on first use.
    const auto add_input_full = [&](const Buffer& b, int64_t lo, int64_t hi,
                                    const std::vector<double>& full) {
        if (lo >= hi || b.cols <= 0) return;
        InputCotangent& in = inputs[b.name];
        if (in.value.empty() && b.rows * b.cols != 0) {
            in.buffer = b.name;
            in.row_lo = 0;
            in.row_hi = b.rows;
            in.shape = b.shape;
            in.value.assign(static_cast<size_t>(b.rows * b.cols), 0.0);
            live_bytes +=
                static_cast<int64_t>(in.value.size()) * static_cast<int64_t>(sizeof(double));
            touch();
        }
        const int64_t cols = b.cols;
        for (int64_t r = lo; r < hi; ++r) {
            const double* src = &full[static_cast<size_t>(r * cols)];
            double* at = in.value.data() + static_cast<size_t>(r * cols);
            for (int64_t k = 0; k < cols; ++k) at[k] += src[k];
        }
    };

    // Resolve a cotangent that enters at the end of the block: a boundary's rows, the version the
    // block left live there, and the writer's dtype. Seeds and the cotangent from above enter the
    // same store and add.
    const auto enter = [&](const std::string& name, int64_t lo, int64_t hi,
                           const std::vector<double>& values, const char* what) {
        if (!boundaries.has(name))
            refuse(std::string(what) + " on '" + name +
                   "', which is not a boundary of the forward");
        const BoundaryRecord& boundary = boundaries.at(name);
        if (lo > hi || lo < boundary.row_lo || hi > boundary.row_hi)
            refuse(std::string(what) + " on '" + name + "' rows " + rows_text(lo, hi) +
                   " lie outside the boundary's window " +
                   rows_text(boundary.row_lo, boundary.row_hi));
        const auto it = plan.by_name.find(name);
        if (it == plan.by_name.end() ||
            plan.buffers[static_cast<size_t>(it->second)].versions.empty())
            refuse(std::string(what) + " on '" + name + "', which the block never wrote");
        Buffer& b = plan.buffers[static_cast<size_t>(it->second)];
        if (lo < 0 || hi > b.rows)
            refuse(std::string(what) + " on '" + name + "' rows " + rows_text(lo, hi) +
                   " lie outside the buffer's " + std::to_string(b.rows) + " rows");
        const int64_t need = (hi - lo) * b.cols;
        if (static_cast<int64_t>(values.size()) != need)
            refuse(std::string(what) + " on '" + name + "' rows " + rows_text(lo, hi) + " has " +
                   std::to_string(values.size()) + " values, not " + std::to_string(need));
        for (const Segment& seg : split_rows(b, lo, hi)) {
            if (seg.version < 0)
                refuse(std::string(what) + " on '" + name + "' rows " +
                       rows_text(seg.row_lo, seg.row_hi) + ", which the block never wrote");
            Version& v = b.versions[static_cast<size_t>(seg.version)];
            add_version_window(v, seg.row_lo, seg.row_hi, lo, values);
        }
    };

    for (const Seed& seed : seeds) {
        const std::vector<double> values(seed.cotangent.begin(), seed.cotangent.end());
        enter(seed.buffer, seed.row_lo, seed.row_hi, values, "seed");
    }
    for (const IncomingCotangent& cot : incoming)
        enter(cot.buffer, cot.row_lo, cot.row_hi, cot.value, "the cotangent from above");

    // The version a record's written operand produced.
    const auto version_at = [&](size_t i, size_t j) -> Version& {
        const int64_t bi = plan.by_name.at(records[i].operands[j].buffer);
        return plan.buffers[static_cast<size_t>(bi)]
            .versions[static_cast<size_t>(plan.write_version[i][j])];
    };

    // The walk itself, latest record first.
    for (size_t i = records.size(); i-- > 0;) {
        const OpRecord& rec = records[i];

        bool on_path = false;
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            if (!rec.operands[j].written) continue;
            if (plan.write_version[i][j] < 0) continue;
            if (version_at(i, j).live) {
                on_path = true;
                break;
            }
        }
        if (!on_path) continue;

        if (stops.count(rec.op)) {
            for (size_t j = 0; j < rec.operands.size(); ++j)
                if (rec.operands[j].written && plan.write_version[i][j] >= 0)
                    release(version_at(i, j));
            continue;
        }

        std::string err;
        const AdjointFn fn = adjoint_find(rec.op, options.side, rec.params, &err);
        if (!fn) refuse(err);

        // The forward values of every operand, as this op saw them.
        std::vector<AdValue> fwd(rec.operands.size());
        int64_t snapshot_bytes = 0;
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            const AdValue* v = snapshots.at(rec.position, j);
            if (!v)
                refuse("op '" + rec.op + "' at position " + std::to_string(rec.position) +
                       " has no snapshot for operand '" + rec.operands[j].buffer + "'");
            // A view operand is snapped as the kernel saw it: its own shape, not the buffer's.
            const std::vector<int64_t>& want = rec.operands[j].view_shape.empty()
                                                   ? rec.operands[j].shape
                                                   : rec.operands[j].view_shape;
            if (v->shape != want)
                refuse("op '" + rec.op + "' at position " + std::to_string(rec.position) +
                       ": operand '" + rec.operands[j].buffer + "' is snapped with another shape");
            if (v->value.size() != static_cast<size_t>(numel(v->shape)))
                refuse("op '" + rec.op + "' at position " + std::to_string(rec.position) +
                       ": operand '" + rec.operands[j].buffer + "' is snapped with " +
                       std::to_string(v->value.size()) + " values for its shape");
            fwd[j] = *v;
            snapshot_bytes += payload_bytes(fwd[j]);
        }
        result.report.peak_snapshot_bytes =
            std::max(result.report.peak_snapshot_bytes, snapshot_bytes);

        // The cotangent of every operand the op wrote, writes' order.
        std::vector<AdValue> cot;
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            if (!rec.operands[j].written) continue;
            Version& v = version_at(i, j);
            AdValue c;
            c.dtype = v.dtype;
            c.shape = v.shape;
            c.value.assign(static_cast<size_t>(numel(v.shape)), 0.0);
            if (v.live && v.cols > 0)
                for (int64_t r = 0; r < v.rows; ++r) {
                    const double* src = &v.cot[static_cast<size_t>(r * v.cols)];
                    std::copy(src, src + v.cols,
                              c.value.begin() + static_cast<size_t>((v.row_lo + r) * v.cols));
                }
            cot.push_back(std::move(c));
        }

        // dc: every operand, in schema order, all zeros for the adjoint to add to.
        std::vector<AdValue> dc(rec.operands.size());
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            dc[j].dtype = fwd[j].dtype;
            dc[j].shape = fwd[j].shape;
            dc[j].value.assign(static_cast<size_t>(numel(dc[j].shape)), 0.0);
        }

        {
            int64_t held = live_bytes;
            for (const AdValue& v : cot) held += payload_bytes(v);
            for (const AdValue& v : dc) held += payload_bytes(v);
            result.report.peak_cotangent_bytes =
                std::max(result.report.peak_cotangent_bytes, held);
        }

        const AdjointCall call{rec, fwd, cot, &dc, options.grads, options.prefix};
        AdjointCall with_slots = call;
        if (options.slot_prefix)
            with_slots.slot_prefix = [&rec, &options](int64_t slot) {
                return options.slot_prefix(rec, slot);
            };
        fn(with_slots);

        // An in-place write's cotangent is also what the value it replaced takes. The adjoint put
        // that in the write's slot; pass it to the versions those rows held before this write.
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            if (!rec.operands[j].written || !rec.operands[j].in_place) continue;
            if (plan.write_version[i][j] < 0) continue;
            const int64_t bi = plan.by_name.at(rec.operands[j].buffer);
            Buffer& b = plan.buffers[static_cast<size_t>(bi)];
            if (dc[j].value.size() != static_cast<size_t>(b.rows * b.cols))
                refuse("op '" + rec.op + "': adjoint left in-place operand '" +
                       rec.operands[j].buffer + "' with " + std::to_string(dc[j].value.size()) +
                       " values, not " + std::to_string(b.rows * b.cols));
            for (const Segment& seg : plan.base_segments[i][j]) {
                if (seg.version >= 0)
                    add_version_full(b.versions[static_cast<size_t>(seg.version)], seg.row_lo,
                                     seg.row_hi, dc[j].value);
                else
                    add_input_full(b, seg.row_lo, seg.row_hi, dc[j].value);
            }
        }

        // The writers' adjoint has run: their cotangent storage is done.
        for (size_t j = 0; j < rec.operands.size(); ++j)
            if (rec.operands[j].written && plan.write_version[i][j] >= 0)
                release(version_at(i, j));

        // Hand what the adjoint produced to the versions each read saw, or out of the block.
        const std::vector<int> frozen = adjoint_frozen(rec.op, options.side, rec.params);
        const std::set<size_t> frozen_at(frozen.begin(), frozen.end());
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            if (rec.operands[j].written) continue;
            if (frozen_at.count(j)) continue;
            const int64_t bi = plan.by_name.at(rec.operands[j].buffer);
            Buffer& b = plan.buffers[static_cast<size_t>(bi)];
            const std::vector<double> dense = dense_view(rec.operands[j], dc[j].value, b.rows,
                                                         b.cols);
            const std::vector<double>& values = dense.empty() ? dc[j].value : dense;
            if (values.size() != static_cast<size_t>(b.rows * b.cols))
                refuse("op '" + rec.op + "': adjoint left operand '" + rec.operands[j].buffer +
                       "' with " + std::to_string(values.size()) + " values, not " +
                       std::to_string(b.rows * b.cols));
            for (const Segment& seg : plan.read_segments[i][j]) {
                if (seg.version >= 0)
                    add_version_full(b.versions[static_cast<size_t>(seg.version)], seg.row_lo,
                                     seg.row_hi, values);
                else
                    add_input_full(b, seg.row_lo, seg.row_hi, values);
            }
        }
    }

    for (auto& [name, cot] : inputs) result.inputs.push_back(std::move(cot));
    return result;
}

}  // namespace r97
