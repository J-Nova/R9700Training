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

#include "parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace r97 {
namespace {

using Clock = std::chrono::steady_clock;

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

std::string shape_text(const std::vector<int64_t>& shape) {
    std::string out = "[";
    for (size_t d = 0; d < shape.size(); ++d) {
        if (d) out += ", ";
        out += std::to_string(shape[d]);
    }
    return out + "]";
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

// A written operand that is also the op's input, though no separate read names it: rope's
// in-place write and the recurrent/conv state buffers. The prior version is the read. The issue
// record marks such an operand (`Operand::in_place`, a RAD_OPD_INOUT operand); a record built from
// a program instead of the live tape may not carry the mark, so the two shapes the ops always
// have -- rope's operand 0, and a `kv:` buffer no separate read names -- are recognised here too.
bool in_place_read(const OpRecord& rec, size_t j) {
    const Operand& o = rec.operands[j];
    if (!o.written) return false;
    if (o.in_place) return true;
    if (rec.op == "rope" && j == 0) return true;
    if (o.buffer.rfind("kv:", 0) != 0) return false;
    for (size_t k = 0; k < rec.operands.size(); ++k)
        if (k != j && !rec.operands[k].written && rec.operands[k].buffer == o.buffer) return false;
    return true;
}

struct Plan {
    std::vector<Buffer> buffers;
    std::map<std::string, int64_t> by_name;
    std::vector<std::vector<int64_t>> write_version;               // [record][operand]
    std::vector<std::vector<std::vector<Segment>>> read_segments;  // [record][operand][segment]
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
    for (size_t i = 0; i < records.size(); ++i) {
        const OpRecord& rec = records[i];
        plan.write_version[i].assign(rec.operands.size(), -1);
        plan.read_segments[i].resize(rec.operands.size());
        // Reads first: an op that writes the buffer it reads sees the version before its own.
        // rope is recorded with only its in-place write, so operand 0 is also its read: name that
        // prior version here, and the walk hands its adjoint's cotangent back to it below.
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            const bool in_place = in_place_read(rec, j);
            if (rec.operands[j].written && !in_place) continue;
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

int64_t StoredSnapshots::bytes() const {
    int64_t total = 0;
    for (const auto& entry : values_) total += payload_bytes(entry.second);
    return total;
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
    std::map<std::string, OpTiming> row_time;  // the row calls per op, for the report
    int64_t live_bytes = 0;                        // version storage plus input accumulators

    const auto touch = [&]() {
        result.report.peak_cotangent_bytes =
            std::max(result.report.peak_cotangent_bytes, live_bytes);
    };

    // A non-finite value in a cotangent the walk is about to hand on: count it against the op, and
    // keep the first one the walk met, with the operand it belongs to and where it came from. The
    // counts and the naming are what turn a NaN gradient into one line a run can be read for; they
    // change no value, no order and no row.
    const auto note_non_finite = [&](const OpRecord& rec, const std::string& op, size_t j,
                                     int64_t at, bool from_row) {
        row_time[op].non_finite += 1;
        if (!result.report.first_non_finite_op.empty()) return;
        result.report.first_non_finite_op = op;
        result.report.first_non_finite_operand = rec.operands[j].buffer;
        result.report.first_non_finite_operand_index = static_cast<int64_t>(j);
        result.report.first_non_finite_position = rec.position;
        result.report.first_non_finite_index = at;
        result.report.first_non_finite_from_row = from_row;
    };

    // The first value of a slot that is not finite, or -1. One pass, and it stops at the hit: the
    // walk already touches every slot it hands on, so this is a scan of what it is holding, not a
    // second pass over the store.
    const auto first_non_finite = [](const std::vector<double>& v) -> int64_t {
        for (size_t e = 0; e < v.size(); ++e)
            if (!std::isfinite(v[e])) return static_cast<int64_t>(e);
        return -1;
    };

    // One unit of work runs on the calling thread: handing a single chunk to a thread costs more
    // than the chunk is worth.
    const auto run_units = [](int64_t n, const auto& body) {
        if (n > 1)
            parallel_for(n, 1, body);
        else if (n == 1)
            body(0);
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

    // The input slot of a buffer the block read but never wrote, created on first use. An existing
    // slot is found, not touched: the distributing workers look a buffer up that they own alone,
    // and only the serial creation ever inserts.
    const auto input_slot = [&](const Buffer& b) -> InputCotangent& {
        auto it = inputs.find(b.name);
        if (it == inputs.end()) it = inputs.emplace(b.name, InputCotangent{}).first;
        InputCotangent& in = it->second;
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
        return in;
    };

    // A cotangent that leaves the block: dense over the buffer, created on first use.
    const auto add_input_full = [&](const Buffer& b, int64_t lo, int64_t hi,
                                    const std::vector<double>& full) {
        if (lo >= hi || b.cols <= 0) return;
        InputCotangent& in = input_slot(b);
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

        // The forward values of every operand, as this op saw them. One worker per operand: each
        // operand's slot is its own, so the copies -- at a real block's rows, the giga-byte frozen
        // weights among them -- run side by side without a value moving. A refusal inside a unit
        // comes back to the caller as itself, after the workers have run out their chunks.
        std::vector<AdValue> fwd(rec.operands.size());
        std::vector<int64_t> fwd_bytes(rec.operands.size(), 0);
        // The forward's own values, per operand: where the first non-finite one sits. Each unit
        // fills its own entry, and the serial read below names the first in operand order.
        std::vector<int64_t> fwd_non_finite(rec.operands.size(), -1);
        run_units(static_cast<int64_t>(rec.operands.size()), [&](int64_t j) {
            const size_t k = static_cast<size_t>(j);
            const AdValue* v = snapshots.at(rec.position, k);
            if (!v)
                refuse("op '" + rec.op + "' at position " + std::to_string(rec.position) +
                       " has no snapshot for operand '" + rec.operands[k].buffer + "'");
            // A view operand is snapped as the kernel saw it: its own shape, not the buffer's.
            const std::vector<int64_t>& want = rec.operands[k].view_shape.empty()
                                                   ? rec.operands[k].shape
                                                   : rec.operands[k].view_shape;
            if (v->shape != want)
                refuse("op '" + rec.op + "' at position " + std::to_string(rec.position) +
                       ": operand '" + rec.operands[k].buffer + "' is snapped with shape " +
                       shape_text(v->shape) + ", not " + shape_text(want));
            // R97_DUMP_VIEW=1: what a view operand's strides say about its span. A view whose
            // numel exceeds its operand's reads past the declared buffer unless every excess
            // stride is zero, and a nonzero row stride over a one-row buffer is that read.
            if (std::getenv("R97_DUMP_VIEW") != nullptr && !rec.operands[k].view_shape.empty()) {
                std::string stride;
                for (int64_t d : rec.operands[k].view_stride)
                    stride += (stride.empty() ? "" : ",") + std::to_string(d);
                std::fprintf(stderr,
                             "VIEW pos=%lld j=%zu buffer=%s shape=%s view=%s stride=[%s] off=%lld "
                             "values=%zu numel=%lld\n",
                             (long long)rec.position, k, rec.operands[k].buffer.c_str(),
                             shape_text(rec.operands[k].shape).c_str(),
                             shape_text(rec.operands[k].view_shape).c_str(), stride.c_str(),
                             (long long)rec.operands[k].view_off, v->value.size(),
                             (long long)numel(rec.operands[k].shape));
                std::fflush(stderr);
            }
            // A plugin-private operand (the w8a8_fwht128 trunk's weight and its scale) has no
            // f64 form: the snapshot carries its shape and the tape's pointer but no values, and
            // a device row decodes the engine's stored layout from that pointer. An operand with
            // neither values nor a pointer is still refused: it is the case the walk cannot
            // differentiate at all.
            if (v->value.size() != static_cast<size_t>(numel(v->shape)) && v->raw == nullptr)
                refuse("op '" + rec.op + "' at position " + std::to_string(rec.position) +
                       ": operand '" + rec.operands[k].buffer + "' is snapped with " +
                       std::to_string(v->value.size()) + " values for its shape");
            fwd[k] = *v;
            fwd_bytes[k] = payload_bytes(fwd[k]);
            fwd_non_finite[k] = first_non_finite(fwd[k].value);
        });
        // A sum of integers: the order the units finished in cannot move the figure.
        int64_t snapshot_bytes = 0;
        for (int64_t bytes : fwd_bytes) snapshot_bytes += bytes;
        result.report.peak_snapshot_bytes =
            std::max(result.report.peak_snapshot_bytes, snapshot_bytes);
        // The forward's finiteness, kept once: the first operand the walk met that was not finite
        // is enough to say a NaN was already in the values, and it is what a run needs to read.
        if (result.report.forward_all_finite)
            for (size_t j = 0; j < rec.operands.size(); ++j)
                if (fwd_non_finite[j] >= 0) {
                    result.report.forward_all_finite = false;
                    result.report.forward_first_non_finite_operand = rec.operands[j].buffer;
                    result.report.forward_first_non_finite_operand_index =
                        static_cast<int64_t>(j);
                    result.report.forward_first_non_finite_position = rec.position;
                    result.report.forward_first_non_finite_index = fwd_non_finite[j];
                    break;
                }

        // The cotangent of every operand the op wrote, writes' order: the copy out of the version
        // store, one worker per written operand. The slots and the versions are each operand's own,
        // so no unit reads what its neighbour writes.
        std::vector<AdValue> cot;
        std::vector<int> cot_index(rec.operands.size(), -1);
        // Where the first non-finite value of each written operand's incoming cotangent sits: the
        // values the walk is handed rather than the ones it produces.
        std::vector<int64_t> cot_non_finite(rec.operands.size(), -1);
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            if (!rec.operands[j].written) continue;
            cot_index[j] = static_cast<int>(cot.size());
            cot.emplace_back();
        }
        run_units(static_cast<int64_t>(rec.operands.size()), [&](int64_t j) {
            const size_t k = static_cast<size_t>(j);
            if (cot_index[k] < 0) return;
            Version& v = version_at(i, k);
            AdValue& c = cot[static_cast<size_t>(cot_index[k])];
            c.dtype = v.dtype;
            c.shape = v.shape;
            c.value.assign(static_cast<size_t>(numel(v.shape)), 0.0);
            if (v.live && v.cols > 0)
                for (int64_t r = 0; r < v.rows; ++r) {
                    const double* src = &v.cot[static_cast<size_t>(r * v.cols)];
                    std::copy(src, src + v.cols,
                              c.value.begin() + static_cast<size_t>((v.row_lo + r) * v.cols));
                }
            cot_non_finite[k] = first_non_finite(c.value);
        });

        // The largest finite magnitude the row is handed, over every written operand's
        // cotangent. The device rows take their operands in f32 and write the cotangent back
        // through the record's dtype; the cotangent is the one value they cannot widen.
        double cot_max = 0.0;
        for (const AdValue& c : cot)
            for (double q : c.value) {
                const double a = std::fabs(q);
                if (std::isfinite(a) && a > cot_max) cot_max = a;
            }
        // A host row cannot read a weight the tape carries behind `raw` (the engine's private
        // encoding): it refuses by name, and the device row is the only side that decodes it. So
        // the range fallback below asks only for records whose forward the host row can read.
        bool host_readable = true;
        for (const AdValue& v : fwd)
            if (v.raw != nullptr) {
                host_readable = false;
                break;
            }

        // The adjoint sees the record the caller prepared when it wants one: the join may hand it
        // names and rows that differ from the tape's, while the walk keeps the tape's record for
        // its versions, boundaries and snapshots. The row is resolved from the prepared record's
        // parameters because that record is the one the adjoint is called with: a predicate set
        // there is what picks a shape-aware row, and asking the tape's parameters for it would
        // always find the catch-all.
        OpRecord named;
        const OpRecord* adjoint_rec = &rec;
        if (options.prepare_record) {
            options.prepare_record(rec, &named);
            adjoint_rec = &named;
        }

        std::string err;
        AdjointFn fn = adjoint_find(adjoint_rec->op, options.side, adjoint_rec->params, &err);
        Side served = options.side;
        if (!fn && options.allow_host_fallback && options.side != Side::host) {
            // The requested side does not serve this op: the host row is the same arithmetic in
            // f64, and the caller asked for the fallback. `err` keeps the requested side's
            // message, so a missing row on both sides still refuses with the one that was asked
            // for.
            std::string host_err;
            AdjointFn host_fn = adjoint_find(adjoint_rec->op, Side::host, adjoint_rec->params,
                                             &host_err);
            if (host_fn) {
                fn = host_fn;
                served = Side::host;
            }
        }
        // A device row computes in f32 (its kernels' operands) and writes the cotangent back
        // through the record's dtype, which the tape holds at 16 bits. A cotangent it is handed
        // outside that range is not representable there: the row's arithmetic overflows and every
        // row below it inherits the Inf. The host row is the same arithmetic in f64, so a record
        // whose incoming cotangent is past the card's range falls back to it. The limit leaves
        // headroom for the row's own amplification -- the gated norm's `sc^3` term, the quant
        // row's division by the group's scale -- not just f32's 3.4e38.
        constexpr double kDeviceCotMax = 1e30;
        if (fn && served == Side::device && options.allow_host_fallback && host_readable &&
            cot_max > kDeviceCotMax) {
            std::string host_err;
            AdjointFn host_fn = adjoint_find(adjoint_rec->op, Side::host, adjoint_rec->params,
                                             &host_err);
            if (host_fn) {
                fn = host_fn;
                served = Side::host;
            }
        }
        if (!fn) refuse(err);
        // What the walk handed this op is already non-finite: a seed, the cotangent from above, or
        // a record later in the forward that the walk has already walked. Recorded before the row
        // runs, so the first non-finite the walk meets is attributed to where it came from and not
        // to the row that merely passes it on.
        for (size_t j = 0; j < rec.operands.size(); ++j)
            if (cot_non_finite[j] >= 0)
                note_non_finite(rec, adjoint_rec->op, j, cot_non_finite[j], false);

        // The input side of one record, in full, so "was this row handed finite data?" is answerable.
        //
        // Two facts settle what can be asked. The walk holds a version only for an operand that
        // WROTE one (`plan.write_version[i][j] >= 0`), so every lookup here is guarded -- an
        // unguarded `version_at` indexes `versions[-1]` and segfaults, which is how the first cut of
        // this instrument died. And a read-only operand has no incoming cotangent at all: the row
        // PRODUCES the gradient for its inputs (`dc`) and CONSUMES the cotangent of its outputs
        // (`cot`, built for written operands alone at `:512`). So the written operands are the whole
        // input side, and the accounting is not in fact blind here -- `h` is written *and* read (an
        // in-place write), so `cot_non_finite[h]` is exactly its incoming cotangent.
        // `R97_DUMP_POS=1176` names one record; `R97_DUMP_POS=all` prints every record in walk
        // order, which is the growth curve: the walk runs 63 -> 0, so the log reads from the head
        // downwards and the op where a magnitude jumps is the one that amplified it.
        if (const char* dp = std::getenv("R97_DUMP_POS");
            dp != nullptr && (std::string(dp) == "all" || rec.position == std::atoll(dp))) {
            for (size_t j = 0; j < rec.operands.size(); ++j) {
                const int64_t wv = plan.write_version[i][j];
                if (wv < 0) {
                    std::fprintf(stderr,
                                 "DUMPPOS pos=%s op=%s opnd=%zu buffer=%s written=%d "
                                 "no_version=1\n",
                                 std::to_string(rec.position).c_str(), adjoint_rec->op.c_str(), j,
                                 rec.operands[j].buffer.c_str(),
                                 static_cast<int>(rec.operands[j].written));
                    continue;
                }
                const Version& v = version_at(i, j);
                int64_t inc = -1;
                int64_t nvals = 0;
                double maxabs = 0.0;
                if (v.live && !v.cot.empty()) {
                    inc = first_non_finite(v.cot);
                    nvals = static_cast<int64_t>(v.cot.size());
                    // Finite is not small. The codes' cotangent is divided by the group's scale in
                    // f32 and the quotient must stay under 3.4e38, so the magnitude of what the row
                    // was HANDED is the other half of that product -- without it, "finite" says
                    // nothing about whether the division can overflow.
                    for (double q : v.cot) {
                        const double a = std::fabs(q);
                        if (std::isfinite(a) && a > maxabs) maxabs = a;
                    }
                }
                std::fprintf(stderr,
                             "DUMPPOS pos=%s op=%s opnd=%zu buffer=%s written=%d live=%d "
                             "inc_cot=%lld inc_nonfinite_at=%lld maxabs=%.6g cot_nonfinite=%lld\n",
                             std::to_string(rec.position).c_str(), adjoint_rec->op.c_str(), j,
                             rec.operands[j].buffer.c_str(),
                             static_cast<int>(rec.operands[j].written), static_cast<int>(v.live),
                             static_cast<long long>(nvals), static_cast<long long>(inc), maxabs,
                             static_cast<long long>(j < cot_non_finite.size() ? cot_non_finite[j]
                                                                             : -1));
            }
            std::fflush(stderr);
        }
        // The frozen operands of the row that serves -- the fallback's, when the fallback took
        // the call -- resolved before the slots are sized: the walk does not ask a row for its
        // frozen operands, and a row that served must not write them anyway.
        const std::vector<int> frozen = adjoint_frozen(adjoint_rec->op, served, adjoint_rec->params);
        const FillsFn fills = adjoint_fills(adjoint_rec->op, served, adjoint_rec->params);
        const std::set<size_t> frozen_at(frozen.begin(), frozen.end());

        // dc: one slot per operand, in schema order. A slot the row may add to -- a read, or an
        // in-place write the record also reads -- is zeroed for it to add into. A frozen slot and
        // a plain write keep their dtype and shape but hold no payload: a record whose operands
        // are the model's big frozen weights therefore allocates no cotangent for them, and a
        // row that wrote such a slot would write into nothing.

        // One worker per operand: each writes only its own slot, and the slots the row may add to
        // -- the activation-sized ones -- are the ones that cost.
        std::vector<AdValue> dc(rec.operands.size());
        std::vector<char> sized(rec.operands.size(), 0);
        run_units(static_cast<int64_t>(rec.operands.size()), [&](int64_t j) {
            const size_t k = static_cast<size_t>(j);
            dc[k].dtype = fwd[k].dtype;
            dc[k].shape = fwd[k].shape;
            // A frozen operand is one the row does not compute. That is a statement about the
            // ROW, not about the cotangent: an in-place write the record also reads still carries
            // the write's cotangent to the value it replaced, and the passthrough below is what
            // fills it. Leaving such a slot unsized is what dropped the residual stream's
            // cotangent at every fused norm -- the row froze it, the passthrough skipped it for
            // being unsized, and only the branch term reached the block's input.
            if (frozen_at.count(k) && !in_place_read(rec, k)) return;
            if (rec.operands[k].written && !in_place_read(rec, k)) return;
            dc[k].value.assign(static_cast<size_t>(numel(dc[k].shape)), 0.0);
            sized[k] = 1;
        });

        {
            int64_t held = live_bytes;
            for (const AdValue& v : cot) held += payload_bytes(v);
            for (const AdValue& v : dc) held += payload_bytes(v);
            result.report.peak_cotangent_bytes =
                std::max(result.report.peak_cotangent_bytes, held);
        }

        std::string record_prefix;
        const char* prefix = options.prefix;
        if (options.record_prefix) {
            record_prefix = options.record_prefix(rec);
            prefix = record_prefix.c_str();
        }
        // The rows a consumer of the whole tape reads, when the caller proved them. A record the
        // caller left out is unproven: the row sees null and treats every row as readable.
        const std::vector<int64_t>* read_rows = nullptr;
        if (options.read_rows) {
            const auto it = options.read_rows->find(rec.position);
            if (it != options.read_rows->end()) read_rows = &it->second;
        }
        AdjointCall::RowCount counted;
        AdjointCall call{*adjoint_rec, fwd, cot, &dc, options.grads, prefix};
        call.fused_slots = options.fused_slots;
        call.read_rows = read_rows;
        call.row_count = &counted;
        AdjointCall with_slots = call;
        if (options.slot_prefix)
            with_slots.slot_prefix = [&rec, &options](int64_t slot) {
                return options.slot_prefix(rec, slot);
            };
        // The row's share of the report: the call, its wall time and the two-way rule's counts.
        // A row that refuses accounts what it did before it threw -- a refusal is one of the two
        // outcomes the report counts, and the throw must not lose it.
        const auto account = [&](double seconds) {
            OpTiming& row_acc = row_time[adjoint_rec->op];
            row_acc.calls += 1;
            row_acc.host = served == Side::host;
            row_acc.seconds += seconds;
            row_acc.rows_zeroed_unreadable += counted.zeroed_unreadable;
            row_acc.rows_readable += counted.readable;
            row_acc.rows_refused_non_finite += counted.refused_non_finite;
            // A row the rule decided something about, named where it happened. A run that zeroes
            // rows is a run whose read set proved something, and that must never be silent: the
            // rule's failure mode is a live row zeroed, which is invisible in the gradient itself.
            if (counted.zeroed_unreadable > 0 || counted.refused_non_finite > 0)
                std::fprintf(stderr,
                             "r97: read set: op '%s' at %lld: %lld of %lld proven rows zeroed, "
                             "%lld readable, %lld refused (read set %s)\n",
                             adjoint_rec->op.c_str(), (long long)rec.position,
                             (long long)counted.zeroed_unreadable,
                             (long long)(counted.zeroed_unreadable + counted.readable),
                             (long long)counted.readable,
                             (long long)counted.refused_non_finite,
                             read_rows != nullptr ? "proven" : "unknown");
        };
        const Clock::time_point row_begin = Clock::now();
        try {
            fn(with_slots);
        } catch (...) {
            account(std::chrono::duration<double>(Clock::now() - row_begin).count());
            throw;
        }
        account(std::chrono::duration<double>(Clock::now() - row_begin).count());

        if (options.in_place_passthrough) {
            // An adjoint either describes what a read takes or leaves the slot it would write empty.
            // An empty slot on an in-place operand is the op's direct path: the value the write
            // replaced carries the write's cotangent unchanged (an expand adds its branch to y
            // without touching the read's slot; a mul, a rope or a state update fill theirs).
            //
            // A row that registered a fills predicate owns the slot: a zero there is the row's
            // answer (kv_store's uncovered pool rows are unread in a single-pass store, so the
            // row's correct fill is all zero), and passing the cotangent through would add the
            // covered rows' cotangent a second time.
            //
            // The direct path is only the answer when no *separate* read of the same buffer
            // already answered it. A record with `read x, write x` gives the row the read's slot to
            // fill, and that slot distributes to the same prior version this write's segments name
            // -- so filling this one too would add the cotangent twice. The check is on the value
            // the row actually produced, not on the shape: a row that leaves the read empty has
            // said nothing about the write either, and the direct path is still the answer.
            const auto read_answered = [&](size_t j) {
                for (size_t k = 0; k < rec.operands.size(); ++k) {
                    if (k == j || rec.operands[k].written || !sized[k]) continue;
                    if (rec.operands[k].buffer != rec.operands[j].buffer) continue;
                    for (double v : dc[k].value)
                        if (v != 0.0) return true;
                }
                return false;
            };
            for (size_t j = 0; j < rec.operands.size(); ++j) {
                if (!in_place_read(rec, j) || cot_index[j] < 0) continue;
                // A slot the walk did not size is not the row's to fill: it is frozen, and the
                // sum below would read it as "untouched" and copy the read's cotangent in.
                if (!sized[j]) continue;
                if (fills && fills(call.rec, j)) continue;
                double written = 0.0;
                for (double v : dc[j].value) written += std::fabs(v);
                if (written == 0.0 && !read_answered(j))
                    dc[j].value = cot[static_cast<size_t>(cot_index[j])].value;
            }
        }

        // What the row produced, scanned where the walk is about to hand it on: the slots the row
        // may add to -- a read, or an in-place write the record also reads. A slot the walk did
        // not size is a frozen operand the row must not write, so there is nothing to scan; one
        // hit per slot is enough to name it. The values the row was handed were scanned above, so
        // a non-finite value recorded here is the row's own.
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            if (!sized[j]) continue;
            const int64_t at = first_non_finite(dc[j].value);
            if (at >= 0) note_non_finite(rec, adjoint_rec->op, j, at, true);
        }

        // R97_DUMP_DC=1: what a row was handed and what it left, per operand, for the ops an
        // adapter's gradient flows through and the one the forward's NaN was born in. `fwd_at`
        // is the first non-finite of the forward value (-1: finite); a sized slot's non-zero
        // count, non-finite count and largest magnitude say whether a cotangent died upstream of
        // the row (all zero) or in the row's own arithmetic (non-zero here, zero in the
        // gradient). Unsized slots (frozen, plain writes) print their forward values only. Read
        // it the way R97_DUMP_SNAP's operand dump is.
        if (const char* d = std::getenv("R97_DUMP_DC");
            d != nullptr && d[0] == '1' &&
            (rec.op == "rope" || rec.op == "lora_expand" || rec.op == "lora_shrink" ||
             rec.op == "rmsnorm" ||
             // The op the 512 case's remaining non-finite is born in, at position 1176 in block
             // 63: not an exact match for "rmsnorm", and outside the last-block window below, so
             // without this line the dump is silent about the one row it is being run for.
             rec.op == "rmsnorm_had_quant_i8g" ||
             // The block boundary, where the per-block factor of ~1.6e3 is introduced rather than
             // inside any one row: the delta-net chain's own rows and the residual add, so a block
             // boundary's contribution shows up in the same table as the rows either side of it.
             rec.op == "add" || rec.op == "gdn_chunk_scan" || rec.op == "gdn_kkt_solve" ||
             rec.op == "gdn_conv_prep" || rec.op == "gdn_gated_rmsnorm" ||
             // The inf hunt: every record of the last blocks, so the earliest forward value that is
             // not finite is named by its op and position rather than guessed. Widened from
             // 1390..1458 (blocks 61..62) to 1300..1458 so the jump between adjacent blocks is
             // inside the window instead of straddling its edge.
             (rec.position >= 1300 && rec.position <= 1458))) {
            for (size_t j = 0; j < rec.operands.size(); ++j) {
                int64_t nonzero = 0, nonfinite = 0;
                double maxabs = 0.0;
                if (sized[j])
                    for (double v : dc[j].value) {
                        if (!std::isfinite(v)) {
                            ++nonfinite;
                            continue;
                        }
                        if (v != 0.0) ++nonzero;
                        maxabs = std::max(maxabs, std::fabs(v));
                    }
                std::fprintf(stderr,
                             "DUMPDC side=%s op=%s pos=%s buffer=%s shape=%s sized=%d nonzero=%lld "
                             "nonfinite=%lld maxabs=%.6g fwd_at=%lld\n",
                             options.side == Side::host ? "host" : "device", rec.op.c_str(),
                             std::to_string(rec.position).c_str(),
                             rec.operands[j].buffer.c_str(), shape_text(dc[j].shape).c_str(),
                             static_cast<int>(sized[j]), static_cast<long long>(nonzero),
                             static_cast<long long>(nonfinite), maxabs,
                             static_cast<long long>(fwd_non_finite[j]));
            }
            std::fflush(stderr);
        }

        // R97_FD_REC=<pos>[,<pos>...]: a live-scale finite difference of the row that just ran, on
        // the tape's own operands and cotangents. Every row check in the suite runs at O(1)
        // synthetic values (`tests/support/adjoint_check.h` seeds with `urand`), and the quantisers'
        // clamp masks are scale-dependent, so a row can pass there and be wrong where the walk
        // actually is. This compares `dc[i][e]` against the twin's own derivative of
        // `<twin, cot>` at that element, at the tape's magnitude, for the largest few elements of
        // each read. Host arm only: the device rows round their cotangents to bf16, so there the
        // comparison would be measuring the rounding.
        if (const char* fr = std::getenv("R97_FD_REC"); fr != nullptr && options.side == Side::host) {
            const std::string list(fr);
            const std::string pos = std::to_string(rec.position);
            bool want = false;
            for (size_t at = 0; at <= list.size();) {
                const size_t comma = list.find(',', at);
                const std::string piece =
                    list.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
                if (piece == pos) want = true;
                if (comma == std::string::npos) break;
                at = comma + 1;
            }
            if (want) {
                std::string terr;
                TwinFn twin = twin_find(rec.op, rec.params, &terr);
                if (twin == nullptr) {
                    std::fprintf(stderr, "FDREC pos=%s op=%s no twin: %s\n", pos.c_str(),
                                 rec.op.c_str(), terr.c_str());
                } else {
                    std::vector<size_t> read_at;
                    std::vector<AdValue> reads;
                    for (size_t i = 0; i < rec.operands.size(); ++i) {
                        const bool inout = rec.operands[i].written && sized[i];
                        if (rec.operands[i].written && !inout) continue;
                        read_at.push_back(i);
                        reads.push_back(fwd[i]);
                    }
                    const auto loss_at = [&](const std::vector<AdValue>& x) {
                        std::vector<AdValue> y;
                        twin(rec, x, &y);
                        double sum = 0.0;
                        for (size_t o = 0; o < y.size() && o < cot.size(); ++o)
                            for (size_t e = 0; e < y[o].value.size() && e < cot[o].value.size(); ++e)
                                sum += y[o].value[e] * cot[o].value[e];
                        return sum;
                    };
                    for (size_t k = 0; k < reads.size(); ++k) {
                        const size_t i = read_at[k];
                        if (!sized[i] || dc[i].value.empty()) continue;
                        // The elements that matter: the largest |dc| and the largest |value| in
                        // this read. A row whose adjoint departs at magnitude shows it on both.
                        size_t by_dc = 0, by_val = 0;
                        for (size_t e = 1; e < dc[i].value.size(); ++e) {
                            if (std::fabs(dc[i].value[e]) > std::fabs(dc[i].value[by_dc])) by_dc = e;
                            if (e < reads[k].value.size() &&
                                std::fabs(reads[k].value[e]) > std::fabs(reads[k].value[by_val]))
                                by_val = e;
                        }
                        for (const size_t e : {by_dc, by_val}) {
                            if (e >= reads[k].value.size() || e >= dc[i].value.size()) continue;
                            const double x0 = reads[k].value[e];
                            // The row's own RMS, which is what sets an RMS norm's scale: the
                            // element's value says nothing about it, and a row at 5e-4 rms and a row
                            // with one element at 5e-4 are different claims. `sc` is the norm's
                            // rsqrt(mean(x^2)+eps) with the op's default eps.
                            double row_ss = 0.0;
                            const int64_t rch = reads[k].shape.empty() ? 0 : reads[k].shape.back();
                            if (rch > 0 && (int64_t)reads[k].value.size() % rch == 0) {
                                const int64_t row = (int64_t)e / rch;
                                for (int64_t t = 0; t < rch; ++t) {
                                    const double v = reads[k].value[(size_t)(row * rch + t)];
                                    row_ss += v * v;
                                }
                                row_ss /= (double)rch;
                            }
                            const double row_sc = row_ss > 0.0 ? 1.0 / std::sqrt(row_ss + 1e-6) : 0.0;
                            const double eps = 1e-6 * std::max(1.0, std::fabs(x0));
                            std::vector<AdValue> xp = reads, xm = reads;
                            xp[k].value[e] += eps;
                            xm[k].value[e] -= eps;
                            const double numeric = (loss_at(xp) - loss_at(xm)) / (2.0 * eps);
                            const double analytic = dc[i].value[e];
                            const double scale = std::max(std::fabs(analytic), std::fabs(numeric));
                            std::fprintf(stderr,
                                         "FDREC pos=%s op=%s read=%zu buffer=%s elem=%zu x=%.6g "
                                         "row_ss=%.6g row_sc=%.6g "
                                         "analytic=%.6g numeric=%.6g ratio=%.6g\n",
                                         pos.c_str(), rec.op.c_str(), i,
                                         rec.operands[i].buffer.c_str(), e, x0, row_ss, row_sc,
                                         analytic, numeric,
                                         scale > 0.0 ? analytic / numeric : 0.0);
                        }
                    }
                }
                std::fflush(stderr);
            }
        }

        // R97_DUMP_REC=1: one line per record, the whole chain in walk order. `cot` is the largest
        // magnitude the record was handed (its writes' incoming cotangents), `dc` the largest it
        // left on its reads; `ratio` is the record's effective Jacobian. A row whose own adjoint is
        // FD-proven still shows a large ratio when the *walk* hands it the wrong cotangent, and the
        // first record whose ratio is out of line with its siblings is where the growth enters.
        // Sums travel beside the maxima because a row that is handed one huge element and returns
        // a dense small answer has the same maximum and a very different chain.
        if (const char* dr = std::getenv("R97_DUMP_REC"); dr != nullptr && dr[0] == '1') {
            double cot_max = 0.0, dc_max = 0.0, cot_sum = 0.0, dc_sum = 0.0;
            int64_t cot_slot = -1, dc_slot = -1;
            for (size_t j = 0; j < rec.operands.size(); ++j) {
                if (rec.operands[j].written && cot_index[j] >= 0)
                    for (double v : cot[static_cast<size_t>(cot_index[j])].value) {
                        const double a = std::fabs(v);
                        if (!std::isfinite(a)) continue;
                        cot_sum += a;
                        if (a > cot_max) {
                            cot_max = a;
                            cot_slot = static_cast<int64_t>(j);
                        }
                    }
                if (!sized[j]) continue;
                for (double v : dc[j].value) {
                    const double a = std::fabs(v);
                    if (!std::isfinite(a)) continue;
                    dc_sum += a;
                    if (a > dc_max) {
                        dc_max = a;
                        dc_slot = static_cast<int64_t>(j);
                    }
                }
            }
            std::fprintf(stderr,
                         "DUMPREC pos=%s op=%s cot_max=%.6g cot_sum=%.6g cot_slot=%lld dc_max=%.6g "
                         "dc_sum=%.6g dc_slot=%lld ratio=%.6g sum_ratio=%.6g\n",
                         std::to_string(rec.position).c_str(), adjoint_rec->op.c_str(), cot_max,
                         cot_sum, (long long)cot_slot, dc_max, dc_sum, (long long)dc_slot,
                         cot_max > 0.0 ? dc_max / cot_max : 0.0,
                         cot_sum > 0.0 ? dc_sum / cot_sum : 0.0);
            std::fflush(stderr);
        }

        // The writers' adjoint has run: their cotangent storage is done. Nothing below reads a
        // written version's cotangent again.
        for (size_t j = 0; j < rec.operands.size(); ++j)
            if (rec.operands[j].written && plan.write_version[i][j] >= 0)
                release(version_at(i, j));

        // Hand what the adjoint produced to the versions each read saw, or out of the block. An
        // in-place write's cotangent is also what the value it replaced takes: its read segments
        // name that prior version, so it travels the same way a read's does.
        //
        // One worker per buffer: a version's storage belongs to one buffer, and an in-place op
        // gives two operands the same buffer, so the buffer -- not the operand -- is the unit that
        // owns every write to one store. The operands inside a unit keep the record's order, which
        // is the serial loop's order, so every output element still sums its terms in the order it
        // did and the bits do not move.
        const auto distribute = [&](size_t j) {
            const int64_t bi = plan.by_name.at(rec.operands[j].buffer);
            Buffer& b = plan.buffers[static_cast<size_t>(bi)];
            const std::vector<double> dense =
                dense_view(rec.operands[j], dc[j].value, b.rows, b.cols);
            const std::vector<double>& values = dense.empty() ? dc[j].value : dense;
            if (values.size() != static_cast<size_t>(b.rows * b.cols))
                refuse("op '" + rec.op + "': adjoint left operand '" + rec.operands[j].buffer +
                       "' with " + std::to_string(values.size()) + " values, not " +
                       std::to_string(b.rows * b.cols));
            for (const Segment& seg : plan.read_segments[i][j]) {
                if (seg.version >= 0) {
                    add_version_full(b.versions[static_cast<size_t>(seg.version)], seg.row_lo,
                                     seg.row_hi, values);
                } else {
                    add_input_full(b, seg.row_lo, seg.row_hi, values);
                }
            }
        };
        std::map<int64_t, std::vector<size_t>> by_buffer;
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            const bool in_place = in_place_read(rec, j);
            if (rec.operands[j].written && !in_place) continue;
            // The same rule the sizing uses: a frozen in-place read's slot is the passthrough's
            // to fill, so it must reach the versions its read names.
            if (frozen_at.count(j) && !in_place) continue;
            by_buffer[plan.by_name.at(rec.operands[j].buffer)].push_back(j);
        }
        // Create the storage the units write before a worker runs: creation moves the live-byte
        // count and, for a block input, inserts into the input map, and neither is a worker's to
        // race on. The versions and slots are the ones the serial loop created, in the same record,
        // so the report's peak stays the walk's figure.
        for (const auto& unit : by_buffer) {
            Buffer& b = plan.buffers[static_cast<size_t>(unit.first)];
            if (b.cols <= 0) continue;
            for (size_t j : unit.second)
                for (const Segment& seg : plan.read_segments[i][j]) {
                    if (seg.version >= 0)
                        (void)version_cot(b.versions[static_cast<size_t>(seg.version)]);
                    else
                        (void)input_slot(b);
                }
        }
        std::vector<std::pair<int64_t, std::vector<size_t>>> units(by_buffer.begin(),
                                                                   by_buffer.end());
        run_units(static_cast<int64_t>(units.size()), [&](int64_t u) {
            for (size_t j : units[static_cast<size_t>(u)].second) distribute(j);
        });
    }

    // The map iterates in op-name order, so the report's list comes out sorted.
    for (auto& [name, t] : row_time) {
        t.op = name;
        result.report.ops.push_back(std::move(t));
    }

    for (auto& [name, cot] : inputs) result.inputs.push_back(std::move(cot));
    return result;
}

}  // namespace r97
