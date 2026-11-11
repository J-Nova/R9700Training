// r97/tape.cpp -- the issue-observer callbacks and the records they build.
//
// The observer comes from a training patch (patches/training/0101-issue-observer.patch); a serving
// build applies none of the training set, so a tape would have nothing to attach to and this file
// compiles to nothing.
#include "r97/tape.h"

#if defined(RAD_ISSUE_OBSERVER)

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace r97 {

// Pack a strided view's values, element by element, for a tape that keeps operand bytes. Returns
// the byte count, or 0 when the view cannot be copied (a dtype with no whole-byte size).
static int64_t copy_view(const RadIssueOperand& o, std::vector<uint8_t>* out) {
    const int64_t es = rad_dtype_bytes(o.dtype, 1);
    if (es <= 0 || rad_dtype_bits(o.dtype) % 8 != 0 || o.data == nullptr || o.view_rank == 0)
        return 0;
    int64_t shape[RAD_MAX_RANK] = {0, 0, 0, 0, 0, 0};
    int64_t n = 1;
    for (uint8_t d = 0; d < o.view_rank; ++d) {
        shape[d] = o.view_shape[d];
        if (d == 0 && o.rows > 0) shape[0] = std::min(o.rows, shape[0]);
        n *= shape[d];
    }
    if (n <= 0) return 0;
    const int64_t bytes = rad_dtype_bytes(o.dtype, n);
    if (bytes <= 0) return 0;
    out->assign(static_cast<size_t>(bytes), 0);
    const auto* base = static_cast<const uint8_t*>(o.data);
    int64_t idx[RAD_MAX_RANK] = {0, 0, 0, 0, 0, 0};
    for (int64_t e = 0; e < n; ++e) {
        int64_t off = o.view_off;
        for (uint8_t d = 0; d < o.view_rank; ++d) off += idx[d] * o.view_stride[d];
        std::memcpy(out->data() + es * e, base + es * off, static_cast<size_t>(es));
        for (uint8_t d = o.view_rank; d-- > 0;) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
    return bytes;
}

namespace {

std::string identity(const RadIssueOperand& o, int index) {
    if (o.name) return o.name;
    switch (o.kind) {
        case RAD_OPK_KV:     return "kv:" + std::to_string(o.handle);
        // A raw or absent operand has no handle that tells it from its neighbours; the position in
        // the schema does.
        case RAD_OPK_RAW:    return "raw:" + std::to_string(index);
        case RAD_OPK_NONE:   return "none:" + std::to_string(index);
        case RAD_OPK_WTAB:   return "wtab";
        default:             return "buffer:" + std::to_string(o.handle);
    }
}

// The row an element offset starts at, for the contract's row range. Radiance counts an offset in
// elements; a buffer's row is everything after dim 0. An offset that is not a whole row has no
// row to name, so it reports row 0 rather than a made-up fraction.
int64_t first_row(const RadIssueOperand& o) {
    if (o.rank == 0 || o.offset <= 0) return 0;
    int64_t row_elems = 1;
    for (uint32_t d = 1; d < o.rank; ++d) row_elems *= o.shape[d];
    if (row_elems <= 0 || o.offset % row_elems != 0) return 0;
    return o.offset / row_elems;
}

}  // namespace

Tape::~Tape() { detach(); }

void Tape::attach(RadCtx* ctx) {
    detach();
    if (!ctx) return;
    RadIssueObserver observer{};
    observer.user = this;
    observer.op = &Tape::observe_op;
    observer.block = &Tape::observe_block;
    rad_set_issue_observer(ctx, &observer);
    ctx_ = ctx;
}

void Tape::detach() {
    if (ctx_) rad_set_issue_observer(ctx_, nullptr);
    ctx_ = nullptr;
}

void Tape::clear() {
    ops_.clear();
    contents_.clear();
    copies_.clear();
    blocks_.clear();
    block_ = -1;
}

const std::vector<OperandContent>& Tape::contents(size_t op) const {
    return contents_.at(op);
}

OperandContent Tape::content(size_t op, size_t operand) const { return contents_.at(op).at(operand); }

void Tape::set_bracket_hook(BracketHook hook) { bracket_hook_ = std::move(hook); }

void Tape::set_issue_hook(IssueHook hook) { issue_hook_ = std::move(hook); }

void Tape::observe_op(void* user, RadCtx* /*ctx*/, const RadIssueOp* issue) {
    if (user && issue) static_cast<Tape*>(user)->record(*issue);
}

void Tape::observe_block(void* user, RadCtx* /*ctx*/, int64_t block, int begin) {
    if (!user) return;
    Tape* tape = static_cast<Tape*>(user);
    if (begin) tape->open_block(block);
    else tape->close_block(block);
}

void Tape::record(const RadIssueOp& issue) {
    OpRecord rec;
    rec.op = issue.name ? issue.name : "?";
    rec.n = issue.n;
    rec.block = block_;
    rec.position = static_cast<int64_t>(ops_.size());

    // OpRecord carries an int or a string per parameter. A resolved geometry is made of those; a
    // range or a double has no slot and is left out rather than flattened into a lie.
    for (int i = 0; i < issue.n_params; ++i) {
        const RadParam& p = issue.params[i];
        if (p.kind == RAD_P_INT) {
            rec.params.emplace_back(p.key, (int64_t)p.ival);
        } else if (p.kind == RAD_P_STR) {
            rec.params.emplace_back(p.key, std::string(p.sval ? p.sval : ""));
        }
    }

    std::vector<OperandContent> content;
    content.reserve(static_cast<size_t>(issue.n_operands));
    rec.operands.reserve(static_cast<size_t>(issue.n_operands));
    for (int i = 0; i < issue.n_operands; ++i) {
        const RadIssueOperand& o = issue.operands[i];
        Operand op;
        op.buffer = identity(o, i);
        op.dtype = rad_dtype_name(o.dtype);
        op.shape.clear();
        if (o.rank > 0) op.shape.assign(o.shape, o.shape + o.rank);
        op.row_lo = first_row(o);
        op.rows = o.rows;
        if (o.view_rank > 0) {
            op.view_shape.assign(o.view_shape, o.view_shape + o.view_rank);
            op.view_stride.assign(o.view_stride, o.view_stride + o.view_rank);
            op.view_off = o.view_off;
        }
        op.written = o.written != 0;
        op.in_place = o.inout != 0;
        rec.operands.push_back(std::move(op));
        if (keep_ && o.data != nullptr) {
            // bytes is 0 where the ABI says the operand is not one span: an op that views its
            // buffer with a shape whose dim 0 it narrows. The declared shape still sizes the rows
            // the issue uses, so a copy of them is the slice the op addressed as long as the
            // view is dense, which is what a copy has to assume. A reader that asked for copies is
            // the one that can tell: its comparison is against these bytes.
            int64_t bytes = o.bytes;
            if (bytes <= 0 && o.view_rank > 0) {
                // A strided view is not one span: the copy is its own values, packed, which the
                // declared shape cannot hold. A reader compares against the view, not the buffer.
                std::vector<uint8_t> view;
                if (copy_view(o, &view) > 0) {
                    const auto* from = view.data();
                    copies_.emplace_back(from, from + view.size());
                    content.push_back(OperandContent{copies_.back().data(), copies_.back().size(),
                                                     o.name != nullptr});
                    continue;
                }
            }
            if (bytes <= 0 && o.rank >= 2 && o.rows > 0) {
                int64_t n = 1;
                for (uint8_t d = 1; d < o.rank; ++d) n *= o.shape[d];
                bytes = rad_dtype_bytes(o.dtype, o.rows * n);
            }
            if (bytes <= 0 && o.rows <= 0 && o.rank >= 1) {
                // A whole-buffer operand -- a carried state pool, addressed by index rather than
                // by row -- is one span of its declared shape, so the copy is all of it.
                int64_t n = 1;
                for (uint8_t d = 0; d < o.rank; ++d) n *= std::max<int64_t>(o.shape[d], 0);
                bytes = rad_dtype_bytes(o.dtype, n);
            }
            if (bytes > 0) {
                const auto* from = static_cast<const uint8_t*>(o.data);
                copies_.emplace_back(from, from + bytes);
                content.push_back(OperandContent{copies_.back().data(),
                                                 static_cast<size_t>(bytes), o.name != nullptr});
                continue;
            }
        }
        content.push_back(OperandContent{o.data, static_cast<size_t>(o.bytes), o.name != nullptr});
    }

    ops_.push_back(std::move(rec));
    contents_.push_back(std::move(content));
    if (issue_hook_) issue_hook_(*this, ops_.back(), contents_.back());
}

void Tape::open_block(int64_t block) {
    block_ = block;
    blocks_.push_back(Block{block, ops_.size(), ops_.size()});
    if (bracket_hook_) bracket_hook_(*this, block, true);
}

void Tape::close_block(int64_t block) {
    if (!blocks_.empty() && blocks_.back().index == block) blocks_.back().end = ops_.size();
    if (block_ == block) block_ = -1;
    if (bracket_hook_) bracket_hook_(*this, block, false);
}

}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER
