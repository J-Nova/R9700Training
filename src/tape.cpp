// r97/tape.cpp -- the issue-observer callbacks and the records they build.
//
// The observer comes from a training patch (patches/training/0101-issue-observer.patch); a serving
// build applies none of the training set, so a tape would have nothing to attach to and this file
// compiles to nothing.
#include "r97/tape.h"

#if defined(RAD_ISSUE_OBSERVER)

#include <cstddef>
#include <string>
#include <vector>

namespace r97 {
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
        op.written = o.written != 0;
        op.in_place = o.inout != 0;
        rec.operands.push_back(std::move(op));
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
