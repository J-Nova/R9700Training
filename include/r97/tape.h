// r97/tape.h -- one step's forward graph, recorded from radiance's issue observer.
//
// The tape copies what the runtime reports for every rad_issue: the op's name, its resolved
// parameters, its operands and the block the architecture was in. It keeps a pointer to each
// operand's data, so a reader can look at what the op saw without a copy; the pointer is valid for
// as long as the buffer lives, which for the training engine is the step.
//
// Block ranges are recorded from the begin/end calls the architecture makes around a layer. Ops
// outside any block have block -1 and appear in ops() but in no range. An architecture writes its
// residual stream in place, so the value a bracket names is gone once the next block has run: a
// caller that must copy a boundary while the step is in flight subscribes with set_bracket_hook.
//
// The observer is a training patch on radiance (patches/training/0101-issue-observer.patch), and a
// serving build applies none of the training set. A runtime without the observer has no graph to
// read, so there is no tape there at all rather than one that records nothing.
#pragma once

#include "r97/op.h"
#include "rad_runtime.h"

#if defined(RAD_ISSUE_OBSERVER)

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace r97 {

// One operand's content at the issue: the slice the kernel saw, not the whole declared buffer.
// `bytes` is 0 when the slice is not one contiguous span. `named` says the operand named a
// declared buffer or weight; the tape's identities for KV, raw and table operands are synthetic, so
// a reader that must find one declared buffer by name (the replay's residual) ignores those.
struct OperandContent {
    const void* data = nullptr;
    size_t bytes = 0;
    bool named = false;
};

class Tape {
public:
    // A half-open range [begin, end) of ops() that one architecture block issued.
    struct Block {
        int64_t index = -1;  // the index the architecture named the block by
        size_t begin = 0;
        size_t end = 0;
    };

    Tape() = default;
    ~Tape();

    Tape(const Tape&) = delete;
    Tape& operator=(const Tape&) = delete;

    // Record `ctx`'s issues into this tape, replacing any observer already there. Attaching twice
    // detaches from the first context. The context must outlive the tape's attachment.
    void attach(RadCtx* ctx);
    void detach();
    bool attached() const { return ctx_ != nullptr; }

    // Drop the recorded step. The observer stays attached, and so does the bracket hook.
    void clear();

    // Copy each operand's bytes at the issue instead of keeping its address. A reader that runs
    // after the step has to turn this on: a later block reuses the same activation buffers, so by
    // the end of the step an address holds the last writer's values rather than the ones the op
    // saw. Off by default, because on it keeps the step's whole operand traffic. The observer runs
    // after the op does, so a written operand's copy is what the op left.
    void keep_contents(bool on) { keep_ = on; }
    bool keeps_contents() const { return keep_; }

    // A block bracket reached while the step runs, called after the bracket is recorded and while
    // the buffers of the step still hold what it names. `begin` is true at the block's entry, when
    // the ops before it are recorded, and false at its exit, when the block's own are.
    using BracketHook = std::function<void(const Tape& tape, int64_t block, bool begin)>;
    void set_bracket_hook(BracketHook hook);

    // Called after an op is recorded and while the step's buffers still hold what it read and
    // wrote, so a reader that must copy an op's output does it here, before the planner reuses the
    // buffer. `operands` is contents(op.position). One hook; a later set replaces it.
    using IssueHook = std::function<void(const Tape& tape, const OpRecord& op,
                                         const std::vector<OperandContent>& operands)>;
    void set_issue_hook(IssueHook hook);

    const std::vector<OpRecord>& ops() const { return ops_; }
    const std::vector<Block>& blocks() const { return blocks_; }

    // The content of every operand of op `op`, in schema order. Out-of-range indices throw
    // std::out_of_range.
    const std::vector<OperandContent>& contents(size_t op) const;

    // The content of one operand. Out-of-range indices throw std::out_of_range.
    OperandContent content(size_t op, size_t operand) const;

private:
    static void observe_op(void* user, RadCtx* ctx, const RadIssueOp* issue);
    static void observe_block(void* user, RadCtx* ctx, int64_t block, int begin);

    void record(const RadIssueOp& issue);
    void open_block(int64_t block);
    void close_block(int64_t block);

    RadCtx* ctx_ = nullptr;
    BracketHook bracket_hook_;
    IssueHook issue_hook_;
    int64_t block_ = -1;
    bool keep_ = false;
    std::vector<OpRecord> ops_;
    std::vector<std::vector<OperandContent>> contents_;
    // One heap block per operand copy; `contents_` points into them, so growth of this vector
    // moves the vector heads and not the bytes.
    std::vector<std::vector<uint8_t>> copies_;
    std::vector<Block> blocks_;
};

}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER
