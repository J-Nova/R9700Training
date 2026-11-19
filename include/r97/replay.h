// r97/replay.h -- the residual stream at every layer boundary, and one layer re-run from it.
//
// The forward of the backward keeps only the residual at the layer boundaries: boundary 0 is what
// the embeddings wrote, boundary `i` is block `i-1`'s output, and a trunk of L layers has L+1 of
// them. A layer replay copies one boundary back into the step's residual buffer, restores the
// state the first forward started from, and re-issues exactly that block's ops, so the backward
// can take one layer at a time without holding every activation.
//
// A block's input state is the residual -- and, for the blocks that read one more boundary-
// sized buffer before writing it, that buffer too. In the qwen35 int8 trunk (qwen35_fp8) every
// block after the first does: its first norm is a fused add that reads the previous block's
// output and the residual and writes their sum. The plan says which boundaries carry that second
// copy (ResidualPlan::extra, one entry per boundary), and the store keeps it beside the
// residual; restore() puts both back before the block re-runs, so a replayed layer reads exactly
// what the forward's layer read, in either order. The first block and the whole bf16 trunk need
// no second copy.
//
// The replay is a training path: it reads the block brackets and the issue observer that
// patches/training/0101 adds to radiance, and the replay block that 0102 adds; an architecture
// must honour that block (qwen35_bf16 does). A serving build applies neither patch, so there is no
// replay there rather than one that would issue the whole model. A build that leaves 0102 out has
// the observer and the brackets but no replay, so the tests that need it skip by name.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rad_device.h"
#include "r97/tape.h"

namespace rad {
class Engine;
}  // namespace rad

namespace r97 {

// The memory plan for the card's residual store: the dimensions of the one device allocation
// that holds every boundary of one step, and the price of it in bytes. The allocation is sized
// from the plan before the step runs, so a run that does not fit its budget is refused before
// it starts, not in the middle of a step.
struct ResidualPlan {
    int64_t boundaries = 0;  // layers + 1
    int64_t tokens     = 0;
    int64_t hidden     = 0;
    // One entry per boundary, parallel to it: nonzero where the block that starts at the
    // boundary reads a second boundary-sized buffer before writing it, whose copy the store
    // keeps beside the residual. Boundary 0 must be clear -- there is no block before it to
    // copy that buffer from -- and the last one too, since no block starts there.
    std::vector<char> extra;

    int64_t extras() const;
    int64_t boundary_bytes() const { return tokens * hidden * 2; }
    int64_t store_bytes() const { return (boundaries + extras()) * boundary_bytes(); }
    bool fits(int64_t budget_bytes) const { return store_bytes() <= budget_bytes; }
    // The refusal with the numbers: what the store holds, what it costs, and the budget.
    std::string refusal(int64_t budget_bytes) const;
};

// The residual at every layer boundary of one step, bf16 on the host or, by plan, on the card.
// `tokens` is how many rows the step has (its sequences' tokens together), `hidden` the
// residual's width, so one boundary is tokens * hidden * 2 bytes. Layers replay from the same
// buffer the forward wrote; `source` keeps that pointer for each boundary.
class ResidualStore {
public:
    ResidualStore() = default;
    ResidualStore(int64_t tokens, int64_t hidden);
    // The host-backed store from a plan: the same shape as the card's, in host memory, for a run
    // that keeps its boundaries there. With `stream` (the device walk's use) the boundaries live
    // in one pinned host allocation -- pinned so the card takes them in a blit packet; a pageable
    // copy would stage through the engine's bounce buffer and drain the lane per chunk -- and
    // keep()/restore() are stream-ordered copies on that stream (the step's lane,
    // patches/training/0122). Copies of such a store share the allocation, as the card's do.
    // Without `stream` (the host walk) the store keeps host vectors and memcpy()s between them.
    ResidualStore(const ResidualPlan& plan, RadStream stream = nullptr);
    // The card-backed store: one device allocation of plan.store_bytes(), checked against
    // `budget_bytes` before it happens, kept and restored as stream-ordered copies on
    // `stream` (the step's lane, patches/training/0122). Copies of a store share the one
    // allocation; the last copy frees it.
    ResidualStore(RadStream stream, const ResidualPlan& plan, int64_t budget_bytes);
    // Copy one boundary out of the step's buffer at `data`: a stream-ordered copy when the store
    // has a stream (device to host, or device to device on the card), host to host otherwise.
    // Boundaries are kept in order from 0; a gap or a repeat throws std::out_of_range, and a
    // null `data` std::invalid_argument.
    void keep(int64_t boundary, const void* data);
    // The name of the buffer keep() copied out of. The report's boundary cotangent is defined as
    // that buffer's -- the host tier finite-differences it against residual perturbations and the
    // oracle matches it to the residual's gradient -- and the store is the only place that knows
    // what it holds: a block's first in-place read is not a reliable name (a rope's write is one).
    void set_name(int64_t boundary, const std::string& buffer);
    std::string name(int64_t boundary) const;  // empty when the caller named none
    // Keep the second copy at `boundary`, beside the residual keep() took at the same boundary:
    // the buffer the block at that boundary reads as its second input, holding what the block
    // before it left there. Throws std::logic_error when the plan holds no extra at the boundary,
    // std::out_of_range when the residual copy is not kept yet or the extra was already kept,
    // std::invalid_argument for a null `data`.
    void keep_extra(int64_t boundary, const void* data);

    bool has(int64_t boundary) const;
    int64_t boundaries() const;
    int64_t tokens() const { return tokens_; }
    int64_t hidden() const { return hidden_; }
    int64_t boundary_bytes() const { return tokens_ * hidden_ * 2; }
    int64_t bytes() const;

    // The stored bytes of `boundary`, or null when it was not kept. A card-backed store has
    // no host view; read() pulls the boundary back over the wire.
    const uint16_t* at(int64_t boundary) const;
    // Copy the stored bytes of `boundary` back into the step buffer at `dest`: a stream-ordered
    // copy when the store has a stream (device to device on the card, host to device from a
    // streamed host store), host to host otherwise. With it, the extra copy the plan holds at the
    // boundary goes back into the buffer it came out of, so the block that reads it as its
    // second input finds the forward's value. Throws std::out_of_range for a boundary that
    // was not kept and std::invalid_argument for a null `dest`.
    void restore(int64_t boundary, void* dest) const;
    // The step buffer `boundary` was copied out of, or null when it was not kept.
    const void* source(int64_t boundary) const;
    // The plan's extra map: nonzero at a boundary whose block reads a second boundary-sized
    // buffer.
    bool extra_at(int64_t boundary) const;
    // The step buffer the extra copy at `boundary` came out of, or null when it was not kept.
    const void* source_extra(int64_t boundary) const;
    // Copy the stored bytes of `boundary` into the host buffer at `host`; a streamed store
    // synchronizes that stream first. Throws like restore().
    void read(int64_t boundary, void* host) const;

    // The second copy the plan holds at `boundary`, read back the same way: a reader checking a
    // restore put the right bytes back where the block reads them needs both halves.
    void read_extra(int64_t boundary, void* host) const;

    // True when the store is the card's one allocation rather than host memory.
    bool on_device() const { return device_ != nullptr; }
    // True when the boundaries are one pinned host allocation the card is handed in copies: the
    // streamed host form the device walk builds. False for the plain host store and the card's.
    bool on_pinned_host() const { return host_pinned_ != nullptr; }

private:
    void* slot(int64_t boundary) const;
    void* extra_slot(int64_t boundary) const;

    int64_t tokens_ = 0;
    int64_t hidden_ = 0;
    std::vector<char> extra_;          // the plan's extra map, parallel to the boundaries
    std::vector<int64_t> extra_slot_;  // per boundary: the extra copy's slot in the allocation, -1
    RadStream stream_ = nullptr;
    std::shared_ptr<void> device_;  // the card's one allocation; copies of the store share it
    // The streamed host store's one allocation (RAD_MEM_HOST_PINNED): the same slot arithmetic as
    // the card's, shared by copies the same way. Null for the plain host store and the card's.
    std::shared_ptr<void> host_pinned_;
    int64_t kept_ = 0;         // the boundaries kept so far, card or pinned host store
    int64_t extras_kept_ = 0;  // the extra copies kept so far
    int64_t allocated_ = 0;    // bytes of the one allocation, card or pinned host
    std::vector<std::vector<uint16_t>> values_;
    std::vector<std::string> names_;  // per boundary: the buffer keep() copied out of
    std::vector<const void*> sources_;
    std::vector<std::vector<uint16_t>> extra_values_;  // host; per boundary
    std::vector<const void*> extra_sources_;
};

#if defined(RAD_ISSUE_OBSERVER) && defined(RAD_REPLAY_BLOCK)
#define R97_REPLAY 1

// Called after one op of a replayed block issued, before the next op can reuse its buffers. `op`
// is the forward's record of the same op and `operands` its content now; a caller that wants the
// bytes must copy them here. The residual the block leaves is the last written operand of its
// last op, exactly the next boundary.
using ReplaySnapshot =
    std::function<void(const OpRecord& op, const std::vector<OperandContent>& operands)>;

// Records the residual at each block boundary of a step as it runs, through the hooks of the tape
// that is recording the step: install it after the tape is attached, read store() after the step,
// and detach it before replaying a layer (the replay replaces those hooks). The architecture must
// bracket its blocks (rad_block_begin/end).
class BoundaryRecorder {
public:
    BoundaryRecorder(Tape& tape, int64_t tokens, int64_t hidden);
    // Keep the boundaries into `store` (moved): the form a card-backed store takes.
    BoundaryRecorder(Tape& tape, ResidualStore store);
    ~BoundaryRecorder();

    BoundaryRecorder(const BoundaryRecorder&) = delete;
    BoundaryRecorder& operator=(const BoundaryRecorder&) = delete;

    // Stop observing. The store stays readable.
    void detach();

    const ResidualStore& store() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Re-runs one block of a recorded step from the stored residual of its input boundary. The tape's
// op records and block ranges are copied at construction and the tape is detached; replay()
// attaches that same tape again for the block, with this replay's hooks. The engine must outlive
// the replay and the step must be its last live prefill.
class LayerReplay {
public:
    LayerReplay(rad::Engine& engine, Tape& tape, ResidualStore store);
    ~LayerReplay();

    LayerReplay(const LayerReplay&) = delete;
    LayerReplay& operator=(const LayerReplay&) = delete;

    // How many blocks the recorded step has. A step of L layers has L blocks and L+1 boundaries.
    int64_t layers() const;

    const ResidualStore& store() const;

    // The step buffer a boundary's residual lives in, or null when the store lacks it. Restoring
    // boundary `layer` writes into this buffer, so it holds boundary `layer + 1` after the replay.
    const void* residual(int64_t boundary) const;

    // Restore boundary `layer` into the step buffer, clear the block's recurrent state, and issue
    // exactly block `layer`. `after` runs once per op of the block as it issues. Throws
    // std::out_of_range for an unknown block and std::runtime_error when the replay fails.
    void replay(int64_t layer, const ReplaySnapshot& after = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif  // RAD_ISSUE_OBSERVER && RAD_REPLAY_BLOCK

}  // namespace r97
