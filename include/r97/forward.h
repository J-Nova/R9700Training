// r97/forward.h -- the training forward: a Batch in, the loss of its objective out.
//
// The driver replaces radiance's server for training. Every sequence of the batch runs as one
// prefill step from position 0 through the container's own architecture on the host kernel tier,
// the sequences of the batch together in that one step, with speculation off, the vision tower off
// and a bf16 key-value pool -- so attention reads only the keys the step wrote and the gated delta
// net starts from zero. Refusing another pool dtype is deliberate: the backward differentiates
// through the pool, and only bf16 has an adjoint here.
//
// What the step gives the backward, and what this header is for:
//
//   * the boundary "logits", in the batch's own row space: L-1 rows per sequence, the rows that
//     predict a token, read a window of positions at a time (ForwardTape::read);
//   * every objective Result's seeds, summed over the windows, and check_seeds run on each one
//     before anything walks them;
//   * the head's input at each scored row -- the final hidden state before the head;
//   * a hook at every block bracket, called while the residual still holds the boundary's value
//     (the architecture writes that stream in place, so it cannot be read after the step);
//   * the step's graph, recorded on a Tape from the architecture's own issues.
//
// The driver holds no loss: it calls the Model's Objective once per window of the boundary and
// sums `loss` and `scored`. It holds no adapter kind: the Model's Delta is what the backward
// trains, and the forward never looks at it.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "r97/batch.h"
#include "r97/model.h"
#include "r97/objective.h"
#include "r97/step.h"
#include "r97/tape.h"

// The driver runs radiance's engine, its prefill mode (patches/training/0103) and the issue
// observer (patches/training/0101); it needs the rank's context on the engine (0101, 0119) to
// record a step. A serving build applies none of the training set, so there is no driver there
// rather than one that cannot record a step -- the same rule tape.h follows.
#if defined(RAD_ISSUE_OBSERVER) && __has_include("train_prefill.h")
#define R97_FORWARD 1

namespace rad {
class Engine;
}  // namespace rad

namespace r97 {

// Where the forward gets the container and how it cuts a step. The defaults are the fixture's
// engine: one step holds up to 512 tokens over at most 8 sequences, and a window is 256 positions.
struct ForwardConfig {
    std::string model;       // the .rad container this run trains
    std::string radiance_home;  // the plugin home radiance loads the architecture from
    std::vector<std::string> kernels = {"libavx", "libref"};  // the host tier, first found wins
    int64_t max_tok = 512;   // tokens one step may hold; a batch over it is refused
    int64_t max_ctx = 512;   // positions the key-value pool covers
    int32_t max_seqs = 8;    // sequences one step may hold
    int64_t window_rows = 256;  // positions of the "logits" boundary one Objective call reads
    std::string kv_dtype = "bf16";  // the key-value pool's dtype: only bf16, nothing else is accepted
};

// One block boundary as it stood during the step: the residual at the block's entry or exit. Its
// contents are live for the length of the hook call only, because the next block writes over them;
// a caller that needs them past that copies them.
struct BlockView {
    int64_t block = -1;   // the index the architecture gave the bracket
    bool begin = true;    // true at the block's entry, false at its exit
    const void* residual = nullptr;  // `rows` x `cols` elements of `dtype`, the step's own buffer
    int64_t rows = 0;     // the step's token rows, which is what a boundary covers
    int64_t cols = 0;
    std::string dtype;    // "bf16", "f32", ...
};

// One batch's step, after it has run: the forward the objective read and what the backward needs
// off the same step. Built by ForwardDriver and owned by it.
class ForwardStep final : public RecordedTape {
public:
    ForwardStep(const ForwardStep&) = delete;
    ForwardStep& operator=(const ForwardStep&) = delete;

    // ---- ForwardTape: the one boundary the plain forward has, "logits"

    const Batch& batch() const override { return *batch_; }
    int64_t total_scored() const override { return total_scored_; }
    std::vector<std::string> boundaries() const override { return {"logits"}; }
    std::vector<int64_t> shape(const std::string& buffer) const override;
    void window(const std::string& buffer, int64_t* row_lo, int64_t* row_hi) const override;
    std::vector<float> read(const std::string& buffer, int64_t row_lo, int64_t row_hi,
                            bool differentiated) const override;
    const std::vector<TapeRead>& reads() const override { return reads_; }

    // ---- what the step leaves behind

    // The seeds of every window's Result, in window order: where the loss enters the tape.
    const std::vector<Seed>& seeds() const { return seeds_; }

    // The loss and the scored count over the whole step, as the windows summed them.
    const StepResult& result() const { return result_; }

    // How many windows the Objective was called with. Chunked by position, so a batch longer than
    // ForwardConfig::window_rows rows is read over more than one call.
    int64_t windows() const { return windows_; }

    // The final hidden state before the head, at the scored rows and in the container's element
    // type widened to f32: `hidden_cols()` values a row, the rows in the batch's order. Row i of
    // `hidden()` is the head's input row `hidden_rows()[i]`, which is the row a walk attaches a
    // cotangent to; it is not the row's index in the batch, because the head's rows are the
    // step's, one a position, and the batch's are the ones it predicts from, one a token but
    // the last.
    const std::vector<float>& hidden() const { return hidden_; }
    const std::vector<int64_t>& hidden_rows() const { return hidden_rows_; }
    int64_t hidden_cols() const { return hidden_cols_; }

    // The step's graph. Its operand contents point into the engine's buffers, which the driver
    // keeps alive for as long as it holds this step.
    const Tape& tape() const { return *tape_; }

private:
    friend class ForwardDriver;

    ForwardStep() = default;

    // The logits boundary, read through to the engine's per-prompt rows: the step keeps no copy.
    // Two row spaces meet here: `off` is the batch's, where a sequence takes L-1 rows because the
    // last position predicts nothing, and `pos` is the step's, where it takes all L positions.
    // The head's rows and its input are in the step's; what an Objective reads is the batch's.
    struct Piece {
        int64_t off;       // the batch row this sequence's first logits row takes
        int64_t pos;       // the step row its first input position takes
        int64_t rows;      // its L-1 rows
        int64_t prompt;    // the index of its vector in the engine's prefill_logits
    };

    const Batch* batch_ = nullptr;
    const Tape* tape_ = nullptr;
    const std::vector<std::vector<float>>* logits_ = nullptr;
    std::vector<Piece> pieces_;
    int64_t vocab_ = 0;
    int64_t rows_ = 0;
    int64_t total_scored_ = 0;

    // The window the driver set for the Objective call in flight; unset means the whole boundary.
    mutable int64_t window_lo_ = 0;
    mutable int64_t window_hi_ = 0;
    mutable bool window_set_ = false;
    mutable std::vector<TapeRead> reads_;

    std::vector<Seed> seeds_;
    StepResult result_{0.0, 0};
    int64_t windows_ = 0;
    std::vector<float> hidden_;
    std::vector<int64_t> hidden_rows_;
    int64_t hidden_cols_ = 0;
};

// The driver: one engine run per batch, one ForwardStep kept for the backward.
//
// One step is in flight at a time. step() runs the batch and replaces last(); the StepFn it returns
// keeps this driver, so the engine that holds the recorded step's buffers lives until the next
// step or the driver's end. Radiance keeps one plugin registry for the process and refuses a
// second live engine, so the step before is released before the next one starts: last()'s rows are
// gone once step() is called again, and only one driver can be open at a time.
class ForwardDriver {
public:
    // Check what can be checked before a step: the container opens and the budgets make room for
    // one. The engine itself is started per step, so what it refuses is refused inside step(). On
    // failure this returns nullptr and, if `err` is given, says why.
    static std::unique_ptr<ForwardDriver> open(const ForwardConfig& cfg, std::string* err = nullptr);

    ~ForwardDriver();

    ForwardDriver(const ForwardDriver&) = delete;
    ForwardDriver& operator=(const ForwardDriver&) = delete;

    // A StepFn that runs one forward step of the batch through this driver and returns the
    // Objective's loss. `grads` is ignored: this is the forward half, and the join that walks the
    // tape is the one that fills it. A step that cannot run throws std::runtime_error.
    StepFn step(const Model& model);

    // The step the last StepFn call ran. Throws std::logic_error before the first one.
    const ForwardStep& last() const;

    // Called at every block bracket of every step. An empty hook (or none) does nothing. The
    // training patches bracket both qwen35 builds, including the bf16 one the fixture uses.
    void set_block_hook(std::function<void(const BlockView&)> hook);

    // The LoRA slots the next steps' engine must offer, and the adapter name the step's
    // sequences acquire. The join sets these from the sites Delta::to_wire yields; the driver
    // itself never looks at a Delta.
    void set_lora(int64_t max_loras, int64_t max_rank, std::vector<std::string> targets,
                  std::string adapter_name = {});

    // Called after the engine starts and before it serves, so a caller can load the live adapter
    // into a slot the engine has already built. The hook runs inside step().
    void set_engine_hook(std::function<void(rad::Engine&)> hook);

    // The engine and the tape the last step ran on. Null before the first step.
    rad::Engine* engine() { return engine_.get(); }
    Tape& tape() { return tape_; }

    const ForwardConfig& config() const { return cfg_; }

private:
    explicit ForwardDriver(const ForwardConfig& cfg) : cfg_(cfg) {}

    StepResult run(const Batch& batch, const ParamSet& params, const Objective& objective);
    void run_engine(const Batch& batch);
    void assemble(ForwardStep& step, const Batch& batch);
    void on_bracket(const Tape& tape, int64_t block, bool begin);
    BlockView residual_at(const Tape& tape, int64_t block, bool begin);
    void head_hidden(ForwardStep& step, const Batch& batch);

    ForwardConfig cfg_;
    // The engine must outlive the tape's pointers into its buffers, so it is declared before them.
    std::unique_ptr<rad::Engine> engine_;
    Tape tape_;
    std::unique_ptr<ForwardStep> last_;
    std::function<void(const BlockView&)> block_hook_;
    std::function<void(rad::Engine&)> engine_hook_;
    int64_t lora_max_loras_ = 0;
    int64_t lora_max_rank_ = 0;
    std::vector<std::string> lora_targets_;
    std::string lora_adapter_name_;
    int64_t step_tokens_ = 0;       // the token rows the step's boundaries cover
    bool residual_seen_ = false;    // a bracket found the residual at least once
    std::string residual_name_;     // the buffer every bracket must answer with
    std::string bracket_error_;     // why a bracket found nothing, reported after the step
};

}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER && train_prefill.h
