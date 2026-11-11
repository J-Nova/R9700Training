// r97/step_host.h -- the host join: one model's adapter gradient as a StepFn.
//
// The forward driver, the layer replay, the backward core and the live adapter meet here. The
// join names neither a loss nor an adapter kind: it takes a Model, asks the delta for the wire
// the live adapter reads, asks the objective for the seeds the backward starts from, and walks
// from the head down through every block, one replay at a time. docs/STEPS-HOST.md owns the
// path and the memory it holds.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "r97/forward.h"
#include "r97/step.h"

namespace r97 {

struct HostStepOptions {
    // Keep the boundary values and cotangents in the report. A training loop may turn this off;
    // the issue's checks need them.
    bool keep_boundaries = true;
};

// What one joined step measured. Boundary 0 is the input of block 0 and boundary `layers` is the
// final hidden state the head reads; each boundary is dense over the step's token rows, in the
// batch's sequence order, so a caller can split it by the batch's lengths.
struct HostStepReport {
    double loss = 0.0;
    int64_t scored = 0;
    int64_t tokens = 0;
    int64_t hidden = 0;
    int64_t layers = 0;
    int64_t residual_store_bytes = 0;    // the boundary residuals (grows with layers)
    int64_t peak_snapshot_bytes = 0;     // the largest op snapshot the walk held
    int64_t peak_cotangent_bytes = 0;    // the largest cotangent set one walk held
    std::vector<std::string> boundary_name;           // the tape buffer of each boundary
    std::vector<std::vector<double>> boundary_value;  // boundary forward values, bf16 widened
    std::vector<std::vector<double>> boundary_cotangent;
};

// The join. The driver must outlive the returned StepFn; the StepFn keeps the Model and, for each
// call, feeds the live adapter from `params`, runs the batch, and fills `grads` with the delta's
// gradients when it is non-null. The report, when given, is overwritten on every call.
StepFn host_step(ForwardDriver& driver, const Model& model,
                 const HostStepOptions& options = {}, HostStepReport* report = nullptr);

}  // namespace r97
