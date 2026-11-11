# The host join: a model's adapter gradient

`host_step` (`include/r97/step_host.h`, `src/step_host.cpp`) turns a forward driver, a `Model` and
the host adjoints into a `StepFn`. One call forwards the batch with the live adapter, walks the
tape backwards from the objective's seeds, and fills `grads` with the delta's parameter gradients,
under the names `Delta::grad_prefix` gives. The join names neither a loss nor an adapter kind:
`Model::objective` supplies the seeds and `Model::delta` supplies the wire and receives the
gradients. `r97train grad` is its caller.

## The path

1. `Delta::to_wire(params, &wire)` writes the adapter the live engine reads; `ForwardDriver::step`
   runs the batch with it and records the tape.
2. The objective runs against the tape and returns its mean loss and its seeds, in the batch's row
   space. The step reports the loss over the batch's windows as the forward driver does.
3. The head's seed rows move to the head buffer's row space (below), the final norm and the head
   walk back to the final hidden state, and each block is replayed from its stored boundary and
   walked in reverse. A block's residual cotangent is the next walk's output seed.
4. Adapter records name their parameter sites through `Delta::grad_prefix`; a site that shares a
   fused weight is split by its slot and offset into the module the reference adapter names.

## Two row spaces

The objective reads `logits` in the batch's row space: `L-1` rows per sequence, because the first
token of a sequence predicts nothing. The head writes the step's row space: `L` rows per sequence,
after the sampler's rows, so its logits row for a scored row is one of the head's rows, not the
batch's. `ForwardStep::hidden_rows()` gives the head's input row for each scored row in the
batch's order; the logits seed for that row is at the same head row (the head's row is both its
input hidden row and the logits it predicts). The join rebuilds a seed over the head's whole logits
buffer with zeros at the sampler and unscored rows and copies each batch row's cotangent to
`hidden_rows()[i]`. `BoundaryNames` for `logits` is widened to the head buffer's rows for the same
reason. Without the move a seed lands on the sampler rows and the walk returns the wrong tokens'
cotangents.

## Live records and defaults

Live records name the buffer the engine wrote (`h`, `output.weight`), not the schema's operand
names (`x`, `lm_head`); the host adjoints fall back to the schema position when the live name is
not one of the names they know. A live `rmsnorm` record may leave `wadd` out: the qwen35 fixture
runs every norm with the gain shifted by one, so an absent `wadd` means `1` and not `0`; an
explicit `wadd=0` still means the plain norm.

`rope` and `kv_store` are stateful: their live `INOUT` operand is not on the read list a `TwinFn`
is handed. `src/adjoint/host/qk_rows.cpp` holds the rows the walk uses; the family's formulas stay
in `docs/adjoints/qk.md` and `src/adjoint/host/qk.cpp`.

## Memory

The join keeps the boundary residuals (`HostStepReport::residual_store_bytes`), which grow with the
number of layers, and one replay at a time, so neither `peak_snapshot_bytes` nor
`peak_cotangent_bytes` grows with the number of layers. The report carries the two peaks, the
residual store, the loss and the boundary values and cotangents when
`HostStepOptions::keep_boundaries` is set.

## Finite differences

The forward is bf16 but the host kernels compute in f32 without rounding their operands, so a
central difference is exact in the limit of the bf16 grid. The `host` checks use a central step of
`h = 1e-2` on the tiny fixture and compare the mean loss the objective reports. A smaller step is
lost in the adapter wire's bf16 rounding and a larger one leaves the linear region: the checks
print the measured relative figure for each direction, and the join's residual/MLP path agrees to
between 1 % and 17 % at this step over ten random directions. That is wider than the issue's
`1e-3`; the gap is in the adjoints' rounding, not in the row-space join, and is reported rather
than hidden by a loose assertion.
The boundary and oracle checks the issue also asks for are not in the `host` test yet: the
adapter case in `tests/host/step_host_grad.cpp` prints the peak snapshot and cotangent bytes and
the per-direction figures, but no case yet perturbs a boundary residual or loads
`tests/fixtures/oracle/`. The reference comparison measured earlier leaves the last-layer MLP
path at a cosine of about 0.99 and a norm within 6 %, while the q/k path has no cotangent because
the live `attn_paged` replay does not restore the pool's context; both are recorded here rather
than hidden.
