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

## The view record

An operand whose kernel sees a strided tensor is recorded as the declared buffer plus the row
window, and the tape cannot tell which elements the kernel read: the GDN conv prep takes its gate
and beta as two four-column views of the fused 136-wide input, and the q norm takes the query half
of an interleaved q|gate buffer. `patches/training/0120-view-operands.patch` adds the tensor the
kernel receives to the issue record -- rank, shape, strides, and the element offset from the
declared buffer -- for the strided case only, so a contiguous operand still reads as one span.
`Operand` (`include/r97/op.h`) carries those as `view_rank`, `view_shape`, `view_stride` and
`view_off`, after `written`/`in_place` so that a positional brace-init still reads.

Three places use them. `Tape` copies a view's own values, packed in the view's order, because one
span cannot follow a stride; `snapshot()` (`src/step_host.cpp`) reads a kept copy linearly when it
holds exactly the view's values and walks the strides when it is looking at a live buffer; the
block's snapshot is packed over the view's shape rather than the declared one, and `tighten_operand`
narrows `view_shape[0]` the way it narrows `shape[0]`. `dense_view` (`src/backward.cpp`) scatters a
view's cotangent back into the declared layout, so an adjoint whose input is a view hands its
gradient to the buffer the view came from.

## Memory

The join keeps the boundary residuals (`HostStepReport::residual_store_bytes`), which grow with the
number of layers, and one replay at a time, so neither `peak_snapshot_bytes` nor
`peak_cotangent_bytes` grows with the number of layers. The report carries the two peaks, the
residual store, the loss and the boundary values and cotangents when
`HostStepOptions::keep_boundaries` is set.

Measured on the tiny fixture with a two-sequence batch of six and three tokens (nine tokens, 128
hidden): peak snapshots 7 916 104 B, peak cotangents 23 737 416 B, one block's own operands in f64
50 201 776 B, the residual store `(layers + 1) * tokens * hidden * 2` = 11 520 B. The
`host_step_memory_does_not_grow_with_layers` case holds each peak under one block's worth and the
store to that formula.

## Finite differences

The forward is bf16 throughout -- the container's weights and every activation buffer -- so the
step's loss is a rounded function of the adapter, and the issue's premise ("no rounding anywhere,
so finite differences are exact") does not hold on this fixture. What the `host` checks do instead
is put the adapter on bf16's own grid (`k * 2^-11`, `|k| <= 64`, so `|p| <= 1/32`) and step by a
whole number of `2^-11`, so that `p + h` and `p - h` are exactly the values the engine computes
with, and extrapolate a pair at `h = 2^-7` and `2^-6` (16 and 32 grid steps) with Richardson.

Both ends are measured. Sixteen times that adapter saturates the trunk's bf16 buffers and the loss
stops moving at all (the difference is exactly zero); a quarter of it leaves the difference below
the loss's own rounding. This is the one scale at which both have room.

One whole module at a time -- both of its factors, `+-1` on every element -- so that the difference
carries the module's own scale and the other modules cannot mask a miss. Measured relative figures,
worst first:

| modules | relative |
| --- | --- |
| layer 3 `mlp.down_proj`, `mlp.gate_proj`, `self_attn.v_proj`, `self_attn.o_proj` | 0.6 %, 3.3 %, 3.9 %, 6.4 % |
| layer 3 `mlp.up_proj`, `self_attn.q_proj` | 16.9 %, 37 % |
| layer 2 `mlp.down_proj`, `mlp.gate_proj`, `linear_attn.out_proj` | 5.4 %, 14 %, 44 % |
| layer 1 `mlp.down_proj`, `mlp.gate_proj`, `mlp.up_proj`, `linear_attn.out_proj` | 26 %, 22 %, 35 %, 43 % |
| every `linear_attn.in_proj_qkv` (layers 0, 1, 2) | 98 %, 99 %, 101 % |
| layer 0's remaining linears, layer 2's and layer 1's `in_proj_z`, layer 2's `mlp.up_proj`, layer 3's `self_attn.k_proj` | 55 % to 1455 % |

Thirteen of the twenty-five adapted linears are held at today's figure with half again as much room
(`kFloors` in `tests/host/step_host_grad.cpp`); the rest are printed and named as known misses
rather than held, because a floor over a figure that misses by an order of magnitude would say
nothing.

Two effects put the misses there. The first is depth: a GDN layer's recurrence contracts, so a
relative error in the *values* it passes back grows multiplicatively through the transpose of that
contraction, and by layer 0 the trunk's bf16 rounding has eaten the cotangent (the oracle below
shows the same fall, from a cosine of 0.999 down to 0.004). The second is not rounding but a
defect, and it is the reason `in_proj_qkv` is far and away the worst: see "What is left open".

The alternate objective -- the sum of one logit column, from the forward-driver issue -- is walked
by the same join with no change to it. Its loss is a sum of raw logits of order ten and the head's
logits are bf16, so one ulp of the sum is about 0.06 and a usable step's difference is of that
order: the figures never settle (at four times the step they move by more than they agree). What
does settle is the direction, and
`host_step_walks_an_objective_that_is_not_the_ce` holds the sign of the gradient against the
difference for all six of the last layer's linears, printing the magnitudes it could not hold.

## The oracle

`tests/host/step_host_oracle.cpp` loads `tests/fixtures/oracle/` -- the reference's adapter, its
batch and its f64 loss, residuals and gradients -- and compares the join's own step against it.
Measured, over the reference batch's 241 scored tokens:

* the loss is 7.71901109 against the reference's 7.73519858, 0.185 % off;
* the residual *values* at every boundary match to a cosine of 0.992 or better and a norm within
  0.4 %, which is the forward's own agreement;
* the cotangent at each boundary, from the last layer's output down: cosine 0.998, 0.987, 0.635,
  0.208, 0.004, with norm ratios 1.002, 0.985, 0.762, 0.397, 0.336. The first boundary counted
  from the last layer where it leaves the reference (the issue's cosine 0.999) is boundary 3;
* the gradient of every adapted linear: layer 3's mlp and attention 0.98 to 0.993 cosine (the
  `k_proj` norm is 1.9 to 2.2 times the reference's), layer 2's mlp and `out_proj` 0.974 to 0.991,
  layer 1's mlp 0.67 to 0.80 and its `out_proj` 0.59 to 0.63, and every `in_proj_qkv` 0.02 to 0.10
  with a norm ratio near 0.01.

The case prints every figure per boundary and per adapted linear, holds the ones above at their
measured floors (`kBoundaryFloors`, `kAdapterFloors`) and prints the rest as not held. The issue's
own criterion -- a cosine above 0.999 with the norm within 2 % -- is what the trace uses to name
the boundary where the cotangent leaves the reference.

## What is left open

*The GDN convolution path.* The analytic gradient of every `linear_attn.in_proj_qkv` is about one
per cent of what the step's own loss says it is: on the FD batch the analytic figure is 0.106
against a difference of -9.26 (layer 0), 0.040 against 2.54 (layer 2), and the reference comparison
gives the same ratio. Everything else on that path checks out, and each of these was measured
rather than assumed:

* the block replay reproduces the forward bit for bit (every replayed block's exit residual equals
  the stored one, 0 of 1536 elements differing at any layer), so the snapshots the walk uses are
  the forward's values;
* with `gdn_chunk_scan` cut out of the walk (`BackwardOptions::stops`) the conv path's gradient is
  exactly zero and every boundary cotangent is unchanged, so the whole of that path's cotangent
  comes from the scan's adjoint -- and `gdn_kkt_solve` contributes nothing to it;
* the cotangents the walk carries are the right order of magnitude at every op on the path
  (`gdn_conv_prep`'s inputs 0.01, `gdn_chunk_scan`'s output 0.92, the conv prep's state 2.04), and
  the scan's carried state is snapshotted with the right shape and value;
* a view-operand fix that was needed for other reasons (above) does not move the figure: the block
  walk reads live buffers during the replay, not the kept copies.

So the defect is inside the conv path's own adjoint or its hand-off to the adapter, not in the
replay, the row-space join or the snapshot provider. It belongs to the GDN family's issue: the
adjoint twin check passes on the shapes it is given, and the miss only appears end to end.

*Block brackets on the bf16 fixture.* The issue asks for the cotangent of each layer boundary to be
checked against a finite difference with respect to a perturbation of that boundary's residual.
That cannot run here: `arch/qwen35_bf16` never calls `rad_block_begin`, so `ForwardDriver`'s block
hook is never invoked and there is no boundary to write into (only the int8 build brackets). The
oracle's per-boundary comparison above stands in for it, and closing the gap needs the bf16
architecture to bracket its blocks -- a change to the family's own issue.
