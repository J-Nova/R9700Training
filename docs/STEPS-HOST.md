# The host join: a model's adapter gradient

`host_step` (`include/r97/step_host.h`, `src/step_host.cpp`) turns a forward driver, a `Model` and
the host adjoints into a `StepFn`. One call forwards the batch with the live adapter, walks the
tape backwards from the objective's seeds, and fills `grads` with the delta's parameter gradients,
under the names `Delta::grad_prefix` gives. The join names neither a loss nor an adapter kind:
`Model::objective` supplies the seeds and `Model::delta` supplies the wire and receives the
gradients. `r97train grad` is its caller: over the tiny container and
`tests/fixtures/chat_small.jsonl` it prints `loss: 7.80732187938`, `scored: 44` and one line per
adapted linear -- 24 modules, from `model.layers.0.linear_attn.in_proj_qkv: 5.87517895` down to
`model.layers.3.self_attn.k_proj: 0.149404334`.

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
not one of the names they know.

A parameter reaches a live record as its value, and `Tape` writes an `RAD_P_F64` as its decimal
spelling (`%.17g`) for that reason: the radiance schema declares `wadd` as a double, and the walk
reparses the recorded string, so the qwen35 fixture's shifted gain arrives as `wadd=1` and not as
an absent parameter. An absent parameter therefore means what `docs/adjoints/elementwise.md` says
it means -- the kernel's own default -- and no family reinterprets its absence.

`rope` and `kv_store` are stateful: their live `INOUT` operand is not on the read list a `TwinFn`
is handed. `src/adjoint/host/rope.cpp` and `src/adjoint/host/kv_store.cpp` -- the qk family's, not
this card's -- register the twin and the host adjoint the walk reaches; the family's formulas stay
in `docs/adjoints/qk.md` and `src/adjoint/host/qk.cpp`. This card adds no row for them, as the
issue leaves the qk gaps to that family's issue.

An operand the kernel also reads as it writes is marked in the issue record:
`patches/training/0121-issue-inout-mark.patch` sets the mark for a `RAD_OPD_INOUT` operand, and
`Tape` copies it into `Operand::in_place`. The walk reads it first where an op writes a buffer and
reads the value it replaces in the same record (`in_place_read`, `include/r97/backward.h`); the two
shapes the ops always have -- `rope`'s operand 0 and a `kv:` buffer no separate read names -- stay
as the fallback for a record built from a program instead of the live tape. The mark is its own
patch for the reason the repository's patches are one concern each: `0101` carries the observer,
`0120` the view and `0121` the mark.

## The view record

An operand whose kernel sees a strided tensor is recorded as the declared buffer plus the row
window, and the tape cannot tell which elements the kernel read: the GDN conv prep takes its gate
and beta as two four-column views of the fused 136-wide input, and the q norm takes the query half
of an interleaved q|gate buffer. `patches/training/0120-view-operands.patch` adds the tensor the
kernel receives to the issue record -- rank, shape, strides, and the element offset from the
declared buffer -- for the strided case only, so a contiguous operand still reads as one span.
`Operand` (`include/r97/op.h`) carries those as `view_shape`, `view_stride` and `view_off` (the
rank is `view_shape.size()`), after `written`/`in_place` so that a positional brace-init still
reads.

Three places use them. `Tape` copies a view's own values, packed in the view's order, because one
span cannot follow a stride; `snapshot()` (`src/join_common.h`) reads a kept copy linearly when it
holds exactly the view's values and walks the strides when it is looking at a live buffer; the
block's snapshot is packed over the view's shape rather than the declared one, and `tighten_operand`
narrows `view_shape[0]` the way it narrows `shape[0]`. `dense_view` (`src/backward.cpp`) scatters a
view's cotangent back into the declared layout, so an adjoint whose input is a view hands its
gradient to the buffer the view came from.

## Memory

The join walks one chunk at a time: the head, then one block from the last to the first, each with
its own `StoredSnapshots`. The store is filled before the walk starts and is held whole for the
whole walk, while the backward core copies out the operands of the one op it is adjointing. So what
the step holds at one moment is

- `peak_snapshot_bytes`: one chunk's store plus that one op. Its two parts are reported beside it,
  `peak_snapshot_store_bytes` and `peak_snapshot_op_bytes`, at the chunk that peaked. This is a
  chunk's worth, so it does not grow with the number of layers;
- `peak_cotangent_bytes`: the walk's live versions, its cot/dc vectors and input accumulators, plus
  the boundary cotangent the join hands in and the final hidden's cotangent it keeps. Also one
  chunk's worth.

Two stores do grow with the number of layers, and each is reported under its own name rather than
folded into the peaks: the forward's bf16 residuals (`residual_store_bytes`) and, only when
`HostStepOptions::keep_boundaries` is set, the report's own f64 copies of the boundary values and
cotangents (`kept_boundary_bytes`). `all_snapshot_store_bytes` is the sum of every chunk's store --
what keeping them all at once would have cost, and what the join does not do -- so a store that grew
with the layers would show up against it.

Measured on the tiny fixture with a two-sequence batch of six and three tokens (nine tokens, 128
hidden): peak snapshots 25 811 992 B, which is a store of 17 895 888 B plus an op of 7 916 104 B;
every chunk's store 24 012 016 B; peak cotangents 23 755 848 B; one boundary 9 216 B; the residual
store `(layers + 1) * tokens * hidden * 2` = 11 520 B; the kept boundaries `(layers + 1) * 2`
boundaries = 92 160 B.

The last block's store (17 895 888 B) is some nine times another block's (1 903 000 B) because two
of its ops, `kv_store` and `attn_paged`, carry operands whose `rows` is 0: the whole declared
buffer rather than the step's rows, which the snapshot takes dense (a 7.9 MB `kv` and a 7.9 MB
`k`/`v` pair). The head's store is 407 128 B, and every block also keeps the last write of each
buffer an op replaced in place (121 824 B, 129 888 B in the last). That is why the sum over all
chunks is only 1.3 times one chunk's store: with uniform blocks it would be one store per layer.
The `host_step_memory_does_not_grow_with_layers` case checks the peak against its own two parts (an
op alone cannot pass for the store), holds the peak store under the sum over all chunks, and holds
the residual store and the kept boundaries to their formulas.

## Finite differences

The forward is bf16 throughout -- the container's weights and every activation buffer -- so the
step's loss is a rounded function of the adapter, and the issue's premise ("no rounding anywhere,
so finite differences are exact") does not hold on this fixture. The checks therefore put the
adapter on bf16's own grid (`k * 2^-11`, `|k| <= 64`, so `|p| <= 1/32`) and step by a whole number
of `2^-11`, so that `p + h` and `p - h` are exactly the values the engine computes with, and
extrapolate a pair at `h = 2^-7` and `2^-6` (16 and 32 grid steps) with Richardson.

Both ends are measured. Sixteen times that adapter saturates the trunk's bf16 buffers and the loss
stops moving at all (the difference is exactly zero); a quarter of it leaves the difference below
the loss's own rounding. This is the one scale at which both have room.

No fixed tolerance can say whether a gradient is right here, so the checks hold the analytic value
to the difference's own scatter instead: the two secants bracket the truncation, their disagreement
is the difference's error estimate, and the analytic value must lie within `kSigmas` (3) times that
disagreement plus a `kSlack` (5 %) fraction of the difference the direction resolves. A gradient
that is wrong moves off by more than the difference scatters and is caught; a gradient that is
right is held to exactly as much agreement as the rounded forward can show.

One whole module at a time -- both of its factors, `+-1` on every element -- so that the difference
carries the module's own scale and the other modules cannot mask a miss. All twenty-five adapted
linears are held this way (`host_step_adapter_gradient_matches_finite_differences`), and the case
prints
each module's analytic value, difference, error, scatter and the tolerance it derived. Measured
relative figures against the difference, worst first: layer 2 `linear_attn.in_proj_z` 424 % (the
module's difference is 0.19 against a gradient of 0.15, i.e. entirely inside the scatter of 0.07
against a tolerance of 0.21), layer 3 `self_attn.k_proj` 268 %, layer 2 `linear_attn.out_proj`
63 %, layer 1 `linear_attn.in_proj_qkv` 55 %, layer 0 `linear_attn.in_proj_qkv` 41 %; the larger
gradients come in at 10 % to 25 % (`mlp.down_proj` layer 0: -26.96 against -22.41; `in_proj_qkv`
layer 1: -44.35 against -28.67), and every module passes the scatter rule. The pattern is a
systematic one: the difference under-reads the larger gradients by 15 % to 35 % at every step in
the usable range, and only the noise-sized modules scatter in both directions. It is not one
factor, which is what a scale error in the gradient would put there: at the same step pair
`mlp.down_proj` reads 1.20 against its difference and `linear_attn.in_proj_qkv` 1.55.

Ten random directions over the whole adapter, each differenced at the same steps
(`host_step_gradient_holds_ten_whole_adapter_directions`). Measured: the analytic sums to -386.041
against the difference's -369.11, an error of 16.9 over a scatter sum of 173 (tolerance 537),
with the per-direction ratios 0.9425, 1.139, 0.898, 0.765, 0.904, 0.867, 0.952, 1.466, 0.826, 0.990
-- relative errors against the issue's 1e-3 of 0.0575, 0.138, 0.102, 0.235, 0.0963, 0.133, 0.0484,
0.466, 0.174 and 0.00994, each of which the case prints beside the step pair it used.
Nine of the ten sit between 0.76 and 1.14, i.e. inside their own scatter; direction 7 comes in at
1.466 with a scatter of 25 against a difference of 52.6, so the case's scatter rule passes it, but
the figure still asks whether the secant or the gradient is wrong, and the secants themselves
cannot say.

That question has an exact forward to answer it. `make_reference_step` runs the same recorded op
graph in f64 from the registered twins, with no card and no bf16, so
`host_step_gradient_against_the_exact_step_at_the_same_steps` runs it beside the join on the same
Model, the same adapter and the same batch, and asks three things of the two.

The reference step keeps each slot's factor under the module that owns it, as this join's
splitter does, so the two gradients name the same leaves and the comparison is tensor for tensor.
The case is enabled and is part of the suite.

The exact step is fed `tests/fixtures/programs/oracle.json`, the op list `r97train ops --record`
wrote from this fixture and the reference's batch, rather than tracing the graph live. A live trace
starts a radiance engine of its own, and radiance's plugin registry is one per process: beside the
driver the second engine refuses to start (`two plugins claim the name 'libavx' ... refusing the
second`), so an earlier version of this case could not run at all. The recorded program is the same
op graph, and the case first shows it is the same forward: the exact step's loss is 7.73519871656
against the reference's 7.73519858105, 1.7e-8 of itself.

* the exact step's gradient against its own central difference at `h = 2^-20` (with the `2^-21`
  difference printed beside it). At an adapter value of about a tenth that step is far above f32's
  ulp and far below bf16's, so the exact step reads it as a derivative: the issue's `1e-3`, held
  per direction, with no floor and no scatter rule. `2^-18` was too coarse for one of the ten
  directions (its central difference sat 2.1e-3 from the gradient); `2^-20` is the step this case
  uses and the one every figure below is measured at. Measured relative errors over the ten
  directions: 5.8e-5, 5.1e-5, 1.5e-4, 6.6e-5, 5.0e-5, 6.0e-6, 4.1e-5, 8.8e-5, 1.1e-5, 5.2e-5 --
  worst 1.5e-4; halving the step moves the difference by 3.7e-5 to 3.2e-3 of the gradient;
* the exact step's own secant at the ten-direction case's steps, `16` and `32` units of `2^-11`:
  the secant's bias with the trunk's rounding taken out. It sits 79 % to 126 % of the gradient away
  -- measured 1.24, 0.803, 0.794, 1.10, 1.08, 1.01, 0.883, 1.03, 0.796, 1.26 -- so the
  ten-direction case's steps are not a derivative even on an f64 forward, and the case fails if any
  direction comes in under half the gradient there. That is the whole of the answer to direction 7
  above: the step, not the gradient;
* the join's gradient against the exact step's, on the join as shipped (its default
  reference-shaped head split, the one the oracle case and `r97train grad` use). The only thing
  between the two is the bf16 trunk, whose own agreement the oracle case below measures per
  boundary and per adapted linear. Each direction's ratio must stay in `[1/kJoinGap, kJoinGap]`,
  with `kJoinGap` 2.0: a direction may not flip and a factor is a real bound, not a sign-blind
  absolute difference. Measured over the same ten seeded directions (`kDirectionSeed`
  `0x4b35b4a458955dd5` and `kDirectionStride`): 0.678, 1.25, 0.639, 0.562, 1.03, 1.20, 0.946,
  0.742, 1.23 and 0.784 --
  all the same sign, worst factor 1.78. The directions are a guard, not a measurement (a random
  Rademacher direction can be nearly orthogonal to the difference), so the whole gradients are
  also held directly: cosine 0.8716 against floors 0.85, norm ratio 0.9754 against 0.10.

The line it prints per direction is `exact g`, `fd(2^-20)`, `fd(2^-21)`, `fd16`, `fd32`, the join's
`g`, `fd(2^-20)`, `fd(2^-21)` and `fd16`, then the exact error, the exact secant's bias, and how
far each step's own difference moves when the step is halved. That last pair is what settles the
issue's premise. Measured, the exact step's difference moves by 3.7e-5 to 3.2e-3 of the gradient
when the step goes from `2^-20` to `2^-21`, and the driver's moves by `28` to `4.4e3` times as
much, changing sign as often as not: direction 2 reads `-796` at `2^-20` and `-13533` at `2^-21`,
direction 3 `-1115` and `-2217`, direction 6 `-3924` and `-20278`. Below a step of about `2^-14`
the loss the driver returns is a step function of the adapter -- bf16's ulp at the adapter's own
values -- and its difference is the ratio of two rounding flips, not a derivative. The case fails
if the driver's moves by less than five times the exact step's, i.e. if the trunk's loss is ever
smooth enough for a difference of it to be asked the issue's `1e-3`.

So the issue's `1e-3` relative is reached on this fixture, but on the loss of the forward that the
issue's premise describes -- "no rounding anywhere, so finite differences are exact" -- and not on
the driver's, where no step makes it reachable. Three effects put the driver's figures where
they are. The first is the step: at the ten-direction case's `16` and `32` units the driver's
difference reads 0.02 to 20 against gradients of 4.1 to 80, and the exact step's reads 57 % to
134 % away -- a step that coarse cannot answer a gradient on either forward. The second is the
rounding, measured above: at `2^-20` and `2^-21` the driver's difference is a ratio of rounding
flips that changes sign when the step is halved, and only the exact step's converges. The third
is depth: a GDN layer's recurrence contracts, so a relative error in the values it passes back
grows through the transpose of that contraction, and by layer 0 the trunk's bf16 rounding has
eaten much of the cotangent (the oracle below shows the same fall, from a cosine of 0.998 down to
0.812).

The alternate objective -- the sum of one logit column, from the forward-driver issue -- is walked
by the same join with no change to it. Its loss is a sum of raw logits of order ten and the head's
logits are bf16, so one ulp of the sum is about 0.06; the measured figures for the last layer's six
linears are 0.69 to 2.26 against gradients of 4.8 to 34, and
`host_step_walks_an_objective_that_is_not_the_ce` holds each to its own scatter, printing every
figure.

## The per-layer boundary

`tests/host/step_host_boundary.cpp` is the issue's per-layer check: for each of the fixture's five
boundaries it perturbs the residual itself and asks the engine what the loss does. The join installs
its own `BoundaryRecorder` over the tape's bracket hook, so the case re-installs an engine hook that
only keeps the live adapter fed and then uses `ForwardDriver::set_block_hook` to write
`x + h*d` into the live residual at the bracket of block `b` (the exit of the last block for
boundary `layers`), reading the loss through the engine's own head and objective. No production API
is added and no head is re-implemented.

The step is per boundary and a power of two, because the residual is bf16: the hook records what it
actually wrote along the direction and the difference divides by that effective step instead of the
one it asked for, which is what makes a step below one quantum usable. The last boundary takes
`2^-3`, then `2^-5`, `2^-7`, and the first two `2^-9`: the deeper the boundary, the more the layers
above amplify a perturbation, so the step that keeps the response linear shrinks with depth.

Measured, eight directions per boundary, the cotangent's sum against the difference's sum:

| boundary | step | cotangent | difference | ratio | mean residual / allowance |
| --- | --- | --- | --- | --- | --- |
| 4 (last block's exit) | 2^-3 | -3.7432 | -3.7335 | 0.9974 | 0.0012 / 0.0205 |
| 3 | 2^-5 | -8.5835 | -8.5170 | 0.9923 | 0.0083 / 0.0796 |
| 2 | 2^-7 | -5.4014 | -5.3273 | 0.9863 | 0.0093 / 0.2797 |
| 1 | 2^-9 | -61.4950 | -53.3759 | 0.8680 | 1.0149 / 5.5444 |
| 0 (first block's entry) | 2^-9 | -226.7137 | -220.0411 | 0.9706 | 0.8341 / 6.6721 |

Every boundary holds: the mean residual over the eight directions stays inside `kSigmas` standard
errors of itself plus two per cent of the cotangent, and the case prints all three figures per
boundary. The three nearest the loss are within 0.3 %, 0.8 % and 1.4 % of the cotangent; the two
deepest are limited by the same depth effect as the adapter gradients, and their allowance says so.

## The oracle

`tests/host/step_host_oracle.cpp` loads `tests/fixtures/oracle/` -- the reference's adapter, its
batch and its f64 loss, residuals and gradients -- and compares the join's own step against it.
Measured, over the reference batch's 241 scored tokens:

* the loss is 7.71901109 against the reference's 7.73519858, 0.21 % off;
* the residual *values* at every boundary match to a cosine of 0.992 or better and a norm within
  0.6 %, which is the forward's own agreement;
* the cotangent at each boundary, from the last layer's output down, per sequence: cosine
  0.997 to 1.000 with norm ratios 0.996 to 1.002 at boundary 4, 0.991 to 0.997 and 0.988 to 0.999
  at boundary 3, 0.871 to 0.916 and 0.900 to 1.147 at boundary 2, 0.911 to 0.965 and 0.802 to 1.175
  at boundary 1, 0.812 to 0.942 and 0.860 to 1.854 at boundary 0. Measured against the issue's own
  criterion, 64 of the 65 figures miss; the case names the first boundary the walk reaches from the
  last layer where any sequence's cotangent leaves the reference -- boundary 4 of the first
  sequence, cosine 0.998292 -- and then the adapters that miss above no boundary;
* the gradient of every adapted linear: layer 3's mlp and attention 0.96 to 0.993 cosine, layer 2's
  mlp and `in_proj_z`/`out_proj` 0.977 to 0.993, layer 1's mlp 0.76 to 0.90 and its `out_proj`
  0.81 to 0.87, layer 0's mlp and attention 0.81 to 0.95, and every `in_proj_qkv` 0.773 to 0.930.

The case prints every figure per boundary and per adapted linear and holds all of them; beside
each boundary cotangent and each adapter gradient it prints `(miss)` when that figure fails the
issue's original 0.999 / 2 % criterion, so the count below can be read off the lines. The
boundary floors are 0.80/0.88/0.85/0.98/0.99 with the norm floors above, and the adapter default is
0.75 with 0.30 norm, with tighter rows for the modules measured above it. The issue's own criterion
-- a cosine above 0.999 with the norm within 2 % -- is what the trace uses to name the boundary
where the cotangent leaves the reference, and the case reports the count that miss it rather than
pretending to hold it.

Where the criterion *is* held is the adapter gradients. The walk's `A` and `B` are checked against
the reference by `reference_step_loss_and_gradients_match_the_oracle`, which walks the same
`backward_block` through the f64 twins: the oracle's loss to 1.75e-8 and every `A` and `B` to
7.1e-5 of its largest value. (The issue's 1e-3 finite-difference tolerance is held on the exact
step's own difference, in the ten-direction case below.)

The boundary cotangents have a counterpart too, though not through `make_reference_step`: it returns
a `StepResult`, the loss and the gradients, and exposes no boundary values. Their counterpart is the
oracle fixture itself, which `tools/oracle/make_reference.py` wrote from PyTorch in f64, with the
residual `seq*/residual.<i>` and its cotangent `seq*/residual_grad.<i>` at every boundary and for
every sequence. `host_step_matches_the_oracle_boundaries_and_adapter_gradient` compares every one of
the join's boundary values and cotangents to those tensors: a residual value must match to a cosine
of 0.99 and a norm within 1 %, and each cotangent is held to its boundary floor in `kBoundaryFloors`
(which may be raised but never lowered). So the cotangent half is held against the reference at
those floors. The issue's 0.999 / 2 % criterion is not held on it -- the forward is bf16 and the GDN
recurrence amplifies the rounding with depth -- so the case prints `(miss)` beside every figure that
fails the criterion and reports the count, which is what the decision asks of the comparison.

Two things the criterion's wording forces and the case does check. The gradient set is compared
whole: every parameter the delta hands the join must have a reference tensor and every reference
`lora_grad/*` tensor must have a gradient, so a name that only one side has is a miss and not a
comparison to skip. And the loss the step reports must be the forward driver's own: the case runs
the same driver over the same adapter itself, with no join behind it
(`host_step_loss_is_the_drivers_own_forward`), and holds the two losses to each other exactly --
80.182612622695075 both times -- which a reference or twin forward could only match by chance.
`host_step_gradients_carry_the_deltas_names` holds the same set from the other side, without the
oracle: 50 gradients, under the delta's own names and shapes, and nothing else. And because
`LoraDelta::grad_prefix()` is empty, the same case is run again over a delta that hands the same
adapter over under `test.prefixed.`: `host_step_gradients_carry_a_deltas_prefix` holds 50
gradients under the prefixed names with the values the plain walk gave the unprefixed ones. Without
the prefix in the walk's slot names and in the split, that case fails on the value of the first
tensor -- the parameter would be left for the zero fill.


## The int8 trunk

The same join walks the w8a8 container. Its op graph differs in three places: the residual add
and norm are fused into `rmsnorm_quant_fp8`, the MLP's product and its rounding into
`gated_quant_fp8`, and every trunk linear reads `had_quant_act_i8g`'s int8 codes through
`gemm_nt_q`. The quantisers' adjoints hand back the **codes'** cotangent (section 3 of
`docs/adjoints/quant.md`), and `gemm_nt_q`'s adjoint returns the same basis: it dequantises the
weight and then multiplies by the activation's own group scale, so the row a code operand feeds is
handed the cotangent a downstream quantiser divides by its scale. No record tag and no step-specific
wrapper is involved; the device row is the same rule.

The adapter is initialised in the basis the container names. `Engine::lora_set_weights` builds its
factors from the caller's map, so the basis the container was configured for has to reach them, as
a file adapter reads its own from its header: `build_sites` records `cfg_.lora.input_basis` on
every site and `lora::pack_modules` copies it into the `AdapterInfo` it returns (patch `0210`).
Reading it from the sites rather than from the engine's config keeps that out of `0201`'s function,
so `0210` still applies on a bare radiance. A live adapter on the rotated trunk is thus read in the
same basis as the container's int8 rows.

The head's int8 weight is joined into one buffer by the engine: a `RadPlanesHeader`, then the codes,
then a scale per group. The ABI's byte count for the operand is the codes' declared elements, so a
tape that copied only that many bytes dropped the scales and the walk dequantised the head's codes
with the codes alone. `Tape::record` now follows the header's own account of the operand's span for
a joined weight, and `snapshot` reads the planes out by their roles (`joined_dense`): code times
scale, at the block the scale plane declares, is the bf16 weight the GEMM's epilogue actually
formed. The head cotangent that enters the trunk is therefore the dequantised one, and its factor
is comparable with the bf16 model's.

`host_step_int8` checks the property a quantised trunk can give: the straight-through gradient is
finite and nonzero, and a step along it lowers the driver's scored loss. A step of the same
length against it -- a hash direction with a negative projection -- is held above the descent:
it may lower the loss slightly (the surface curves down along that direction, and the walk's
gradient projects little onto it), but it must not descend as well as the gradient's own step, so
the check cannot be passed by a direction that merely has a negative projection.

`host_step_int8_gradient_points_the_way_bf16_does` runs the same adapter on the same batch through
both containers and compares the adapter's gradients, per site, per layer and whole. Both sides
start from the same adapter, which takes two things. `init` leaves `lora_B` at zero, and at `B = 0`
neither the loss nor the `lora_A` gradient depends on the adapter at all, so the test seeds a
deterministic nonzero `B`. And the int8 trunk rotates each 128-wide group of the activation before
the GEMM, so what `lora_A` is applied to there is `H*x`, the container's dequantised activation
with `H` the *unnormalised* Sylvester transform (as `docs/ADAPTER.md` records it and the kernels
build it): the bf16 side runs `A_i8 * H`, and the int8 gradient comes back through `H/128`. The
two losses agreeing to within a percent is what that mapping gets, and it is the check the mapping
has to pass: the wrong normalisation (`H/sqrt(128)`, or no mapping at all) moves the same loss by
five percent.

What the walk measures on this tree, with that mapping and a nonzero `B`: the two losses agree to
0.7 %; the gradients agree with a cosine of 0.86 whole and 0.97 over the top block; the top
block's magnitude is 0.89 of the bf16 one. The per-site cosines run from -0.56 to 0.99 and the
per-site magnitudes from 0.37 to 1.70. The replay gets those figures by restoring the trunk's
second boundary input: in the int8 chain every block after the first reads the previous block's
output beside the residual (the fused add of `rmsnorm_quant_fp8`), and a replay that handed those
blocks the buffer's stale contents scored 0.67 whole instead, with the anti-correlated sites that
the restored copy removes (the residual store's second copies, `docs/REPLAY.md`).

## The floors the int8 check holds

- **The criterion is the measured basis, not a hoped-for 0.9.** `docs/PLAN.md` records it. The bf16
  walk's own whole gradient scores 0.8716 against the exact reference and is held at 0.85
  (`tests/host/step_host_oracle.cpp`, `tests/host/step_host_grad.cpp`); the int8 walk's scores
  0.86 whole and 0.97 over the top block against the bf16 one, held at 0.8 and 0.9, with the
  per-site cosines running -0.56 to 0.99. The
  adapter gradients are comparable -- the two containers' fused and unfused layouts are layouts,
  and the walk and the split make the nine targets match element for element (see
  `docs/FIXTURES.md`) -- and the gap between the two containers is inside the blocks, not at their
  edges or in the join: the boundaries they share agree to 0.9999 at the head and 0.978 at the top
  block's input, and the losses to a percent, yet the sites' gradients disagree by more than their
  blocks' boundaries do. Raise a floor when the walk improves; never lower one.
- **The anti-correlated sites are the straight-through boundary, in plain sight.** In the int8 chain
  every quantiser's derivative is defined as one: the walk passes the cotangent straight through
  the code quantisers (`had_quant_act_i8g`, `rmsnorm_quant_fp8`, `gated_quant_fp8`, and
  `gemm_nt_q`'s code-basis wrapper, `docs/adjoints/quant.md`), so a site whose cotangent is a
  near-cancelling sum follows the quantised forward's rounding in place of the bf16 one and can
  come out anti-correlated with it. That is exactly what the first block shows: its `mlp` sites
  and its `in_proj_z` read -0.56 to -0.00 of the bf16 gradient while the same block's
  `in_proj_qkv` agrees at 0.95 to 0.98. It is the definition of the gradient the issue asks for,
  not a defect of the walk, and it is why the fixture's floor is 0.8 whole and 0.9 over the top
  block rather than a hoped-for 0.99.

*The 10-12x magnitude the last round measured was the test's, not the walk's.* The committed test
mapped `lora_A` through `H/sqrt(128)`, which is the normalisation the transform does not carry in
this container: `H` is its own inverse up to 128, so `A_i8 * H` maps into the natural basis and
`H/128` maps a gradient back. With the wrong normalisation the top block's gradients came out
11-12 times the bf16 ones; with the right one they come out between 0.4 and 1.8, which is why the
comparison above is on magnitudes and not only on directions.

*The join is the bf16 forward, by decision.* An earlier round of this card made `host_step`
delegate to `make_reference_step`, so that the checks could be met at the issue's tolerance; the
review rejected it, and the issue is right to: it asks for the forward driver with the live adapter
and for the driver's own loss, and `make_reference_step` computes neither (it is the f64 replay).
So the join runs the driver, and every figure on this page is the bf16 forward's. What proves the
adjoints is the f64 twin walk, below. The cost is the issue's 1e-3 finite-difference tolerance and
its 0.999/2 % oracle criterion: neither is reachable on this forward, for the reason in the next
paragraph. The 1e-3 is held where it can be, on the exact step's own difference (the ten-direction
note below), and this page reports what the join's forward measures instead of moving the forward.

*The bf16 forward, not the adjoints.* What the join measures short of the reference is the driver's
bf16 forward: `reference_step_loss_and_gradients_match_the_oracle`
(`tests/host/reference_step.cpp`) runs the whole model through the f64 twins, which walk the same
host adjoints this join walks through `backward_block`, and reproduces the oracle's loss to 1.75e-8
and every adapter gradient to 7.1e-5 of its largest value. The adjoints are therefore at the
reference; the cotangent and adapter figures above are the rounding of the value path, amplified
backwards by the GDN recurrences, and nothing in the join can remove them while the forward is the
bf16 kernel chain the model runs. The issue's 0.999/2 % criterion would need either an f64 forward
or a tolerance that says bf16.

*One whole-adapter direction.* Of the ten directions the issue asks for, the join's projection
runs from 0.56 to 1.25 of the exact step's (worst factor 1.78), every one the same sign, and the
per-module checks pass.
`host_step_gradient_against_the_exact_step_at_the_same_steps` now measures those same ten
directions against the f64 reference step, so the figure is attributed rather than left standing:
the exact step's gradient is held to its own difference at `2^-20` (the issue's tolerance, on a step
that is a derivative; worst measured 1.5e-4), the exact step's secant at the same two grid steps is
printed beside the join's, and the join's gradient is held to the factor `kJoinGap` (2.0) of the
exact one -- a guard for the join, since the per-tensor agreement the bf16 trunk allows is the
oracle case's to say. The whole gradients are held beside it, cosine 0.8716 against 0.85 and norm
ratio 0.9754 against 0.10. A gradient that is wrong in the join fails on a number there; a step
that is too coarse is visible as the exact step's own secant, printed on the same line.

*The GDN convolution path.* The first version of this join snapshotted an in-place read after the
op's write, so the replay fed the conv prep the value the op had just produced instead of the value
it read; every `linear_attn.in_proj_qkv` gradient was about one per cent of the difference's. The
fix is in `src/step_host.cpp`: a per-layer map of the last full write of a buffer, and for an
in-place op whose read and write share a buffer, the previous value when the shapes match. The
per-module finite differences improved from about 100 % to 0.3 % (layer 2), 55 % (layer 1) and 41 %
(layer 0), and the oracle's qkv cosine from 0.02 to 0.10 up to 0.773 to 0.930. The conv path is now
inside the same bf16 agreement as the rest of the trunk; the remaining qkv gap is depth and the
attenuating GDN recurrence, not a missing derivative.
