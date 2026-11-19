# Device backward: the 512-token walk is finite with the right sign, the magnitude is still wrong; 4096 overflows at `gemm_nt_q`

`2026-11-18`, issue #55.

## What the issue asks

The device backward as a `StepFn` on one R9700: one step over the real container, the adapter's
gradient finite on every leaf, and one AdamW step that lowers the loss. The join is #137; #146
made the model cases assert the contract and report the value checks; this report is the
gradient itself.

## The state on current main before this branch

`device_step_tape` (512 tokens): `all_finite 0`, first non-finite `gdn_gated_rmsnorm`/`gdn_o[0]`
at tape position **1143** (11-17), 398/400 A leaves and 399/400 B leaves finite.
`device_step_seq4096`: `gdn_chunk_scan`/`gdn_k[1]` at 463, step 3054 s. #145's verdict was that
the cotangent growth is the checkpoint's mathematics; #149 settled the GDN pre-norm's ~1e-3 as
the checkpoint's property. The walk's in-place reads were served from the op's own output (#148),
and #157 put the in-place mark on the tape and restored the 63 reads with no earlier writer from
the block's stored boundary.

## What changed here

1. **The device rows' in-place handling.** `kv_store` now fills only the pool rows the store
   does not overwrite (mirroring `host/kv_store.cpp`); `lora_expand` fills its in-place slot with
   the write's cotangent. A new `tests/gpu/adjoint_in_place.cpp` holds both at their aliased and
   plain-write records against the host row, and `tests/support/device_check.h`'s `make_call` now
   places twin outputs at their schema positions, which an aliased record needs.
2. **A fills signal in the registry.** The walk's in-place passthrough replaced a row's all-zero
   slot with the whole incoming cotangent, which cannot be told from "the row left it empty". An
   adjoint row now registers a `FillsFn` (`R97_REGISTER_ADJOINT_FF`), and the walk skips the
   passthrough for a slot the row owns. `kv_store` and `lora_expand` register on both sides.
3. **A readable-host range fallback.** A device row's kernels take their operands in f32 and
   write the cotangent back through the record's dtype; a record handed a cotangent past f32's
   range hands on Inf. `src/backward.cpp` now falls back to the op's host row (the same
   arithmetic in f64) when the incoming cotangent exceeds 1e30 and every forward operand is
   host-readable — a plugin-private weight (`raw != nullptr`) keeps the device row, because the
   host row refuses it by name.
4. **`clip_impl`'s norm is double.** `const float norm = float(std::sqrt(sq))` became Inf for the
   7.5e38 gradient, the scale became 1/Inf = 0, and a clip zeroed the whole gradient. The compare
   and the scale are double now; the scale is a representable f32 subnormal.
5. **The descent probes.** The test prints one AdamW step at the default 1e-3, unclipped 3e-4 and
   1e-4, and clipped (norm 1) 1e-3 and 1e-4; the value check is the direction at a step the
   gradient's scale admits, and the default's overshoot is reported. The changed-leaf checks hold
   that both factors move and the gradient reaches every leaf, not that every leaf moves: the
   unclipped AdamW's f32 `v = (1-b2)g^2` is Inf above |g| ~1.8e19, so the dominant leaves take no
   update at the walk's scale.

## The 512-token measurement

`device_step_tape`, one held `local-radiance-window.sh` window per run, prod restored by `close`.

| | 11-17 (pre-#157) | this branch |
|---|---|---|
| `all_finite` | 0 | **1** |
| finite leaves | 398/400 A, 399/400 B | **400/400 A, 400/400 B** |
| first non-finite | `gdn_gated_rmsnorm`/`gdn_o[0]`@1143 | **none** |
| in-place restores | — | **63 of 63** |
| grad norm | nan | 7.63e38 |
| step | 325.7 s | 342.4 s (fwd 47.0, bwd 292.3, head 0.8) |
| VRAM peak | 19,222 MiB | 19,489 MiB |

The walk's cotangent is finite on every op (`non-finite: cotangent none | per op: (none)`); the
forward is finite. The first non-finite before the range fallback was at position 9, and
`R97_DUMP_POS=9` printed the incoming cotangent's `maxabs = 4.23e34` — above the 1e30 threshold,
so the host row served it and the walk became finite.

The descent, same run (`descent_lrs` in the results JSON):

| lr | clip | loss after | delta |
|---|---|---|---|
| 1e-3 | 0 | 7403.47027 | +31.88 |
| 3e-4 | 0 | 7289.34732 | −82.24 |
| 1e-4 | 0 | 7317.57839 | −54.01 |
| 1e-3 | 1 | 7221.76164 | −149.83 |
| 1e-4 | 1 | 7299.87113 | −71.72 |

The direction is right at every step the gradient's scale admits; the unclipped default 1e-3
overshoots. **The magnitude is not.** The same-tree FD (`R97_CENTRAL`, the bf16-sized eps) at the
two ends of the stack:

| leaf | walk g (mean-loss) | central FD (sum-loss) | FD / 511 | walk vs true |
|---|---|---|---|---|
| `model.layers.63.self_attn.q_proj.lora_A` | 0.00744629 | 948.917 | 1.857 | **250x too small** |
| `model.layers.0.linear_attn.in_proj_qkv.lora_A` | -6.66657e35 | -74358.9 | -145.5 | **4.6e33x too large** |

The last layer is already wrong — 250x too small, above the in-place reads #157 fixed — and the
error compounds downward, crossing 1 and reaching 4.6e33x at the bottom. The true leaf gradients
are O(1)..O(150) mean-loss; the walk's ~1e38 norm is the compounding, not the checkpoint's
Jacobian. The clipped probes are a **first-order estimate**, not direction-only: `clip_impl` clips
the gradient before the moments (`src/optim.cpp:264`) and AdamW's `mh/sqrt(vh)` cancels the clip's
scale (`:289-293`), so a dominant leaf's step is ~`lr*sign(g)`. The FD run's clipped 1e-3 loss
change is −175.09, i.e. `sum(g_true*sign(g_walk)) = 1.75e5` sum-loss (342 mean over 511); the FD's
two leaves — block 0 74,358.9 and block 63 948.9 — are 43% of it. The 1e-4 clipped probe gives
4.39e5 (859 mean); the 1e-3 step is past the linear regime, so the two bracket the nonlinearity,
and both are O(1e2)..O(1e3) mean-loss, not 1e38. The *unclipped* probes are the direction-only
ones: the dominant leaves' f32 `v` is Inf above |g| ~1.8e19, so they move only the moderate
leaves. The two runs differ — the `R97_CENTRAL` run's walk norm 7.92e38 against the gate run's
7.63e38, its default-loss after 7400.96052 against 7403.47027, its clipped probes −175.09/−43.93
against −149.83/−71.72 — consistent with the f32 accumulation's order dependence at the walk's
~1e38 scale, while the FD's central differences agree to the digit (948.917 and −74358.9 in both).
#55 stays open on the magnitude; the section below carries what the error is — a uniform ~x3.7 per
block gain, not a top-block offset — and the candidate site.

## The 4096-token ticket case

`device_step_seq4096` (zero-B start), same window harness: step **3610.2 s** (fwd 106.2,
bwd 3494.5, head 6.4), store 5120 MiB, VRAM 21,422 MiB after load / **23,510 MiB peak**, loss
34,682.089 → 50,870.029. The contract passes (the `StepFn` runs, the params keep their names and
shapes, the forward is finite, the optimiser applied one update); the walk is **non-finite**:
`all_finite 0`, first non-finite `gemm_nt_q`/`gdn_o.q8`[0] at position **690**, from the row.
The per-op counts are in the log (`gemm_nt_q 271/288`, `lora_expand 405/288`, ...; the forward
is finite).

Why the range fallback does not reach it: `gemm_nt_q`'s `b` weight is plugin-private
(`raw != nullptr`); the host row refuses a stored operand by name, so `src/backward.cpp`'s
fallback keeps the device row, and that row's f32 output overflows at 4096's larger cotangent.
The value checks stay gated: `n_a_finite_nonzero 0`, `n_b_finite_nonzero 204`, `n_zero_a 211`,
`grad_norm` nan, and all five descent probes report the default's post-step loss because the
gradient is non-finite. The restores still happen (`in_place_restores 63`; the case does not run
the coverage probe, so `in_place_expected -1`). The next piece of work is the `gemm_nt_q` row's
range — an f64 carry (or a per-row scale) for the activation cotangent and the staged weight
gradient — not a change to the host fallback.

## The per-block gain, against the full device walk

`R97_WALK_BLOCKS` limits **the host walk only** (`src/step_device.cpp:1220-1231`, deliberate: a
truncated walk zero-fills what it skipped, which would silence the very A/B it exists for). A device
arm run with `R97_WALK_BLOCKS=14` therefore still walks all 64 blocks, and its `grad norm 8.10e38`
is a full-walk norm — it is **not** comparable with a host 14-block `6.25e8`. Reading those two as an
A/B is a scope error, not a device/host disagreement.

What *is* comparable is the cotangent table at the positions both arms reach. The host 14-block
(blocks 50..63) and the device full walk agree at all 28 of those positions, within 1.25-1.45x — the
device slightly the larger, as its bf16 operands predict:

| pos | block | host | device |
|---|---|---|---|
| 1449 | 63 ffn | 7.25985e-06 | 7.24196e-06 |
| 1429 | 63 attn | 2.62647e-06 | 2.563e-06 |
| 1366 | 60 attn | 4.81279e-05 | 4.98295e-05 |
| 1358 | 59 ffn | 2.0516e-03 | 2.2583e-03 |
| 1148 | 50 ffn | 180.998 | 262 |
| 1135 | 50 attn | 40.267 | 72.5 |

The device walk continues below block 50 with no host counterpart: 1127 (49 ffn) 1344, 1106 (48
ffn) 9792, 1023 9.37e04, 924 5.58e08, 812 5.03e11, 693 6.33e14, 560 2.83e18, 448 1.78e21, 357
2.98e23, 238 1.56e27, 126 3.69e30, **14 (block 0 ffn) 7.05e33**. Head to foot is 40.7 decades over
64 blocks: the walk multiplies the `h` cotangent by **~4.3 per block** (median per-block ratio
4.2; 60->59 is x7.3, 50->49 x2.8).

**The true cotangent is flat.** The FD ladder puts the true leaf gradients at O(1)..O(150)
*mean-loss* across all 64 blocks, so the true per-block gain is ~x1.0, and the walk's ~x4.3 is the
whole error: block 63 too small by 249x and block 0 too large by 4.6e33 is a *pure* per-block gain
error of (4.6e33 x 249)^(1/63) = **x3.7 per block**, with no separate top-block offset needed. The
walk's measured x4.0-4.3 per block is the same number to within the FD's two-point estimate.

The 11-15 report's signature survives: the residual `x` path stays flat at 1.31e-07 while the `h`
path grows. `h` crosses exactly **two** fused norms per block (its `ffn_norm` and `attn_norm`), so a
factor of two at each is a x4/block gain on the `h` path alone, and the in-place `h` fill is the
candidate. The fused norms are **not** amplifying in their own right: `DUMPNORM`'s `dy_sum` against
`DUMPQUANT`'s `quant_dx + dx` is 1.0 within 3% at every record (1429: 181.144 vs 181.57; 1408:
391.66 vs 387.92; 1400: 485.412 vs 498.63; 1387: 950.038 vs 940.05), so the rows are unit-gain.

That measurement does **not** clear the walk, which is why the in-place `h` fill stays the candidate:
a double-count at the walk level would hand each norm an already-doubled cotangent, so `dy_sum`
inflates with it and the ratio above stays ~1.0. The sites to separate are the row's own write to
the `h` slot beside the passthrough fill, the record's two `h` slots being summed instead of one
being taken, and the distribution of the write's cotangent to the replaced version.

A `gdn_gated_rmsnorm` instance amplifies x1.5e4 in one step (pos 1416 leaves `dc[gdn_o] = 3.14662`
against the `2.09146e-04` handed to it at 1417), but that is not the per-block gain: the `dc[gdn_o]`
sequence across gated norms climbs x2-8 per block (3.15, 6.50, 23.4, 299, 925, 7304, 6.28e4,
1.88e5, 4.48e5, 1.52e7), the same x3.8/block as the `h` path, so something inside each block
compensates it by ~x1e4. The row's internal `dc[gdn_o]/dc[gdn_in]` is ~1800-3000 ~= `sc` at every
instance.

### The gain is the container's weight, and the norm is the only row family that disagrees

Four readings close every element-level candidate on the fused norm's gain:

- **The operand is rank 1.** `DUMPDC` prints `buffer=blk.63.ffn_norm.weight shape=[5120]` for all 28
  fused-norm records, so the record declares the weight itself, not a `[rows, n]` per-row tensor.
  `R97_GAIN_ROW=1` is therefore inert (`FDRMS ... w_rank=1 w0=5120 per_row=0`), and "no collapse"
  in that A/B says nothing about the kernel. The `elementwise.cpp:387-390` comment describing a
  `[512,5120]` arrival is stale symptom text of the #157 mis-taken-intermediate bug that
  `quant.cpp:1376-1403` fixed.
- **Its values are the container's**, read offline by `tools/dump_weight.cpp` (`rad::RadFile`, mmap,
  no engine, no GPU): `blk.63.ffn_norm.weight` `0.2451171875, 0.146484375, 0.2099609375,
  0.2138671875`, `blk.63.attn_norm.weight` `0.07568359375, 0.171875, 0.11328125, 0.080078125`,
  `output_norm.weight` `0.9609375, 0.75, 0.9375, 0.828125` -- all `scheme=plain`, bf16, one plane.
  The gain the walk took is the same to the bf16 digit (`GAINPOS pos=1449 ... vals=[0.245117,
  0.146484, 0.209961, 0.213867]`, and the same line for the twin and the adjoint).
- **Gain-after-rotation is closed.** `y2_rel` tracks `deq_rel` at every norm (1449: 1.362 vs 1.364;
  1379: 1.479 vs 1.490), and at the gates `y2_rel` is 1e4 while `deq_rel` is 0.004, so the shape test
  does not discriminate.
- **The twin's elementwise output is not the kernel's value.** `deq_vs_write` of 3e-4 / 1e-5 proves
  `dq` -- reconstructed from the kernel's own codes and scale -- *is* the value the kernel stored,
  and the twin's `y` differs from that same tensor by 1.36 at every fused norm, while all 14
  `gated_had_quant_i8g` and 6 `gate_had_quant_i8g` records match at 0.0034-0.0078. The difference is
  not a per-row scale (`row_ratio` 0.988-1.056) and sorting does not remove it (`sort_rel` 0.33-0.52,
  against 0.0037-0.0045 at the gates), so it is not a permutation either.

With `w` correct and `s = 1/rms(x)` derived from `x`, `y = x*s*(w+1)`: a *scaled* wrong `x` cannot
explain the difference, because `s` cancels the scale. What is left is the `h` snapshot's **vector**
or a structural difference in the kernel's norm.

### The fused norm's `residual` operand: the row differentiates the wrong vector

`rmsnorm_had_quant_i8g`'s operands are `x`, `residual`:inout?, `w`w -> `q`, `scale`, `out_bf16?`
(`third_party/radiance/docs/OPS.md:739`), and `third_party/radiance/libref/ref_quant.cpp:149-206` --
the fused op's own reference implementation, "what a plugin asks for first" -- defines the forward:

```cpp
const float v = res ? ldt(x, xb + i*xs) + ldt(res, rb + i*rst) : src(i);
if (res) stt(res, rb + i*rst, v);      // residual = x + residual
ss += v * v;                            // the variance is over the SUM
...
auto normed = [&](int64_t i) { return src(i) * sc * (ldt(w, i*ws) + wadd); };
```

with `src(i)` reading the residual *after* the update. Its comment states it outright: "With
`residual` it is the fused_add_rms_norm form: the sum is written back to `residual` (INOUT; the next
layer's shortcut reads it) and the norm is over the sum."

**The row takes operand 0 alone.** `src/adjoint/host/elementwise.cpp:461` binds `x = op[0]` and the
loop at `:518-527` never reads operand 1, so on every fused norm whose record carries a residual --
every one on this container -- the row differentiates `rmsnorm(x)` where the kernel computed
`rmsnorm(x + residual)`. That single wrong operand accounts for every exclusion above:

| observation | explained by normalising `x` instead of `x + residual` |
|---|---|
| `deq_rel` 0.99-1.62 at all 28 fused norms, 0.0034-0.0078 at all 20 gate records | gates have no residual operand |
| `row_ratio` 0.988-1.056 | `x` and `x + residual` are both residual-scaled vectors |
| `sort_rel` 0.33-0.52, `y2_rel` tracking `deq_rel` | neither a permutation nor a rotation placement |
| `bestrow` offsets scattered | not a row shift |
| plain `rmsnorm` at 1458 matches to 0.28% (`NORMELEM`) | it has no residual operand |

**Measured, not inferred.** The fused norm rebuilt from each candidate operand, against the
kernel's own reconstructed activation (`DEQELEM ... operandN_rel` = `max|y_N - dq| / max|dq|`):

| pos | `operand0_rel` (what the row uses) | `operand1_rel` |
|---|---|---|
| 1449 | 1.364 | **0.0080** |
| 1429 | 1.046 | **0.0054** |
| 1421 | 1.394 | **0.0087** |
| 1408 | 1.203 | **0.0062** |
| 1400 | 1.363 | **0.0088** |
| 1387 | 1.426 | **0.0071** |
| 1379 | 1.490 | **0.0087** |
| 1366 | 0.986 | **0.0060** |

and at a gate record the relationship inverts as the operands predict (operand 0 right at 0.0078,
operand 1 meaningless at 31, because a gate has no residual operand). 0.5-0.9% is the same floor the
gates and the plain `rmsnorm` sit at, so operand 1's snapshot *is* the norm's input: the tape
snapshots an operand after the op issues, so the in-out residual's value is the narrowed sum
`x + residual_old`, exactly what the reference's `src(i)` reads. The row therefore does not need to
sum anything -- it needs to take operand 1 instead of operand 0, and split the input cotangent
across the two summands.

No test caught it: `tests/gpu/adjoint_quant.cpp:154-187` builds the fused shape **without** the
in-place mark on operand 1, so the residual path is unexpressible in the fixture, and the host
fixtures carry no residual at all.

The backward has the matching hole: the sum's cotangent must reach *both* `x` and the residual's
prior value, and the residual's own in-place write cotangent must be added to its slot. Today the
residual slot is filled by the direct-path passthrough (`src/backward.cpp:803-812`), which is why
the earlier trace of that slot found no row involvement.

### The fix

Both sides now take the residual operand's snapshot as the norm's input `v` -- no summation in the
row, because the tape's post-issue snapshot already holds the narrowed sum -- and split the input
cotangent as `add` does, one copy of each term per slot:

- `src/adjoint/host/elementwise.cpp` -- `rmsnorm_residual` (by-role, with a named refusal on a size
  mismatch); `rmsnorm_twin` and `rmsnorm_host_adjoint` take `v` from that operand; the row adds
  `J_norm(dy) + cot_res` to **both** `dc[0]` and `dc[residual]` (`dv = 0` reproduces `add`); a
  `NORMSCALE` print under `R97_GAIN_POS`.
- `src/adjoint/host/quant.cpp` -- `fused_twin`'s residual slot carries the sum; `fused_adjoint`
  builds `ecot = {g_r, qdc[0] + out_cot}` and **skips the residual in the fold**, which used to
  mis-fold it into the intermediate's cotangent -- a second defect on the same record.
- `src/adjoint/device/quant.hip` -- the mirror: kernel unchanged, `v` uploaded in place of operand
  0, by-role `res_at` with a named refusal, the residual's cotangent kept off the rotation, both
  slots receiving `dv + g_r` once (`:977-1006`), pure superset with no residual.
- `tests/host/adjoint_rmsnorm_residual.cpp` (new) -- forward exact (`residual rel 0, normed rel 0`)
  and gradient `rel 1.33e-10` on both slots against a finite difference of the same decomposition.
  The FD zeroes the quantiser's `codes`/`scale` cotangents, which are piecewise constant in the
  input and cannot be finite-differenced; with them live the test failed at 5.5% while measuring a
  different function from the one the row differentiates.

Two defects sat on this one record, and they are separable in kind. The **wrong input operand**
changes *which function* the row differentiates -- the direction. The **fold in `fused_adjoint` that
leaked the residual's cotangent `g_r` into the elementwise cotangent** is a walk-level double count
*outside* the row: the residual stream's dominant cotangent was added to the norm's input cotangent
at every fused norm. Only the second can explain the magnitude, and only the second is invisible to
the in-row check -- `DUMPNORM`'s `dy_sum` against `DUMPQUANT`'s `quant_dx + dx` compares the row's
output L1 with its input, both computed from whatever input the row was handed, so a gain error
outside the row and a wrong input both leave it at ~1.0. That is also why that measurement, and the
in-place passthrough's insensitivity, are consistent rather than contradictory.

The growth's disappearance (below) was measured with **both** fixes in place. Which of the two
removed it is the mechanism above -- consistent with the insensitivity, and the only one of the two
that can produce a per-block factor -- but it has not been A/B'd against the input change alone.

Two follow-ups carried, not fixed: `rmsnorm_quant_fp8` is a separate row in `quant_fp8.cpp` and is
not in this container's program; and `R97_REGISTER_ADJOINT_IF_F(..., rotated, 1)` freezes operand 1,
which is the gain in the no-residual layout and the residual in the residual layout -- it works
because the walk sizes an in-place frozen slot anyway, but it is misaligned with the op's schema.

### The remaining per-block factor is the residual handoff, not the row

After the residual-operand fix the walk's cotangents decay ~x0.3 per block instead of growing ~x3.8
(the FD ladder at blocks 63 and 0 then read 2.2e7x and ~1e39x too small, a uniform 3.2x/block;
**both ratios are retracted** in "The reference is not a derivative" below -- the FD reference on the
real container is not a derivative, so no ratio can be formed from it).
The row is not the cause: its split is proven against a finite difference of `v = x + residual_old`
(both slots 1.33e-10) and the same-term A/B is refuted by that test -- `dc[0]` reads 0.247 with `g_r`
off the branch slot. The live walk confirms the rule too: `l1_dx == l1_dx_res` at all eight fused
norms (`R97_GAIN_POS=1`, `SPLIT`).

What is broken is which terms reach the row (`SPLIT` at the eight fused norms):

| pos | block | `l1_dy` | `l1_cot_res` | `l1_dx` |
|---|---|---|---|---|
| 1449 | 63 ffn | 15.71 | 0.02845 | 0.02926 |
| 1429 | 63 attn | 3.994 | 0.02926 | 0.02931 |
| 1421 | 62 ffn | 16.88 | **0** | 0.007908 |
| 1408 | 62 attn | 0.5828 | 0.007908 | 0.007912 |
| 1400 | 61 ffn | 4.745 | **0** | 0.002366 |
| 1387 | 61 attn | 0.1752 | 0.002366 | 0.002367 |
| 1379 | 60 ffn | 1.295 | **0** | 0.000647 |
| 1366 | 60 attn | 0.04298 | 0.000647 | 0.000648 |

`l1_cot_res` -- the residual write's incoming cotangent -- is **exactly zero at every ffn norm**, so
the row's output there is `dv` alone (0.007908 against an input of 0.02926): the accumulated
residual cotangent is discarded as the chain crosses each block. That is the x0.27/block, measured
rather than inferred.

The mechanism is in the distribution (`src/backward.cpp:1021-1047`): an in-place write's cotangent
travels to the version its **read segments** name, and `DEVRESTORE` shows the attn norm's residual
read being served from the **block-boundary store** rather than from the version the previous
block's ffn norm wrote. With `seg.version < 0` that cotangent leaves the block via `add_input_full`
instead of feeding the previous writer's version, so the ffn norm's write has no reader in the
version graph, `cot_res` is 0, and its `dc[residual]` carries only the norm's own small term. The
block handoff still works -- the cotangent does reach the next block's attn norm through `incoming`
-- but the ffn norm's slot, and with it the feedback the `add` half of the fusion owes that write,
is dropped every block.

**The constraint on any fix, stated before one is attempted.** The boundary store is not a mistake to
be removed: it exists because the engine reuses its arena between ops, so an in-place read genuinely
cannot be served from the live buffer and the stored copy is what the op consumed. The attn norm's
residual read *is* served from that store, correctly. So the fix is **not** to repoint those read
segments at the previous writer's version -- that would delete the boundary path and reintroduce
whatever it was built for. It is to credit the writer's version **in addition** to
`add_input_full`: the cotangent must both leave the block through `incoming` (as it does now) and
land on the version the previous block's ffn norm wrote, so that write's slot carries its consumers'
cotangent. Anything that moves the term rather than duplicating it will trade this decay for the
growth the boundary path was introduced to stop.

This is walk-side, so it is reachable from the host tier: `SPLIT` is a host-row print and reproduces
the same `l1_cot_res = 0` on the host arm, and `tests/host/backward_compose.cpp` drives the walk core
with a synthetic tape (`BoundaryNames::from(VectorTape)` + `StoredSnapshots` + `Seed`) in seconds. A
fixture there should reproduce the missing back-credit, so the fix can be iterated without a GPU
window each time; the GPU arm is then only for confirming it on the real container.

**That fixture now exists and corrects the mechanism.** `tests/host/backward_resid_handoff.cpp` (own
rows, `BoundaryNames` + `StoredSnapshots` + `Seed`, two `backward_block` calls joined by the join's
own pick rule) reproduces the live pattern in seconds:

```
JOIN below=m l1=0.203692 | the residual input x left block B with l1=0.203692
SPLIT pos=1 role=ffn l1_cot_res=0 (want the next block's residual cot l1=0.203692)
SPLIT pos=0 role=attn l1_cot_res=0.0676224 (the in-block reader; control)
```

`JOIN below=m` is the defect in one token. **Both** boundary-sized buffers of the upper block leave
it -- the residual `x` through `add_input_full` and the branch `m` -- and `pick_boundary` keeps the
first in **name order** (`src/backward.cpp:1080-1083`: the inputs map iterates in op-name order). The
block's carried cotangent therefore lands on the branch's write version, the ffn norm's residual
write has no reader in the version graph, and its slot is exactly zero. The earlier reading of this
section put the defect in the distribution's `seg.version < 0` routing; that routing is correct --
the residual *does* leave the block -- and the loss is one step later, in which of the qualifying
outputs the join chooses to carry.

The test asserts the defect's observable (the ffn slot exactly zero, the attn slot non-zero, the
boundary input non-zero), so it passes now and **fails once the pick is fixed**: the zero assertion
becomes an equality against the block's residual input cotangent, printed as `want` above. That is
the fix's acceptance test.

**How the fix chooses, as landed.** The carry is not a name or shape preference. The set is the
upper walk's own input list (`BackwardResult::inputs`) filtered by the boundary-shape gate, and each
entry is then checked against the block below: if the lower block **writes** that buffer, the carry
is refused by name. That list is tape-derived and is exactly "the buffers the block above consumed
from below", because every read with no earlier writer in the block leaves through
`add_input_full` (`src/backward.cpp:1014-1032`). On this container family it is the residual **and**
the branch.

My first instruction -- "the buffers the next block's records read" -- was wrong, and the agent
found it: the lower block never reads its own mlp output, it **writes** it; the branch is read by
the upper block. A literal read-of-the-lower predicate yields `{residual}` alone and starves the
branch, which is the regression the two-buffer rule exists to avoid.

Both paths are now pinned by the test, which is what makes this safe to land: a reproducer that
asserted only the residual would go green while the branch was zeroed.

```
JOIN below=m,x l1_m=0.203692 l1_x=0.203692
SPLIT pos=1 role=ffn l1_dy=0.203692 l1_cot_res=0.203692 (want the residual cot l1=0.203692)
SPLIT pos=0 role=attn l1_cot_res=0.271315 (the in-block reader; control)
SPLIT pos=2 role=mlp l1_cot_m=0.203692 (want the branch cot l1=0.203692)
```

The ffn norm's residual write now receives block B's residual cotangent **element-wise** and its own
`dy` is non-zero again; the mlp receives the branch cotangent; the attn control stays non-zero. The
equalities are bit-exact f64 copies, no tolerance.

Two shared-interface notes: `in_place_read` moved from `src/backward.cpp`'s anonymous namespace into
`include/r97/backward.h:41-50`, so the walk's version graph and the join derive from one rule; and
`store_boundary` now takes the whole vector and selects from it, since the boundary vector holds
`IncomingCotangent` rather than `InputCotangent`.

That selection matters more than it looks. `report.boundary_cotangent` is **defined** as the
residual's cotangent: `tests/host/step_host_boundary.cpp:245` finite-differences it against residual
perturbations, `tests/host/step_host_oracle.cpp:261` matches it to `residual_grad.<i>`, and the
criterion-two table above reads it. With two entries carried, "the first one" is the branch as often
as not, which would have silently redefined all three.

It selects the entry named by the **store's own record** (`ResidualStore::set_name`, filled at
`keep()` time from the buffer `residual_bytes` recognised), refusing by name when the walk left no
cotangent for it. Two earlier attempts were wrong and the host tier said so: the block's first
in-place read is a rope's write on every model with a rope -- eleven cases refused, naming
`attn_qg` -- and `BoundaryNames` is name-keyed, so a residual repeated at every boundary collapses
to one id and answers `logits`. The store cannot disagree with itself about what it holds.

### After the handoff fix: the boundary moves, the top block does not

The 512 gate improved by 143x in magnitude, stayed finite, and every descent probe still descends
and descends harder:

| | before the handoff fix | after |
|---|---|---|
| grad norm | 5.98934 | 858.248 |
| descent probes | -272.13, -387.26, -214.04, -334.80, -219.37 | -257.82, -741.63, -780.25, -258.80, -783.34 |
| loss after one step | 7099.46 | 7113.77 |
| step | 328.5 s | 311.3 s |

The FD ladder separates two things the fix could not both address.

**Block 0** (`linear_attn.in_proj_qkv.lora_A`): the walk's gradient moved from `7.6652e-32` to
`0.898438`, so the handoff fix reaches the walk. Its central difference is not a usable arbiter
though: at eps=1e-4 the loss moves 7381.51 -> 7396.38, a discrete 15-unit jump, so
`central=-74358.9` is a quantisation or token flip rather than a derivative; at eps=1e-6 both
central differences are exactly 0, below bf16's resolution. The walk's `0.898438` is neither
confirmed nor refuted by this ladder.

**Block 63** (`self_attn.q_proj.lora_A`): the walk is **unchanged** at `4.33922e-05` while the FD
reference reads `948.917`. That the value did not move is expected and diagnostic: block 63 is the
top block, so its leaf gradient never crosses a boundary below it and no handoff change can move it.
Whether `4.33922e-05` is *wrong* is not established by this ladder -- see below.

**The reference is not a derivative.** The third point settles it:

| eps | central |
|---|---|
| 1.14e-6 | 0 (the bf16 floor) |
| 1.14e-4 | 948.917 |
| 4.56e-4 | 152.981 |

A linear response gives the same central at every step. This one falls 6.2x when eps grows 4x, so the
finite difference is dominated by something that shrinks with eps and `948.917` is not a usable
derivative. The real container's forward is bf16 weights through an int8 path: it is piecewise
constant at this scale, and no FD on it can arbitrate a magnitude.

**The ratio does not survive, and it must not drive the next step.** I first wrote that because the
central falls as eps grows, the true gradient must be at least 948.917 and the walk therefore
>=2.2e7x too small. That inference assumes a smooth, concave response; for a piecewise-constant
forward it fails outright, because a single quantisation flip inside the interval scales the finite
difference as 1/eps and says nothing about the derivative. So **the 2.2e7x shortfall at block 63 is
retracted**: neither point is a derivative, no ratio can be formed from them, and criterion two is
open and unmeasured at this leaf rather than refuted.

That does mean the magnitude needs a different arbiter: a finite difference on a quantised forward
cannot supply one. The candidate already in the tree is the **reference step** (`src/reference_step.cpp`),
the f64 exact step the host tier's oracle test compares against -- a gradient-to-gradient comparison
there would be a far better instrument than FD on the real container.

The host walk is still the right discriminator, and it does not depend on the FD: the same batch's
block-63 leaf through `R97_HOST_WALK_AB=2 R97_WALK_BLOCKS=1`, the path the host tier already
finite-differences on models where FD is valid. Agreement with the device walk means the walk is not
what differs and the whole magnitude question is in the instrument; disagreement hands over a value
to bisect the device walk against, which is far cheaper than probing it blind.

The arbiter is `R97_DUMP_LEAF=<name>`: each arm prints its own leaf's gradient, labelled `(walk arm)`
or `(host arm)`. Its failure mode is worth recording, because it is silent. `R97_HOST_WALK_AB=2`
returns from its own arm -- before the device walk's block -- and its gradients live in `host_grads`,
never `grads`. A print placed after that return therefore reports the *device* value in both runs:
two identical numbers, indistinguishable from agreement. The labels exist so that cannot recur.

## Tests

- Host tier serial (`ctest --test-dir build -L host`): **25/25**; the new
  `adjoint_registry_fills_follow_the_served_row` and the huge-norm clip check in
  `tests/host/optim.cpp`.
- GPU tier in one held window: `adjoint_in_place_rows` (4 cases), `adjoint_lora_fixture`,
  `adjoint_quant_device`, `adjoint_qk_fixture` — all pass, 27.8 s.

## Files changed

`include/r97/adjoint.h`, `src/adjoint_registry.cpp`, `src/backward.cpp`, `src/optim.cpp`,
`src/adjoint/device/{kv_store,lora,plugin}.hip`, `src/adjoint/host/{kv_store,lora}.cpp`,
`tests/support/device_check.h`, `tests/gpu/adjoint_in_place.cpp` (new), `tests/gpu/device_step.cpp`,
`tests/host/{adjoint_registry_precedence,optim}.cpp`, `docs/STEPS-DEVICE.md`.
