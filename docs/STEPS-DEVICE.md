# The device join: one step's adapter gradient on one card

`r97::device_step` (`include/r97/step_device.h`) is the card's `StepFn`: it forwards a batch
through the real container on the device kernel tier, walks the step back with the device adjoints
and returns the adapter's gradient. It is the device twin of `r97::host_step`
([`STEPS-HOST.md`](STEPS-HOST.md)), with the same block-walk structure and the same pure helpers
(`src/join_common.h`), and it differs in three things: the forward values the walk reads come back
from the card (pulled at the replay for a block, copied at the issue for the head's records), the
head is never held whole, and the gradient is staged on the card.

## The tier it needs

The driver must name the card's library **and** `libr97`, the plugin the device rows load from
(`kernels = {"libr4d", "libr97", "libavx", "libref"}`). `libr97` serves no forward kernel; it is
in the hierarchy so the engine `dlopen`s it, and its static initialisers register the device rows
(and the head's chunk kernel, and the vocabulary head's kernels) into this process's registry.
A driver opened on the host tier has none of them and the walk refuses by name.

The join also needs the training patch set: the issue observer (0101) for the tape, the replay
block (0102) for a block's operands, and the context stream (0122) so the residual store keeps its
boundaries on the stream the step's ops issue on. `R97_DEVICE_STEP` is defined only when all three
are in the build.

## The path

1. **Forward, with the adapter live.** The join installs the driver's engine hook, opens the live
   adapter from the delta's `WireBuffers` and fills it from `params` before `serve()`, exactly as
   the host join does. The step's tape is the engine's own record of the issue.
2. **The residual store.** After the model's load and before the serve, the store is opened on
   the step's stream: `L+1` boundaries, plus the second copy at every boundary a block after the
   first starts at (`L-1` of them, the int8 trunk reads the previous block's output beside the
   residual). `2L` copies of `tokens x hidden x 2 B` — 5 GiB at 4096 tokens, 128 copies. The
   boundaries live in one **pinned host allocation**, shared by the store's copies, and move to
   and from the card as stream-ordered copies on the step's lane
   (`ResidualStore(plan, stream)`); pinned, so each copy is a blit packet rather than a bounce
   through the engine's 8 MiB buffer with a stream drain per chunk. The card's VRAM stays free for
   the walk's per-op scratch: with the store in VRAM, the 85 MiB the int8 decode's weight plane
   asks for was refused at position 565 of the walk (issue #143). The budget the store is checked
   against is the host's available memory (`/proc/meminfo`'s `MemAvailable`, or
   `DeviceStepOptions::residual_budget_bytes`), and the walk refuses with the plan's numbers when
   it does not fit, before the store is allocated.
3. **Each block, replayed and walked.** In reverse order: the replay re-issues the block's ops
   from the stored residual on the card, and the walk reads that block's operands through the
   tape, widens them to the f64 `AdValue` contract (`snapshot`), and runs `backward_block` with
   `BackwardOptions::side = Side::device`. The cotangents are the block above's; the block's
   device rows add into the step's card staging. One block's snapshots live on the host at a
   time, as in the host join.
4. **The head, chunk by chunk.** A logit row is 248320 wide, so its cotangent is never
   materialised. The scored rows go through `head_ce_loop` in chunks of `kHeadChunkRows` (64):
   the fill copies the engine's f32 logits chunk into the chunk's bf16 (`vocab_head_cast`), the
   CE kernel writes `(softmax - onehot) / scored`, and the sink consumes that gradient with the
   transposed product (`vocab_head_transpose`): per class, `dy`'s columns gathered by the class's
   ids, the class's IQR weight decoded to its int8 grid chunk by chunk, the int8 x bf16 product,
   `scale.col`, the sum over the four classes, and the inverse fwht per row — accumulated into
   `d_hout` `[tokens, hidden]` f32. `d_hout` seeds the head's remaining records (the final
   `rmsnorm` and the `gather_rows` before the logits op), which are walked with the device rows.
   Their operands are not pulled: the issue hook copied them at their issue, while the buffers
   still held what the op left (see below).
5. **The flush.** The card's staging holds one f32 buffer per leaf the step's rows named. It is
   copied out once, rounded to bf16, and split into the delta's leaves by the naming the walk
   used (`fused_slots` / `slot_prefix` / the record's prefix), then ordered like `params`. Any
   error path discards it, so the next step does not add to a dead total.

`grads == nullptr` is a forward-only call: the forward runs with the adapter fed and the report is
filled, but the head's gradient pass and the block walk are skipped and nothing is staged. That is
the evaluation pass of a training loop, and the pre-step reference of a check.

## What it holds, and what it does not

| | at 4096 tokens |
| --- | --- |
| residual store (pinned host, streamed) | 5 GiB (`2 x 64` copies of 4096 x 5120 x 2 B) in one pinned allocation, each boundary over the step's lane |
| engine (weights, KV, arena) | 20.2 GiB, measured ([`reports/2026-11-12-device-forward.md`](reports/2026-11-12-device-forward.md)) |
| `d_hout` (card) | 84 MB |
| the CE chunk (card) | 64 x 248320 x 2 B for the logits and the same for the gradient |
| one block's f64 snapshots (host) | ~2.5 GB |
| the card's gradient staging | one f32 buffer per leaf, 320 KiB + 768 KiB per `q_proj` at rank 16 |
| the head's captured operands (host) | one copy per head operand, the logits record excluded: 42 MB per 4096 x 5120 bf16 |

### The head's operands, copied at the issue

The engine reuses the head's buffers after their ops issue and before the walk runs: at its issue
the final norm's input held a maximum magnitude of 14720 and its output 11.06, and after `serve()`
the same two addresses read -7.87e33 and 42.25. No forward op computed a non-finite value, so a
pull taken then copies another writer's bytes -- and the head's read set rightly refused 512 of
512 rows it could not prove. `device_step` therefore wraps the tape's issue hook (the boundary
recorder's, which it keeps) and copies the operands of every record issued outside a block after
the tape has recorded `L` blocks: the head's trailing records, `ops[blocks.back().end..)`. That
criterion is exact -- an op inside a layer carries that layer's block index, the block list
reaches `L` entries at the last layer's begin, and after the last layer's end every op is recorded
with no block open. The logits record is left out: the CE loop reads its output through the tape's
pointer, the step's last write of it, and the walk never snapshots the record -- its output is
4 GB at 4096 tokens. `snap_record` reads the capture for a record at or after `head_start` and
refuses by name -- record, operand and position -- when a head record has no capture, rather than
pulling a stale buffer. The copies are dropped after the head walk; the block walk pulls as
before. The report carries `head_capture_records` and `head_capture_bytes`, and a forward-only
call installs no hook.

### The frozen operands, widened once

An operand of a buffer the step never writes -- a block's weights, a table, a norm's scale -- is
the same value in every block that reads it. `r97::SharedSnapshots`
([`../include/r97/snapshot_share.h`](../include/r97/snapshot_share.h)) holds each one once for the
step, keyed by `snapshot_key` (the buffer, dtype, shape, row window and view), and every block's
walk binds that value instead of pulling the bytes back from the card and widening them again; a
block's own store holds only what that block sees. The report says both: the
`peak_snapshot_store_bytes` / `peak_snapshot_op_bytes` / `peak_snapshot_bytes` figures are one
block's, and `shared_snapshot_bytes` is the single copy the step holds beside them -- at a block
the join holds `peak_snapshot_bytes + shared_snapshot_bytes`. On the real container the frozen
share is ~3.7 GiB of f64, which used to be pulled and widened again for each of the 64 blocks; that
work sits outside the walk's timed rows and is a large part of the untimed backward.

`DeviceStepOptions::keep_boundaries` is off by default. On, the report keeps every boundary's f64
value **and** cotangent: 65 boundaries x 4096 x 5120 x 8 B is ~10 GiB each way, ~21 GiB for both.
A check turns it on
for a batch that fits (512 tokens is 2.7 GiB); a training loop never asks.

Not in this step: the narrow-fold `gemm_nt_q_gated` (a 4096-token step takes the `gemm_nt_q` +
`gated_had_quant_i8g` pair, whose rows exist), the `rope_table` op (off the walk's path), and the
logits op itself (the join handles it; no adjoint row exists for it, by design).

## Where a non-finite gradient is attributed

A step whose gradient holds a NaN or an inf reports where the walk first met one, instead of
leaving it to a bisect. `BackwardReport` (and, summed over the step's walks, `DeviceStepReport`)
carries `first_non_finite_op` / `first_non_finite_operand` / `first_non_finite_operand_index` /
`first_non_finite_position` / `first_non_finite_index` and `first_non_finite_from_row`: the op whose
row was being walked, the tape buffer of the operand whose cotangent held the value, its index in
the record and the value's offset in that operand's payload, and whether the row produced it or the
walk handed it in (a seed, the cotangent from above, or a record later in the forward). Each
`OpTiming` in `row_seconds` counts the slots of its op that carried one (`non_finite`). The same
question is asked of the forward's own operands (`forward_all_finite` and `forward_first_*`): a NaN
born in the forward is a different bug from one born in the walk. The scan is one pass over each
slot the walk already holds, stopping at the slot's first non-finite value, and changes no value,
order or row.

`device_step_tape` prints it as one line, after the walk's per-op split:

```
device_step_tape: non-finite: cotangent op gemm_nt_q, operand 'blk.3.ffn_gate_up.weight'[2] at position 812, value 4097, from the row | forward all finite | per op: gemm_nt_q 2/64
```

`cotangent none` and `per op: (none)` mean the walk handed on only finite values; `from the walk's
input` instead of `from the row` means the op merely passed the value on. The results JSON carries
the same numbers under `non_finite` and per op under `row_seconds[].non_finite`.

The device join also scans what each op *wrote*, so a forward NaN is named at its producer and not
only at the first reader downstream of it: `forward_first_non_finite_op` is the op that wrote the
first non-finite value the snapshot loop met (empty when no walked op wrote one -- a value that
entered from a boundary no walked op writes). The line then reads `forward rmsnorm wrote operand
'h'[0] at position 1458, value 62` rather than naming only the read of `h` at position 1459, and
the JSON carries it as `non_finite.forward_op`.

## How it is checked

`tests/gpu/device_step.cpp`, model tier, a child process per case (ROCR is initialised once):

- `device_step_tape` (512 tokens) — the op-coverage probe: every op the 64 blocks issued has a
  device row, and the head's `rmsnorm`/`gather_rows` do; the exceptions are named. Then a full
  step with the report's boundary copies on.
- `device_step_seq4096` — the ticket: one 4096-token step. The loss equals the forward-only call's,
  the gradient names every leaf of the delta, every `lora_A` leaf's gradient is exactly zero (the
  init leaves B at zero) and every `lora_B` leaf's is not, one AdamW step moves only the `lora_B`
  leaves, and a second forward with the updated adapter is strictly lower — the gradient's
  direction, not just its shape.

Both cases assert the contract issue #146 asks: the `StepFn` runs, the returned `ParamSet` has
the params' names and shapes, the forward is finite, and the optimiser applied one update. The
checks on the gradient's values — `all_finite`, the zero / non-zero split, the descent — are
printed always and asserted when the walk's gradient is finite, so they re-arm the moment it
becomes one. On the current tree the **512-token** walk's cotangent is finite: #157's in-place
restore serves the 63 reads with no earlier writer from the block's stored boundary, and a
record handed a cotangent past f32's range falls back to its host row (the same arithmetic in
f64; see [`reports/2026-11-18-device-backward-55.md`](reports/2026-11-18-device-backward-55.md)).
The **4096-token** case is still non-finite, at `gemm_nt_q`'s `gdn_o.q8`: that weight is
plugin-private, the host row refuses it by name, so the fallback cannot serve the record and the
device row's f32 output overflows. The walk's norm is ~1e38, but the same-tree FD measures the
true leaf gradients at O(1)..O(150) mean-loss (block 63: 1.86, block 0: −145.5) against the
walk's 0.0074 and −6.7e35 — the last layer 250x too small, the bottom 4.6e33x too large — so the
norm is the walk's per-block amplification (#145), not the checkpoint's. The descent is therefore
held only as a **direction** check — the unclipped 1e-4/3e-4 probes or a clipped step — while the
config's 1e-3 is printed and reported, not asserted. The magnitude is still wrong and #55 stays
open.
`R97_DSTEP_ASSERT_FINITE=1` forces the value checks.

The 512-token case's row coverage is a separate check, because a device row is an optimisation
and the walk falls back to the host row where `libr97` has none (`allow_host_fallback` in
`src/step_device.cpp`). The ops that took the host path are named always, and only an op outside
the case's `host_served_op()` list — today `rope`, whose imrope rows are the live container's one
host-served op — fails; `R97_DSTEP_ASSERT_FINITE=1` also forces the coverage, naming even that.

`device_step_tape` also prints which leaf of each factor has an exactly-zero gradient
(`zero-gradient leaves: lora_A <name> (n of N), lora_B ...`), and the JSON carries
`first_zero_a_leaf` / `first_zero_b_leaf` and the counts `n_zero_a` / `n_zero_b`: a
`b_all_nonzero` false names one leaf out of 400, and the name is what a window needs to look at it.

The sharing itself is pinned without a card: `tests/host/snapshot_share.cpp` walks a two-block
tape whose blocks read one frozen weight, once with a per-block widen and once through
`SharedSnapshots`. The walk's outputs are bit-identical, the weight's f64 values are the ones the
per-block widen produced, and the per-block peak store falls by exactly the frozen share.

The row-level checks are the kernel families' own: `tests/gpu/adjoint_{linear,lora,attention,...}.cpp`
and `tests/gpu/vocab_head.cpp`.

## Reproducing

```sh
cmake -B build -G Ninja && cmake --build build -j8
ctest --test-dir build -L host                      # no card
ctest --test-dir build -L gpu                       # one card, serial; the model cases skip without the container
```

The numbers this file's tables carry are in
[`reports/2026-11-13-device-backward.md`](reports/2026-11-13-device-backward.md).
