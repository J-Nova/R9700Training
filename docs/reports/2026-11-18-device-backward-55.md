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
Jacobian. The clipped 1e-3 probe agrees: its -175.09 sum-loss change over the leaves the unit
step moves puts their true mean-loss gradients at O(0.1)..O(1), not 1e38. The unclipped
3e-4/1e-4 probes only move leaves with |g| < 1.8e19 (f32 `v` is Inf above), so their descent
tests the moderate leaves' direction only. (The `R97_CENTRAL` run's step was 963.0 s — the host
slower that evening, bwd 886.5 — against the gate run's 342.4 s; the values agree.) #55 stays
open on the magnitude; the next fix is the per-block amplification, starting at the last block
(the first divergent one).

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
