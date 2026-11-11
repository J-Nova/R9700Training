# Ops the Qwen3.5 training forward issues, on the host

This is the inventory `r97train ops <file.rad>` prints, for the two tiny fixtures the host tests
build: the bf16 checkpoint and the w8a8 (int8) container converted from it with
`tests/fixtures/w8a8_fwht128.recipe`. It is the list every adjoint issue is checked against: what
the forward actually issues, in what form, how often, in which kind of block, with which adjoint
rows already registered, and where the backward walk can start.

```
cmake --build build
build/r97train ops <fixture>.rad --home build/radiance-build/radiance_home --kernels libavx,libref
build/r97train ops --registered    # the registry rows the twin/host/device columns read
```

`tests/host/ops_inventory.cpp` runs the same step twice for each fixture and checks the two
inventories agree; that is what makes the tables below comparable with a later issue's.

## The training default

The step is a prefill of one 8-token prompt with `n_spec = 0` and `mm_max_patches = 0`
(src/op_inventory.cpp's `record_prefill_step`): text only, speculation off, and a vision tower the
container might carry stays off the card. The engine pads the step to its `max_tok` budget (512), so
the kernels run at `M = 512`; that is a geometry parameter and does not change the set of ops or
their counts.

With that configuration the tape holds no drafter op (DFlash2) and no MTP-head op: the fixtures
carry neither, `n_spec = 0` keeps a drafter out, and a container that carried an MTP layer would
need a speculator to issue its head. No vision-tower op can be issued either: the fixtures carry no
tower, and `mm_max_patches = 0` keeps one off the card even if a container has it.

The engine's sampler tail is not in that list -- and it is absent because the step asks for none,
not because the inventory leaves it out. The prefill driver's requests are prefill-only
(`Request::prefill_only`, `patches/training/0105-prefill-not-sampled.patch`): a prefill-only
request draws no token, so the engine hands it to no sampler, the step issues no `sample_*` op, and
the request finishes at the commit of the step that places its last position. What the step reads
is the request's scored rows: the logits at every prompt position. `docs/FIXTURES.md` names that
chain -- the one a completion would run after the prefill -- "the sampling tail", and says the
training forward does not reach it; so the tape ends at the logits gemm and the inventory lists
every issue the step made, nothing filtered out by name.

The step's logits buffer still carries the row such a sequence would sample from: the batch sizes
its out rows as the scored rows plus one per sampled sequence, and 0105 says why the row is kept.
Moving the scored rows changes the head gemm's row count, which moves its split-K rounding, and
`tests/host/fixture_prefill.cpp` checks that a prompt in a batch reads the rows it reads alone bit
for bit. So the row exists and is never read: no token, no end-of-text test, no grammar step.

## Columns

- **op** -- the op name the runtime issued.
- **block** -- where in the model it ran: `embedding` (before the first layer), `full-attention`
  and `gated-delta-net` (the first half of a layer, by whether it names `attn_*` or `ssm_*`
  weights), `mlp` (from the first `ffn` tensor of the layer on) and `head` (the output norm, the
  gather and the logits gemm). qwen35 does not call `rad_block_begin`, so the ranges are read off
  the `blk.<n>.<tensor>` names on the tape rather than `Tape::blocks()`.
- **form** -- the op's string parameters, `dtype` first. They are what selects a kernel form
  (bf16 against w8a8, `act=silu`, `mode=mrope`, `q_dtype`/`kv_dtype`); the integer parameters are
  geometry and are left out. `-` means every parameter is an integer: the op name itself already
  fixes the form (the `gdn_*` ops have no dtype parameter).
- **count** -- issues in the recorded step. One step of the tiny fixture is 512 padded rows, hence
  the multiples of the layer count.
- **twin / host / device** -- whether `r97train ops --registered` holds a forward twin, a host
  adjoint or a device adjoint for the op. The family issues merged so far give a twin and a host
  adjoint to every op here except `rope_table`, `rmsnorm_quant_fp8`, `gated_quant_fp8` and
  `quant_act_fp8`; `rope` and `kv_store` get their rows from `src/adjoint/host/qk_rows.cpp`, the
  stateful rows the walk needs. The device column is read by a host-only run with
  `--kernels libavx,libref`, which never loads the device plugin: `src/adjoint/device/add.hip`
  registers `add`'s device adjoint when that plugin loads, but every `device` cell below is `-`
  because this inventory does not load it. The rest is what the adjoint issues are for.
- **seed** -- the named boundary under which the forward driver exposes the buffer the op writes,
  so an objective can seed the backward walk there, or `-`. In the plain forward only the op that
  writes the logits has one, `logits`; the rollout driver of the second milestone adds `ce` and
  `gate_logits`. The inventory is built with all three names, and neither fixture writes `ce` or
  `gate_logits`.
- **on path** -- whether a backward walk from the seed's buffer through every op that wrote a
  buffer the walk reads reaches this op. Weights and raw operands are leaves: no op writes them,
  so the walk ends there. `yes` means the op is on the path from the loss back to a trunk linear.
  An op that only prepares state the loss does not read is marked `no`, with the reason; there are
  two such groups in the int8 inventory (below) and none in the bf16 one.
- **adjoint** -- the family file the op's backward belongs to (`src/adjoint/host/<family>.cpp`),
  from the table at the end.

`kv_store` is on the path even though it writes decode state, because this step's attention reads
what it wrote; an op is marked off only when *nothing* on the loss path reads its writes. The
decode-only GDN ops the int8 container declares -- `gdn_conv_update`, `gdn_conv_recurrent_update`,
`gdn_recurrent_update` -- are not issued in prefill at all, so they are not rows.

## bf16 fixture

One prefill step: 65 issues, 23 op forms.

| op | block | form | count | twin | host | device | seed | on path | adjoint |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `embed_lookup` | embedding | dtype=bf16 | 1 | yes | yes | - | - | yes | linear |
| `rmsnorm` | gated-delta-net | dtype=bf16 | 3 | yes | yes | - | - | yes | elementwise |
| `gemm_nt` | gated-delta-net | dtype=bf16 | 9 | yes | yes | - | - | yes | linear |
| `gdn_conv_prep` | gated-delta-net | - | 3 | yes | yes | - | - | yes | gdn_prep |
| `gdn_kkt_solve` | gated-delta-net | - | 3 | yes | yes | - | - | yes | gdn_kkt |
| `gdn_chunk_scan` | gated-delta-net | - | 3 | yes | yes | - | - | yes | gdn_scan |
| `gdn_gated_rmsnorm` | gated-delta-net | act=silu | 3 | yes | yes | - | - | yes | gdn_prep |
| `add` | gated-delta-net | dtype=bf16 | 3 | yes | yes | - | - | yes | elementwise |
| `rmsnorm` | mlp | dtype=bf16 | 4 | yes | yes | - | - | yes | elementwise |
| `gemm_nt` | mlp | dtype=bf16 | 8 | yes | yes | - | - | yes | linear |
| `silu_mul` | mlp | dtype=bf16 | 4 | yes | yes | - | - | yes | elementwise |
| `add` | mlp | dtype=bf16 | 4 | yes | yes | - | - | yes | elementwise |
| `rmsnorm` | full-attention | dtype=bf16 | 3 | yes | yes | - | - | yes | elementwise |
| `gemm_nt` | full-attention | dtype=bf16 | 4 | yes | yes | - | - | yes | linear |
| `rope` | full-attention | mode=mrope | 2 | - | yes | - | - | yes | qk |
| `kv_store` | full-attention | kv_dtype=bf16 | 1 | - | yes | - | - | yes | qk |
| `attn_paged` | full-attention | q_dtype=bf16 kv_dtype=bf16 | 1 | yes | yes | - | - | yes | attention |
| `sigmoid` | full-attention | dtype=bf16 | 1 | yes | yes | - | - | yes | elementwise |
| `mul` | full-attention | dtype=bf16 | 1 | yes | yes | - | - | yes | elementwise |
| `add` | full-attention | dtype=bf16 | 1 | yes | yes | - | - | yes | elementwise |
| `rmsnorm` | head | dtype=bf16 | 1 | yes | yes | - | - | yes | elementwise |
| `gather_rows` | head | dtype=bf16 | 1 | yes | yes | - | - | yes | linear |
| `logits_gemm` | head | dtype=bf16 | 1 | yes | yes | - | logits | yes | linear |

## int8 (w8a8) fixture

One prefill step: 78 issues, 29 op forms.

| op | block | form | count | twin | host | device | seed | on path | adjoint |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `embed_lookup` | embedding | dtype=bf16 | 1 | yes | yes | - | - | yes | linear |
| `rope_table` | full-attention | dtype=f32 | 1 | - | - | - | - | no: nothing it writes is read on the walk from the loss back to the trunk linears | qk |
| `rmsnorm_quant_fp8` | gated-delta-net | dtype=bf16 | 3 | - | - | - | - | yes | quant |
| `gemm_nt` | gated-delta-net | dtype=bf16 | 6 | yes | yes | - | - | yes | linear |
| `had_quant_act_i8g` | gated-delta-net | dtype=bf16 | 3 | yes | yes | - | - | yes | quant |
| `gemm_nt_q` | gated-delta-net | dtype=i8a8h | 3 | yes | yes | - | - | yes | linear |
| `gdn_conv_prep` | gated-delta-net | - | 3 | yes | yes | - | - | yes | gdn_prep |
| `gdn_kkt_solve` | gated-delta-net | - | 3 | yes | yes | - | - | yes | gdn_kkt |
| `gdn_chunk_scan` | gated-delta-net | - | 3 | yes | yes | - | - | yes | gdn_scan |
| `gdn_gated_rmsnorm` | gated-delta-net | act=silu | 3 | yes | yes | - | - | yes | gdn_prep |
| `quant_act_fp8` | gated-delta-net | dtype=bf16 | 3 | - | - | - | - | no: nothing it writes is read on the walk from the loss back to the trunk linears | quant |
| `rmsnorm_quant_fp8` | mlp | dtype=bf16 | 4 | - | - | - | - | yes | quant |
| `had_quant_act_i8g` | mlp | dtype=bf16 | 8 | yes | yes | - | - | yes | quant |
| `gemm_nt_q` | mlp | dtype=i8a8h | 8 | yes | yes | - | - | yes | linear |
| `gated_quant_fp8` | mlp | dtype=bf16 act=silu | 4 | - | - | - | - | yes | quant |
| `rmsnorm_quant_fp8` | full-attention | dtype=bf16 | 1 | - | - | - | - | yes | quant |
| `had_quant_act_i8g` | full-attention | dtype=bf16 | 4 | yes | yes | - | - | yes | quant |
| `gemm_nt_q` | full-attention | dtype=i8a8h | 4 | yes | yes | - | - | yes | linear |
| `rmsnorm` | full-attention | dtype=bf16 | 2 | yes | yes | - | - | yes | elementwise |
| `rope` | full-attention | mode=mrope | 2 | - | yes | - | - | yes | qk |
| `kv_store` | full-attention | kv_dtype=bf16 | 1 | - | yes | - | - | yes | qk |
| `attn_paged` | full-attention | q_dtype=bf16 kv_dtype=bf16 | 1 | yes | yes | - | - | yes | attention |
| `sigmoid` | full-attention | dtype=bf16 | 1 | yes | yes | - | - | yes | elementwise |
| `mul` | full-attention | dtype=bf16 | 1 | yes | yes | - | - | yes | elementwise |
| `quant_act_fp8` | full-attention | dtype=bf16 | 1 | - | - | - | - | no: nothing it writes is read on the walk from the loss back to the trunk linears | quant |
| `add` | mlp | dtype=bf16 | 1 | yes | yes | - | - | yes | elementwise |
| `rmsnorm` | head | dtype=bf16 | 1 | yes | yes | - | - | yes | elementwise |
| `gather_rows` | head | dtype=bf16 | 1 | yes | yes | - | - | yes | linear |
| `logits_gemm` | head | dtype=bf16 | 1 | yes | yes | - | logits | yes | linear |

The int8 forward issues six ops the bf16 one does not: `rope_table`, `rmsnorm_quant_fp8`,
`had_quant_act_i8g`, `gemm_nt_q`, `gated_quant_fp8` and `quant_act_fp8`. The residual add of a
layer is fused into the following `rmsnorm_quant_fp8` (which writes both the normed `h` and the
residual `x`), so only the last layer's plain `add` survives.

Two groups are off the path, both because their writes are dead on the tape:

- `rope_table` writes the rotary cos/sin table. The `rope` kernels take that table through a raw
  parameter, so the tape has no operand edge from `rope_cs` to `rope`; and a constant table is not
  something the loss differentiates through. It is listed in the `full-attention` block because it
  prepares the rope kernels of that layer.
- `quant_act_fp8` (three in the gated-delta-net layers, one in the full-attention layer) writes
  `gdn_o.q`/`.s` and `attn_out.q`/`.s`, and nothing on the tape reads either copy. In a
  gated-delta-net layer the out-projection that follows is a bf16 `gemm_nt` reading the bf16
  `gdn_o` against `blk.<n>.ssm_out.weight`, not the i8a8h form that would read the copy. In the
  full-attention layer the out-projection is an i8a8h `gemm_nt_q` against
  `blk.3.attn_output.weight`, but it reads `attn_out.q8`/`.s8`, the copy the `had_quant_act_i8g`
  between the two quantisers writes: that op quantises the same bf16 `attn_out` again under the
  Hadamard rotation. So the fp8 copy of both outputs is written and never read, and the tape, not
  the declared graph, is what shows it.

No op in either inventory is without a family: every op the step issues maps to one of the nine
families, which the table below names in its last column, and an op that fitted none would be
printed as `none` and fail `tests/host/ops_inventory.cpp`. The family is claimed by name in one
table, `src/adjoint_family.cpp` (`R97_REGISTER_ADJOINT_FAMILY`, `src/adjoint_family.h`), so the
column is a claim the family's owner made, not a guess from the op's name. The claims sit in that
one file, and not beside each twin and adjoint, because the op files they describe belong to the
adjoint issues that wrote them and were merged in parallel: a change adds a file rather than
editing files other people are working in (`CONTRIBUTING.md`), and a new op whose backward is in
its own file may still claim its family there. `lora` is the one family no op here belongs to: no
adapter is loaded, so no `lora_*` op reaches either tape, though `lora_shrink`, `lora_expand` and
`lora_expand_mm` are claimed and their adjoint is `src/adjoint/host/lora.cpp`.

## Op to adjoint issue

The adjoint issues add one file per family under `src/adjoint/host/`, and there is now one for
each of the nine families (`add.cpp` is the worked example). The family each op belongs to (the
claims are in `src/adjoint_family.cpp`):

| op | blocks | family file |
| --- | --- | --- |
| `embed_lookup` | embedding | `linear` |
| `rmsnorm` | gated-delta-net, mlp, full-attention, head | `elementwise` |
| `add` | gated-delta-net, mlp, full-attention | `elementwise` |
| `mul` | full-attention | `elementwise` |
| `sigmoid` | full-attention | `elementwise` |
| `silu_mul` | mlp | `elementwise` |
| `gemm_nt` | gated-delta-net, mlp, full-attention | `linear` |
| `gemm_nt_q` | gated-delta-net, mlp, full-attention | `linear` |
| `logits_gemm` | head | `linear` |
| `gather_rows` | head | `linear` |
| `rmsnorm_quant_fp8` | gated-delta-net, mlp, full-attention | `quant` |
| `had_quant_act_i8g` | gated-delta-net, mlp, full-attention | `quant` |
| `gated_quant_fp8` | mlp | `quant` |
| `quant_act_fp8` | gated-delta-net, full-attention | `quant` |
| `rope` | full-attention | `qk` |
| `rope_table` | full-attention | `qk` |
| `kv_store` | full-attention | `qk` |
| `attn_paged` | full-attention | `attention` |
| `gdn_conv_prep` | gated-delta-net | `gdn_prep` |
| `gdn_gated_rmsnorm` | gated-delta-net | `gdn_prep` |
| `gdn_kkt_solve` | gated-delta-net | `gdn_kkt` |
| `gdn_chunk_scan` | gated-delta-net | `gdn_scan` |

Three readings worth stating because the mapping is not mechanical:

- `embed_lookup` and `gather_rows` are `linear`: both are row movement against a weight the
  linear walk already reaches (`embed_lookup`'s table and the head's `lm_head`), so the family that
  owns the operands owns the gather too. It is where the merged linear issue put them
  (`src/adjoint/host/linear.cpp`).
- `gdn_gated_rmsnorm` is `gdn_prep`, not `elementwise`: it is the norm inside the conv prep's
  block, and its adjoint runs with the prep's row layout. A `rmsnorm` outside a GDN layer is
  `elementwise`.
- `rmsnorm_quant_fp8` and `gated_quant_fp8` are `quant`, not `elementwise`: their adjoints share
  the quantiser's backward (the scale's cotangent and the rounding's), which is the same walk for
  every op that fuses a quantiser into a norm or an activation. `rmsnorm` without the quantiser is
  `elementwise`.
- `kv_store` is `qk`, not `attention`: it writes the paged KV pool that `attn_paged` later reads,
  and its backward sits with the rope and the gate in `src/adjoint/host/qk.cpp`
  (`docs/adjoints/qk.md`); only the paged attention itself is `attention`.

