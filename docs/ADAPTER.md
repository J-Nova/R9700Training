# ADAPTER.md — the LoRA adapter contract on the rotated int8 trunk

This is the contract six parallel issues implement together: the two radiance patches (the site
filter's lifted refusals and the `radiance_input_basis` config key), the live training loop's
adapter, the adapter files on disk, the host delta encoder, and the device LoRA adjoints. It is a
document, not code. Every file reference below is into radiance at
`451b136874084650992668bed9e8662deb58de88`. The line numbers are exact at that commit; newer
revisions of radiance shift them.

Nothing here changes what the base GEMM computes. The adapter adds a separate low-rank product to
each targetable GEMM's output; the base path, its codes and its scales are untouched.

---

## 1. The formula

For a trunk linear of `qwen35_w8a8` whose declared weight carries the
`i8*bf16[1x128]/fwht128` encoding, with a LoRA module `(A, B)` of rank `r` and scale
`alpha / r`:

```
y = y_base + (alpha / r) * B * A * x
```

- `x` is the **dequantised rotated activation** the base GEMM reads: the int8 codes of the site's
  activation buffer times their per-(row, 128-of-K) group scale, living in the 128-wide Hadamard
  basis. `A` is stored in that same basis. The product `B * A * x` therefore needs **no** extra
  Hadamard: the basis the base GEMM works in is the basis the adapter works in.
- `A` is `[r, K]` bf16, `B` is `[out_rows, r]` bf16, both held at bf16 with the same round-off
  freedom any PEFT serving path has.
- `alpha / r` is PEFT's `lora_alpha / r` (`rsLoRA`, `alpha / sqrt(r)`, is not this contract's
  default; radiance carries the per-module scale either way — `core/lora/lora.h:98`,
  `AdapterInfo::Module::scale`).

Equivalently, in the natural (unrotated) weight domain the adapter is the familiar
`ΔW = (alpha / r) · B · A · H₁₂₈`, where `H₁₂₈` is the unnormalised 128-wide Hadamard the weight
encoding is stored under (`transform=fwht128`, `data/recipes/q38-27b-w8a8-df2.recipe`).

### 1.1 What `x` is, exactly, at every GEMM site of `qwen35_w8a8`

`qwen35_w8a8` is `arch/qwen35_w8a8/qwen35_w8a8.cpp` (five lines of `#define` over
`arch/qwen35_fp8/qwen35_fp8.cpp`, which carries the difference behind `#if QWEN35_W8A8`). The
builder's quantiser-to-producer wiring for the whole trunk is stated in one comment block and the
four declarations under it (`arch/qwen35_fp8/qwen35_fp8.cpp:871-889`). Every trunk linear is declared by
`LinearFP8::declare` (`arch/common/rad_fp8.h:853`): with `i8_rot` set (`rad_fp8.h:768-771`) it
issues `gemm_nt_q` at dtype **`i8a8h`** (`rad_fp8.h:1386`, `arch/common/rad_fp8.h:1991`) whose
inputs are one buffer pair (the site's activation's `ActFP8::q8` / `ActFP8::s8`, `rad_fp8.h:57-70`),
read as `{ a.q8, a.s8 }` (`rad_fp8.h:1380-1384`).

The dequantisation of `x` at all eight sites, host and card:

- `q8` is `[max_tok, n]` int8, **row-major in the activation's column order**: column `c` of a row
  sits at byte `row * n + c` on the host and on the card alike. This is `quant_act_i8g`'s /
  `had_quant_act_i8g`'s layout (`libref/ref_quant.cpp:9-11`, writer `libref/ref_quant.cpp:74ff`;
  the card's writer keeps the same order, `libr4d/r4d_quant_act_fp8.hip:37`).
- `s8` is `[max_tok, n/128]` f32, one scale per (row, 128-column block).
- `x[m, 128*b + c] = float(q8[m, 128*b + c]) * s8[m, b]`, rounded to bf16 for the adapter's use.

The sites read these buffers of these activations. The site/activation pairing is stated by the
builder's `--calib-dir` mirror, which records exactly these (codes, scale) planes per site
(`arch/qwen35_fp8/qwen35_fp8.cpp:1157-1184`, one site for `h` shared by q/k/v and in_proj_qkv/z,
one for the attention or delta-net output, one for the ffn input, one for down):

| site (declared weight)   | reads                     | codes producer |
|--------------------------|---------------------------|----------------|
| `blk.<l>.attn_qg`        | `m.a_h` (`a_h.q8`, `a_h.s8`)  | the boundary norm (`rmsnorm_had_quant_i8g`), `qwen35_fp8.cpp:879-881` |
| `blk.<l>.attn_k`, `blk.<l>.attn_v` | `m.a_h`         | same codes as `attn_qg` (`q8_fed`/`q8_rot`, one launch for q, k, v) |
| `blk.<l>.attn_output`    | `m.a_attn`                    | the gated attention's output gate (`gate_had_quant_i8g`), `qwen35_fp8.cpp:888-889` |
| `blk.<l>.ssm_inz`        | `m.a_h`                       | same codes as `attn_qg` |
| `blk.<l>.ssm_out`        | `m.a_go`                      | the delta net's conv-fold epilogue `gdn_conv_recurrent_update_had` plus `had_quant_act_i8g` on the other rows, `qwen35_fp8.cpp:885-886` |
| `blk.<l>.ffn_gate_up`    | `m.a_h` (the ffn norm's codes)| same codes as `attn_qg` — one norm feeds both the attention and the ffn (`rad_block_mlp_fp8.h:110`) |
| `blk.<l>.ffn_down`       | `m.a_ffn`                     | the ffn's gated product (`gated_had_quant_i8g`), `qwen35_fp8.cpp:882` |

Where a `_had_quant_i8g` form is not declared (an older device, the CPU kernels) the site
quantises its own input from the bf16 plane with a plain `quant_act_i8g`
(`qwen35_fp8.cpp:871-889`, `rad_fp8.h:1379-1386`); the codes then dequantise identically — the
bytes are the same rotated codes, whoever wrote them (`rad_fp8.h:71-74`).

**The step order is not in play here, and the contract keeps it that way.** `core/lora/lora.cpp:604`
notes, for `quant_act_i8` (per-row scale, dtypes `w4a8` / `w8a8` / `w2a8`), that those codes are
"quant_act_i8's codes in the int8 WMMA's step order (the middle dwords of every 16-byte step
exchanged" — the middle two dwords of each sixteen-byte step are exchanged, host and card both
(`docs/OPS.md:623-624`, `libref/ref_common.h:577-580` (`i8_step_pos`), host writer
`libavx/avx_quant.cpp:53`). That order belongs to the per-row quantiser, **not** to
`quant_act_i8g` / `had_quant_act_i8g`, which `qwen35_w8a8`'s sites use: their codes are plain
column order. `lora_shrink` reads codes by column, so an adapter over a step-order site would be
wrong — which is why radiance refuses it (see §3). The first patch lifts this refusal *by
unexchanging the middle dwords when it builds `x`* for the step-order dtypes, and the
`i8a8h` sites of this contract need only the column-order path above. What says which order a
shrink's codes are in is the encoding of the GEMM whose activation it reads, so the patch has the
core declare `lora_shrink` with a `gemm_dtype` parameter holding that GEMM's own `dtype` string
(`i8a8h`, `w8a8`, ...). `dtype` alone cannot: it is the activation's element dtype, `i8` at every
coded site.

---

## 2. The targetable linears, their PEFT names and their shapes

The recipes name the tensors `attn_qg`, `attn_k`, `attn_v`, `attn_output`, `ssm_inz`, `ssm_out`,
`ffn_gate_up`, `ffn_down` (`data/recipes/q38-27b-w8a8-df2.recipe:30-37`), exactly the eight
`block_weight` entries the plugin declares (`arch/common/rad_block_attn_gated_fp8.h:280-288, 497`,
`rad_block_gdn_fp8.h:247-252, 360-362`, `rad_block_mlp_fp8.h:112-114, 155-157`). Not part of the adapter's target
set: the norms, the delta net's `conv1d`, `A_log`, `dt_bias`, `in_proj_a`/`in_proj_b`, the
vocabulary edges (`output`/`lm_head`, a vocabulary weight, `core/lora/lora.cpp:87`), and the
MTP head and `dflash` draft blocks — the adapter's `target_modules` names exactly the ten PEFT
modules of the table below, so none of those ever enters a site.

A declared weight is targeted when **one of its name-map sources** is `<prefix>.<leaf>.weight`
with `leaf` in the target set (`core/lora/lora.h:32-33`, `core/lora/lora.cpp:147`). Targets are
checkpoint-tensor leaf names, i.e. PEFT's module names as HuggingFace spells them.

| PEFT module (`qwen35_w8a8` and `qwen35_bf16` alike)      | declared weight it lands in | weight shape at 27B |
|---|---|---|
| `self_attn.q_proj`          | `blk.<l>.attn_qg`   | `[12288, 5120]` — query **and** its gate, `docs/ARCHITECTURES.md:302` |
| `self_attn.k_proj`          | `blk.<l>.attn_k`    | `[1024, 5120]` |
| `self_attn.v_proj`          | `blk.<l>.attn_v`    | `[1024, 5120]` |
| `self_attn.o_proj`          | `blk.<l>.attn_output` | `[5120, 6144]` |
| `linear_attn.in_proj_qkv`   | `blk.<l>.ssm_inz`   | rows `[0, 10240)` of `[16384, 5120]` |
| `linear_attn.in_proj_z`     | `blk.<l>.ssm_inz`   | rows `[10240, 16384)` |
| `linear_attn.out_proj`      | `blk.<l>.ssm_out`   | `[5120, 6144]` |
| `mlp.gate_proj`             | `blk.<l>.ffn_gate_up` | rows `[0, 17408)` of `[34816, 5120]` |
| `mlp.up_proj`               | `blk.<l>.ffn_gate_up` | rows `[17408, 34816)` |
| `mlp.down_proj`             | `blk.<l>.ffn_down`  | `[5120, 17408]` |

### 2.1 Fused weights

Three declared weights hold more than one PEFT module, each module owning a **contiguous run of the
weight's rows**, in the order the plugin concatenated them:

- `ffn_gate_up` holds PEFT's `gate_proj` and `up_proj` (`arch/common/rad_block_mlp.h:52`,
  `map_concat { gate, up }`): gate rows `[0, 17408)`, up rows `[17408, 34816)`.
- `ssm_inz` holds `in_proj_qkv` (rows `[0, 10240)`, which is itself `[q|k|v]` =
  `[k_dim | k_dim | v_dim]` = `[2048 | 2048 | 6144]`: q and k 16 heads × 128 each, v 48 heads × 128)
  and `in_proj_z` (rows `[10240, 16384)`); the declared weight's four row parts are
  `{ k_dim, k_dim, v_dim, v_dim }` (`arch/common/rad_block_gdn_fp8.h:243-252`).
- `attn_qg` is **not** split for PEFT: the checkpoint's `q_proj` tensor already carries query and
  gate interleaved per head (`docs/ARCHITECTURES.md:302, 330-337`), so `q_proj` is one module
  owning all 12288 rows of `attn_qg`. Targeting `q_proj` adapts query and gate together; a
  gate-only or query-only adapter is not expressible in the module set (the rows are strided,
  head pitch 512, not contiguous runs, and `SitePart` is runs-based, `core/lora/lora.h:35-44`).

Radiance keeps one `Site` per declared weight, with `n_modules = |targeted sources|`, capacity
`R = n_modules * max_rank`, and one t-offset slot per module in the sources' order
(`core/lora/lora.h`, struct `Site`; `SitePart::t_off = slot * max_rank`). Concretely: `A` is
`[R, K_r]` (each module's `[r, K_r]` block at its slot), `B` is `[N_r, R]` with each module's rows
put only on its own slot's columns (`core/lora/lora.h`, struct `PackedSite`; packing
`build_sites` / `load_adapter` in `core/lora/lora.cpp`). The device adjoint must therefore treat
`R`-wide `A`/`B` buffers with zero regions, not one dense `[r, ·]` pair per site.

---

## 3. The two refusals the patches lift, and the ones that stay

Both are in `source_checks` in `core/lora/lora.cpp` (`find_unsupported_uses` runs it per site).

**Lifted, refusal 1 — `lora.cpp:608-610`** (the step-order dtype check, §1.1):

```c
if (dt && (!std::strcmp(dt, "w4a8") || !std::strcmp(dt, "w8a8") || !std::strcmp(dt, "w2a8")))
    return no(fmt("%s at %s reads quant_act_i8's codes in the int8 WMMA's step order, not "
                  "in the activation's column order", op.op.c_str(), dt));
```

**Lifted, refusal 2 — `lora.cpp:619-621`** (the transform-basis check, the one that blocks the
whole `fwht128` trunk today):

```c
if (w.enc_known && w.enc.transform[0])
    return no(fmt("weight '%s' is stored in the '%.*s' transform basis, so its GEMM is fed a "
                  "transformed activation", w.name.c_str(), (int)RAD_ENC_STR, w.enc.transform));
```

The patch lifts them for sites that satisfy the conditions of §1 and §4: the adapter declares
`"radiance_input_basis": "fwht128"` and every targeted site is an `i8a8h` (or, after patch 1,
step-order) `gemm_nt_q`. Lifting refusal 1 is the shrink's business: what order a row's codes are in
is the reading GEMM's encoding, so the patch declares `lora_shrink` with a `gemm_dtype` (§1.1) and
the rows that can read the step order exchange it back. `libref`'s and `libavx`'s rows do;
`libr4d`'s and `libhalo`'s rows read a row's codes by column, so they refuse a step-order reading by
name (`RAD_E_UNSUPPORTED`) rather than fold the codes of the wrong columns. The `i8a8h` sites of
this trunk read column order, and run on both. **Staying** (no patch touches these):

- the rotated-parts refusal `lora.cpp:599-601` — `gemm_nt_q` over rotated parts (`pb1`/`pb2`),
  which is the IQR-style decode, not an adapter site;
- the K-width check `lora.cpp:615-617` — a `gemm_nt_q` whose `K` is wider than the weight's
  columns is a widened, rotated input the adapter frame does not describe;
- `excluded_weight` (`lora.cpp:86-91`): vocabulary weights and anything that is not a rank-2
  weight; the draft (`dflash.`) and MTP weights are simply never named by a target set.
- non-`LORA` adapters: the PEFT-type check `lora.cpp:821-825` and the suffix table of
  `core/lora/adapter.cpp:230-263` (DoRA, LoHa, LoKr, ...), unchanged.

---

## 4. The adapter files and the config key

The directory keeps PEFT's layout and module names, so radiance's loader stays in use unchanged
(`core/lora/lora.cpp:805` and `:913`):

```
adapter/
  adapter_config.json
  adapter_model.safetensors
```

`adapter_config.json` is read for `target_modules` (`core/engine_lora.cpp:32-36`), `r`,
`lora_alpha`, `rank_pattern`/`alpha_pattern`, `use_rslora`, `base_model_name_or_path`
(`core/lora/adapter.cpp`, `AdapterInfo`). One **extra key** marks the rotated trunk:

```json
"radiance_input_basis": "fwht128"
```

- The value names the Hadamard the weights were rotated with (`transform=fwht128` in the recipe,
  weight encoding `i8*bf16[1x128]/fwht128`). Its presence tells radiance: the `A` factors of this
  adapter are stored in the site's transformed-input basis, per §1, and the site filter's
  transform-basis refusal (`lora.cpp:619-621`) may be lifted for these sites.
- **Without the key, radiance refuses the adapter on a transformed trunk, as today** — the
  refusal stays exactly as quoted in §3. The key on a non-transformed trunk (bf16, plain `i8a8`)
  must also be refused: the file claims a basis the sites do not have. Both checks live in the
  site-filter patch, and they run for an adapter loaded after startup too: `Engine::lora_load`
  and `lora_set_weights` end in `lora::check_placed`, which runs the site filter again with the
  loaded adapter's basis, so a keyless adapter is refused there with the same message instead of
  being placed on the rotated trunk.

Defaults the trainer writes (PEFT spellings): `r: 16` — radiance's own
`--max-lora-rank` default is 16 (`core/lora/config.h:31`, `kDefaultRank`), `lora_alpha: 32` (so every site's scale is `32/16 = 2.0`),
`A` random (kaiming-uniform on the input side, as PEFT initialises it), `B` zero so the served
model is unchanged at load. In a `ParamSet` (the live training loop's parameter table) the
parameters are named `<module>.lora_A` and `<module>.lora_B`, with `<module>` the PEFT module
path of §2, e.g.:

```
model.layers.0.self_attn.q_proj.lora_A      [16, 5120]   bf16
model.layers.0.self_attn.q_proj.lora_B      [12288, 16]  bf16
model.layers.0.mlp.gate_proj.lora_A         [16, 5120]
model.layers.0.mlp.gate_proj.lora_B         [17408, 16]
...
```

(The safetensors on disk keep PEFT's full spellings,
`base_model.model.model.layers.<l>.self_attn.q_proj.lora_A.weight`, which radiance's
`AdapterFile` suffix table reads, `core/lora/adapter.cpp:230-263`.)

### 4.1 The same formula on a bf16 trunk

On `qwen35_bf16` (`arch/qwen35_bf16/qwen35_bf16.cpp`) — a separate plugin composing the same
blocks against bf16 GEMMs (`arch/common/rad_block_attn_gated.h`, `rad_block_gdn.h`,
`rad_block_mlp.h`) — every site is a plain `gemm_nt`: no quantiser op at all, so `x` is
**the plain bf16 activation** in the
natural basis, no codes and no group scales. The formula is unchanged:

```
y = y_base + (alpha / r) * B * A * x_natural
```

`A` is in the natural basis, which is the only basis there is, so `"radiance_input_basis"` is
**absent** — the key's presence on a bf16 trunk is refused (§4). This is the case radiance serves
today: the bf16 trunk passes both site filters untouched (dtype is not `w4a8`/`w8a8`/`w2a8`, the
encodings carry no `transform`), and every module name, fused weight and packing rule of §2
is the same.

---

## 5. `Delta::to_wire`: the site layout on `include/r97/delta.h`

`Delta::to_wire` packs a `ParamSet` into the sites of `WireBuffers` (`include/r97/delta.h`, the
trainer's header). The layout the issue sketch proposes — **one site per targetable linear, with
its module name, rank, scale, `A` and `B`** — cannot express the fused weights of §2.1: a site of
`ffn_gate_up` has two modules with two independent `(A, B)` pairs, and a site of `ssm_inz` has two
modules at disjoint row runs. One `(module, rank, scale, A, B)` per site would silently drop one
of the two.

The layout that can — and the one this contract fixes; `include/r97/delta.h` **changes** with a
note at the top of this pull request, per the issue — is **one site per targeted PEFT module**:

```
struct Site {                 // delta.h site record, one per <module> of §2
  const char* weight_name;    // the declared weight it lands in: "blk.5.ffn_gate_up"
  int32_t     slot;           // index among the weight's targeted sources, in the order
                              // the plugin concatenates them (§2.1)
  const char* module;         // the PEFT module path: "model.layers.5.mlp.gate_proj"
  int32_t     rank;           // r of this module
  float       scale;          // alpha / r of this module
  bf16  A[];                  // [rank, K]   the module's own factor
  bf16  B[];                  // [module_rows, rank]
};
```

- `weight_name` + `slot` is the address: radiance's `Site::sources` holds the targeted
  checkpoint tensors in declaration order, and `SitePart::t_off = slot * max_rank` places the
  module's `A` rows and `B` columns (`core/lora/lora.h:49-56`). Two records of the same
  `weight_name` (`ffn_gate_up`, `ssm_inz`) are merged by the host loader into one radiance site,
  one per module, in slot order.
- `A`/`B` may be sent full-size; the loader slices to the rank's `[K_r]` columns and `[N_r]` rows
  (`Site::K_r / k_lo / N_r`, `core/lora/lora.h`, struct `Site`). The row-shard weights
  (`attn_qg`, `attn_k`, `attn_v`, `ssm_inz`, `ffn_gate_up` are `RAD_SHARD_ROW`,
  `rad_block_attn_gated_fp8.h:280-285`, `rad_block_gdn_fp8.h:247-249`, `rad_block_mlp_fp8.h:113`)
  slice `A` by K and `B` by its rows; the column-shard ones (`attn_output`, `ssm_out`, `ffn_down`)
  slice `A` by its columns. The wire record is the whole module's factors either way, a per-rank
  slice of it, so the layout needs no world knowledge.
- Everything a module needs is per-module (`rank`, `scale`), because rank patterns and
  `alpha_pattern` make them differ (`core/lora/adapter.cpp`, `AdapterInfo::Module`).

If `include/r97/delta.h`'s `ParamSet` keys carry the fused weight only once (one entry per
*declared weight*), `to_wire` must emit two sites for it instead — the merge above is then
implicit in `to_wire`'s output, and the header keeps a single `ParamSet` key per weight with the
module list inline. Either form expresses the site; what may **not** change is the one-site-per-
linear, one-`(A,B)`-per-site shape the sketch implies.

---

## 6. Worked example: Qwen3.8-27B, one full layer and one delta layer, rank 16

Qwen3.8-27B (`docs/ARCHITECTURES.md:260-270`): 64 layers, `full_attention_interval 4` — layers
3, 7, …, 63 are gated attention (16), the other 48 are the gated delta net. `n_embd = 5120`,
`n_ff = 17408`, head_dim 256, 24 attention heads, 4 KV heads (GQA 6, `docs/ARCHITECTURES.md:266`); the delta net has 16 key
heads and 48 value heads of 128 (`k_dim = 2048`, `v_dim = 6144`, `docs/ARCHITECTURES.md:357-360`). Defaults of §4: rank 16, alpha 32,
`alpha / r = 2.0`, `A` random bf16, `B` zero.

### Full-attention layer `model.layers.3` (the base weight's shape first)

| module | site | A | B | parameters |
|---|---|---|---|---|
| `self_attn.q_proj`   | `blk.3.attn_qg`     | `[16, 5120]` | `[12288, 16]` | 81,920 + 196,608 = 278,528 |
| `self_attn.k_proj`   | `blk.3.attn_k`      | `[16, 5120]` | `[1024, 16]`  | 81,920 + 16,384 = 98,304 |
| `self_attn.v_proj`   | `blk.3.attn_v`      | `[16, 5120]` | `[1024, 16]`  | 81,920 + 16,384 = 98,304 |
| `self_attn.o_proj`   | `blk.3.attn_output` | `[16, 6144]` | `[5120, 16]`  | 98,304 + 81,920 = 180,224 |
| `mlp.gate_proj`      | `blk.3.ffn_gate_up` | `[16, 5120]` | `[17408, 16]` | 81,920 + 278,528 = 360,448 |
| `mlp.up_proj`        | `blk.3.ffn_gate_up` | `[16, 5120]` | `[17408, 16]` | 360,448 |
| `mlp.down_proj`      | `blk.3.ffn_down`    | `[16, 17408]` | `[5120, 16]`  | 278,528 + 81,920 = 360,448 |

Layer total: 1,736,704. Over the 16 full-attention layers: 27,787,264.

### Gated-delta-net layer `model.layers.1`

| module | site | A | B | parameters |
|---|---|---|---|---|
| `linear_attn.in_proj_qkv` | `blk.1.ssm_inz` | `[16, 5120]` | `[10240, 16]` | 81,920 + 163,840 = 245,760 |
| `linear_attn.in_proj_z`   | `blk.1.ssm_inz` | `[16, 5120]` | `[6144, 16]`  | 81,920 + 98,304 = 180,224 |
| `linear_attn.out_proj`    | `blk.1.ssm_out` | `[16, 6144]` | `[5120, 16]`  | 98,304 + 81,920 = 180,224 |
| `mlp.gate_proj` / `up_proj` / `down_proj` | as layer 3 | | | 1,081,344 |

Layer total: 1,687,552. Over the 48 delta-net layers: 81,002,496.

### Adapter total, all layers all modules, rank 16

**108,789,760 parameters** — 27,787,264 on the attention layers, 81,002,496 on the delta-net
layers — 218 MB (207.5 MiB) of bf16 per copy; held twice at training time (base copy + gradient),
435 MB of fp32 if the trainer keeps a master. At training time the site's `R` for a layer with
two targeted modules (`ssm_inz`, `ffn_gate_up`) is `2 * 16 = 32`, so the device adjoint sees
`A` `[32, 5120]`-shaped slices and `B` `[N_r, 32]`-shaped slices with one module's block live per
slot, exactly as `PackedSite` packs them (`core/lora/lora.h`).

## 7. ops

A step with a live adapter still runs the base GEMMs; the adapter is two ops around each targeted
GEMM, issued by radiance's `core/lora/declare.cpp`:

- `lora_shrink` reads the GEMM's own activation operand(s) and multiplies by the slot's A factors.
  A fused weight may need one per row part (for example `in_proj_qkv`'s three parts).
- `lora_expand` multiplies by the slot's B factors and adds the result to the GEMM's output `y`.

Both are present at every targeted site, and both are issued by a step that carries the adapter:
`tests/host/live_adapter.cpp` fills the slot from memory, runs a prefill, and checks the step's
tape (`Program::ops[i].issued`) for one issued `lora_shrink` and one issued `lora_expand` per
entry in `Program::lora_sites`. A step that never names the slot leaves both unissued, so the
check is what holds this section to a step rather than to the declaration alone. The int8 trunk
follows with patch `0210` and the same two ops.
