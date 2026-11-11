# The qk family: rope, kv_store and the attention gate

This is the host adjoint note for the ops the gated full-attention block
(`arch/common/rad_block_attn_gated_fp8.h`) issues around attention. The card fuses the prologue
into `qk_norm_rope_gate`; the host runs `rmsnorm` on queries and keys, then `rope`, then
`kv_store`, and finally `sigmoid` and `mul` for the gate. The qk family owns `rope`, `kv_store`,
`sigmoid` and `mul`; the two RMS norms belong to the elementwise/norm family and are not repeated
here. `src/adjoint/host/qk.cpp` is the f64 twin and adjoint, `src/adjoint/host/qk_rows.cpp`
registers the `rope`/`kv_store` rows the host step's backward walk uses, `tests/host/adjoint_qk.cpp`
checks the family, and this file carries every formula.

All operands are f64 and row-major in the twin, with the same layouts as radiance's host kernels.
The adjoint takes the cotangent of the output and writes the cotangent of each differentiable
input. The rope table and the slot mapping are constants and have no cotangent.

## rope

### Shape and mode

A rope operand has `T` token rows. Each row is

```text
(n_q + 2 n_kv) * D
```

values: `n_q` query heads, then `n_kv` key heads, then the same number of value heads. The host
issues the q-only and k-only cases separately, which are `n_kv = 0` and `n_q = n_kv` with no
value heads present. For Qwen3.5, `D` is the attention head width and only the first `R` lanes of
each q and k head turn; the remaining `D - R` lanes are copied. `R` is even, and the frequency
ladder divides by `R` (radiance's default `rotary` ladder, not the proportional `head` ladder).
The pairing is NeoX: lane `i` pairs with lane `i + R/2`. A value head is never rotated.

Let `p_t` be the (single-component) position of token `t`, `s` the position scale parameter
(Qwen3.5 uses 1), `theta` the rope base, and `h = R/2`. The angle and the two constants are

```text
inv_i = theta^(-2i/R),        i = 0 .. h-1
a_i   = (p_t / s) * inv_i
c_i   = cos(a_i)
s_i   = sin(a_i)
```

Those constants are the `rope_table` plane: row `p` holds `[c_0 .. c_{h-1}, s_0 .. s_{h-1}]`. The
twin builds it with `rope_table_f64`, and the test checks every row against radiance's
`rope_table` kernel. The table is position-indexed and constant, so it is not differentiated.

### Forward

For each rotated head and each `i < h`, with `x_0` and `x_1` the two lanes of the NeoX pair,

```text
y_0 = c_i x_0 - s_i x_1
y_1 = s_i x_0 + c_i x_1
```

or, per pair,

```text
[y_0]   [ c_i  -s_i ] [x_0]
[y_1] = [ s_i   c_i ] [x_1]
```

Lanes `R .. D-1` and all value heads satisfy `y = x`.

### Adjoint

The adjoint of a real rotation is its transpose. With `g_0`, `g_1` the output cotangent pair,

```text
dx_0 =  c_i g_0 + s_i g_1
dx_1 = -s_i g_0 + c_i g_1
```

Lanes `R .. D-1` and value heads pass their cotangent through unchanged. There is no cotangent
for the table or the positions.

## kv_store

### Shape and addressing

The step's keys and values are `[T, H, D]`; the paged pool is

```text
[blocks, H, block_size, 2D]
```

with key lane `d` in `0 .. D-1` and value lane `d` in `D .. 2D-1`. `slot[t]` names the physical
pool row of token `t`; a negative slot is padding and writes nothing. One token row is

```text
pool[slot[t], h, d]      = k[t, h, d]
pool[slot[t], h, D + d]  = v[t, h, d]
```

for every head `h` and lane `d`.

The cotangent of the pool is addressed by **sequence and position**, not by a flat slot number.
For sequence `q` and position `j`,

```text
block  = block_table[q][j / block_size]
offset = j % block_size
slot   = block * block_size + offset
```

`kv_slot` performs that resolution; a negative block is an absent sequence. `kv_store`'s slot
mapping is exactly this address for the tokens of the current step. Consequently a cotangent of
the pool reaches only the keys and values of a token whose `(sequence, position)` the step wrote;
a pool slot outside the step's mapping, including padding, sends and receives nothing. The
attention family fills `d_pool` through the same `(sequence, position)` view, so the two families
never disagree about which pool row a token owns.

### Forward

The forward is the copy above. It touches only mapped slots; the rest of the pool is INOUT and
remains what it was.

### Adjoint

With `g_pool` the pool cotangent,

```text
dk[t, h, d] = g_pool[slot[t], h, d]          if slot[t] >= 0, else 0
dv[t, h, d] = g_pool[slot[t], h, D + d]      if slot[t] >= 0, else 0
```

If a slot is written twice it receives the sum of both rows' cotangents. The adjoint never writes
the pool cotangent; it only reads it.

## The gate

The host gate is `sigmoid` followed by an elementwise `mul`:

```text
u = sigmoid(z)
y = u * x
```

### sigmoid

Forward and adjoint, with `g` the cotangent of `u`:

```text
u_i = 1 / (1 + exp(-z_i))
dz_i = g_i * u_i * (1 - u_i)
```

The twin evaluates the positive and negative branches separately for numerical stability; that is
the same function.

### mul

With `y = x * u`, elementwise or with `u` broadcast down the rows:

```text
dx = dy * u
du = dy * x
```

When `u` is the broadcast row (`b_broadcast` in the twin, `numel(u) == n` and `rows > 1` in
radiance's `mul`), every row's `dy * x` contributes to the same `du`, so the adjoint sums over
the rows:

```text
du_i = sum_r dy[r, i] * x[r, i]
```

The attention gate is same-shaped, so the broadcast sum is not on its path; it is implemented and
tested because `mul` itself has that one broadcast rule.

## What the test checks

For each op, `tests/host/adjoint_qk.cpp`:

1. runs the f64 twin and radiance's host kernel on the same input and compares them within
   bf16 tolerance (the rope and pool cases include a bf16-stored case; the f32 cases are held to
   a tighter tolerance);
2. checks the adjoint against central differences of the twin, with a fixed random cotangent,
   to `1e-6` relative;
3. uses at least three shapes, including the tiny fixture's: rope with `T = 5`, `n_q = 4`,
   `n_kv = 2`, `D = 32`, `R = 16`, `theta = 10000`, and the host's separate q-only and k-only
   forms (`n_kv = 0`); kv_store with 5 tokens, 2 kv heads, `D = 32`, `block_size = 4`; and a gate
   over the fixture's `5 * 4 * 32` query values.

The host kernel is loaded from the build's `libref` plugin, so the comparison is against the
same host path the engine uses rather than a second copy of the formula.
