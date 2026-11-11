# Upstream: the patches this tree carries

r97 builds radiance from the pinned submodule at `third_party/radiance`
(`451b136874084650992668bed9e8662deb58de88`) and applies the patches under `patches/serving/` and
`patches/training/` to a copy in the build tree. Each patch is one concern and is written to apply
alone, so it can be submitted upstream as it is; this file is the ledger that keeps them small.

`ci/patch_audit.sh` reads this table and every patch: the patch must begin with a `Concern:` and an
`Upstream:` line, must have a row here whose concern, status and size agree with the patch, and
must not cost more than its budget. It also fails when a serving patch touches a file a training
patch introduces, because `-DR97_PATCH_SET=serving` must apply only the serving set.

## Sets

- **serving** (`02xx`): what a radiance needs to serve an adapter trained here. It stands alone on
  the pinned radiance and is what goes upstream.
- **training** (`01xx`): runtime hooks only the trainer needs. A patch that only the trainer needs
  never goes into the serving set.

## Budgets

A patch's size is the lines it adds and removes, summed over its files (`git apply --numstat`); its
budget is a hard cap on that sum. The budget is the size at the review that last changed the patch, plus
about a tenth of headroom: room to fix a bug, not a target to fill. The small patches get a little
more, where one or two lines are a large fraction of the size. A patch that must grow
raises its budget in the same pull request, with the reason in the row. Prose above the diff (the
two header lines and any explanation) is free; only the diff counts.

`-DR97_PATCH_SKIP=<number>` leaves out every patch whose number (the leading digits of its file
name) is `<number>`, which is how a build proves that a patch is optional. A build with
`-DR97_PATCH_SKIP=0102` has the issue observer but no layer replay; the host tier skips the two
replay tests by name. See `docs/reports/2026-11-11-patch-audit.md`.

## The patches

| Patch | Set | Added | Removed | Budget | Upstream | Concern |
| --- | --- | --- | --- | --- | --- | --- |
| `patches/serving/0201-lora-set-weights.patch` | serving | 75 | 0 | 85 | not-submitted | put an adapter slot in service from factors already in memory, not only from a directory |
| `patches/serving/0201-narrow-head-gate.patch` | serving | 6 | 3 | 15 | not-submitted | an int8 attention gate on a head narrower than the 128-column quantiser group |
| `patches/serving/0210-lora-int8-trunk-host.patch` | serving | 290 | 28 | 325 | not-submitted | a LoRA adapter on the rotated int8 trunk, through the host kernels |
| `patches/training/0101-issue-observer.patch` | training | 154 | 0 | 170 | not-submitted | an optional issue observer and layer block brackets on the runtime context |
| `patches/training/0102-replay-block.patch` | training | 114 | 19 | 150 | not-submitted | re-running one block of a step from a restored residual, after its ops have run |
| `patches/training/0103-prefill-logits.patch` | training | 201 | 20 | 240 | not-submitted | a training-fixture prefill that keeps every position's logits, driven through the engine |
| `patches/training/0104-prefill-lora.patch` | training | 16 | 0 | 25 | not-submitted | let the fixture prefill run one named adapter slot, so an in-memory update is one step away |
| `patches/training/0105-prefill-not-sampled.patch` | training | 38 | 8 | 55 | not-submitted | a prefill that draws no token -- no sampler op, and a prompt of exactly the context fits |
| `patches/training/0119-issue-observer-hook.patch` | training | 12 | 0 | 20 | not-submitted | the training tape attaches to a RadCtx, and the Engine is the only owner of it |
| `patches/training/0120-view-operands.patch` | training | 21 | 0 | 25 | not-submitted | the issue record names a narrowing view but not where its values are |

Every patch is `not-submitted` at this pin. None is `merged`: none is in radiance's `origin/main`.
Moving one to `open` or `merged` changes only its own row and `Upstream:` line, in the pull request
that submits it.

## Why the runtime hooks are still patches

radiance's plugin documentation at the pin (`docs/PLUGIN.md`) offers two places a plugin can attach:
optional kernel hooks per op (§4) and a new op defined by a schema (§7); §13 covers building,
installing and the search path for a plugin library. `abi/rad_plugin.h` has no callback on
`rad_issue`, and no header in `abi/` lets a caller read the ops an architecture issued. An observer,
a block bracket and a replay block are therefore not expressible through the plugin ABI, and the
patches that add them stay, deliberately small and generic. The review that found this, and
what it removed from each patch, is in `docs/reports/2026-11-11-patch-audit.md`.

- `0101` is the one hook all three consumers share: one optional observer on the context, called
  once per issue after the op has run, plus a pair of block brackets. It names nothing of this
  repository; its types are radiance's `Rad*` and its comment is written as a candidate for
  upstream. The tape, the boundary recorder and the replay all read it.
- `0102` adds only what the observer cannot give: a way to run one recorded block alone, plus the
  architecture's agreement to honour it. It adds no observer of its own.
- `0201` is serving: radiance's service API loads an adapter from a directory, so a slot from
  factors already in memory needs an entry point. It reuses the slot and the swap the engine
  already has.
