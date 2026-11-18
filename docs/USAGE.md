# USAGE — running `r97train`

`r97train` is the trainer's one binary: a subcommand per job, each registered in
`tools/commands.h`. This file is the user's walk through it — the build, what a run needs, every
subcommand, the configuration schema — and where each part is documented in depth:

| what | where |
| --- | --- |
| what this project is, and the pinned engine | `README.md`, `docs/UPSTREAM.md` |
| the training loop, its log and its horizons | `docs/TRAINING.md` |
| a checkpoint on disk, and resuming one | `docs/CHECKPOINT.md` |
| operating a real run: preflight, launch, failures | `docs/RUNBOOK.md` |
| the loss and the held-out measurement | `docs/EVAL.md` |
| the training sequences | `docs/DATA.md`, `docs/CHATDATA.md` |
| what an adapter is, and how radiance serves it | `docs/ADAPTER.md` |
| the tiny containers the tests run on | `docs/FIXTURES.md` |
| the three test labels and what each may assume | `docs/TESTING.md` |

## Build, and where the binary is

```sh
git submodule update --init          # fresh clone only
cmake -B build -G Ninja              # HIP from /opt/rocm, target gfx1201
cmake --build build                  # writes build/r97train
```

A host-only build — no card, no ROCm — is the same tree with `-DR97_WITH_HIP=OFF`. The engine is a
pinned submodule; the build copies it, applies `patches/` onto the copy and builds that, so nothing
under `third_party/radiance/` is ever edited. Which patches a build gets is `docs/UPSTREAM.md`'s
subject, not this file's.

The binary finds radiance's plugins (`libref`, `libavx`, …) through `RADIANCE_HOME`, else beside
itself in `<build>/radiance-build/radiance_home`; `docs/FIXTURES.md` says how the build puts them
there. A command that needs them says so, with the directory it looked in.

## What a run needs

- **A container**: a `.rad` file the engine serves. `r97train info` reads one without a card. The
  tests make theirs from the tiny checkpoint (`docs/FIXTURES.md`); a real run uses the 27B container
  under `data/`.
- **Training sequences**: one chat-format JSONL file, tokenised with the container's own chat
  template. `docs/CHATDATA.md` describes the format, what is scored and what is skipped.
- **An output directory**: the run's log and its checkpoints go here, and it must be writable
  before the first step.

## The commands

| command | what it does |
| --- | --- |
| `config` | the configuration schema, or the configuration those arguments resolve to |
| `data` | the tokenisation statistics, and the first N conversations with their scored runs marked |
| `deltas` | the registered trainable kinds, one per line |
| `grad` | one loss and its per-module gradient norms through the host adjoints |
| `info` | architecture, quantisation, layers, hidden size |
| `loss` | the held-out loss, cross-checked against radiance's own KL record |
| `objectives` | the registered losses, one per line |
| `ops` | the op inventory of a container's forward pass, or the registry |
| `train` | the training loop, with the host step or, under `--reference`, the reference step |

A command returns `0` on success, `2` for a usage error, and something other than `0` for a check
it could not complete; the message names the file, the key or the parameters at fault. `r97train`
with no argument prints the command table above; `r97train --help` prints it as an answer rather
than an error; `r97train <command> --help` prints that command's usage line — every flag it takes —
and points here. The sections below open with that same line, so the document and the binary carry
one string: `tests/cmake/config_schema_test.sh` compares them, and every flag listed under them,
with what the binary prints. A flag on one side and not the other fails `ctest --test-dir build -L
host`.

## Every command's flags

A row is either a flag this build has now, or a flag of an issue that has not merged. A **planned**
row is not in `--help` and not accepted by the binary: it is here so that a reader knows which
command the issue lands on and what the flag will be called. It carries the issue's title and the
issue's tracker number, `(#<n>)` — a number on its own says nothing — and `docs/PLAN.md` cites its
own list of positions as `R97 #<n>`, so the two cannot be mistaken for each other. `--config <file>`
and `--config-file <file>` are the same flag under two spellings; both are kept because a file path
reads better as the last word of a long line.

### `config`

```text
usage: r97train config [--config <file>] [--config-file <file>] [configuration keys as flags]
```

| flag | state | what it does |
| --- | --- | --- |
| `--config <file>` | now | read the file's `key = value` lines before anything else |
| `--config-file <file>` | now | the same flag |

Every key of the table under [Configuration](#configuration) is a flag too, `--<key> <value>` —
`--lr 1e-4` — which is what the usage line's `[configuration keys as flags]` says; the table, not
this one, is where they are listed.

With no argument the command prints the configuration schema — the table below, from the binary, so
a reader can check the document against what the build accepts. With `--config` or any
configuration flag it loads the file, applies the flags over it, validates the result and prints
one `key = value` per line: what a `train` with those arguments will use. A key the file leaves
alone keeps its `config.h` default, and an unknown key or an out-of-range value stops it with the
key's name. It is the preflight `docs/RUNBOOK.md` runs before a long run.

`config` applies no fallbacks of its own. `train` has four — `lr = 1e-3`, `steps = 1`,
`token_budget = 2048`, `accumulation = 1` — so a run may leave those out of its file and still
start, while `r97train config --config` on the same file refuses it: their `config.h` default is
outside the valid range. Write them in the file and the two agree.

### `data`

```text
usage: r97train data --model <file.rad> --data <file.jsonl> [--show N] [--kld-corpus <file>]
```

| flag | state | what it does |
| --- | --- | --- |
| `--model <file.rad>` | now | the container whose chat template and vocabulary tokenise the file |
| `--data <file.jsonl>` | now | the chat-format file to read |
| `--show N` | now | print the first N conversations, the scored runs marked `[[…]]`; default 0 |
| `--kld-corpus <file>` | now | also write the file as radiance's KL corpus, one document per scored run, and report its `kld documents` and `kld rows` |

The cheap read of a run's input, beside `info`. It prints the counts the batcher will see —
`conversations`, `kept`, `shortened`, `skipped`, `tokens`, `scored` — and `--show` prints the
conversations with what the loss is taken over marked. A `skipped` count above zero is a template or
format problem, not something to train through: `docs/CHATDATA.md` says which. `--kld-corpus`
writes the same scored runs in the shape radiance's KL mode reads, so `tools/kld_loss.py` prints the
same loss radiance serves: `docs/EVAL.md` holds the two numbers against each other, and
`docs/RUNBOOK.md` §7 runs it on a trained adapter.

### `deltas`

```text
usage: r97train deltas
```

Takes no flags. Prints one line per registered trainable kind, sorted; `docs/OBJECTIVES.md` and the
`include/r97/delta.h` contract say what a new one is and where it registers.

### `dump-weight`

```text
usage: r97train dump-weight <file.rad> <name> [<name>...]
```

Reads the container as a file — no engine, no kernel library, no card — and prints each named
weight's rank, shape, encoding and planes and, for a one-element-a-value plane, its first eight
values with the whole tensor's L1 and sum. A name that misses lists the directory entries containing
it, so a spelling the tape uses and the container does not is found rather than read as an absent
weight; a code plane is shown as its raw codes, never as values, so a quantised weight cannot be
mistaken for a plain one. Several names may be given at once, which is how a walk operand is held to
the container's own tensor — the check that settled whether the fused norm's gain was the weight the
container holds.

### `grad`

```text
usage: r97train grad --model <file.rad> --data <file.jsonl> [--delta lora] [--rank N] [--alpha A] [--limit N]
```

| flag | state | what it does |
| --- | --- | --- |
| `--model <file.rad>` | now | the container |
| `--data <file.jsonl>` | now | the sequences the loss is taken over |
| `--delta lora` | now | the trainable kind to attach; `deltas` lists them |
| `--rank N` | now | the factors' rank |
| `--alpha A` | now | the LoRA scale, applied as `alpha / rank` |
| `--limit N` | now | stop after N sequences |

One loss and its per-module gradient norms through the host adjoints — the check that a step's
backward is wired where it should be. `docs/STEPS-HOST.md` owns what it prints.

### `info`

```text
usage: r97train info <file.rad>
```

Takes no flags. Prints the container's `arch`, `quant`, `layers` and `hidden` and exits `3` when it
does not open: the one command that needs neither data nor a card, and the first one to run on a new
container.

### `loss`

```text
usage: r97train loss --model <file.rad> --data <file.jsonl> [--lora <dir>] [--limit N] [--device N]
```

| flag | state | what it does |
| --- | --- | --- |
| `--model <file.rad>` | now | the container |
| `--data <file.jsonl>` | now | the held-out sequences to score |
| `--lora <dir>` | now | an `adapter/` directory, served through radiance's own loader |
| `--limit N` | now | stop after N sequences |
| `--device N` | now | score on the card, through `ROCR_VISIBLE_DEVICES` |

The held-out loss: `--lora` on a checkpoint's `adapter/` loads it through the same radiance loader
a serving radiance uses. `docs/RUNBOOK.md` section 7 holds it to the trainer's held-out loss and
then serves the adapter with the serving-only build, whose own KL record of the same adapter is held
to the same number; `docs/EVAL.md` has both measurements and how they agree.

### `objectives`

```text
usage: r97train objectives
```

Takes no flags. Prints one line per registered loss, sorted; `docs/OBJECTIVES.md` says what a new
one is and where it registers.

### `ops`

```text
usage: r97train ops <file.rad> or --registered [--home <dir>] [--kernels <list>] [--record <file>] [--batch <file>] [--device N]
```

| flag | state | what it does |
| --- | --- | --- |
| `--registered` | now | print the registry instead of a container's op graph |
| `--home <dir>` | now | the radiance home to take the kernel libraries from |
| `--kernels <list>` | now | the kernel libraries to try, in the order given |
| `--record <file>` | now | record the pass's inputs and outputs to `<file>` |
| `--batch <file>` | now | replay a recorded pass instead of fresh inputs |
| `--device N` | now | take the inventory on the card |

Answers "can this build run this container, on the host and on the card": the inventory the host
tier prints is `docs/OPS-qwen35-host.md`, the card's is `docs/OPS-qwen35-device.md`, and
`docs/RUNBOOK.md` section 2 runs both forms (`R97_DEVICE` names the card) and section 10 says when
to ask it.

### `train`

```text
usage: r97train train --model <file.rad> --out <dir> [--data <file.jsonl>] [--heldout <file.jsonl>] [--steps N] [--reference] [--dry-run] [--resume <dir>] [--program <file>] [--config <file>] [--config-file <file>] [configuration keys as flags]
```

| flag | state | what it does |
| --- | --- | --- |
| `--model <file.rad>` | now | the `model` key: the container to train against |
| `--out <dir>` | now | the `out` key: where the log and the checkpoints go |
| `--data <file.jsonl>` | now | the `data` key: the training sequences |
| `--heldout <file.jsonl>` | now | the `heldout` key: what `eval_every` scores |
| `--steps N` | now | the `steps` key: the run's horizon |
| `--reference` | now | the reference step instead of the host step: the op graph in `f64`, no card, slower |
| `--dry-run` | now | resolve everything to the first step, print the settings and the first step's shape, write nothing |
| `--resume <dir>` | now | continue from a checkpoint directory, at the horizon the configuration names |
| `--program <file>` | now | the op program a reference step replays |
| `--config <file>` | now | read the configuration file first; a flag overrides it |
| `--config-file <file>` | now | the same flag |
| `--devices <N>` | planned: Two cards, data-parallel: the whole container on each card, adapter gradients summed every step (#57) | one sub-batch per card |
| `--placement <spec>` | planned: Second placement: a layer split across the two cards, for containers that do not fit one card (#59) | which layers live on which card |

```sh
build/r97train train --model <file.rad> --data <train.jsonl> --heldout <heldout.jsonl> \
    --out runs/lora-v1 --steps 1000 --lr 1e-4 --token_budget 4096 --eval_every 100
```

The step is the ordinary host step: the engine's forward on the host kernels and the host adjoints
(`docs/STEPS-HOST.md`). It needs no card, and it is the one a real run uses. `--reference` replaces
it with the container's op graph replayed in `f64` from the engine's twins and the host adjoints,
with the `libavx` and `libref` kernel libraries: no card, slower, and the same loop, checkpoint and
resume. A build without radiance's forward tier has only the reference step, and `train` without
`--reference` stops and says so.

Every configuration key is accepted as a flag too, as `--key value`. The file is read first and a
flag overrides it. Without `--data` the sequences are `tests/fixtures/chat_small.jsonl`, relative to
the working directory — give a real run its own file. `heldout` (or `--heldout <file>`) names the
file `eval_every` evaluates on; `eval_every = 0` turns the evaluation off. The loop logs to
`<out>/train.jsonl` and to the terminal, writes checkpoints under `<out>/`, and its last line is
`train: wrote <checkpoint>`.

`--dry-run` is the preflight a long run gets: it opens the tokenizer, reads both files, builds the
delta and the step's engine, prints the resolved configuration and one line saying what the first
step would read, and stops before the step. It writes nothing — `<out>` is made by the run — so a
mistyped path or a key out of range costs seconds instead of a night. It refuses `--resume`: a
resume starts a run, and a dry run does not.

`docs/TRAINING.md` says what the loop does with the log, the horizons and the evaluation;
`docs/CHECKPOINT.md` says what the directory it wrote contains and what `--resume` needs;
`docs/RUNBOOK.md` runs the whole procedure this section's flags serve.

## Configuration

One `key = value` per line. A line whose first non-space character is `#` is a comment, blank
lines are ignored, and a value is everything after the first `=` with the ends trimmed (so a path
may contain spaces; `#` is a comment only at the start of a line). Targets are a comma-separated
list.

A flag is `--key=value` or `--key value`, with the same key names. `--config` is the command
line's own flag, not a configuration key.

| key | type | default (`config.h`) | valid range and meaning |
| --- | --- | --- | --- |
| `model` | string | empty | the container to train against; the loop does not read it |
| `data` | string | empty | the training sequences; the loop does not read it |
| `heldout` | string | empty | the held-out sequences; the caller turns them into `TrainerOptions::heldout` |
| `out` | string | empty | **required**: the output directory for the log and checkpoints |
| `objective` | string | `ce` | **non-empty**; the name is the `StepFn`'s business, the loop only carries it |
| `base_id` | string | empty | opaque caller identifier stored in every checkpoint; `resume` refuses a different one |
| `delta.kind` | string | `lora` | **non-empty**; the delta's name, not read by the loop |
| `delta.rank` | integer | 16 | **> 0**: the rank of the trainable factors |
| `delta.alpha` | number | 32 | **finite, > 0**: the LoRA scale, applied as `alpha / rank` |
| `delta.targets` | list | empty | comma-separated non-empty module names; empty means the delta's default |
| `delta.gate` | boolean | false | `true`/`false` (also `1`/`0`, `yes`/`no`, `on`/`off`): a gate beside the sites; no delta builds one yet |
| `lr` | number | 0 | **finite, > 0**: the peak learning rate; must be set |
| `weight_decay` | number | 0 | finite, >= 0: the optimiser's weight decay |
| `warmup_steps` | integer | 0 | >= 0, counted in optimiser updates |
| `token_budget` | integer | 0 | **> 0**: the batcher's tokens per sub-batch; must be set |
| `accumulation` | integer | 0 | **>= 1**: micro-steps per optimiser update; must be set |
| `steps` | integer | 0 | >= 0: the loop-step horizon |
| `epochs` | integer | 0 | >= 0: the epoch horizon |
| `seed` | integer | 0 | >= 0: the batcher shuffle and the loop's random state |
| `eval_every` | integer | 0 | >= 0: evaluate after every N steps; 0 disables evaluation |
| `checkpoint_every` | integer | 0 | >= 0: checkpoint after every N steps; 0 means only when a run stops |

Exactly one of `steps` and `epochs` is positive. `out`, and the values marked **must be set**,
are required because their `Config` default is outside the valid range.

The `empty` in the default column is the table's word for an empty string or an empty list; a file
writes an empty value as nothing after the `=`. So `out = ` is empty — refused, since `out` is
required — while `out = empty` is the directory named `empty`, like any other value: what
`r97train config` prints reads back to the same `Config`, the word included.

The default column is `include/r97/config.h`: a default lives there and nowhere else, and
`r97train config` prints it from a default-constructed `Config`. `tests/cmake/config_schema_test.sh`
keeps this table and the binary's schema equal — a key, a type, a default or a range edited here is
edited in `src/config_schema.cpp` too, or the test says which row differs.

A complete file:

```
model = /srv/models/qwen3.8-27b-iq4r-s-v4.rad
data = /data/chat/train.jsonl
heldout = /data/chat/heldout.jsonl
out = runs/lora-v1
objective = ce
base_id = qwen3.8-27b-iq4r-s-v4
delta.kind = lora
delta.rank = 16
delta.alpha = 32
delta.targets = q_proj, v_proj
delta.gate = false
lr = 1e-4
weight_decay = 0.0
warmup_steps = 20
token_budget = 4096
accumulation = 1
steps = 1000
seed = 7
eval_every = 100
checkpoint_every = 100
```

`r97train config --config that-file` prints the same keys in schema order and validates them; the
run is then `build/r97train train --config that-file`. `train` writes the container's SHA-256 into
`base_id`, over whatever the file names; the key is for a caller that drives `Trainer` itself
(`docs/TRAINING.md`).

## A first run, end to end

The smallest run there is, over the tiny fixture container — no card and no data of your own; the
checks a run needs around the commands are `docs/RUNBOOK.md`'s. `docs/FIXTURES.md` owns the
fixture: the host tests make it, and it lands in the build's fixture directory as
`build/fixture/tiny_qwen35_bf16.rad`. These commands name it and keep the path in `$tiny`:

```sh
ctest --test-dir build -L host          # once: the host tests make the tiny container
tiny=build/fixture/tiny_qwen35_bf16.rad
[ -e "$tiny" ] || {
    echo "no tiny container at $tiny; run ctest --test-dir build -L host" >&2
    exit 1
}
```

`$tiny` is the tiny run's `model`; the file below is its configuration, not the real one above:

```sh
mkdir -p runs
cat > runs/tiny.conf <<CONF
model = $tiny
data = tests/fixtures/chat_small.jsonl
out = runs/tiny
objective = ce
delta.kind = lora
delta.rank = 4
delta.alpha = 8
delta.targets = q_proj, v_proj
lr = 1e-3
token_budget = 2048
accumulation = 1
steps = 20
checkpoint_every = 10
seed = 7
CONF

build/r97train info "$tiny"                                  # arch, quant, layers, hidden
build/r97train data --model "$tiny" --data tests/fixtures/chat_small.jsonl --show 2
build/r97train config --config runs/tiny.conf                # the schema, resolved
build/r97train train --reference --config runs/tiny.conf     # 20 steps; checkpoint-000000010, -000000020
build/r97train train --reference --config runs/tiny.conf --resume runs/tiny/checkpoint-000000010
build/r97train loss --model "$tiny" --data tests/fixtures/chat_small.jsonl \
    --lora runs/tiny/checkpoint-000000020/adapter
```

The first `train` writes `checkpoint-000000010` and the one it stops on,
`checkpoint-000000020`, with `run.json` beside them. The second resumes from step 10 and stops at
the file's horizon, 20 — `steps` is the whole run's, so the resumed run does steps 11 to 20 again
and the log gets a second copy of them (`docs/CHECKPOINT.md`). `loss` prints `loss: <mean per
scored token>` and the counts behind it.

The terminal line is one per step and `train.jsonl` holds the same numbers as fields;
`docs/TRAINING.md`'s Logging owns the line and what each field means.

A checkpoint directory holds `trainer.state` and `adapter/`; the adapter directory is what `--lora`
reads and what the serving side loads. `docs/CHECKPOINT.md` says what the state file holds and what
`--resume` requires. `docs/RUNBOOK.md` takes the commands through a real run's checks and failures.

## What the host build can and cannot do

- No card: everything in this file except `--device`, the `gpu` tests and the card's own inventory.
  `train` (the host step, or `--reference`), `loss`, `grad`, `data`, `info`, `ops`, `config` are
  host commands.
- A build without radiance's forward tier has no ordinary host step; `train` without `--reference`
  stops with `train: this build has no forward driver; use --reference with the host step`. The
  device step is a later issue's; `docs/PLAN.md` says which.
- `loss` and `grad` need a build with radiance's forward tier, and a plugin home; a build without it
  does not register them. `train` and `loss` find the plugin home from `RADIANCE_HOME` or the one
  the build keeps beside the binary.
- `--device N` selects the card through `ROCR_VISIBLE_DEVICES`; a `gpu` test takes `gpu_guard` so
  two runs never share the one card.

## The checks behind this file

`ctest --test-dir build -L host` runs them all. The ones a reader of this file will want by name:

| what | test |
| --- | --- |
| every command's listing and registration | `listings`, `tests/host/test_registrations.cpp` |
| `info` and `data` on a container | `info_cli`, `data_cli` |
| the schema, against `Config` and against this file | `tests/host/config_schema.cpp`, `config_schema` |
| every command's usage line and flags, against this file | `config_schema` |
| `--dry-run`, the served loss and the serve's own flags | `train_host_cli_serves_the_int8_trunk`, `train_cli_dry_run_resolves_and_writes_nothing` |
| the runbook's fenced blocks, all `sh`, extracted and run in order | `runbook_script_runs_as_the_document_orders_it` |
| the file/flags/defaults order | `config_file_overrides_command_fallbacks` |
| `train` (the host step), the held-out evaluation and the resume | `train_host_cli_overfits_and_resumes` |
| `train --reference`, overfit and resume | `train_reference_cli_overfits_and_resumes` |
| the log and the checkpoint directory | `trainer_log_and_checkpoint_files_match_the_document` |
| resume identity and the non-finite stops | `trainer_resume_is_bit_identical`, `trainer_resume_refuses_a_different_base_id`, `trainer_non_finite_loss_stops_and_keeps_the_last_good_checkpoint` |
| the held-out loss | `loss_cli_matches_radiance_kld` |
