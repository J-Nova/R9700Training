# Testing

Three test labels, and the variables that run them. ctest runs each tier's binary; the sources are
globs of their directories (`tests/<tier>/*.cpp`), so a new test file needs no edit anywhere, and
the tests inside one binary register themselves with `R97_TEST(name)` (`tests/support/test.h`).
Each tier's binary holds one `main` in its own directory: `tests/host/main.cpp`,
`tests/gpu/main.cpp`, `tests/model/info_meta.cpp`. Checks are `R97_CHECK`, not
`assert`: the default build is `RelWithDebInfo`, which defines `NDEBUG` and compiles `assert` out.

## The three labels

| Label | Binary | Needs | When it cannot run |
|---|---|---|---|
| `host` | `r97_host_tests` | no card | never — runs anywhere |
| `gpu` | `r97_gpu_tests` | one card, under the guard | never — a missing card is a failure |
| `model` | `r97_model_tests`, and `gpu_model_tests` (see below) | the real container; one entry also a card | skipped, exit 77, naming where it looked |

```sh
cmake -B build -G Ninja && cmake --build build
ctest --test-dir build -L host    # no card needed
ctest --test-dir build -L gpu     # the card, serially
ctest --test-dir build -L model   # the container, exit 77 (skip) when absent
```

### A test that needs the container *and* the card

The labels above promise one need each, so a case that needs both registers with `R97_MODEL_TEST`
(`tests/support/model_case.h`) instead of `R97_TEST`. The gpu binary is then started a second time,
as ctest's `gpu_model_tests`, with `R97_MODEL_CASES=1` set, and that is the only run that registers
those cases. The entry carries the `model` label and not `gpu`: it skips with 77 when the container
is absent, which a `gpu` test may never do, and the plain `gpu` run does not hold the case at all,
so nothing there reports a case it did not start. One consequence: on a machine that has the
container and no card, `ctest -L model` fails in that entry, because a `model` run promises the
container and not the card; exclude it there (`ctest -L model -E gpu_model_tests`).

## The GPU test guard

Several people and several agents run GPU tests on one machine. Every `gpu` test takes the guard
(`tests/support/gpu_guard.h`) before it touches a card:

- **Card selection.** `R97_DEVICE` names the card, default `0`. A test reads it with
  `r97test::parse_device`, which fails a bad value at once instead of silently reading card 0.
- **A lock per card.** The test's `r97test::DeviceGuard` holds an exclusive `flock` on
  `<lock_dir>/r97-card<N>.lock` for its lifetime. Two runs started together queue: the second
  blocks in the constructor until the first's guard is destroyed. The tier also runs serially
  under ctest (`RUN_SERIAL`); the lock is for runs ctest does not start, such as two agents
  building and testing at once. Different users queue on the same lock: the file is left
  world-writable, and `flock` needs only an open descriptor, so a run that may not write another
  user's lock file opens it read-only and waits there.
- **A card is required.** When the sysfs tree holds no card with VRAM the guard throws, naming the
  directory it looked in and `R97_DEVICE`: the `gpu` tier is for card machines, so a missing card
  fails rather than passing on radiance's host backend.
- **The free-VRAM check.** A test declares what it needs in `GuardOptions::need_vram`, in bytes.
  Before anything happens, the guard reads
  `/sys/class/drm/card*/device/mem_info_vram_{total,used}` for the selected card and, when free
  VRAM (total − used) is less than needed, fails at once naming the free and the needed amounts
  and the processes that hold the card (from `amd-smi process`, whose CSV nests one
  per-process record per line under `process_info`, so its pid, name and `vram_mem` are read out
  of that record), instead of dying with an allocation error halfway:

  ```
  card1: free VRAM 70 MiB, the test needs 1024 MiB. Processes that hold the card:
    gpu 0 pid 1543186 radiance 32500 MiB
  ```

  A card nothing holds reads `0,No running processes detected`; the guard then says so instead of
  listing holders.

The guard's logic itself is tested in the `host` tier (`tests/host/gpu_guard.cpp`) against fake
sysfs trees and a stub `amd-smi`, so it runs with no card at all. Those tests make every file they
touch under a directory of their own (`mkdtemp` under `TMPDIR`) and hand every `DeviceGuard` that
directory as both sysfs root and lock directory: they never read `/sys`, never take
`<lock_dir>/r97-card<N>.lock`, and so never queue behind a real `gpu` run — nor do two copies of
the `host` tier collide. Several copies can run at once. The same file checks the two-user case
by becoming another user in a child (skipped, with a note, when the suite does not run as root).

### Using the guard in a gpu test

```cpp
#include "support/gpu_guard.h"
#include "support/test.h"

R97_TEST(my_gpu_test) {
    r97test::GuardOptions o;
    o.device    = r97test::parse_device(getenv("R97_DEVICE") ? getenv("R97_DEVICE") : "0");
    o.need_vram = int64_t(1) << 30;                // what this test needs, in bytes
    o.bdf       = r97test::hip_bdf(o.device);      // HIP device -> PCI bus id -> sysfs card
    r97test::DeviceGuard guard(o);                 // locks (queues), then checks VRAM; throws

    // ... open the card through radiance's device layer (rad_dev_set, rad_dev_alloc, ...);
    // the guard is released when it leaves scope.
}
```

`o.bdf` is how the guard maps the HIP device to its `/sys/class/drm` card: two cards with equal
VRAM cannot be told apart by size, so on a machine with more than one card a test must pass it
(the smoke test derives it from `R97_DEVICE`). A single-card machine is unambiguous without it.
The guard finds HIP's `hipDeviceGetPCIBusId` at run time rather than linking to it, so a host-only
configuration (`-DR97_WITH_HIP=OFF`) still builds and links the `gpu` binary; its smoke test then
fails by name, because radiance's device there is its host backend, not a card.

## The model container

A `model` test resolves the container first thing through `tests/support/model_dir.h`:

```cpp
const std::string rad = r97test::container_or_skip(R97_MODEL_RAD);
```

`R97_MODEL_DIR` wins when set, else the build's `R97_MODEL_RAD`, else
`/srv/models/qwen3.8-27b-iqr-v4`. When the container is absent the helper prints `skip: …` with
the message naming what it looked for and the `hf download` line that fetches it, and ends the
process with exit 77, which ctest records as skipped (`SKIP_RETURN_CODE`). Every `model` test needs
the container, so the skip is the whole run. `r97test::resolve_container` is the same resolution
without the exit, for a caller that reports the absence itself.

## Variables

| Variable | Meaning |
|---|---|
| `R97_DEVICE` | the card ordinal the tests use; default `0` |
| `R97_LOCK_DIR` | where the guard keeps the per-card locks; default `/tmp` (set it when `/tmp` is not writable) |
| `R97_MODEL_DIR` | where the model tests find the container: a `.rad` file, or a directory that holds one (the first `*.rad` inside; the `qwen3.8-27b-iq4r-s-v4.rad` name wins if present) |
| `R97_MODEL_RAD` | the same thing, resolved by the build at configure time; `R97_MODEL_DIR`, when set, wins over it |

The default directory is `/srv/models/qwen3.8-27b-iqr-v4`. Fetch the container (15.4 GiB):

```sh
hf download StillDeadcode/qwen3.8-27b-iqr-v4 qwen3.8-27b-iq4r-s-v4.rad
```

## Notes

- A `gpu` test that cannot fit the card stops before opening it and says so; it does not retry.
- `tests/gpu/gpu_guard_smoke.cpp` is the tier's smoke test: it opens the card through radiance's
  device layer, allocates 1 GiB and frees it, and proves the per-card lock from a child process
  (a second copy asking with `fail_fast` is refused at once, rather than opening the card too).
  To see two copies serialise, start two: `./build/r97_gpu_tests & ./build/r97_gpu_tests & wait`.
- The guard holds a lock, never ownership: it never stops a process it did not start. If a stale
  holder is wedged, kill only processes you started.
- `r97_gpu_tests` runs its tests from `main` (so does `r97_host_tests`), not from the at-exit
  runner `tests/support/test.h` arms, which only the model tier uses (`tests/gpu/main.cpp` says
  why: a device test `dlopen`s `libr97.so`, and a handler armed during static initialisation would
  register the plugin's rows after the r97 registry they go into is destroyed).
  `tests/gpu/adjoint_add.cpp` is the example of a device test.
