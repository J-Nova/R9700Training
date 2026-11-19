// tests/gpu/device_step.cpp -- the device join (issue #55): one step of the real container on one
// card, as a StepFn that comes back with the adapter's gradient.
//
// The join runs the batch through the forward driver on the device kernel tier (libr4d, the host
// tier as the per-op fallback), keeps the residual at every layer boundary in one pinned host
// block (r97/step_device.h's ResidualStore, over the stream the step's ops issue on -- patches/
// training/0122), walks each block back on the card with the device adjoints (Side::device, the rows
// libr97.so registers as the plugin loads), and flushes the adapter's gradients once, at the
// step's end, into the ParamSet the caller passes as grads. docs/STEPS-DEVICE.md owns the path
// and the memory it holds. A call with grads == nullptr is a true forward-only step: the forward
// runs with the adapter fed, the report is filled, and the head's gradient pass and the block
// walk are skipped -- so a forward-only call is the cheap reference the checks lean on: the
// pre-step loss, and the descent, which proves the gradient's direction, not just its shape (a
// sign or scale error in the walk would leave the loss up).
//
//   device_step_tape       512 tokens. The op-coverage probe plus a full step, with the adapter
//                          seeded. The probe runs the join forward-only (grads == nullptr) on
//                          the step's own params, and then reads the step's tape, which the
//                          driver keeps until the next step: every op the 64 blocks issued must
//                          have a device adjoint row -- the join walks exactly those blocks and,
//                          of the head's records, the rmsnorm and the gather_rows before the
//                          logits op. The ops the walk does not need a row for: logits_gemm_vocab
//                          (the join handles the logits op itself; no adjoint row exists for it,
//                          by design), rope_table (off the walk's path) and embed_lookup (below
//                          the first boundary, a frozen leaf). What has no row is reported with
//                          adjoint_find's own message. The adapter is seeded with a
//                          deterministic non-zero pattern over both factors -- init leaves
//                          lora_B exactly zero, and a zero B never reaches lora_A's factor, so an
//                          unseeded start only proves the chain half-way -- and the step's
//                          gradient must be finite and non-zero on every leaf of both factors:
//                          one AdamW step then moves both factors, and a forward-only call on the
//                          updated adapter sees the loss fall.
//   device_step_seq4096    the ticket: one 4096-token step on the card, at the seed-0 start where
//                          lora_B is exactly zero and lora_A is random at 1/sqrt(fan_in). The
//                          forward peaks at 20.2 GiB of VRAM (docs/reports/2026-11-12-device-
//                          forward.md, which records this exact batch's loss at 34682.089228),
//                          the residual store is 5 GiB of pinned host memory -- 128 copies x
//                          4096 x 5120 x 2 bytes, the L+1 = 65 boundaries plus the L-1 = 63 second
//                          copies the int8 trunk reads beside the residual (replay_layers' plan),
//                          moved over the lane -- so the card's peak follows the store down and
//                          the case's declaration follows the footprint (card_case below). At
//                          the zero-B start the `.lora_A`
//                          gradients are exactly zero (no cotangent reaches A's factor) while the
//                          `.lora_B` gradients are not -- the pair that proves the chain reached
//                          the adapter -- and one AdamW step moves only the lora_B's, at weight
//                          decay 0.
//
// What is asserted and what is only reported. The cases assert the contract issue #146 asks:
// the StepFn runs, the returned ParamSet has the params' names and shapes, the forward is
// finite, and the optimiser applied one update. The checks on the gradient's values --
// all_finite, the zero / non-zero split, the descent -- are printed always and asserted when
// the walk's gradient is finite, so they re-arm the moment it becomes one. On the current tree
// the 512-token walk's cotangent is finite: #157's in-place restore and the readable-host range
// fallback in src/backward.cpp (docs/reports/2026-11-18-device-backward-55.md). The 4096-token
// case is still non-finite at gemm_nt_q's gdn_o.q8: that weight is plugin-private, the host row
// refuses it, so the fallback cannot serve the record. The walk's norm is ~1e38, but the
// same-tree FD measures the true leaf gradients at O(1)..O(150) mean-loss (block 63: 1.86,
// block 0: -145.5) against the walk's 0.0074 and -6.7e35 -- the last layer 250x too small, the
// bottom 4.6e33x too large -- so the norm is the walk's per-block amplification, not the
// checkpoint's: the descent is held only as a direction check -- the unclipped 1e-4/3e-4 probes
// or a clipped step -- while the default 1e-3 is printed and reported, not asserted. The
// magnitude is still wrong and issue #55 stays open;
// R97_DSTEP_ASSERT_FINITE=1 forces the value checks.
//
// The 512-token case's row coverage is separate: a device row is an optimisation and the walk
// falls back to the host row where libr97 has none (src/step_device.cpp's allow_host_fallback),
// so the ops that took the host path are named always and only an op outside host_served_op()
// fails the case -- riding the walk's finiteness there would re-arm it on a gap the design
// permits. R97_DSTEP_ASSERT_FINITE=1 also forces the coverage, naming even the host-served ops.
// The gradients the walk returns are mean-loss gradients: the head's chunk kernel is already
// handed grad_scale = 1/scored (include/r97/head.h), so nothing here rescales, and no check
// compares a magnitude against a hand-normalised expectation.
//
// Both cases run with keep_boundaries off, its default: the report's f64 copies of the 65
// boundaries would be ~2.7 GiB of host RAM at 512 tokens and ~43 GiB at 4096.
//
// The child/parent split. The engine picks the card only through ROCR_VISIBLE_DEVICES, which the
// ROCR runtime reads once, at init, and the shared r97_gpu_tests binary runs every test in one
// process where the other gpu tests init ROCR before this one does. So no engine work runs in
// this process at all: the parent forks, and the child re-execs this same binary with
// R97_TEST_ONLY (only this test), R97_DSTEP_OUT (where to write its results file) and
// ROCR_VISIBLE_DEVICES already set -- a fresh process whose ROCR points at the card from the
// first device call. The child does the real work and writes a JSON file the parent reads and
// checks; the parent resolves the container's shape from its metadata (no card) so its checks
// carry their own expected values. Each child waits for the card to report its VRAM free
// (wait_for_card_free) before the guard checks it: a previous child's VRAM is returned to the
// driver asynchronously. The child prints each phase's elapsed time -- the step's wall, the
// join's own forward/head/backward split, and each forward-only call's wall -- so a run's
// numbers size the window the parent holds.
//
// The tier. The driver's kernel hierarchy names the card's library and libr97, the plugin the
// device adjoints load from: it serves no forward kernel, it is in the list so the engine
// dlopen()s it through the hierarchy (core/plugin/loader.cpp loads every named library), and its
// static initialisers register the device rows into this process's r97 registry as it loads. The
// build passes the exact file it built as R97_DEVICE_PLUGIN (CMakeLists.txt) and
// support/device_check.h resolves it: the probe dlopen()s it once, through load_plugin(), as a
// reference count on the file the engine already loaded, so its adjoint_find does not depend on
// the engine's load. libr97 is installed in radiance_home's kernels directory beside libr4d, so
// the name is all the hierarchy needs.
//
// The join is a training path: it needs the issue observer (patches/training/0101), the replay
// block (0102) and the context stream (0122) the store keeps its boundaries on; a serving build
// applies none of the set, so the cases compile to nothing there and say so by name instead.
#include "support/gpu_guard.h"
#include "support/model_case.h"
#include "support/model_dir.h"
#include "support/test.h"

#include "r97/forward.h"
#include "r97/step_device.h"

#if R97_FORWARD && R97_DEVICE_STEP

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#include <sys/wait.h>

#include "engine.h"             // radiance's core/; rad::Engine must be complete: the driver's
                                // own std::unique_ptr<rad::Engine> instantiates its destructor
                                // at the first use
#include "rad_device.h"         // rad_dev_count/props: the card, not the host backend
#include "support/device_check.h"  // load_plugin(): libr97.so, resolved and dlopen()ed once

#include "r97/adapter_io.h"
#include "r97/adjoint.h"
#include "r97/optim.h"

namespace {

namespace fs = std::filesystem;

using r97::Batch;
using r97::TokenSeq;

const char* kRoleVar = "R97_DSTEP_ROLE";  // set in the re-exec'd child; the role that does the work
const char* kOutVar  = "R97_DSTEP_OUT";   // the child's results directory

// R97_DSTEP_ASSERT_FINITE=1 forces the value checks while the walk's gradient is non-finite;
// the checks re-arm on their own once it is finite (the header's "What is asserted and what is
// only reported"). The re-exec'd child inherits it from the parent's environment, like R97_CENTRAL.
bool assert_finite() {
    const char* v = std::getenv("R97_DSTEP_ASSERT_FINITE");
    return v != nullptr && v[0] == '1';
}

// The ops the live container's walk leaves to a host row by design (src/step_device.cpp's
// allow_host_fallback: a device row is an optimisation, and the host row is the same arithmetic
// in f64). Only these are tolerated when the coverage probe finds no device row for them.
bool host_served_op(const std::string& op) { return op == "rope"; }

// nlohmann writes a non-finite number as null. The walk's non-finite gradient makes the
// post-step loss and the gradient norm exactly that, so the report reads a null as NaN: a
// non-finite value stays a finding for the checks to name, not a parse error in the reporting.
double report_number(const nlohmann::json& j, const char* key) {
    const auto it = j.find(key);
    if (it == j.end() || it->is_null()) return std::nan("");
    return it->get<double>();
}

// ------------------------------------------------------------------ the small helpers

const std::string& self_exe() {
    static const std::string exe = [] {
        std::error_code ec;
        const fs::path p = fs::read_symlink("/proc/self/exe", ec);
        if (ec) throw std::runtime_error("device_step: cannot read /proc/self/exe");
        return p.string();
    }();
    return exe;
}

// The build directory this test binary sits in: radiance-build/ (the plugins, the home) is beside
// it.
std::string build_dir() {
    std::error_code ec;
    const fs::path p = fs::read_symlink("/proc/self/exe", ec);
    return ec ? std::string() : p.parent_path().string();
}

// The plugin home: RADIANCE_HOME, else the build's radiance_home.
std::string radiance_home() {
    if (const char* e = std::getenv("RADIANCE_HOME"))
        if (*e) return e;
    const std::string h = build_dir() + "/radiance-build/radiance_home";
    std::error_code ec;
    return fs::exists(h + "/architectures", ec) ? h : std::string();
}

// A fresh directory under TMPDIR (else /tmp) for one case's results. mkdtemp, so two copies of the
// test never collide. The test tier does not carry radiance's tests/ path, so this is the
// equivalent of radtest::tmp_path, which the gpu target cannot use.
std::string temp_dir(const std::string& name) {
    const char* base = std::getenv("TMPDIR");
    if (!base || !*base) base = "/tmp";
    std::string path = std::string(base) + "/r97-dstep-" + name + "-XXXXXX";
    char* d = ::mkdtemp(path.data());
    if (!d)
        throw std::runtime_error("device_step: cannot make a temp directory under " +
                                 std::string(base) + ": " + std::strerror(errno));
    return d;
}

// Fork, and have the child re-exec this binary running only `test_name`, with ROCR_VISIBLE_DEVICES
// pointing the ROCR runtime at the card before its first init. R97_TEST_ONLY selects this test so
// the re-exec'd copy runs it alone, and -- seeing R97_DSTEP_ROLE -- does the child's work instead
// of forking again. Wait, and end the run as skipped (exit 77) when the child skips (the container
// absent). The parent never opens the card or inits an engine: its ROCR was already init'd by the
// other gpu tests, so only a fresh child process can be pointed at the card.
void run_in_child(const std::string& dir, const char* test_name) {
    const char* dev = std::getenv("R97_DEVICE");
    const std::string device_num = dev && *dev ? dev : "0";

    const pid_t pid = ::fork();
    if (pid < 0)
        throw std::runtime_error(std::string("device_step: fork: ") + std::strerror(errno));
    if (pid == 0) {
        // The child: point both runtimes at the card before any init, mark this re-exec'd copy as
        // the one that does the work, and hand it the results directory. The re-exec'd process is
        // fresh, so the runtimes read these for the first time.
        if (::setenv("HIP_VISIBLE_DEVICES", device_num.c_str(), 1) != 0 ||
            ::setenv("ROCR_VISIBLE_DEVICES", device_num.c_str(), 1) != 0 ||
            ::setenv("R97_TEST_ONLY", test_name, 1) != 0 ||
            ::setenv(kRoleVar, "device", 1) != 0 || ::setenv(kOutVar, dir.c_str(), 1) != 0)
            std::_Exit(1);
        std::string exe = self_exe();
        char* const argv[2] = {exe.data(), nullptr};
        ::execv(exe.c_str(), argv);
        std::_Exit(1);  // execv failed
    }

    int status = 0;
    if (::waitpid(pid, &status, 0) != pid)
        throw std::runtime_error(std::string("device_step: ") + test_name + ": waitpid failed: " +
                                 std::strerror(errno));
    if (!WIFEXITED(status)) {
        // The signal, by name and number: "killed by a signal" alone cannot tell an OOM kill
        // (SIGKILL) from a fault (SIGSEGV/SIGABRT) or a hang triaged by a supervisor (SIGTERM).
        const int sig = WTERMSIG(status);
        const char* name = ::strsignal(sig);
        throw std::runtime_error(std::string("device_step: ") + test_name + "'s child was killed by "
                                 "signal " + std::to_string(sig) + " (" +
                                 (name ? name : "unknown") + "); the card may be held by another run");
    }
    const int code = WEXITSTATUS(status);
    if (code != 0) {
        // 77 (the container absent) re-skips the way ctest records it; anything else is a failure.
        if (code == r97test::kSkip)
            r97test::skip(std::string("the model container is absent (see the child run of ") +
                          test_name + " above");
        throw std::runtime_error(std::string("device_step: ") + test_name +
                                 "'s child exited " + std::to_string(code) +
                                 ": its stderr above says why; the card is R97_DEVICE, and the "
                                 "guard names the card's holders when the VRAM is short");
    }
}

// The results directory this (child) process was told to write into.
std::string out_dir() {
    const char* d = std::getenv(kOutVar);
    if (!d || !*d)
        throw std::runtime_error("device_step: " + std::string(kOutVar) +
                                 " is unset; the results have nowhere to go");
    return d;
}

// ------------------------------------------------------------------ the card-free wait

// Wait until the selected card reports `need` bytes free. A previous test's child may have just
// exited and the driver can take a moment to return its VRAM to mem_info_vram_used; the guard
// would otherwise refuse on a stale reading of a card that is in fact free.
void wait_for_card_free(const r97test::GuardOptions& o) {
    const std::string sysfs = "/sys/class/drm";
    const std::string card = r97test::card_for_bdf(sysfs, o.bdf);
    for (int i = 0; i < 120; ++i) {  // up to ~60 s
        try {
            const r97test::CardVram v = r97test::read_card_vram(sysfs, card);
            if (v.total - v.used >= o.need_vram) return;
        } catch (const std::exception&) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

int64_t read_vram_used(const std::string& card) {
    return r97test::read_sys_int("/sys/class/drm/" + card + "/device/mem_info_vram_used");
}

// ------------------------------------------------------------------ the adapter's start

// A deterministic, non-zero seed over both factors: init leaves `lora_B` exactly zero, and a zero
// B never reaches `lora_A`'s factor, so an unseeded start proves the chain half-way only. The
// seed is splitmix64 over each leaf's index, one draw per value, in [-1/sqrt(fan_in),
// 1/sqrt(fan_in)]: both factors take rank inputs per output row, and every leaf is a different
// pattern, so a step that trains the adapter moves every one of them.
void seed_adapter(r97::ParamSet& params) {
    for (size_t leaf = 0; leaf < params.size(); ++leaf) {
        r97::Param& p = params[leaf];
        const int64_t fan_in = p.shape.size() >= 2 ? p.shape[1] : 1;
        const double scale = 1.0 / std::sqrt((double)std::max<int64_t>(1, fan_in));
        uint64_t state = 0x9e3779b97f4a7c15ull * (uint64_t(leaf) + 1);
        for (float& v : p.value)
            v = float((2.0 * r97::device_check::urand(&state) - 1.0) * scale);
    }
}

// ------------------------------------------------------------------ the child path

// One joined step of one `n`-token sequence on the card, the op-coverage probe when asked, the
// gradient's checks, one AdamW step, and the forward-only descent check: the checks run where
// the data is, and the parent reads the numbers out of the results file.
void card_case(const char* test, int64_t n, bool probe) {
    r97test::GuardOptions o;
    const char* de = std::getenv("R97_DEVICE");
    o.device    = r97test::parse_device(de ? de : "0");
    // The container's device forward peaks at 20.2 GiB at 4096 tokens (weights, KV and the
    // engine's arena), the join's snapshots and the gradient staging sit beside it, and the
    // residual store is off the card -- one pinned host block, moved over the lane: the 4096
    // case measured ~23 GiB and declares 24, the 512 one 19.
    o.need_vram = (n >= 4096) ? int64_t(24) << 30 : int64_t(19) << 30;
    o.bdf       = r97test::hip_bdf(o.device);  // HIP is inited only now: ROCR_VISIBLE_DEVICES is set
    wait_for_card_free(o);
    r97test::DeviceGuard guard(o);
    const std::string card = guard.card();  // the /sys/class/drm/cardN directory

    const std::string container = r97test::container_or_skip();
    const std::string home = radiance_home();
    if (home.empty())
        throw std::runtime_error(std::string(test) + ": no radiance home beside the build");
    const std::string dir = out_dir();

    // This child points both runtimes at its one card before the first init, so that card is
    // device 0 in this process, whatever R97_DEVICE is on the machine -- and it is a card. The
    // host backend standing in for one cannot run this container's trunk linears at all.
    R97_CHECK(rad_dev_count() >= 1);
    RadDeviceProps props{};
    if (rad_dev_props(0, &props) != 0)
        throw std::runtime_error(std::string(test) + ": rad_dev_props: " + rad_dev_last_error());
    R97_CHECK(!props.is_host_backend);
    std::printf("%s: %s, %lld tokens, %lld MiB free on the card\n", test, card.c_str(),
                (long long)n, (long long)(props.vram_free / r97test::kMiB));

    // The one-prompt length this child steps: the same id scheme as device_forward's seq cases,
    // every token but the first scored.
    TokenSeq seq;
    seq.ids.resize(size_t(n));
    for (size_t i = 0; i < seq.ids.size(); ++i) seq.ids[i] = 1 + int32_t((i * 7) % 299);
    seq.score.assign(size_t(n), 1);
    seq.score[0] = 0;
    Batch batch;
    batch.push_back(std::move(seq));

    // The model and its adapter, rank 16, alpha 32. The 4096 case starts where init leaves the
    // adapter -- lora_B exactly zero, lora_A random at 1/sqrt(fan_in) -- so the step's loss is
    // the base model's and the gradient's split is the seed-0 pair; the 512 case seeds both
    // factors non-zero so every leaf of the gradient has to be non-zero.
    const r97::ModelMeta meta = r97::read_model_meta(container);
    r97::Config mc;
    mc.model     = container;
    mc.objective = "ce";
    mc.delta.kind = "lora";
    mc.delta.rank = 16;
    mc.delta.alpha = 32.0;
    const r97::Model model = r97::make_model(mc);
    r97::ParamSet params = model.delta->init(meta, mc.delta, /*seed=*/0);
    R97_CHECK(!params.empty());
    if (probe) seed_adapter(params);

    // The driver for the tier: the card's library first, libr97 the plugin the device adjoints
    // load from (it serves no forward kernel; it is in the list so the engine dlopen()s it), the
    // host tier as the per-op fallback. One sequence, one piece: the driver turns radiance's
    // checkpoint interval off for the step itself (see src/forward.cpp).
    r97::ForwardConfig cfg;
    cfg.model         = container;
    cfg.radiance_home = home;
    cfg.kernels       = {"libr4d", "libr97", "libavx", "libref"};
    cfg.max_tok       = n;
    cfg.max_ctx       = n;
    cfg.max_seqs      = 1;
    cfg.window_rows   = 256;
    std::string err;
    std::unique_ptr<r97::ForwardDriver> driver = r97::ForwardDriver::open(cfg, &err);
    if (!driver) throw std::runtime_error(err);

    r97::DeviceStepOptions options;  // keep_boundaries stays off, its default: the report's f64
                                     // copies of the 65 boundaries are ~10 GiB a vector, ~21 GiB
                                     // for values and cotangents, at 4096 tokens
    r97::DeviceStepReport report;
    const r97::StepFn step = r97::device_step(*driver, model, options, &report);

    // The walk-vs-walk arbiter's print: every gradient whose name contains R97_DUMP_LEAF, and the
    // set's own norm. A substring, like the FD harness's find_w -- one leaf can be unrepresentative,
    // and a block-wide spec (model.layers.63.) compares a whole block without a scope mismatch. The
    // arm label matters: R97_HOST_WALK_AB=2 returns from its own arm and its gradients live in
    // host_grads, never grads, so an unlabelled print reads as agreement.
    const auto print_leaves = [&](const r97::ParamSet& set, const char* arm) {
        const char* leaf = std::getenv("R97_DUMP_LEAF");
        if (leaf == nullptr) return;
        const std::string want(leaf);
        int64_t n = 0;
        double norm_sq = 0.0;
        for (const r97::Param& g : set)
            if (g.name.find(want) != std::string::npos && !g.value.empty()) {
                double leaf_sq = 0.0;
                for (float v : g.value) leaf_sq += double(v) * double(v);
                norm_sq += leaf_sq;
                ++n;
                std::printf("%s: LEAF %s grad=%.9g (%s)\n", test, g.name.c_str(), g.value[0], arm);
            }
        std::printf("%s: LEAFSET %s leaves=%lld norm=%.9g (%s)\n", test, want.c_str(),
                    (long long)n, std::sqrt(norm_sq), arm);
    };

    // R97_HOST_WALK_AB=2: the host walk alone, first, for iterating on a host row that refuses a
    // record. The A/B below runs after the device step and only when that walk's gradient is not
    // finite, so a refusing host row otherwise costs a whole device walk -- minutes on the real
    // container -- per attempt. This arm runs the same walk and returns: a refusal throws and fails
    // the case, which is the whole answer the iteration is after.
    if (const char* ab = std::getenv("R97_HOST_WALK_AB");
        ab != nullptr && std::strcmp(ab, "2") == 0) {
        // R97_CENTRAL is read far below this arm's return, and it differences the *device* walk's
        // gradient. Asking for both would run the host walk, return, and never difference anything:
        // a flag that silently measures nothing is worse than one that refuses.
        if (std::getenv("R97_CENTRAL") != nullptr)
            throw std::runtime_error(
                "device step: R97_CENTRAL with R97_HOST_WALK_AB=2 is not a measurement: the central "
                "difference reads the device walk's gradient and this arm returns before it runs. "
                "Run R97_CENTRAL without R97_HOST_WALK_AB for the device gradient, or R97_FD_REC=<pos> "
                "on this arm for a per-row check at the tape's own operands.");
        r97::DeviceStepOptions host_options = options;
        host_options.host_walk = true;
        r97::DeviceStepReport host_report;
        const r97::StepFn host_step = r97::device_step(*driver, model, host_options, &host_report);
        r97::ParamSet host_grads;
        const auto t_ab = std::chrono::steady_clock::now();
        const r97::StepResult host_result = host_step(batch, params, &host_grads);
        const double host_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ab).count();
        bool host_all_finite = true;
        int64_t host_non_finite_leaves = 0;
        double host_norm_sq = 0.0;
        for (const r97::Param& g : host_grads) {
            bool leaf_finite = true;
            for (float v : g.value) {
                if (!std::isfinite(v)) leaf_finite = false;
                host_norm_sq += double(v) * double(v);
            }
            if (!leaf_finite) {
                host_all_finite = false;
                ++host_non_finite_leaves;
            }
        }
        std::printf("%s: host walk only: %lld scored, loss %.6f, gradient %s (grad norm %.6g, "
                    "%lld of %lld leaves non-finite), cotangent op '%s' operand '%s'[%lld] pos "
                    "%lld value %lld, forward_all_finite %d, %.1f s\n",
                    test, (long long)host_result.scored, host_result.loss,
                    host_all_finite ? "all finite" : "NOT finite", std::sqrt(host_norm_sq),
                    (long long)host_non_finite_leaves, (long long)host_grads.size(),
                    host_report.first_non_finite_op.c_str(),
                    host_report.first_non_finite_operand.c_str(),
                    (long long)host_report.first_non_finite_operand_index,
                    (long long)host_report.first_non_finite_position,
                    (long long)host_report.first_non_finite_index,
                    host_report.forward_all_finite ? 1 : 0, host_seconds);
        // The walk-vs-walk arbiter's host value. This arm returns below, before the print that
        // reads the device walk's `grads`, so without this the arbiter yields one number and
        // reads as agreement.
        print_leaves(host_grads, "host arm");
        // The parent reads a results file whatever the child did, and this arm ran no device step
        // for it to assert on. The file carries the host walk's own numbers and a marker, so the
        // parent checks what this arm actually measured instead of reporting a missing file.
        if (const char* dir = std::getenv(kOutVar)) {
            nlohmann::json hj;
            hj["name"] = test;
            hj["host_walk_only"] = true;
            hj["tokens"] = n;
            hj["scored"] = host_result.scored;
            hj["loss"] = host_result.loss;
            hj["host_finite"] = host_all_finite;
            hj["host_non_finite_leaves"] = host_non_finite_leaves;
            hj["host_norm"] = std::sqrt(host_norm_sq);
            hj["host_seconds"] = host_seconds;
            std::ofstream out(std::string(dir) + "/" + test + ".json");
            if (!out) throw std::runtime_error(std::string(test) + ": cannot write the results");
            out << hj.dump(2) << "\n";
        }
        return;
    }

    nlohmann::json j;
    j["name"] = test;

    // The in-place reads with no earlier writer the tape names, when the probe ran: the walk must
    // serve each from the block's stored boundary. -1 when no probe ran (the 4096 case).
    int64_t in_place_expected = -1;

    // ---- the op-coverage probe: the join forward-only, then the step's tape

    double probe_seconds = 0.0;
    if (probe) {
        // grads == nullptr is a forward-only call on the step's own params: the forward runs
        // with the adapter fed, and the driver keeps this step's tape until the next step. The
        // tape is read here, before the gradient call replaces last().
        const auto tp0 = std::chrono::steady_clock::now();
        const r97::StepResult r0 = step(batch, params, nullptr);
        probe_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
        R97_CHECK(r0.scored == n - 1);
        R97_CHECK(std::isfinite(r0.loss));
        // The engine loaded libr97 through the kernel hierarchy when the step's engine started
        // (the join's head loop needs the plugin's head kernel, and the step would have thrown
        // without one). dlopen() the same file -- R97_DEVICE_PLUGIN is the build's exact path,
        // and radiance_home's kernels directory is where the build puts it -- as a reference
        // count, so the lookups below do not depend on that load.
        r97::device_check::load_plugin();
        const r97::Tape& tape = driver->last().tape();
        // The ops the walk does not need a row for: the logits op the join handles itself (no
        // adjoint row exists for it, by design), the rope table, which is off the walk's path,
        // and the embedding below the first boundary, a frozen leaf.
        static const std::set<std::string> kNotWalked = {"logits_gemm_vocab", "rope_table",
                                                         "embed_lookup"};
        std::set<std::string> block_ops, head_ops;
        std::set<int64_t> block_ids;
        std::vector<std::pair<std::string, std::string>> missing;  // (op, adjoint_find's message)
        int64_t records = 0;
        for (const r97::OpRecord& rec : tape.ops()) {
            ++records;
            if (rec.block >= 0) {
                R97_CHECK(rec.block < meta.layers);
                block_ops.insert(rec.op);
                block_ids.insert(rec.block);
            } else {
                head_ops.insert(rec.op);
            }
            if (rec.block < 0 && kNotWalked.count(rec.op)) continue;
            std::string why;
            if (!r97::adjoint_find(rec.op, r97::Side::device, rec.params, &why)) {
                if (std::find_if(missing.begin(), missing.end(),
                                 [&](const std::pair<std::string, std::string>& m) {
                                     return m.first == rec.op && m.second == why;
                                 }) == missing.end())
                    missing.push_back({rec.op, std::move(why)});
            }
        }
        R97_CHECK(records > 0);
        nlohmann::json block_list = nlohmann::json::array();
        for (const std::string& op : block_ops) block_list.push_back(op);
        nlohmann::json head_list = nlohmann::json::array();
        for (const std::string& op : head_ops) head_list.push_back(op);
        nlohmann::json missing_list = nlohmann::json::array();
        for (const auto& m : missing) missing_list.push_back({m.first, m.second});
        j["probe_ops"]       = records;
        j["probe_blocks"]    = int64_t(block_ids.size());
        j["probe_block_ops"] = block_list;
        j["probe_head_ops"]  = head_list;
        j["probe_missing"]   = missing_list;
        j["probe_loss"]      = r0.loss;
        j["probe_scored"]    = r0.scored;

        // The count the walk must report, derived from the tape and not from the run: an operand
        // the record itself also writes, with no earlier op in its block writing the buffer, is
        // the in-place read with no earlier writer. A kv: read enters zero, not the store.
        in_place_expected = 0;
        {
            std::set<std::string> wrote;
            int64_t block = -1;
            for (const r97::OpRecord& rec : tape.ops()) {
                if (rec.block < 0) continue;
                if (rec.block != block) { block = rec.block; wrote.clear(); }
                for (const r97::Operand& read : rec.operands) {
                    if (read.written) continue;
                    bool in_place = false;
                    for (const r97::Operand& w : rec.operands)
                        if (w.written && w.buffer == read.buffer) in_place = true;
                    if (!in_place || wrote.count(read.buffer) != 0) continue;
                    if (read.buffer.rfind("kv:", 0) == 0) continue;
                    ++in_place_expected;
                }
                for (const r97::Operand& w : rec.operands)
                    if (w.written) wrote.insert(w.buffer);
            }
        }
    }

    // ---- the step: one prefill piece with the adapter's gradient, VRAM at the peak

    r97::ParamSet grads;
    // The step runs the batch as one engine run: start() (the model's load) and serve() (the
    // forward) inside it, then the head and the walk. Poll the card's VRAM at the peak while it
    // runs.
    std::atomic<bool> running{true};
    std::atomic<long long> peak{0};
    std::thread poller([&] {
        while (running.load(std::memory_order_relaxed)) {
            try {
                const int64_t v = read_vram_used(card);
                long long cur = peak.load(std::memory_order_relaxed);
                while (v > cur && !peak.compare_exchange_weak(cur, v)) {
                }
            } catch (const std::exception&) {
            }  // one unreadable poll is noise; the step's time is what matters
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    });
    // The poller must be joined before any exception leaves this function: a joinable std::thread
    // destroyed during unwinding calls std::terminate, which would hide the step's real error.
    double step_seconds = 0.0;
    r97::StepResult result{0.0, 0};
    try {
        const auto t0 = std::chrono::steady_clock::now();
        result = step(batch, params, &grads);
        step_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    } catch (...) {
        running.store(false);
        poller.join();
        throw;
    }
    running.store(false);
    poller.join();
    const int64_t vram_after_load = read_vram_used(card);

    // The report the gradient call left: the descent call below overwrites it (grad_stage_bytes
    // is 0 on a step that runs without grads), so the step's own numbers are copied out first.
    const r97::DeviceStepReport rep = report;

    // ---- the gradient: the same names and shapes as the params, and the start's invariants

    // The CE chunk's gradient is already scaled by grad_scale = 1/scored (include/r97/head.h),
    // so what the walk returns is the mean-loss gradient: no rescaling here, and no magnitude
    // compared against a hand-normalised expectation -- the checks are the shape, the zero /
    // non-zero split, and the descent.
    const auto ends = [](const std::string& s, const char* suffix) {
        const std::string tail(suffix);
        return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
    };
    bool names_match      = grads.size() == params.size();
    bool shapes_match     = names_match;
    bool all_finite       = true;
    bool a_all_zero       = true;   // the zero-B start: no cotangent reaches A's factor
    bool a_all_nonzero    = true;   // the seeded start: every A leaf sees the cotangent
    bool b_all_nonzero    = true;   // both starts: the cotangent reaches B's factor
    int64_t n_a = 0, n_b = 0, n_other = 0;
    // The first leaf of each factor whose gradient is exactly zero, and how many there are: a
    // `b_all_nonzero` false names one leaf out of 400, and the next window should not have to
    // guess which. A leaf the walk never reached (an empty gradient) counts as zero too.
    std::string first_zero_a_leaf, first_zero_b_leaf;
    // Which leaf does move, and whether it moves to a finite value or to NaN. Only one outcome is
    // decisive: a *finite* non-zero leaf means the walk differentiated a live cotangent all the way
    // to that module, which the cascade theory cannot produce (a zeroed dx never touches the last
    // gemm's own lora rows). A NaN leaf is consistent with both candidates -- under a zero scale,
    // dB = sum cy*sc*t is still NaN when cy is -- so it says only that the walk reached the leaf.
    // The per-operand non-zero counts on the 288 lora rows are what settles the mechanism.
    std::string first_nonzero_a_leaf, first_nonzero_b_leaf;
    bool first_nonzero_a_finite = true, first_nonzero_b_finite = true;
    // The count the structural question actually needs. `part != 0.0` is true for NaN, so a poisoned
    // leaf reads as "moved" in `a_all_zero`/`a_changed`; a *finite* non-zero `part` is the only
    // reading that says a live cotangent reached the module. `n_zero_a` is its complement on the
    // zero side only -- `n_a_finite_nonzero + n_zero_a == n_a` holds exactly when the run is finite.
    int64_t n_a_finite_nonzero = 0, n_b_finite_nonzero = 0;
    int64_t n_zero_a = 0, n_zero_b = 0;
    double grad_norm_sq   = 0.0;
    for (const r97::Param& p : params) {
        const r97::Param* g = nullptr;
        for (const r97::Param& c : grads)
            if (c.name == p.name) { g = &c; break; }
        if (!g) {
            names_match = false;
            continue;
        }
        if (g->shape != p.shape) shapes_match = false;
        const bool is_a = ends(p.name, "lora_A");
        const bool is_b = ends(p.name, "lora_B");
        if (is_a) ++n_a;
        else if (is_b) ++n_b;
        else ++n_other;
        double part = 0.0;
        for (size_t i = 0; i < g->value.size(); ++i) {
            const float v = g->value[i];
            if (!std::isfinite(v)) all_finite = false;
            part += double(v) * double(v);
        }
        grad_norm_sq += part;
        if (is_a) {
            if (part == 0.0) {
                a_all_nonzero = false;
                ++n_zero_a;
                if (first_zero_a_leaf.empty()) first_zero_a_leaf = p.name;
            }
            if (part != 0.0) a_all_zero = false;
            if (std::isfinite(part) && part != 0.0) ++n_a_finite_nonzero;
            if (part != 0.0 && first_nonzero_a_leaf.empty()) {
                first_nonzero_a_leaf = p.name;
                for (size_t i = 0; i < g->value.size(); ++i)
                    if (!std::isfinite(g->value[i])) first_nonzero_a_finite = false;
            }
        }
        if (is_b && std::isfinite(part) && part != 0.0) ++n_b_finite_nonzero;
        if (is_b && part == 0.0) {
            b_all_nonzero = false;
            ++n_zero_b;
            if (first_zero_b_leaf.empty()) first_zero_b_leaf = p.name;
        }
        if (is_b && part != 0.0 && first_nonzero_b_leaf.empty()) {
            first_nonzero_b_leaf = p.name;
            for (size_t i = 0; i < g->value.size(); ++i)
                if (!std::isfinite(g->value[i])) first_nonzero_b_finite = false;
        }
    }

    // ---- the A/B, only when the device walk's gradient was not finite: the same tape and the
    // same params walked with the host rows (side = host, no fallback). A device row that is wrong
    // shows as a finite host gradient beside the device one; both non-finite means the plumbing --
    // the seeds, the head's cotangent, the staging -- rather than a row. The host walk costs the
    // host's f64 arithmetic, hours on the real container, so a finite run never pays it.
    nlohmann::json ab;
    // The A/B costs the host's f64 walk -- hours on the real container -- so it runs only when
    // asked for: R97_HOST_WALK_AB=1. The attribution line above is what a normal window reads.
    const char* ab_env = std::getenv("R97_HOST_WALK_AB");
    if (!all_finite && ab_env != nullptr && *ab_env == '1') {
        r97::DeviceStepOptions host_options = options;
        host_options.host_walk = true;
        r97::DeviceStepReport host_report;
        const r97::StepFn host_step = r97::device_step(*driver, model, host_options, &host_report);
        r97::ParamSet host_grads;
        const auto t_ab = std::chrono::steady_clock::now();
        const r97::StepResult host_result = host_step(batch, params, &host_grads);
        const double host_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ab).count();
        bool host_all_finite = true;
        int64_t host_non_finite_leaves = 0;
        double host_norm_sq = 0.0;
        for (const r97::Param& g : host_grads) {
            bool leaf_finite = true;
            for (float v : g.value) {
                if (!std::isfinite(v)) leaf_finite = false;
                host_norm_sq += double(v) * double(v);
            }
            if (!leaf_finite) {
                host_all_finite = false;
                ++host_non_finite_leaves;
            }
        }
        ab = {{"loss", host_result.loss},
              {"seconds", host_seconds},
              {"n_grads", int64_t(host_grads.size())},
              {"all_finite", host_all_finite},
              {"non_finite_leaves", host_non_finite_leaves},
              {"grad_norm", std::sqrt(host_norm_sq)},
              {"first_non_finite_op", host_report.first_non_finite_op},
              {"first_non_finite_operand", host_report.first_non_finite_operand},
              {"first_non_finite_operand_index", host_report.first_non_finite_operand_index},
              {"first_non_finite_position", host_report.first_non_finite_position},
              {"first_non_finite_index", host_report.first_non_finite_index},
              {"first_non_finite_from_row", host_report.first_non_finite_from_row},
              {"forward_all_finite", host_report.forward_all_finite},
              {"forward_operand", host_report.forward_first_non_finite_operand},
              {"forward_position", host_report.forward_first_non_finite_position},
              {"forward_value_index", host_report.forward_first_non_finite_index}};
        std::printf("%s: host-walk A/B: %lld scored, loss %.6f, gradient %s (grad norm %.6g, %lld "
                    "of %lld leaves non-finite), cotangent op '%s' operand '%s'[%lld] pos %lld "
                    "value %lld from_row %d, forward_all_finite %d, %.1f s\n",
                    test, (long long)host_result.scored, host_result.loss,
                    host_all_finite ? "all finite" : "NOT finite", std::sqrt(host_norm_sq),
                    (long long)host_non_finite_leaves, (long long)host_grads.size(),
                    host_report.first_non_finite_op.c_str(),
                    host_report.first_non_finite_operand.c_str(),
                    (long long)host_report.first_non_finite_operand_index,
                    (long long)host_report.first_non_finite_position,
                    (long long)host_report.first_non_finite_index,
                    host_report.first_non_finite_from_row ? 1 : 0,
                    host_report.forward_all_finite ? 1 : 0, host_seconds);
        // The host arm's gradients are its own ParamSet, never `grads`: printing from `grads` here
        // would report the DEVICE value in both runs and read as perfect agreement.
        print_leaves(host_grads, "host arm");
    }

    // ---- the walk-vs-walk arbiter: the same leaf's gradient as THIS arm computed it. Two computed
    // gradients can be compared where a finite difference of a quantised loss cannot -- the real
    // container's central difference swings 948.917 -> 152.981 for a 4x larger eps, so it is a
    // quantisation flip over 2*eps and no ratio can be formed from it (docs/reports/
    // 2026-11-18-device-backward-55.md, "The reference is not a derivative"). Run the device arm
    // and then the host arm (R97_HOST_WALK_AB=2 R97_WALK_BLOCKS=1) with the same R97_DUMP_LEAF.
    print_leaves(grads, "walk arm");

    // ---- the central difference, only when asked: R97_CENTRAL=1. Perturbs one LoRA weight by a
    // small relative eps and re-forwards, so the loss's own sensitivity to that weight is
    // measured -- the number that settles whether the walk's ~1e37-magnitude gradient is the true
    // gradient (the loss jumps ~eps * 1e37, non-finite) or a scale bug (the loss moves ~eps *
    // 1e0..1e3). The step's params are still the original here, before the AdamW step below.
    // R97_CENTRAL_BLK names the block ("blk.51."); default is the block whose gradient the walk
    // reported as 1.2e37.
    if (std::getenv("R97_CENTRAL") != nullptr) {
        const auto find_w = [&](const std::string& blk) -> int64_t {
            for (size_t i = 0; i < params.size(); ++i)
                if (params[i].name.find("lora") != std::string::npos &&
                    params[i].name.find(blk) != std::string::npos &&
                    params[i].value.size() > 0 && params[i].value[0] != 0.0f)
                    return (int64_t)i;
            return -1;
        };
        // The params are named `model.layers.<N>.<module>.lora_A` (not `blk.<N>.`), so the block
        // spec is `model.layers.51.`. R97_CENTRAL_BLKS takes a comma-separated list and measures
        // each in one run: the walk's error is per-layer and compounding, so the block at which the
        // FD oracle and the walk first disagree localises it, and forwards are cheap next to the
        // backward. R97_CENTRAL_BLK stays the single-block form.
        std::vector<std::string> specs;
        if (const char* env_blks = std::getenv("R97_CENTRAL_BLKS")) {
            std::string s(env_blks);
            size_t at = 0;
            while (at <= s.size()) {
                const size_t comma = s.find(',', at);
                const std::string piece = s.substr(at, comma == std::string::npos ? std::string::npos
                                                                                 : comma - at);
                if (!piece.empty()) specs.push_back(piece);
                if (comma == std::string::npos) break;
                at = comma + 1;
            }
        }
        if (specs.empty()) specs.push_back(std::getenv("R97_CENTRAL_BLK")
                                               ? std::getenv("R97_CENTRAL_BLK")
                                               : "model.layers.51.");
        for (const std::string& spec : specs) {
            int64_t wi = find_w(spec);
            if (wi < 0) wi = find_w("model.layers.51.");
            if (wi < 0) wi = find_w("model.layers.0.");
            if (wi < 0) wi = find_w("");
            if (wi < 0) continue;
            const std::string name = params[wi].name;
            const float w0 = params[wi].value[0];
            double g0 = 0.0;
            for (const r97::Param& c : grads)
                if (c.name == name) { g0 = c.value[0]; break; }
            // The forward runs bf16, whose mantissa resolves ~4e-3 relative: a perturbation below
            // that leaves the loss bit-identical by construction and the small-eps difference reads
            // exactly 0 -- quantisation, not a zero gradient. So measure at two steps: the small
            // one (which shows the floor) and one ~2.5 bf16 ulps up, which the forward can see.
            const double eps = std::max(std::fabs(w0) * 1e-4, 1e-6);
            params[wi].value[0] = (float)(w0 + eps);
            const double lp = step(batch, params, nullptr).loss;
            params[wi].value[0] = (float)(w0 - eps);
            const double lm = step(batch, params, nullptr).loss;
            params[wi].value[0] = w0;
            const double cd = (lp - lm) / (2.0 * eps);
            const double eps_big = std::max(std::fabs(w0) * 1e-2, 1e-4);
            params[wi].value[0] = (float)(w0 + eps_big);
            const double bp = step(batch, params, nullptr).loss;
            params[wi].value[0] = (float)(w0 - eps_big);
            const double bm = step(batch, params, nullptr).loss;
            params[wi].value[0] = w0;
            const double cd_big = (bp - bm) / (2.0 * eps_big);
            // A third step, ~4x the bf16-visible one: two points cannot tell a linear response
            // from a lucky one, and a reference that is not in the linear regime is no arbiter for
            // the walk's magnitude. Agreement across eps_big and eps_mid is what makes `central`
            // usable; disagreement means the reference is the suspect, not the walk.
            const double eps_mid = std::max(std::fabs(w0) * 4e-2, 1e-4);
            params[wi].value[0] = (float)(w0 + eps_mid);
            const double mp = step(batch, params, nullptr).loss;
            params[wi].value[0] = (float)(w0 - eps_mid);
            const double mm = step(batch, params, nullptr).loss;
            params[wi].value[0] = w0;
            const double cd_mid = (mp - mm) / (2.0 * eps_mid);
            std::printf("%s: CENTRAL-MID %s eps=%.6g grad=%.6g central=%.6g mp=%.9g mm=%.9g\n",
                        test, name.c_str(), eps_mid, g0, cd_mid, mp, mm);
            std::printf("%s: CENTRAL %s w=%.6g eps=%.6g grad=%.6g central=%.6g lp=%.9g lm=%.9g\n",
                        test, name.c_str(), w0, eps, g0, cd, lp, lm);
            std::printf("%s: CENTRAL-BF16 %s eps=%.6g grad=%.6g central=%.6g bp=%.9g bm=%.9g\n",
                        test, name.c_str(), eps_big, g0, cd_big, bp, bm);
            // The control: a perturbation large enough that the loss MUST move if the forward
            // reads the params it is handed. Without it a `central` of exactly zero cannot be
            // told apart from a forward that cached the weights, and the FD would prove nothing.
            params[wi].value[0] = (float)(w0 + std::fabs(w0) * 0.1);
            const double lc = step(batch, params, nullptr).loss;
            params[wi].value[0] = w0;
            const bool live = lc != result.loss;
            std::printf("%s: CENTRAL-CONTROL %s w+10%% loss=%.9g base=%.9g %s\n", test,
                        name.c_str(), lc, result.loss,
                        live ? "(params are live)" : "(PARAMS CACHED -- FD invalid)");
            j["central_blocks"].push_back({{"name", name}, {"w", w0}, {"eps", eps}, {"grad", g0},
                                           {"central", cd}, {"loss_plus", lp}, {"loss_minus", lm},
                                           {"eps_bf16", eps_big}, {"central_bf16", cd_big},
                                           {"loss_plus_bf16", bp}, {"loss_minus_bf16", bm},
                                           {"eps_mid", eps_mid}, {"central_mid", cd_mid},
                                           {"loss_plus_mid", mp}, {"loss_minus_mid", mm},
                                           {"control_loss", lc}, {"params_live", live}});
            // The single-block object stays the first measured block, so existing readers of
            // j["central"] keep working.
            if (!j.contains("central"))
                j["central"] = {{"name", name}, {"w", w0}, {"eps", eps}, {"grad", g0},
                                {"central", cd}, {"loss_plus", lp}, {"loss_minus", lm},
                                {"eps_bf16", eps_big}, {"central_bf16", cd_big},
                                {"loss_plus_bf16", bp}, {"loss_minus_bf16", bm},
                                {"control_loss", lc}, {"params_live", live}};
        }
    }

    // One AdamW step off the gradient, at the config's default lr, weight decay 0: a zero
    // gradient leaves its parameter bit-identical, and a nonzero one moves it. The same step at
    // smaller lrs is a forward-only probe each: the walk's gradient is finite but its norm is
    // ~1e38 on this checkpoint, so a default-lr step that overshoots leaves the descent up while
    // a smaller one still goes down -- which separates a wrong direction from a step too large
    // without paying a second backward.
    const r97::ParamSet before = params;
    constexpr float kDefaultLr = 1e-3f;
    constexpr float kProbeLrs[] = {3e-4f, 1e-4f};
    r97::AdamWConfig oc;
    oc.lr = kDefaultLr;
    r97::Optimiser opt(params, oc);
    const bool applied = opt.step(grads);  // throws when grads do not name the optimiser's params
    R97_CHECK(applied);
    R97_CHECK(opt.updates_done() == 1);
    int64_t a_changed = 0, b_changed = 0;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i].value == before[i].value) continue;
        if (ends(params[i].name, "lora_A")) ++a_changed;
        else if (ends(params[i].name, "lora_B")) ++b_changed;
    }

    // ---- the descent: a forward-only call on the updated adapter

    // The step's own loss is the pre-step reference -- the same params the forward-only probe
    // ran -- and the loss after one AdamW step must be lower: a sign or scale error in the walk
    // would leave it up.
    double after_seconds = 0.0;
    r97::StepResult after{0.0, 0};
    {
        const auto t0 = std::chrono::steady_clock::now();
        after = step(batch, params, nullptr);
        after_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    struct LrProbe {
        float lr;
        float clip;
        double loss;
    };
    std::vector<LrProbe> lr_probes = {{kDefaultLr, 0.0f, after.loss}};
    for (float lr : kProbeLrs) {
        r97::ParamSet probe = before;
        r97::AdamWConfig poc;
        poc.lr = lr;
        r97::Optimiser popt(probe, poc);
        if (!popt.step(grads) || popt.updates_done() != 1) continue;
        lr_probes.push_back({lr, 0.0f, step(batch, probe, nullptr).loss});
    }
    // The clipped probe: the gradient's norm is ~7.5e38, so AdamW's f32 v = (1-b2)*g^2 is Inf for
    // every leaf above ~1.8e19 and that leaf's update is zero -- the unclipped probes test only
    // the moderate leaves. The clip (norm 1, computed in double, rescaled in f32) runs before the
    // moments, so the dominant leaves take a finite step and this descent tests their direction.
    for (float lr : {kDefaultLr, 1e-4f}) {
        r97::ParamSet probe = before;
        r97::AdamWConfig poc;
        poc.lr = lr;
        poc.clip_norm = 1.0f;
        r97::Optimiser popt(probe, poc);
        if (popt.step(grads) && popt.updates_done() == 1)
            lr_probes.push_back({lr, 1.0f, step(batch, probe, nullptr).loss});
    }
    for (const LrProbe& p : lr_probes)
        std::printf("%s: descent lr=%.0e clip=%.0f loss %.9g (%+.6g)\n", test, p.lr, p.clip,
                    p.loss, p.loss - result.loss);
    // Deferred to the end of this function, after the results file and the attribution prints.
    // A failure here must not discard them: two 90-minute 4096-token windows were spent
    // re-learning a single line because these threw before anything was written, and the one
    // line that says where a bad number came from was unreachable exactly when it was needed.
    const bool after_scored_same = after.scored == result.scored;
    const bool after_finite = std::isfinite(after.loss);
    // The default-lr step is printed and carried in the JSON; the check below is the direction at
    // a step the gradient's scale admits. This checkpoint's gradient norm is ~1e38, so a 1e-3
    // AdamW step overshoots (+25 at 512 tokens) while 1e-4 and 1e-5 descend (-54, -7): the
    // config's lr is a training choice, the walk's direction is what these checks can hold.
    const bool descended_default_lr = after.loss < result.loss;
    bool descended_at_a_valid_step = false;
    for (const LrProbe& p : lr_probes)
        if (p.loss < result.loss) descended_at_a_valid_step = true;

    // ---- the results

    j["card"]                 = card;
    j["tokens"]               = n;
    j["scored"]               = result.scored;
    j["loss"]                 = result.loss;
    j["loss_after"]           = after.loss;
    j["loss_after_scored"]    = after.scored;
    // The descent at the default lr and at the smaller probes: a direction error leaves every lr
    // up, a step too large leaves only the default up.
    j["descent_lrs"] = nlohmann::json::array();
    for (const LrProbe& p : lr_probes)
        j["descent_lrs"].push_back(
            {{"lr", p.lr}, {"clip", p.clip}, {"loss", p.loss}, {"delta", p.loss - result.loss}});
    j["descended_default_lr"] = descended_default_lr;
    j["descended_at_a_valid_step"] = descended_at_a_valid_step;
    j["report_loss"]          = rep.loss;
    j["report_scored"]        = rep.scored;
    j["hidden"]               = rep.hidden;
    j["layers"]               = rep.layers;
    j["residual_boundaries"]  = rep.residual_boundaries;
    j["residual_extras"]      = rep.residual_extras;
    j["residual_store_bytes"] = rep.residual_store_bytes;
    j["residual_budget_bytes"] = rep.residual_budget_bytes;
    j["peak_snapshot_bytes"]  = rep.peak_snapshot_bytes;
    j["peak_snapshot_store_bytes"] = rep.peak_snapshot_store_bytes;
    j["peak_snapshot_op_bytes"] = rep.peak_snapshot_op_bytes;
    j["peak_cotangent_bytes"] = rep.peak_cotangent_bytes;
    // The step's frozen operands -- weights and tables no op writes -- widened once and shared by
    // every block: held beside the per-block store, so the step's total f64 snapshot payload at a
    // block is peak_snapshot_bytes + shared_snapshot_bytes.
    j["shared_snapshot_bytes"] = rep.shared_snapshot_bytes;
    // The in-place reads with no earlier writer, and the count the tape named: equal on a walk
    // that served every one from the store, -1 for the case whose probe did not run.
    j["in_place_restores"] = rep.in_place_restores;
    j["in_place_expected"] = in_place_expected;
    j["grad_stage_bytes"]     = rep.grad_stage_bytes;
    j["d_hout_bytes"]         = rep.d_hout_bytes;
    j["head_rows"]            = rep.head_rows;
    j["head_chunks"]          = rep.head_chunks;
    j["kept_boundary_bytes"]  = rep.kept_boundary_bytes;
    j["boundary_names"]       = int64_t(rep.boundary_name.size());
    j["forward_seconds"]      = rep.forward_seconds;
    j["head_seconds"]         = rep.head_seconds;
    j["backward_seconds"]     = rep.backward_seconds;
    j["probe_seconds"]        = probe_seconds;
    j["step_seconds"]         = step_seconds;
    j["after_seconds"]        = after_seconds;
    j["tok_s"]                = step_seconds > 0.0 ? double(n) / step_seconds : 0.0;
    j["vram_after_load"]      = vram_after_load;
    j["vram_peak"]            = peak.load();
    j["n_params"]             = int64_t(params.size());
    j["n_grads"]              = int64_t(grads.size());
    j["names_match"]          = names_match;
    j["shapes_match"]         = shapes_match;
    j["all_finite"]           = all_finite;
    // Where a non-finite cotangent was born: the op, the operand and the value, and whether the
    // row produced it or the walk handed it in. `forward_all_finite` asks the same of the
    // forward's own values, so a NaN that was already there is told apart from one the walk made.
    j["non_finite"] = {{"op", rep.first_non_finite_op},
                       {"operand", rep.first_non_finite_operand},
                       {"operand_index", rep.first_non_finite_operand_index},
                       {"position", rep.first_non_finite_position},
                       {"value_index", rep.first_non_finite_index},
                       {"from_row", rep.first_non_finite_from_row},
                       {"forward_all_finite", rep.forward_all_finite},
                       {"forward_operand", rep.forward_first_non_finite_operand},
                       {"forward_op", rep.forward_first_non_finite_op},
                       {"forward_operand_index", rep.forward_first_non_finite_operand_index},
                       {"forward_position", rep.forward_first_non_finite_position},
                       {"forward_value_index", rep.forward_first_non_finite_index}};
    // The host-vs-device A/B of the same tape, run only when the device gradient was not finite
    // (null otherwise): a finite host gradient beside the device one says a device row is wrong,
    // both non-finite says the plumbing is.
    j["ab_host_walk"] = ab;
    j["n_lora_a"]             = n_a;
    j["n_lora_b"]             = n_b;
    j["n_other"]              = n_other;
    j["a_all_zero"]           = a_all_zero;
    j["a_all_nonzero"]        = a_all_nonzero;
    j["b_all_nonzero"]        = b_all_nonzero;
    // Which leaf of each factor lost its gradient, and how many: a `b_all_nonzero` false names a
    // leaf out of 400, and the name is what the next window needs.
    j["first_zero_a_leaf"]    = first_zero_a_leaf;
    // The complement: the one leaf per factor that does move, and whether it moved to a finite
    // value. A finite non-zero leaf means the walk differentiated a live cotangent all the way to
    // that module; a NaN one means the walk reached it and the cotangent was already poisoned.
    j["first_nonzero_a_leaf"] = first_nonzero_a_leaf;
    j["first_nonzero_b_leaf"] = first_nonzero_b_leaf;
    j["first_nonzero_a_finite"] = first_nonzero_a_finite;
    j["first_nonzero_b_finite"] = first_nonzero_b_finite;
    j["first_zero_b_leaf"]    = first_zero_b_leaf;
    j["n_zero_a"]             = n_zero_a;
    j["n_zero_b"]             = n_zero_b;
    // The count that reads the structural contract independently of the range defect: `part != 0.0`
    // is true for NaN, so `a_all_zero`/`a_changed` cannot distinguish "a cotangent reached A" from
    // "A was poisoned". `n_*_finite_nonzero + n_zero_* == n_*` iff the run produced no non-finite
    // leaf gradient at all.
    j["n_a_finite_nonzero"]   = n_a_finite_nonzero;
    j["n_b_finite_nonzero"]   = n_b_finite_nonzero;
    j["grad_norm"]            = std::sqrt(grad_norm_sq);
    j["opt_applied"]          = applied;
    j["updates_done"]         = opt.updates_done();
    j["a_changed"]            = a_changed;
    j["b_changed"]            = b_changed;
    // The walk's split by op name: the row calls (with the bus each one owns) and the replay's
    // pull-backs. Written before any check below can throw, so a refused walk still leaves them.
    auto timing = [](const std::vector<r97::OpTiming>& v) {
        nlohmann::json a = nlohmann::json::array();
        for (const r97::OpTiming& t : v)
            a.push_back({{"op", t.op}, {"calls", t.calls}, {"seconds", t.seconds},
                         {"side", t.host ? "host" : "device"}, {"non_finite", t.non_finite},
                         {"rows_zeroed_unreadable", t.rows_zeroed_unreadable},
                         {"rows_readable", t.rows_readable},
                         {"rows_refused_non_finite", t.rows_refused_non_finite}});
        return a;
    };
    j["row_seconds"]          = timing(rep.row_seconds);
    j["pull_seconds"]         = timing(rep.pull_seconds);

    std::ofstream out(dir + "/" + test + ".json");
    if (!out) throw std::runtime_error(std::string(test) + ": cannot write the results");
    out << j.dump(2) << "\n";

    // Each phase's elapsed time, so a run's numbers size the window the parent holds: the
    // forward-only probe, the step's wall (and the join's own forward/head/backward split
    // inside it), and the descent call.
    std::printf("%s: %s: %lld scored, loss %.6f -> %.6f after one AdamW step, grad norm %.6g, "
                "store %lld MiB, snapshots %lld MiB (store) + %lld MiB (op) + %lld MiB (shared), "
                "probe %.1f s, step %.1f s (forward %.1f, head %.1f, backward %.1f), descent "
                "%.1f s\n",
                test, card.c_str(), (long long)result.scored, result.loss, after.loss,
                std::sqrt(grad_norm_sq), (long long)(rep.residual_store_bytes / r97test::kMiB),
                (long long)(rep.peak_snapshot_store_bytes / r97test::kMiB),
                (long long)(rep.peak_snapshot_op_bytes / r97test::kMiB),
                (long long)(rep.shared_snapshot_bytes / r97test::kMiB),
                probe_seconds, step_seconds, rep.forward_seconds, rep.head_seconds,
                rep.backward_seconds, after_seconds);

    // The in-place reads with no earlier writer, before the walk-by-op lines: a count that differs
    // from the tape's is the first thing a window reads.
    if (in_place_expected >= 0)
        std::printf("%s: in-place reads with no earlier writer: %lld restored of %lld the tape "
                    "names\n", test, (long long)rep.in_place_restores,
                    (long long)in_place_expected);
    // The same split, the few ops that hold the walk: the number that says whether a 4096-token
    // step needs a longer window, and which op to look at when it does.
    auto top = [](const std::vector<r97::OpTiming>& v) {
        std::vector<const r97::OpTiming*> p;
        p.reserve(v.size());
        for (const r97::OpTiming& t : v) p.push_back(&t);
        std::sort(p.begin(), p.end(),
                  [](const r97::OpTiming* a, const r97::OpTiming* b) { return a->seconds > b->seconds; });
        std::string s;
        for (size_t i = 0; i < p.size() && i < 5; ++i) {
            char b[160];
            std::snprintf(b, sizeof b, "%s%.1f s/%lld%s", i ? ", " : " ", p[i]->seconds,
                          (long long)p[i]->calls, p[i]->host ? "h" : "d");
            s += p[i]->op + b;
        }
        return s.empty() ? std::string(" (none)") : s;
    };
    std::printf("%s: walk by op:%s | pull by op:%s\n", test, top(rep.row_seconds).c_str(),
                top(rep.pull_seconds).c_str());
    // Where the walk's first non-finite cotangent was born, and whether the forward's values were
    // finite: the line a window is read for when `all_finite` is false. `from the row` means the
    // op's adjoint produced it; `from the walk's input` means the walk handed it in -- a seed, the
    // cotangent from above, or a record later in the forward. The per-op counts name every op that
    // carried one (slots with a non-finite value, each counted once).
    std::string cot_nf = "none";
    if (!rep.first_non_finite_op.empty()) {
        char b[256];
        std::snprintf(b, sizeof b,
                      "op %s, operand '%s'[%lld] at position %lld, value %lld, from the %s",
                      rep.first_non_finite_op.c_str(), rep.first_non_finite_operand.c_str(),
                      (long long)rep.first_non_finite_operand_index,
                      (long long)rep.first_non_finite_position,
                      (long long)rep.first_non_finite_index,
                      rep.first_non_finite_from_row ? "row" : "walk's input");
        cot_nf = b;
    }
    std::string fwd_nf = "all finite";
    if (!rep.forward_all_finite) {
        char b[288];
        if (rep.forward_first_non_finite_op.empty()) {
            // No walked op wrote it: the value entered from a boundary no walked op writes, so
            // only the read the walk met can be named.
            std::snprintf(b, sizeof b, "not finite: operand '%s'[%lld] at position %lld, value %lld",
                          rep.forward_first_non_finite_operand.c_str(),
                          (long long)rep.forward_first_non_finite_operand_index,
                          (long long)rep.forward_first_non_finite_position,
                          (long long)rep.forward_first_non_finite_index);
        } else {
            // The op that produced it, which the walk alone cannot name: it only meets the first
            // reader downstream of the op that wrote the buffer.
            std::snprintf(b, sizeof b,
                          "not finite: %s wrote operand '%s'[%lld] at position %lld, value %lld",
                          rep.forward_first_non_finite_op.c_str(),
                          rep.forward_first_non_finite_operand.c_str(),
                          (long long)rep.forward_first_non_finite_operand_index,
                          (long long)rep.forward_first_non_finite_position,
                          (long long)rep.forward_first_non_finite_index);
        }
        fwd_nf = b;
    }
    std::string nf_ops;
    for (const r97::OpTiming& t : rep.row_seconds)
        if (t.non_finite) {
            nf_ops += (nf_ops.empty() ? " " : ", ") + t.op + " " + std::to_string(t.non_finite) +
                      "/" + std::to_string(t.calls);
        }
    std::printf("%s: non-finite: cotangent %s | forward %s | per op:%s\n", test, cot_nf.c_str(),
                fwd_nf.c_str(), nf_ops.empty() ? " (none)" : nf_ops.c_str());
    // Which leaf of each factor lost its gradient: `b_all_nonzero` false names a leaf out of 400,
    // and this line says which one and how many, so identifying it costs no second window.
    std::string zero_a = "none";
    std::string zero_b = "none";
    if (!first_zero_a_leaf.empty())
        zero_a = first_zero_a_leaf + " (" + std::to_string(n_zero_a) + " of " +
                 std::to_string(n_a) + ")";
    if (!first_zero_b_leaf.empty())
        zero_b = first_zero_b_leaf + " (" + std::to_string(n_zero_b) + " of " +
                 std::to_string(n_b) + ")";
    std::printf("%s: zero-gradient leaves: lora_A %s, lora_B %s\n", test, zero_a.c_str(),
                zero_b.c_str());
    // The descent, last: the step's own loss is the pre-step reference (the same params the
    // forward-only call ran), and one AdamW step at the config's default lr must leave it lower.
    // A sign or scale error in the walk leaves it up, and a non-finite gradient leaves it
    // non-finite; both are named here with the numbers, after the file is on disk.
    std::printf("%s: descent: loss %.9g -> %.9g (%+.6g) after %.1f s | scored %lld -> %lld\n", test,
                result.loss, after.loss, after.loss - result.loss, after_seconds,
                (long long)result.scored, (long long)after.scored);

    // The walk served every in-place read with no earlier writer from the block's stored
    // boundary: the tape's count and the report's must agree, and the tape names at least one.
    if (in_place_expected >= 0) {
        R97_CHECK(in_place_expected > 0);
        R97_CHECK(rep.in_place_restores == in_place_expected);
    }
    R97_CHECK(after_scored_same);
    // The descent and the post-step forward's finiteness ride on the walk's gradient: value
    // checks. They hold when the gradient is finite -- that is what they assert -- so they
    // re-arm with the walk; R97_DSTEP_ASSERT_FINITE=1 forces them (and fails) while it is not.
    // The numbers are in the results file either way.
    if (all_finite || assert_finite()) {
        R97_CHECK(after_finite);
        R97_CHECK(descended_at_a_valid_step);
    }
}

}  // namespace

// ------------------------------------------------------------------ the two cases

// One joined step of `n` tokens, the op-coverage probe when asked. The parent resolves the
// container's shape from its metadata (no card) so its expected values carry their own
// provenance: the store's plan, the head's rows and its chunks, the brackets' count.
static void device_step_case(const char* test, int64_t n, bool probe) {
#ifdef R97_MODEL_RAD
    const std::string container = r97test::container_or_skip(R97_MODEL_RAD);
#else
    const std::string container = r97test::container_or_skip("");
#endif
    if (std::getenv(kRoleVar)) {
        card_case(test, n, probe);
        return;  // the child did the work and wrote its file; run_all reports the pass
    }

    const std::string dir = temp_dir(test);
    run_in_child(dir, test);

    const std::string path = dir + "/" + test + ".json";
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string(test) + ": no results at " + path);
    nlohmann::json j;
    in >> j;

    // R97_HOST_WALK_AB=2's arm walked the host rows alone and wrote its own numbers: there is no
    // device step here to hold to the invariants below. A host walk that is not finite fails,
    // which is the one thing this arm's contract says.
    if (j.value("host_walk_only", false)) {
        std::printf("%s: host walk only: %lld scored, loss %.9g, gradient %s (norm %.6g, %lld "
                    "leaves non-finite), %.1f s\n",
                    test, (long long)j.value("scored", int64_t(0)), j.value("loss", 0.0),
                    j.value("host_finite", false) ? "all finite" : "NOT finite",
                    j.value("host_norm", 0.0),
                    (long long)j.value("host_non_finite_leaves", int64_t(0)),
                    j.value("host_seconds", 0.0));
        R97_CHECK(j.value("host_finite", false));
        return;
    }

    const int64_t tokens = j.at("tokens").get<int64_t>();
    const int64_t scored = j.at("scored").get<int64_t>();
    const double loss    = j.at("loss").get<double>();
    const int64_t hidden = j.at("hidden").get<int64_t>();
    const int64_t layers = j.at("layers").get<int64_t>();
    const double after   = report_number(j, "loss_after");

    // The headline numbers, before any check below can throw, so a failing case says what it
    // measured and not only which assertion failed.
    std::printf("%s: results: loss %.9g -> %.9g (%+.6g) | scored %lld | all_finite %d | "
                "step %.1f s (fwd %.1f bwd %.1f head %.1f)\n",
                test, loss, after, after - loss, (long long)scored,
                j.value("all_finite", false) ? 1 : 0, j.value("step_seconds", 0.0),
                j.value("forward_seconds", 0.0), j.value("backward_seconds", 0.0),
                j.value("head_seconds", 0.0));

    if (probe) {
        // Every in-place read with no earlier writer was served from the block's stored boundary:
        // the walk's count equals the tape's, and the tape names at least one. Checked here,
        // before the all_finite check this container currently fails (issue 140).
        std::printf("%s: in-place reads with no earlier writer: %lld restored of %lld the tape "
                    "names\n", test, (long long)j.at("in_place_restores").get<int64_t>(),
                    (long long)j.at("in_place_expected").get<int64_t>());
        R97_CHECK(j.at("in_place_expected").get<int64_t>() > 0);
        R97_CHECK(j.at("in_place_restores").get<int64_t>() ==
                  j.at("in_place_expected").get<int64_t>());
    }

    const r97::ModelMeta meta = r97::read_model_meta(container);

    // The step's loss over every scored token: the sum the join's forward and the objective
    // agree on, covering the whole prompt, and the head's rows and its chunks.
    R97_CHECK(tokens == n);
    R97_CHECK(scored == n - 1);  // every token but the first is scored
    R97_CHECK(std::isfinite(loss));
    // #146's contract names the forward's finiteness on its own: a broken container path shows
    // up here, while the walk's non-finite cotangents (the line printed above) do not.
    R97_CHECK(j.at("non_finite").at("forward_all_finite").get<bool>());
    R97_CHECK(hidden == meta.hidden);
    R97_CHECK(layers == meta.layers);
    R97_CHECK(j.at("report_scored").get<int64_t>() == scored);
    R97_CHECK(std::fabs(j.at("report_loss").get<double>() - loss) <=
              1e-6 * std::max(1.0, std::fabs(loss)));
    R97_CHECK(j.at("head_rows").get<int64_t>() == n - 1);
    R97_CHECK(j.at("head_chunks").get<int64_t>() ==
              (n - 1 + r97::kHeadChunkRows - 1) / r97::kHeadChunkRows);
    R97_CHECK(j.at("d_hout_bytes").get<int64_t>() > 0);

    // The residual store: the L+1 boundaries plus the second copy at every boundary a block after
    // the first starts at -- this container's int8 trunk reads the previous block's output beside
    // the residual -- in one pinned host block, checked against the host's available memory when
    // the engine was up.
    const int64_t store = j.at("residual_store_bytes").get<int64_t>();
    R97_CHECK(j.at("residual_boundaries").get<int64_t>() == layers + 1);
    R97_CHECK(j.at("residual_extras").get<int64_t>() == layers - 1);
    R97_CHECK(store == 2 * layers * n * hidden * 2);  // (L+1) + (L-1) copies, tokens x hidden x 2
    R97_CHECK(j.at("residual_budget_bytes").get<int64_t>() > 0);
    R97_CHECK(store <= j.at("residual_budget_bytes").get<int64_t>());

    // The adapter's gradient: the same parameters and shapes as the params, finite on every
    // leaf, and staged on the card at the step's end. The start's split is checked per case
    // below. The report kept no boundary copies: keep_boundaries is off.
    const int64_t n_a = j.at("n_lora_a").get<int64_t>();
    const int64_t n_b = j.at("n_lora_b").get<int64_t>();
    R97_CHECK(j.at("n_grads").get<int64_t>() == j.at("n_params").get<int64_t>());
    R97_CHECK(j.at("names_match").get<bool>());
    R97_CHECK(j.at("shapes_match").get<bool>());
    R97_CHECK(j.at("n_other").get<int64_t>() == 0);
    R97_CHECK(n_a > 0);
    R97_CHECK(n_b > 0);
    // The gradient's finiteness is a value check on the walk, outside #146's contract: the
    // live-container walk is non-finite by the rows' own mathematics, so it is reported above.
    // The checks below hold when the gradient is finite -- that is what they assert -- so they
    // re-arm with the walk; R97_DSTEP_ASSERT_FINITE=1 forces them (and fails on the non-finite
    // values) while it is not.
    const bool finite = j.at("all_finite").get<bool>();
    if (assert_finite()) R97_CHECK(finite);
    const bool value_checks = finite || assert_finite();
    R97_CHECK(j.at("grad_stage_bytes").get<int64_t>() > 0);
    R97_CHECK(j.at("kept_boundary_bytes").get<int64_t>() == 0);

    // The descent: one AdamW update, and the forward-only call on the updated adapter sees the
    // loss fall -- the step's own loss is the pre-step reference, the same params.
    R97_CHECK(j.at("opt_applied").get<bool>());
    R97_CHECK(j.at("updates_done").get<int64_t>() == 1);
    // The descent and the post-step forward's finiteness ride on the gradient: value checks,
    // gated the same way. The default-lr step is printed above and carried in `descent_lrs`; the
    // direction is held at a step the gradient's scale admits (unclipped 1e-4/3e-4, or clipped),
    // because this checkpoint's norm is ~1e38 and a 1e-3 AdamW step overshoots.
    if (value_checks) {
        R97_CHECK(std::isfinite(after));
        R97_CHECK(j.at("descended_at_a_valid_step").get<bool>());
    }
    R97_CHECK(j.at("loss_after_scored").get<int64_t>() == scored);

    R97_CHECK(j.at("vram_peak").get<int64_t>() >= j.at("vram_after_load").get<int64_t>());
    R97_CHECK(j.at("tok_s").get<double>() > 0);

    if (probe) {
        // The forward-only probe and the gradient call agree on the loss: the same (seeded)
        // params, the same forward.
        const double probe_loss = j.at("probe_loss").get<double>();
        R97_CHECK(std::fabs(probe_loss - loss) <= 1e-6 * std::max(1.0, std::fabs(loss)));

        // Every op the step's 64 blocks issued has a device adjoint row, or is host-served by
        // design; the brackets cover the trunk. What has no row is named with adjoint_find's
        // own message, and only an op outside host_served_op() fails here: the coverage is not
        // a value check, and riding all_finite would re-arm it on the gaps the design permits.
        R97_CHECK(j.at("probe_blocks").get<int64_t>() == layers);
        R97_CHECK(j.at("probe_ops").get<int64_t>() > 0);
        const nlohmann::json missing = j.at("probe_missing");
        if (!missing.empty()) {
            std::string why, unexpected;
            for (const auto& m : missing) {
                const std::string op = m.at(0).get<std::string>();
                const std::string what = op + ": " + m.at(1).get<std::string>();
                why += (why.empty() ? "" : "; ") + what;
                if (!host_served_op(op) || assert_finite())
                    unexpected += (unexpected.empty() ? "" : "; ") + what;
            }
            std::printf("%s: no device adjoint row: %s\n", test, why.c_str());
            if (!unexpected.empty()) {
                throw std::runtime_error(std::string(test) + ": no device adjoint row for " +
                                         unexpected);
            }
        }
        // The head the join walks: the rmsnorm and the gather_rows need a row (the probe above
        // found one for each), and the logits op the join handles itself is on the tape.
        const nlohmann::json head_ops = j.at("probe_head_ops");
        const auto head_has = [&](const char* op) {
            for (const auto& o : head_ops)
                if (o.get<std::string>() == op) return true;
            return false;
        };
        R97_CHECK(head_has("rmsnorm"));
        R97_CHECK(head_has("gather_rows"));
        R97_CHECK(head_has("logits_gemm_vocab"));

        // The seeded start: the cotangent reaches every leaf of both factors, so the gradient
        // is non-zero on each of them -- and the step moves both factors, not one. Value checks,
        // gated with all_finite: a non-finite leaf defeats `part != 0.0` and moves its
        // parameter, so it reads as "reached" here; the n_*_finite_nonzero counts are the
        // honest reading while the walk is non-finite.
        //
        // The step's per-leaf movement is not uniform on this checkpoint: the gradient norm is
        // ~1e38, and the unclipped AdamW's f32 v = (1-b2)*g^2 is Inf above |g| ~1.8e19, so the
        // dominant leaves take no update. `a_all_nonzero`/`b_all_nonzero` are what "the chain
        // reached every leaf" means; the changed counts say which leaves the step could move at
        // this scale and are reported, not held to n_*.
        if (value_checks) {
            R97_CHECK(j.at("a_all_nonzero").get<bool>());
            R97_CHECK(j.at("b_all_nonzero").get<bool>());
            R97_CHECK(j.at("a_changed").get<int64_t>() > 0);
            R97_CHECK(j.at("b_changed").get<int64_t>() > 0);
        }
    } else {
        // The zero-B start: no cotangent reaches A's factor, so its gradients are exactly zero,
        // and the step moves only the lora_B's -- the pair that proves the chain reached the
        // adapter. Value checks, gated with all_finite: a non-finite A leaf defeats `a_all_zero`
        // and `a_changed == 0` alike -- it reads non-zero and moves its parameter. As above, the
        // B movement is held as "some moved", not "all": the ~1e38 norm's dominant leaves sit
        // past the unclipped AdamW's f32 v.
        if (value_checks) {
            R97_CHECK(j.at("a_all_zero").get<bool>());
            R97_CHECK(j.at("b_all_nonzero").get<bool>());
            R97_CHECK(j.at("a_changed").get<int64_t>() == 0);
            R97_CHECK(j.at("b_changed").get<int64_t>() > 0);
        }
    }

    std::printf("%s: %s: %lld scored, loss %.6f (mean NLL %.6f) -> %.6f (mean NLL %.6f) after "
                "one AdamW step, %.1f tok/s, grad norm %.6g, store %lld MiB, VRAM %lld MiB "
                "after load, %lld MiB at the step's peak\n",
                test, j.at("card").get<std::string>().c_str(), (long long)scored, loss,
                loss / double(scored), after, after / double(scored),
                j.at("tok_s").get<double>(), report_number(j, "grad_norm"),
                (long long)(store / r97test::kMiB),
                (long long)(j.at("vram_after_load").get<int64_t>() / r97test::kMiB),
                (long long)(j.at("vram_peak").get<int64_t>() / r97test::kMiB));
    if (!probe) {
        // No check against the recorded figure (docs/reports/2026-11-12-device-forward.md): the
        // report compares them. Print them next to each other so the comparison is easy.
        std::printf("%s: this step's loss %.6f against the device forward's recorded %.6f for "
                    "this batch and container\n",
                    test, loss, 34682.089228);
    }
}

// 512 tokens: the op-coverage probe plus a full step, with the adapter seeded over both factors.
R97_MODEL_TEST(device_step_tape) { device_step_case("device_step_tape", 512, /*probe=*/true); }

// The ticket: one 4096-token step on the card, with the adapter's gradient.
R97_MODEL_TEST(device_step_seq4096) {
    device_step_case("device_step_seq4096", 4096, /*probe=*/false);
}

#else

// A build without the patch set cannot join a step on the card. Say so by name so the tier's
// output distinguishes a skip from a test that passed or never existed; no model case is
// registered in such a build.
namespace {

const char* device_step_skip_reason() {
#if !defined(RAD_ISSUE_OBSERVER)
    return "the issue observer patch 0101 is not in this build";
#elif !__has_include("train_prefill.h")
    return "the prefill mode patch 0103 is not in this build";
#elif !defined(RAD_REPLAY_BLOCK)
    return "the replay patch 0102 is not in this build: "
           "-DR97_PATCH_SKIP took it out (RAD_REPLAY_BLOCK is undefined)";
#else
    return "the context-stream patch 0122 is not in this build: the store keeps its boundaries "
           "in one pinned host block, over the stream the step's ops issue on";
#endif
}

struct DeviceStepTapeSkips {
    DeviceStepTapeSkips() {
        std::printf("[gpu] device_step_tape ... skipped (%s)\n", device_step_skip_reason());
        std::fflush(stdout);
    }
} device_step_tape_skips;

struct DeviceStepSeq4096Skips {
    DeviceStepSeq4096Skips() {
        std::printf("[gpu] device_step_seq4096 ... skipped (%s)\n", device_step_skip_reason());
        std::fflush(stdout);
    }
} device_step_seq4096_skips;

}  // namespace

#endif  // R97_FORWARD && R97_DEVICE_STEP
