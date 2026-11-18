// tests/support/device_check.h -- the device-versus-host harness every device adjoint uses.
//
// A device adjoint is an ordinary AdjointFn (include/r97/adjoint.h): it is handed the record, the
// operands and a cotangent, and adds to the same dc buffers the host adjoint does. It reads the
// cotangent and computes in f32, then rounds its result to the op's dtype, so it cannot be held to
// f64 equality; it is held to the bf16 tolerance below instead, against the host adjoint of the
// same op (or a composition of host adjoints the test supplies, for a kernel that fuses several
// ops).
//
// The harness:
//   * builds a case's forward values with the op's f64 twin, so the device adjoint reads the same
//     values a host adjoint would, and starts both dc sets at zero;
//   * runs the registered device adjoint (or one handed in, for a deliberately wrong kernel) and
//     the host reference on those values;
//   * resets the device scratch ledger, then reports the peak the device kernel asked for and
//     fails the case when a kernel leaked;
//   * with --bench, times the device adjoint and the op's forward twin in the same run and prints
//     the line docs/kernels/README.md asks a kernel's doc to carry.
//
// The plugin: the device adjoints live in libr97.so, next to libr4d. load_plugin() dlopen()s it
// and leaves it open -- r97's registry holds function pointers into it, and HIP keeps the code
// object it registered at load for as long as the handle lives. R97_DEVICE_SO overrides the path;
// the build passes the exact file it built as R97_DEVICE_PLUGIN.
#pragma once

#include "r97/adjoint.h"

#include "adjoint/device/scratch.h"
#include "rad_device.h"

#include "bench.h"
#include "gpu_guard.h"

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace r97 {
namespace device_check {

// The tolerance a device kernel that rounds to bf16 once and accumulates in f32 is held to.
// bf16 carries 8 significand bits, so one unit roundoff is 2^-8; a reduction over a few hundred
// terms moves its result by a few of those, and 2% leaves that room without hiding a kernel that
// computes another function. The floor keeps a value that rounds to zero out of a relative test.
inline constexpr double kRelTol = 0.02;
inline constexpr double kAbsTol = 1e-3;

inline int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

// The values a case hands both sides: operands in schema order, cotangents in the writes' order.
struct Case {
    OpRecord rec;
    std::vector<AdValue> in;   // read operands, values in [-1, 1)
    std::vector<AdValue> cot;  // one per written operand, the same shapes, values in [-1, 1)
    std::string label;         // for the failure message and the bench line
};

// A host reference that is not simply the op's host row: run host adjoints in the order the fused
// kernel undoes them. The harness hands it a call with zeroed dc, the same as the registry row.
using HostRef = std::function<void(AdjointCall&)>;

struct Report {
    std::string label;
    bool ok = true;
    std::string note;
    int64_t first_slot = -1;   // the operand the first mismatch is in
    int64_t first_index = -1;  // the element within it
    double first_device = 0.0;
    double first_host = 0.0;
    int64_t mismatches = 0;
    int64_t scratch_peak = 0;  // bytes of device scratch the kernel held at its high-water mark
};

inline void require(const Report& r) {
    if (r.ok) return;
    throw std::runtime_error(r.note);
}

// -- card and plugin ---------------------------------------------------------

inline r97test::GuardOptions guard_options(int64_t need_vram = int64_t(1) << 30) {
    r97test::GuardOptions o;
    const char* env = std::getenv("R97_DEVICE");
    o.device = r97test::parse_device(env && *env ? env : "0");
    o.need_vram = need_vram;
    o.bdf = r97test::hip_bdf(o.device);
    return o;
}

// Opens the card through radiance's device layer, after the caller's DeviceGuard took the lock.
// A build that fell back to the host backend has no device kernels to test.
inline void require_device(int device) {
    if (rad_dev_count() < 1)
        throw std::runtime_error("no device behind radiance's device layer (rad_dev_count=0); "
                                 "the gpu tier needs a card");
    RadDeviceProps props{};
    if (rad_dev_props(device, &props) != 0)
        throw std::runtime_error(std::string("rad_dev_props: ") + rad_dev_last_error());
    if (props.is_host_backend)
        throw std::runtime_error(std::string("radiance runs its host backend on ") + props.name
                                 + "; the gpu tier needs a card");
    if (rad_dev_set(device) != 0)
        throw std::runtime_error("rad_dev_set(" + std::to_string(device)
                                 + "): " + rad_dev_last_error());
}

inline bool file_exists(const std::string& p) {
    return !p.empty() && access(p.c_str(), R_OK) == 0;
}

inline std::string exe_dir() {
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n <= 0) return std::string();
    buf[n] = '\0';
    std::string p(buf);
    const size_t slash = p.find_last_of('/');
    return slash == std::string::npos ? std::string() : p.substr(0, slash);
}

inline std::string plugin_path() {
    if (const char* env = std::getenv("R97_DEVICE_SO"); env && *env) return env;
#ifdef R97_DEVICE_PLUGIN
    if (file_exists(R97_DEVICE_PLUGIN)) return R97_DEVICE_PLUGIN;
#endif
    if (const char* home = std::getenv("RADIANCE_HOME"); home && *home) {
        const std::string p = std::string(home) + "/kernels/libr97.so";
        if (file_exists(p)) return p;
    }
    const std::string dir = exe_dir();
    if (!dir.empty()) {
        for (const char* rel : {"/kernels/libr97.so", "/radiance-build/radiance_home/kernels/libr97.so",
                                "/../radiance-build/radiance_home/kernels/libr97.so"}) {
            const std::string p = dir + rel;
            if (file_exists(p)) return p;
        }
    }
    return std::string();
}

// dlopen()s libr97.so once and leaves it open. The plugin registers each device adjoint into this
// process's r97 registry from a static initializer, so the lookup below finds it; HIP registers
// the code object the plugin carries when the file loads.
inline void* load_plugin() {
    static void* handle = []() -> void* {
        const std::string p = plugin_path();
        if (p.empty())
            throw std::runtime_error(
                "no libr97.so: this build has no device plugin (configured without HIP) and "
                "RADIANCE_HOME holds none; set R97_DEVICE_SO");
        void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) throw std::runtime_error("dlopen(" + p + "): " + dlerror());
        return h;
    }();
    return handle;
}

// -- values ------------------------------------------------------------------

inline uint64_t splitmix64(uint64_t* s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

inline double urand(uint64_t* s) {
    return (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
}

inline AdValue random_value(const std::string& dtype, std::vector<int64_t> shape, uint64_t* seed) {
    AdValue v;
    v.dtype = dtype;
    v.shape = std::move(shape);
    v.value.resize((size_t)numel(v.shape));
    for (double& x : v.value) x = urand(seed) * 2.0 - 1.0;
    return v;
}

// -- the comparison ----------------------------------------------------------

// Builds one call over a case: forward values from the twin, cotangents as the case gives them,
// and a zeroed dc with one slot per operand in schema order. The caller owns the vectors the
// returned call refers to.
inline AdjointCall make_call(const Case& c, std::vector<AdValue>* fwd, std::vector<AdValue>* dc) {
    TwinFn twin = twin_find(c.rec.op, c.rec.params, nullptr);
    if (!twin) throw std::runtime_error("no twin for op '" + c.rec.op + "'");

    std::vector<AdValue> out;
    twin(c.rec, c.in, &out);

    // The forward in schema order: a read takes the caller's next read operand, a write the twin's
    // next output. An in-place write is both -- the caller carries the pre-op value as an extra
    // last read (the host rows' convention), the twin consumes it, and the schema position the
    // adjoint reads holds the post-op value the twin produced. Appending every output after every
    // read, as this did, is the same order only when the record's writes all follow its reads; an
    // aliased slot puts a write before later reads, and the row then differentiated at the wrong
    // operand.
    fwd->clear();
    size_t read_at = 0, write_at = 0;
    for (const Operand& op : c.rec.operands) {
        if (op.written) {
            if (write_at >= out.size())
                throw std::runtime_error("device_check: op '" + c.rec.op + "' write '" +
                                         op.buffer + "' has no twin output");
            fwd->push_back(out[write_at++]);
        } else {
            if (read_at >= c.in.size())
                throw std::runtime_error("device_check: op '" + c.rec.op + "' read '" +
                                         op.buffer + "' has no value");
            fwd->push_back(c.in[read_at++]);
        }
    }

    dc->clear();
    for (const Operand& op : c.rec.operands) {
        AdValue v;
        v.dtype = op.dtype;
        v.shape = op.shape;
        v.value.assign((size_t)numel(op.shape), 0.0);
        dc->push_back(std::move(v));
    }
    return AdjointCall{c.rec, *fwd, c.cot, dc, nullptr, ""};
}

inline std::string op_name(const Case& c, size_t slot) {
    if (slot < c.rec.operands.size()) return c.rec.operands[slot].buffer;
    return "slot " + std::to_string(slot);
}

// Runs `device` and the host reference and compares every slot of dc. An empty `device` looks up
// the op's registered device row; an empty `host` looks up its registered host row. Both lookups
// are reported rather than thrown, so a test can assert that a missing registration fails too.
inline Report check(const Case& c, AdjointFn device = nullptr, const HostRef& host = HostRef()) {
    Report r;
    r.label = c.label;

    std::string err;
    if (!device) device = adjoint_find(c.rec.op, Side::device, c.rec.params, &err);
    if (!device) {
        r.ok = false;
        r.note = c.label + ": device adjoint: " + err;
        return r;
    }
    HostRef reference = host;
    if (!reference) {
        AdjointFn h = adjoint_find(c.rec.op, Side::host, c.rec.params, &err);
        if (!h) {
            r.ok = false;
            r.note = c.label + ": host adjoint: " + err;
            return r;
        }
        reference = [h](AdjointCall& call) { h(call); };
    }

    std::vector<AdValue> device_fwd, device_dc;
    device::scratch_reset();
    {
        AdjointCall call = make_call(c, &device_fwd, &device_dc);
        device(call);
    }
    const device::ScratchStats stats = device::scratch_stats();
    r.scratch_peak = stats.peak_bytes;
    if (stats.live_bytes != 0) {
        r.ok = false;
        r.note = c.label + ": device kernel leaked " + std::to_string(stats.live_bytes)
                 + " bytes of scratch";
        return r;
    }

    // The reference sees the same forward values and the same cotangent as the device did; only
    // the dc slots start at zero again.
    std::vector<AdValue> host_dc = device_dc;
    for (AdValue& v : host_dc) std::fill(v.value.begin(), v.value.end(), 0.0);
    AdjointCall host_call{c.rec, device_fwd, c.cot, &host_dc, nullptr, ""};
    reference(host_call);

    if (device_dc.size() != host_dc.size()) {
        r.ok = false;
        r.note = c.label + ": device wrote " + std::to_string(device_dc.size())
                 + " dc slots, host " + std::to_string(host_dc.size());
        return r;
    }
    for (size_t slot = 0; slot < device_dc.size(); ++slot) {
        const AdValue& d = device_dc[slot];
        const AdValue& h = host_dc[slot];
        if (d.value.size() != h.value.size()) {
            r.ok = false;
            r.note = c.label + ": dc[" + std::to_string(slot) + "] (" + op_name(c, slot)
                     + ") has " + std::to_string(d.value.size()) + " values, host "
                     + std::to_string(h.value.size());
            return r;
        }
        for (size_t i = 0; i < d.value.size(); ++i) {
            const double dv = d.value[i];
            const double hv = h.value[i];
            const double tol = kAbsTol + kRelTol * std::abs(hv);
            if (std::isfinite(dv) && std::abs(dv - hv) <= tol) continue;
            if (r.mismatches++ == 0) {
                r.first_slot = (int64_t)slot;
                r.first_index = (int64_t)i;
                r.first_device = dv;
                r.first_host = hv;
                r.ok = false;
            }
        }
    }
    if (!r.ok) {
        char buf[256];
        std::snprintf(buf, sizeof buf, " (tolerance %.3g + %.3g*|host|)", kAbsTol, kRelTol);
        r.note = c.label + ": dc[" + std::to_string(r.first_slot) + "] (" + op_name(c, r.first_slot)
                 + ") index " + std::to_string(r.first_index) + ": device "
                 + std::to_string(r.first_device) + ", host " + std::to_string(r.first_host) + buf
                 + (r.mismatches > 1 ? ", " + std::to_string(r.mismatches) + " mismatches"
                                     : std::string());
    }
    return r;
}

// -- the bench ---------------------------------------------------------------

struct Timing {
    double adjoint_ms = 0.0;   // the whole call: upload, launch, download
    double forward_ms = 0.0;   // the op's forward twin, f64 on the host
    int64_t scratch_peak = 0;  // bytes of device scratch the adjoint held
};

// Times the device adjoint and the forward op it belongs to, in the same run over the same values.
// The adjoint's number is the whole call, because the contract hands it f64 operands and takes f64
// cotangents back; the forward is the f64 twin until a device forward exists for the op.
inline Timing time_case(const Case& c, int reps = 20) {
    AdjointFn device = adjoint_find(c.rec.op, Side::device, c.rec.params, nullptr);
    if (!device) throw std::runtime_error("no device adjoint for op '" + c.rec.op + "'");
    TwinFn twin = twin_find(c.rec.op, c.rec.params, nullptr);
    if (!twin) throw std::runtime_error("no twin for op '" + c.rec.op + "'");

    Timing t;
    if (reps < 1) reps = 1;

    std::vector<AdValue> fwd, dc;
    AdjointCall call = make_call(c, &fwd, &dc);
    device::scratch_reset();
    device(call);  // warm-up: the first launch is where the kernel's module is finalized
    auto zero_dc = [&dc] {
        for (AdValue& v : dc) std::fill(v.value.begin(), v.value.end(), 0.0);
    };
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        zero_dc();
        device(call);
    }
    auto t1 = std::chrono::steady_clock::now();
    t.adjoint_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    t.scratch_peak = device::scratch_stats().peak_bytes;

    std::vector<AdValue> out;
    twin(c.rec, c.in, &out);
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        out.clear();
        twin(c.rec, c.in, &out);
    }
    auto t3 = std::chrono::steady_clock::now();
    t.forward_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / reps;
    return t;
}

// The line a kernel's doc carries: the op, the shape, both times, the ratio, and the scratch the
// kernel asked for. One line per case, so a doc can quote the run's output.
inline void print_bench(const Case& c, int reps = 20) {
    const Timing t = time_case(c, reps);
    const double ratio = t.forward_ms > 0.0 ? t.adjoint_ms / t.forward_ms : 0.0;
    std::printf("%-28s %8.3f ms adjoint  %8.3f ms forward twin  %6.2fx forward  "
                "scratch peak %lld KiB\n",
                (c.rec.op + " " + c.label).c_str(), t.adjoint_ms, t.forward_ms, ratio,
                (long long)(t.scratch_peak / 1024));
    std::fflush(stdout);
}

}  // namespace device_check
}  // namespace r97
