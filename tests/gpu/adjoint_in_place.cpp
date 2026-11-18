// tests/gpu/adjoint_in_place.cpp -- the device rows on the records the live tape marks in place.
//
// Five ops carry the tape's INOUT mark (`written=1 in_place=1`) on the live container
// (docs/reports/2026-11-16-in-place-fd-checks.md): `lora_expand` 288 records, `rmsnorm_had_quant_i8g`
// 127, `gdn_conv_prep` 48, `rope` 32 and `kv_store` 16. The host rows read the mark; the device
// rows did not, and these are the two whose direct path is a value the walk can get wrong:
//
//   * `kv_store`: the store overwrites the pool rows its slots name, so those rows of the old pool
//     are dead -- their cotangent is the k's and the v's. Only the rows the write leaves alone
//     carry the pool's cotangent on. A device row that leaves the slot empty lets the walk's
//     passthrough copy the whole pool into it, counting the covered rows twice.
//   * `lora_expand`: the write adds the adapter's branch to the buffer it replaces, so the buffer's
//     old value takes the write's cotangent unchanged. The walk's passthrough fills it when the
//     row leaves the slot empty; the row filling it says the same thing when a check calls it
//     without the walk, which is how the aliased record is held to the host row here.
//
// Each op is checked at its aliased record and, where the same row serves it, at the plain write:
// the two differ only in the mark, so a failure names which one it is in. The harness places the
// twin's output at the write's schema position, which is what an aliased slot needs.
//
// Needs one R9700 (gfx1201, 32 GB) or similar; runs under the GPU test guard. No container is
// read: the geometry is a fixture, so this test carries the `gpu` label only.
#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace r97;

// ------------------------------------------------------------------ kv_store

// Four blocks of four slots, two kv heads, eight lanes each way.
constexpr int64_t kBlocks = 4, kHeads = 2, kBlockSize = 4, kHeadDim = 8, kTokens = 6;

device_check::Case kv_store_case(bool in_place, uint64_t seed) {
    device_check::Case c;
    c.rec.op = "kv_store";
    c.rec.position = 0;
    c.rec.params = {{"n_head_kv", kHeads}, {"head_dim", kHeadDim}, {"block_size", kBlockSize}};

    const std::vector<int64_t> kv_shape{kTokens, kHeads * kHeadDim};
    const std::vector<int64_t> slot_shape{kTokens};
    const std::vector<int64_t> pool_shape{kBlocks, kHeads, kBlockSize, 2 * kHeadDim};

    Operand k{"k", "bf16", kv_shape, 0, kTokens, false};
    Operand v{"v", "bf16", kv_shape, 0, kTokens, false};
    Operand slot{"slot", "i32", slot_shape, 0, kTokens, false};
    Operand pool{"pool", "bf16", pool_shape, 0, kBlocks, true};
    pool.in_place = in_place;
    c.rec.operands = {k, v, slot, pool};

    c.in = {device_check::random_value("bf16", kv_shape, &seed),
            device_check::random_value("bf16", kv_shape, &seed),
            device_check::random_value("i32", slot_shape, &seed),
            device_check::random_value("bf16", pool_shape, &seed)};
    // The slots: two tokens share slot 0, one is padded, the rest are distinct. Written as f64
    // values because that is what the tape carries; a slot past the pool is a padded row too.
    const int32_t slots[kTokens] = {0, 1, 3, -1, 2, 0};
    for (int64_t t = 0; t < kTokens; ++t) c.in[2].value[(size_t)t] = (double)slots[t];
    if (!in_place) c.in.pop_back();  // a plain write: the pool is not among the reads
    c.cot = {device_check::random_value("bf16", pool_shape, &seed)};
    c.label = std::string("kv_store ") + (in_place ? "in place" : "plain write") + " " +
              std::to_string(kTokens) + " tokens";
    return c;
}

// ------------------------------------------------------------------ lora_expand

// Six tokens, rank 4, a two-row pool of eight columns.
constexpr int64_t kM = 6, kR = 4, kS = 2, kN = 8;

device_check::Case lora_expand_case(bool in_place, uint64_t seed) {
    device_check::Case c;
    c.rec.op = "lora_expand";
    c.rec.position = 0;

    const std::vector<int64_t> t_shape{kM, kR};
    const std::vector<int64_t> slot_shape{kM};
    const std::vector<int64_t> b_shape{kS, kN, kR};
    const std::vector<int64_t> scale_shape{kS, kN};
    const std::vector<int64_t> y_shape{kM, kN};

    Operand t{"t", "bf16", t_shape, 0, kM, false};
    Operand slot{"slot", "i32", slot_shape, 0, kM, false};
    Operand lb{"lora_b", "bf16", b_shape, 0, kS, false};
    Operand sc{"scale", "f32", scale_shape, 0, kS, false};
    Operand y{"y", "bf16", y_shape, 0, kM, true};
    y.in_place = in_place;
    c.rec.operands = {t, slot, lb, sc, y};

    c.in = {device_check::random_value("bf16", t_shape, &seed),
            device_check::random_value("i32", slot_shape, &seed),
            device_check::random_value("bf16", b_shape, &seed),
            device_check::random_value("f32", scale_shape, &seed),
            device_check::random_value("bf16", y_shape, &seed)};
    // Row 2 is a base row (-1): it is neither read nor written, so its old value is kept.
    const int32_t slots[kM] = {0, 1, -1, 0, 1, 0};
    for (int64_t m = 0; m < kM; ++m) c.in[1].value[(size_t)m] = (double)slots[m];
    if (!in_place) c.in.pop_back();  // a plain write: the buffer is not among the reads
    c.cot = {device_check::random_value("f32", y_shape, &seed)};
    c.label = std::string("lora_expand ") + (in_place ? "in place" : "plain write") + " " +
              std::to_string(kM) + " tokens";
    return c;
}

// ------------------------------------------------------------------ the cases

void run() {
    const auto options = device_check::guard_options(int64_t(1) << 30);
    const r97test::DeviceGuard guard(options);
    (void)device_check::load_plugin();
    device_check::require_device(options.device);

    const device_check::Case cases[] = {kv_store_case(true, 0x9e3779b97f4a7c15ull),
                                        kv_store_case(false, 0x9e3779b97f4a7c15ull),
                                        lora_expand_case(true, 0x123456789abcdefull),
                                        lora_expand_case(false, 0x123456789abcdefull)};
    for (const device_check::Case& c : cases) {
        const device_check::Report r = device_check::check(c);
        std::printf("adjoint_in_place %s: %s\n", r.label.c_str(), r.ok ? "ok" : r.note.c_str());
        device_check::require(r);
    }
}

}  // namespace

R97_TEST(adjoint_in_place_rows) { run(); }
