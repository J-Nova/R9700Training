// tests/host/lora_shrink_step_order.cpp -- the reference lora_shrink reads a step-order
// activation.
//
// radiance declares lora_shrink with two dtype parameters: `dtype`, the activation's element dtype
// (always "i8" for codes, whatever the site), and `gemm_dtype`, the encoding of the GEMM whose
// activation it reads -- the same string that GEMM carries. It is `gemm_dtype` that says how the
// codes are stored: quant_act_i8 writes them in the int8 WMMA's step order at w4a8/w8a8/w2a8 (the
// middle dwords of every 16-byte step exchanged) and the rotated producers write column order.
//
// patches/serving/0210-lora-int8-trunk-host.patch makes libref's lora_shrink unexchange the step
// order when `gemm_dtype` names one of those. The int8 fixture's trunk is i8a8h, whose codes are
// already in column order, so the fixture test exercises only the column-order reading; this runs
// the reference kernel directly, with the parameter set radiance declares at such a site: the
// same codes, once in column order and once in step order, must produce the same f32 accumulator,
// and that accumulator must agree with an independent f64 dot in which every code is multiplied
// by its group's scale.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "support/adjoint_check.h"
#include "support/test.h"

namespace {

constexpr int64_t kM = 4, kK = 128, kR = 2, kS = 1, kG = 128;

// i8_step_pos from libref/ref_common.h: the middle two dwords of each 16-code step exchanged.
int64_t step_pos(int64_t c) {
    return (c & ~int64_t(12)) | ((c & 4) << 1) | ((c & 8) >> 1);
}

struct Result {
    int rc = RAD_E_INVAL;
    std::vector<float> t;
};

// Run libref's lora_shrink over `codes` (row-major [M,K], column order) as a site whose GEMM is
// `gemm_dtype` reads it. With `step_order` the codes are stored the way quant_act_i8 writes them.
Result shrink(const RadKernelInfo* k, const std::vector<int8_t>& codes,
              const std::vector<float>& scale, const std::vector<float>& a, const char* gemm_dtype,
              bool step_order) {
    std::vector<int8_t> x((size_t)(kM * kK));
    for (int64_t m = 0; m < kM; ++m)
        for (int64_t c = 0; c < kK; ++c)
            x[(size_t)(m * kK + (step_order ? step_pos(c) : c))] = codes[(size_t)(m * kK + c)];

    std::vector<int32_t> slot((size_t)kM, 0);
    std::vector<float> t((size_t)(kM * kR), 0.0f);

    RadTensor tensors[5]{};
    tensors[0].data = x.data();
    tensors[0].dtype = RAD_I8;
    tensors[0].rank = 2;
    tensors[0].shape[0] = kM;
    tensors[0].shape[1] = kK;
    tensors[0].stride[0] = kK;
    tensors[0].stride[1] = 1;

    tensors[1].data = const_cast<float*>(scale.data());
    tensors[1].dtype = RAD_F32;
    tensors[1].rank = 2;
    tensors[1].shape[0] = kM;
    tensors[1].shape[1] = kK / kG;
    tensors[1].stride[0] = kK / kG;
    tensors[1].stride[1] = 1;

    tensors[2].data = slot.data();
    tensors[2].dtype = RAD_I32;
    tensors[2].rank = 1;
    tensors[2].shape[0] = kM;
    tensors[2].stride[0] = 1;

    tensors[3].data = const_cast<float*>(a.data());
    tensors[3].dtype = RAD_F32;
    tensors[3].rank = 3;
    tensors[3].shape[0] = kS;
    tensors[3].shape[1] = kR;
    tensors[3].shape[2] = kK;
    tensors[3].stride[0] = kR * kK;
    tensors[3].stride[1] = kK;
    tensors[3].stride[2] = 1;

    tensors[4].data = t.data();
    tensors[4].dtype = RAD_F32;
    tensors[4].rank = 2;
    tensors[4].shape[0] = kM;
    tensors[4].shape[1] = kR;
    tensors[4].stride[0] = kR;
    tensors[4].stride[1] = 1;

    // `dtype` is the element dtype radiance declares (`rad_dtype_name` of the codes), and
    // `gemm_dtype` the site GEMM's own name: the two are different strings at every coded site.
    RadParam params[] = {
        RAD_INT("K", kK),  RAD_INT("R", kR),
        RAD_INT("S", kS),  RAD_INT("M", kM),
        RAD_INT("group", kG),
        RAD_STR("dtype", "i8"),
        RAD_STR("gemm_dtype", gemm_dtype),
    };
    RadArgs args{};
    args.t = tensors;
    args.n_t = 5;
    args.p = params;
    args.n_p = (int)(sizeof params / sizeof params[0]);
    args.world_size = 1;

    Result r;
    r.rc = k->launch(&args, nullptr);
    r.t = t;
    return r;
}

}  // namespace

R97_TEST(lora_shrink_reads_step_order_codes) {
    const RadKernelInfo* k = r97::adjoint_check::find_host_kernel("lora_shrink");
    R97_CHECK(k != nullptr);

    uint64_t seed = 0x0210c0deull;
    std::vector<int8_t> codes((size_t)(kM * kK));
    std::vector<float> scale((size_t)kM, 0.0f);
    std::vector<float> a((size_t)(kS * kR * kK));
    for (int64_t m = 0; m < kM; ++m) {
        scale[(size_t)m] = 0.003f + (float)(m + 1) * 0.0007f;
        for (int64_t c = 0; c < kK; ++c)
            codes[(size_t)(m * kK + c)] = (int8_t)((int)(r97::adjoint_check::splitmix64(&seed) % 255) - 127);
    }
    for (float& v : a)
        v = (float)r97::adjoint_check::urand(&seed) * 0.25f;

    // The column-order readings: the rotated trunk's i8a8h, and the plain i8a8 codes of
    // quant_act_i8g. Both must agree, bit for bit, with the same kernel run over the codes in
    // each step-order encoding -- w4a8, w8a8 and w2a8 are what quant_act_i8 feeds and what
    // radiance's model of an A in the fwht128 basis is served with.
    const Result column = shrink(k, codes, scale, a, "i8a8h", false);
    R97_CHECK(column.rc == RAD_OK);
    const struct {
        const char* gemm_dtype;
        bool        step_order;
    } cases[] = {
        { "i8a8", false }, { "i8a8h", false }, { "w4a8", true },
        { "w8a8", true },  { "w2a8", true },
    };
    for (const auto& c : cases) {
        const Result got = shrink(k, codes, scale, a, c.gemm_dtype, c.step_order);
        R97_CHECK(got.rc == RAD_OK);
        R97_CHECK(got.t.size() == column.t.size());
        R97_CHECK(std::memcmp(column.t.data(), got.t.data(), column.t.size() * sizeof(float)) == 0);
    }

    // The independent f64 dot: t[m,j] = sum_k (code[m,k] * scale[m,k]) * A[0,j,k].
    for (int64_t m = 0; m < kM; ++m) {
        for (int64_t j = 0; j < kR; ++j) {
            double want = 0.0;
            for (int64_t c = 0; c < kK; ++c)
                want += (double)codes[(size_t)(m * kK + c)] * (double)scale[(size_t)m] *
                        (double)a[(size_t)(j * kK + c)];
            const double got = (double)column.t[(size_t)(m * kR + j)];
            R97_CHECK(std::fabs(got - want) <= 1e-5 * std::max(std::fabs(got), std::fabs(want)) + 1e-6);
        }
    }
}
