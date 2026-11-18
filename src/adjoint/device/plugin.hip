// src/adjoint/device/plugin.hip -- the kernel-plugin table libr97 exports to radiance's loader.
//
// libr97 is a radiance kernel plugin whose rows are not radiance ops: its kernels are r97's device
// adjoints, and each of them registers itself into r97's registry (R97_REGISTER_ADJOINT) from its
// own translation unit when the plugin loads. The radiance-side table is therefore empty -- the
// plugin is here so the engine finds it and loads it next to libr4d, and HIP registers the code
// objects it carries when the file loads, as it does for any kernel plugin.
//
// It is a .hip file because radiance's out-of-tree shape compiles a plugin's sources as HIP
// through rad_hipify; the runtime header comes first so the abi's host/device annotations are
// defined. There is no device code here, only the exports the loader reads.
#include <hip/hip_runtime.h>

#include "rad_plugin.h"
#include "rad_abi_layout.h"

#include "r97/adjoint.h"
#include "r97/head.h"

#include "grad_stage.h"
#include "scratch.h"

// The build passes the architecture list it is building for; a standalone compile of this file
// names the card this repository's kernels target.
#ifndef R97_DEVICE_BUILD_TARGET
#define R97_DEVICE_BUILD_TARGET "gfx1201"
#endif

namespace {

const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "r97", "0.1",
    "r97 device adjoints (add today); no radiance op rows", R97_DEVICE_BUILD_TARGET,
};

}  // namespace

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }

// Empty: r97's registry is the table that matters here, and it has no schema rows of its own.
extern "C" int rad_kernel_schema_count(void) { return 0; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
extern "C" int rad_kernel_count(void) { return 0; }
extern "C" const RadKernelInfo* rad_kernel_at(int) { return nullptr; }

// libr97 lives in the radiance home beside libr4d, so radiance's other tools (rad-convert, ...)
// dlopen it too, and those processes do not link r97. These fallback definitions are what lets the
// plugin load there. A process that links r97 exports the real functions -- r97train and the gpu
// tests build with ENABLE_EXPORTS -- and the dynamic loader prefers the executable's definition to
// a library's, so an r97 process gets the real registry and allocator. Where the fallbacks are in
// force there is no registry to register into and no device adjoint runs, which is right: the
// plugin serves no rows to a process that is not r97. Add no fallback until a kernel uses one.
namespace r97 {

void adjoint_register(const char*, Side, AdjointFn, const std::vector<int>&, ParamPred, FillsFn) {}

// The head's chunk kernel is not an op's adjoint: the loss registers it (include/r97/head.h) so the
// walk in src/head.cpp can run a chunk on the card.
void head_ce_register(HeadCeKernel) {}

// The vocabulary head's kernels are not an op's adjoint either: the device join runs them on the
// card (src/adjoint/device/vocab_head.hip registers them into src/vocab_head.cpp). The struct is
// forward-declared: the real one is src/vocab_head.h's, and this file need not drag the header in.
struct VocabHeadKernels;
void register_vocab_head_kernels(const VocabHeadKernels&) {}

// The fused producers' adjoint (quant.hip) rebuilds the value the quantiser rotated through the
// parts' twin. A process with no registry has no twin, and no device adjoint runs there either.
TwinFn twin_find(const std::string&, const std::vector<OpParam>&, std::string*) { return nullptr; }

namespace device {

void* scratch_alloc(int64_t) { return nullptr; }
void scratch_free(void*) {}
float* grad_stage_slot(const std::string&, const std::vector<int64_t>&) { return nullptr; }

}  // namespace device
}  // namespace r97
