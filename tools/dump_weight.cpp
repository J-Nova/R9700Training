// tools/dump_weight.cpp -- print named .rad container tensors' own values, so what the walk takes
// for a weight can be compared against what the container holds.
//
// Issue #55 diagnostic. The device backward's gradient grows ~x3.7-4.3 per block against a flat
// finite difference, and the fused norm's twin activation disagrees with the kernel's own
// reconstructed activation (`DUMPROT deq_rel` 0.99-1.62 at all 28 fused norms) while every
// gate/gated record matches at 0.0034-0.0078. The row and the adjoint both take the gain as
// `call.fwd[2]`, the record declares it rank 1 `[5120]`, and whether those 5120 values are the
// container's `blk.N.*_norm.weight` is the last unexplained element-level term. `r97train info`
// prints only the header and `rad-info` prints encodings and shapes, never values.
//
// Several names may be given at once, and a name that misses lists the directory entries that
// contain it: the container's names come from the declaration's `source` and need not be spelled
// the way the tape spells the operand (`backward.cpp`'s buffer names), so a single exact lookup
// would read a spelling mismatch as "the weight is absent". A one-element-a-value plane is
// decoded; a code plane is shown as its raw codes, never as values.
//
// Reader only: `rad::RadFile` mmaps the file and needs no engine, no plugin and no device, so this
// runs on the CPU and needs no GPU window. `src/vocab_head.cpp` uses the same API the same way.
#include "commands.h"

#include "radfile.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// One element of a plane, decoded as a double for the dtypes a weight is plausibly stored in as
// itself. A code plane of an affine encoding is not a value and is reported as its integer code.
double element_as_double(uint32_t dtype, const uint8_t* plane, int64_t i) {
    switch (dtype) {
        case RAD_F32: {
            float f = 0.0f;
            std::memcpy(&f, plane + 4 * i, 4);
            return (double)f;
        }
        case RAD_BF16: {
            uint16_t b = 0;
            std::memcpy(&b, plane + 2 * i, 2);
            const uint32_t u = (uint32_t)b << 16;
            float f = 0.0f;
            std::memcpy(&f, &u, 4);
            return (double)f;
        }
        case RAD_I8:
            return (double)(int8_t)plane[i];
        case RAD_U8:
            return (double)plane[i];
        default:
            return 0.0;
    }
}

bool is_value_dtype(uint32_t dtype) {
    return dtype == RAD_F32 || dtype == RAD_BF16 || dtype == RAD_I8 || dtype == RAD_U8;
}

// A miss is the one result that must not be silent: list what the directory does hold.
void list_matches(const rad::RadFile& f, const std::string& query, const char* exact) {
    const int64_t n = f.entry_count();
    std::printf("  directory holds %lld weights;", (long long)n);
    int shown = 0;
    for (int64_t i = 0; i < n && shown < 40; ++i) {
        const char* name = f.str(f.entry(i).name);
        const std::string hay = name != nullptr ? name : "";
        // The tape spells a fused operand `blk.63.ffn_norm.weight`; the container may spell the
        // same weight differently. Match on the tail after the last '.' too, so a prefix
        // difference does not hide it.
        const std::string tail = query.substr(query.rfind('.') == std::string::npos
                                                  ? 0
                                                  : query.rfind('.') + 1);
        if (hay.find(query) == std::string::npos && hay.find(tail) == std::string::npos) continue;
        std::printf(" [%lld] %s", (long long)i, name != nullptr ? name : "(unnamed)");
        ++shown;
    }
    if (shown == 0) std::printf(" none contains '%s'", query.c_str());
    std::printf("\n");
    std::printf("  no entry named exactly '%s'\n", exact);
}

void dump_one(const rad::RadFile& f, const std::string& name, int64_t want) {
    const RadFileEntry* e = f.find(name);
    if (e == nullptr) {
        std::printf("name=%s FOUND=0\n", name.c_str());
        list_matches(f, name, name.c_str());
        return;
    }
    const RadEncoding& enc = f.encoding(*e);
    int64_t numel = 1;
    std::string shape;
    for (uint32_t i = 0; i < e->rank; ++i) {
        numel *= e->shape[i];
        shape += (i ? "," : "") + std::to_string(e->shape[i]);
    }
    std::printf("name=%s FOUND=1 rank=%u shape=[%s] numel=%lld stored_name=%s quantizer=%s "
                "options=%s scheme=%s transform=%s planes=%d\n",
                name.c_str(), e->rank, shape.c_str(), (long long)numel, f.str(e->name),
                f.str(e->quantizer), f.str(e->options), enc.scheme, enc.transform, enc.n_planes);
    for (int k = 0; k < enc.n_planes; ++k) {
        const RadFilePlane& p = f.plane(*e, (int)k);
        const RadEncPlane& d = enc.plane[k];
        std::printf("  plane %d role=%s dtype=%s(%u) kind=%u block=[%lld,%lld] extent=[%lld,%lld] "
                    "bytes=%llu\n",
                    k, d.role, rad_dtype_name(d.dtype), d.dtype, d.kind, (long long)d.block[0],
                    (long long)d.block[1], (long long)d.extent[0], (long long)d.extent[1],
                    (unsigned long long)p.bytes);
    }
    const RadEncPlane& d0 = enc.plane[0];
    const RadFilePlane& p0 = f.plane(*e, 0);
    const uint8_t* base = f.base() + p0.offset;
    if (!(d0.kind == RAD_PLANE_TILED && d0.block[0] == 1 && d0.block[1] == 1 &&
          is_value_dtype(d0.dtype))) {
        // A code or scale plane. Show it as it is: reading it as a value is the misreading this
        // tool exists to prevent, and "the weight is stored quantised" must be visible at once.
        std::printf("  plane 0 is not one-element-a-value (scheme=%s role=%s dtype=%s): first "
                    "bytes/codes:",
                    enc.scheme, d0.role, rad_dtype_name(d0.dtype));
        const int64_t bits = rad_dtype_bits(d0.dtype);
        const int64_t shown = bits > 0 && bits < 8 ? 16 : 8;
        for (int64_t i = 0; i < shown && (int64_t)(bits > 0 ? (bits * i + 7) / 8 : i) <
                                       (int64_t)p0.bytes;
             ++i) {
            if (bits == 8) {
                std::printf(" %d", (int)(int8_t)base[i]);
            } else if (bits > 0) {
                const int64_t bit = i * bits;
                const uint32_t byte = base[bit / 8];
                std::printf(" %u", (byte >> (bit % 8)) & ((1u << bits) - 1u));
            } else {
                std::printf(" %02x", base[i]);
            }
        }
        std::printf("\n");
        return;
    }
    const int64_t need = rad_dtype_bytes(d0.dtype, numel);
    if (need <= 0 || (int64_t)p0.bytes < need) {
        std::printf("  plane 0 holds %llu bytes for %lld %s elements\n",
                    (unsigned long long)p0.bytes, (long long)numel, rad_dtype_name(d0.dtype));
        return;
    }
    std::printf("  values:");
    double l1 = 0.0, s = 0.0;
    for (int64_t i = 0; i < numel; ++i) {
        const double v = element_as_double(d0.dtype, base, i);
        l1 += std::fabs(v);
        s += v;
        if (i < want) std::printf(" %.10g", v);
    }
    std::printf("\n  l1=%.10g sum=%.10g elements=%lld\n", l1, s, (long long)numel);
}

int run(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: r97train dump-weight <file.rad> <name> [<name>...]\n"
                     "  prints each weight's shape, encoding, planes and -- for a one-element-a-"
                     "value plane --\n  its first 8 values plus the whole tensor's L1 and sum. A "
                     "name that misses lists\n  the directory entries containing it.\n");
        return 2;
    }
    rad::RadFile f;
    if (f.open(argv[1]) != RAD_OK) {
        std::fprintf(stderr, "dump-weight: '%s' will not open\n", argv[1]);
        return 1;
    }
    for (int a = 2; a < argc; ++a) dump_one(f, argv[a], 8);
    f.close();
    return 0;
}

}  // namespace

R97_REGISTER_COMMAND("dump-weight", "<file.rad> <name> [<name>...]", run);
