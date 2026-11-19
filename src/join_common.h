// src/join_common.h -- internal: the join helpers both joins share.
//
// The host join (src/step_host.cpp, docs/STEPS-HOST.md) and the device join (src/step_device.cpp)
// walk one recorded step the same way, and this file is the part the walk shares: how one operand
// reads dense over the shape the op saw, how a quantised weight the engine joined from planes
// dequantises, how the head's seeds move between the batch's row space and the head's, and how
// the walk's adapter gradients split back into the delta's per-site names. The host join and its
// tests held the behavior these carry; the device join calls them rather than carry a copy.
#pragma once

#if defined(RAD_ISSUE_OBSERVER) && defined(RAD_REPLAY_BLOCK)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "rad_encoding.h"  // the stored form a plugin-private weight's decode derives (0124)
#include "rad_planes.h"  // the joined form a quantised lm_head reaches the tape in
#include "read_set.h"  // the read-set derivation (r97 headers only, so the host tier tests it)
#include "r97/adjoint.h"
#include "r97/backward.h"
#include "r97/batch.h"
#include "r97/delta.h"
#include "r97/forward.h"
#include "r97/objective.h"
#include "r97/op.h"
#include "r97/params.h"
#include "r97/tape.h"

namespace r97 {

inline int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

// The payload of a stored value or cotangent: what the join's memory figures count.
inline int64_t value_bytes(const std::vector<double>& v) {
    return static_cast<int64_t>(v.size()) * static_cast<int64_t>(sizeof(double));
}

inline int64_t elem_size(const std::string& dtype) {
    if (dtype == "bf16" || dtype == "f16") return 2;
    if (dtype == "f32" || dtype == "i32" || dtype == "u32") return 4;
    if (dtype == "f64" || dtype == "i64" || dtype == "u64") return 8;
    if (dtype == "i8" || dtype == "u8" || dtype == "f8" || dtype == "fp8_e4m3" ||
        dtype == "fp8_e5m2")
        return 1;
    return 0;
}

inline double read_elem(const std::string& dtype, const void* data, int64_t i) {
    if (dtype == "f32") return static_cast<const float*>(data)[i];
    if (dtype == "f64") return static_cast<const double*>(data)[i];
    if (dtype == "i32") return static_cast<const int32_t*>(data)[i];
    if (dtype == "u32") return static_cast<const uint32_t*>(data)[i];
    if (dtype == "i64") return static_cast<double>(static_cast<const int64_t*>(data)[i]);
    if (dtype == "u64") return static_cast<double>(static_cast<const uint64_t*>(data)[i]);
    if (dtype == "i8") return static_cast<const int8_t*>(data)[i];
    if (dtype == "u8") return static_cast<const uint8_t*>(data)[i];
    if (dtype == "fp8_e4m3") return rad_fp8e4m3_to_f32(static_cast<const uint8_t*>(data)[i]);
    if (dtype == "fp8_e5m2") return rad_fp8e5m2_to_f32(static_cast<const uint8_t*>(data)[i]);
    if (dtype == "bf16") {
        const uint32_t u = static_cast<uint32_t>(static_cast<const uint16_t*>(data)[i]) << 16;
        float f;
        std::memcpy(&f, &u, sizeof f);
        return f;
    }
    if (dtype == "f16") {
        const uint16_t h = static_cast<const uint16_t*>(data)[i];
        const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1fu;
        uint32_t mant = h & 0x3ffu;
        uint32_t bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign;
            } else {
                exp = 127 - 15 + 1;
                while ((mant & 0x400u) == 0) {
                    mant <<= 1;
                    --exp;
                }
                bits = sign | (exp << 23) | ((mant & 0x3ffu) << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7f800000u | (mant << 13);
        } else {
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &bits, sizeof f);
        return f;
    }
    throw std::runtime_error("host step: cannot snapshot operand dtype '" + dtype + "'");
}

// A quantised weight the engine joins from planes (abi/rad_planes.h) reaches the tape as one
// buffer: the header, the entries, the codes, then the scales. The adjoints take the dequantised
// weight, the product the GEMM's epilogue forms, so read the planes out by their roles. Only a
// whole-buffer operand can be joined -- a ranged or viewed one is a slice of something else --
// and the codes' extents have to be the operand's own, or this is not the weight the op read.
inline bool joined_dense(const Operand& op, const OperandContent& content, std::vector<double>* out) {
    if (content.data == nullptr || content.bytes < sizeof(RadPlanesHeader)) return false;
    if (op.rows > 0 || op.shape.size() < 2) return false;
    const RadPlanesEntry* codes = nullptr;
    const RadPlanesEntry* scales = nullptr;
    const void* code_data = rad_planes_find(content.data, "codes", &codes);
    const void* scale_data = rad_planes_find(content.data, "scale", &scales);
    if (code_data == nullptr || codes == nullptr) return false;
    const int64_t rows = codes->rows;
    const int64_t cols = codes->cols;
    if (rows <= 0 || cols <= 0 || rows * cols != numel(op.shape)) return false;
    if (scale_data != nullptr && (scales->rows <= 0 || scales->cols <= 0)) return false;

    const char* code_name = rad_dtype_name(codes->dtype);
    const int64_t code_es = rad_dtype_bytes(codes->dtype, 1);
    const char* scale_name = scale_data != nullptr ? rad_dtype_name(scales->dtype) : nullptr;
    const int64_t scale_es = scale_data != nullptr ? rad_dtype_bytes(scales->dtype, 1) : 0;
    if (code_es <= 0 || (scale_data != nullptr && scale_es <= 0)) return false;
    // The block is the dequantisation group: a scale per block[1] columns, block[0] rows.
    const int64_t block_r = scales != nullptr ? std::max<int64_t>(scales->block[0], 1) : rows;
    const int64_t block_c = scales != nullptr ? std::max<int64_t>(scales->block[1], 1) : cols;

    const auto* code_bytes = static_cast<const uint8_t*>(code_data);
    const auto* scale_bytes = static_cast<const uint8_t*>(scale_data);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            const double code =
                read_elem(code_name, code_bytes + r * codes->row_bytes + c * code_es, 0);
            double scale = 1.0;
            if (scale_bytes != nullptr) {
                const int64_t sr = r / block_r;
                const int64_t sc = c / block_c;
                if (sr >= scales->rows || sc >= scales->cols)
                    throw std::runtime_error("host step: a joined weight's scale plane is short");
                scale =
                    read_elem(scale_name, scale_bytes + sr * scales->row_bytes + sc * scale_es, 0);
            }
            (*out)[static_cast<size_t>(r * cols + c)] = code * scale;
        }
    }
    return true;
}

// One operand as the op saw it, dense over the operand's full declared shape. The tape keeps the
// slice the kernel saw, which for a ranged operand starts at row_lo; the walk's adjoints expect
// the full buffer with the untouched rows zero.
inline AdValue snapshot(const Operand& op, const OperandContent& content,
                        const char* op_name = nullptr, int64_t pos = -1) {
    AdValue v;
    // The family adjoints read the record's dtype off the value too (an absent scale is spelled
    // "invalid"), so keep it with the numbers.
    v.dtype = op.dtype;
    // R97_DUMP_SNAP=1 prints a plugin-private operand's whole description -- the dtype string, the
    // tape's byte count and pointer, and the view -- for the first few records of the ops whose
    // stored operands decide a gradient: gemm_nt_q's weight and scale, and every lora row's
    // (a lora operand the tape does not name keeps zeros, and that is a silent zero). It is how the
    // trunk weight's stored layout was identified and how a zero operand is told from a wrong span;
    // see docs/adjoints/linear.md. The non-zero count is over the span the tape gave, so a named
    // operand that is all zeros shows as nz=0 of n rather than as an absent line.
    if (const char* d = std::getenv("R97_DUMP_SNAP"); d != nullptr && (d[0] == '1' || d[0] == '2')) {
        static long n_gemm = 0, n_lora = 0;
        const std::string name = op_name != nullptr ? op_name : "";
        const bool lora = name == "lora_expand" || name == "lora_shrink";
        long& n = lora ? n_lora : n_gemm;
        // =1 is the historical few-records form. =2 prints every record's every operand, which is
        // what lets a snapshot be joined against DUMPFWD's own read of the same
        // (op, pos, buffer, data) in one process -- the only way a stale or mismatched snapshot is
        // told from a correct one, since two separate logs cannot be joined.
        const bool all = d[0] == '2';
        if (all || ((name == "gemm_nt_q" || lora) && n < (lora ? 40 : 4))) {
            ++n;
            std::string sh, vs, vst;
            for (int64_t x : op.shape) sh += std::to_string(x) + " ";
            for (int64_t x : op.view_shape) vs += std::to_string(x) + " ";
            for (int64_t x : op.view_stride) vst += std::to_string(x) + " ";
            long nz = -1;
            long long span = 0;
            double maxabs = 0.0;
            const int64_t es = elem_size(op.dtype);
            if (content.data != nullptr && es > 0 && content.bytes > 0) {
                nz = 0;
                span = static_cast<long long>(content.bytes) / es;
                for (long long i = 0; i < span; ++i) {
                    const double x = read_elem(op.dtype,
                                               static_cast<const uint8_t*>(content.data) + i * es, 0);
                    if (x != 0.0) ++nz;
                    // maxabs over the same span DUMPFWD reports it over, so a snapshot and the
                    // forward's own read of the same buffer are comparable in one process rather
                    // than in two logs that cannot be joined. A stale snapshot shows as a maxabs
                    // that disagrees with the forward's for the same (op, pos, buffer, data).
                    if (x == x) {
                        const double a = x < 0 ? -x : x;
                        if (a > maxabs) maxabs = a;
                    }
                }
            }
            std::fprintf(stderr,
                         "DUMPSNAP op=%s pos=%lld buffer=%s dtype=%s shape=[%s] bytes=%zu named=%d "
                         "data=%p view_shape=[%s] view_stride=[%s] view_off=%lld nonzero=%ld of %lld "
                         "maxabs=%.6g written=%d in_place=%d\n",
                         op_name, static_cast<long long>(pos), op.buffer.c_str(), op.dtype.c_str(),
                         sh.c_str(), content.bytes, static_cast<int>(content.named), content.data,
                         vs.c_str(), vst.c_str(), static_cast<long long>(op.view_off), nz, span,
                         maxabs, static_cast<int>(op.written), static_cast<int>(op.in_place));
            // The stored form's inputs (patch 0124): the engine's own planes, so a form the row
            // derives is checkable against what the layout hook saw.
            if (op.enc != nullptr) {
                const RadEncoding* e = static_cast<const RadEncoding*>(op.enc);
                std::string pl;
                for (uint32_t k = 0; k < e->n_planes; ++k) {
                    const RadEncPlane& p = e->plane[k];
                    char one[160];
                    std::snprintf(one, sizeof one, " [%s kind=%u dtype=%u e=%lld,%lld b=%lld,%lld]",
                                  p.role, (unsigned)p.kind, (unsigned)p.dtype,
                                  (long long)p.extent[0], (long long)p.extent[1],
                                  (long long)p.block[0], (long long)p.block[1]);
                    pl += one;
                }
                std::fprintf(stderr, "DUMPSNAP   enc=%p transform=%s n_planes=%u%s\n",
                             op.enc, e->transform, (unsigned)e->n_planes, pl.c_str());
            }
            std::fflush(stderr);
        }
    }
    // A dtype with no f64 form -- a plugin-private quantised tensor the engine decodes in its own
    // stored layout -- has no value a snapshot could hold, and the tape's byte count cannot name
    // one: the refusal below is for a *readable* operand whose span was lost, not for this. Return
    // the shape, no values, and the pointer and encoding the tape recorded.
    //
    // The stored path is the default: the IQR decode it feeds is proven against the engine's own
    // `iqr_gemm_case` on both of the container's forms (tests/gpu/adjoint_iqr_plane.cpp, whose
    // hand-read plane is the one that test measures), so a shipped build decodes the trunk instead
    // of refusing every weight of it. `R97_NO_STORED_WEIGHTS=1` is the kill switch: the operand
    // then has neither values nor a pointer and the walk refuses it by name, exactly as before the
    // pointer was threaded. The *form* gate lives in the row (src/adjoint/device/linear.hip): an
    // encoding this build cannot decode throws with its name rather than reading the plane.
    if (content.bytes == 0 && content.named && elem_size(op.dtype) == 0) {
        v.shape = op.view_shape.empty() ? op.shape : op.view_shape;
        const char* off = std::getenv("R97_NO_STORED_WEIGHTS");
        if (off == nullptr || off[0] != '1') {
            v.raw = content.data;
            v.enc = op.enc;
        }
        return v;
    }
    // A strided or narrowed read -- the q norm and the gate read the two halves of the q|gate
    // projection, the gated-delta layer reads q|k|v out of one fused projection -- is packed over
    // its own shape: the adjoint that follows normalises the last dim the kernel normalised, and
    // the declared buffer's last dim is not that one. backward.cpp scatters the cotangent back.
    if (!op.view_shape.empty()) {
        v.shape = op.view_shape;
        const int64_t es = elem_size(op.dtype);
        if (es == 0)
            throw std::runtime_error("host step: cannot snapshot operand dtype '" + op.dtype + "'");
        int64_t n = 1;
        for (int64_t d : v.shape) n *= std::max<int64_t>(d, 0);
        v.value.assign(static_cast<size_t>(std::max<int64_t>(n, 0)), 0.0);
        if (!content.data || n == 0) return v;
        const auto* base = static_cast<const uint8_t*>(content.data);
        int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const size_t rank = op.view_shape.size();
        // A kept copy of a strided view is the view's own values, packed in its own order (the
        // tape cannot copy a stride out of one span); a live buffer still has to be walked with
        // the strides. The size tells which one this is.
        const bool packed = content.bytes >= static_cast<size_t>(n) * static_cast<size_t>(es);
        for (int64_t e = 0; e < n; ++e) {
            int64_t off = packed ? e : op.view_off;
            if (!packed)
                for (size_t d = 0; d < rank; ++d) off += idx[d] * op.view_stride[d];
            v.value[static_cast<size_t>(e)] = read_elem(op.dtype, base + static_cast<int64_t>(es) * off, 0);
            for (size_t d = rank; d-- > 0;) {
                if (++idx[d] < op.view_shape[d]) break;
                idx[d] = 0;
            }
        }
        return v;
    }
    v.shape = op.shape;
    const int64_t total = numel(op.shape);
    v.value.assign(static_cast<size_t>(std::max<int64_t>(total, 0)), 0.0);
    if (!content.data || total == 0) return v;
    if (joined_dense(op, content, &v.value)) return v;
    // A declared buffer or weight the tape handed a pointer for, but could not describe as one
    // span: `OperandContent::bytes` is 0 for a slice that is not contiguous, and for any dtype with
    // no element size (a plugin-private quantised weight reports bytes 0 while being contiguous).
    // Reading it as zeros is silent and wrong: the adjoint then differentiates a zero operand, and
    // that is one of the two ways the device step's adapter gradient came back zero for 399 of 400
    // leaves. Refuse and name it; the tape's own description of such a tensor is what has to
    // improve (patch 0123), not this fallback.
    // An operand the tape does not name (a raw batch field, a KV group, a table) keeps the zeros:
    // its values are not a declared buffer's, and the raw slot's zero is a valid slot index.
    if (content.bytes == 0) {
        if (!content.named || op.shape.empty()) return v;
        std::string shape = "[";
        for (size_t d = 0; d < op.shape.size(); ++d)
            shape += (d == 0 ? "" : ",") + std::to_string(op.shape[d]);
        shape += "]";
        throw std::runtime_error("snapshot: op '" + std::string(op_name ? op_name : "?") +
                                 "': operand '" + op.buffer + "' declares " + shape +
                                 " but the tape carries no bytes for it");
    }

    const int64_t es = elem_size(op.dtype);
    if (es == 0)
        throw std::runtime_error("host step: cannot snapshot operand dtype '" + op.dtype + "'");
    const int64_t count = static_cast<int64_t>(content.bytes) / es;
    if (count <= 0) return v;

    int64_t cols = total;
    int64_t declared_rows = 1;
    if (!op.shape.empty()) {
        declared_rows = op.shape[0];
        cols = declared_rows > 0 ? total / declared_rows : 0;
    }
    for (int64_t i = 0; i < count; ++i) {
        int64_t at;
        if (op.rows == 0 || count == total) {
            at = i;
        } else {
            const int64_t r = i / (cols > 0 ? cols : 1);
            const int64_t c = cols > 0 ? i % cols : 0;
            at = (op.row_lo + r) * cols + c;
        }
        if (at < 0 || at >= total) continue;
        v.value[static_cast<size_t>(at)] = read_elem(op.dtype, content.data, i);
    }
    return v;
}

inline bool residual_shape(const InputCotangent& in, int64_t tokens, int64_t hidden) {
    // The head's ops see the engine's whole buffer, which is longer than the step's rows; its
    // first `tokens` rows are the step's. A block boundary is already the step's row count.
    return in.value.size() >= static_cast<size_t>(tokens * hidden) && !in.shape.empty() &&
           in.shape.back() == hidden && in.row_hi >= tokens;
}

inline std::vector<double> first_rows(const InputCotangent& in, int64_t tokens, int64_t hidden) {
    const size_t n = static_cast<size_t>(tokens * hidden);
    if (in.value.size() < n) return {};
    return std::vector<double>(in.value.begin(), in.value.begin() + static_cast<ptrdiff_t>(n));
}

// True when any record of the block wrote the buffer.
inline bool block_wrote(const std::vector<OpRecord>& ops, const Tape::Block& block,
                        const std::string& buffer) {
    for (size_t i = block.begin; i < block.end; ++i)
        for (const Operand& o : ops[static_cast<size_t>(i)].operands)
            if (o.written && o.buffer == buffer) return true;
    return false;
}

// The cotangents the block below enters, one per buffer the block above left on the boundary.
// The walk's input cotangents are, by construction, the cotangents of every buffer the block
// above consumed from below -- a read with no earlier writer in the block leaves through
// `add_input_full` (src/backward.cpp) -- so the boundary is the upper walk's own input list
// filtered to the boundary's shape: the residual (its in-place read) and the branch (a plain
// read by the same fused norm) both qualify. No name- or shape-preference over what qualifies:
// that pick carried the branch and dropped the residual's cotangent at every block.
//
// Each entry enters the block below at the version its last write produced (src/backward.cpp's
// `enter`): the residual's entry credits the ffn norm's in-place write, the branch's entry the
// mlp's write, so both writers' slots carry their consumers' cotangents. The entries keep the
// walk's input order (the names are distinct there, so nothing is collected twice). A
// boundary-shaped input the block below never wrote is a refusal by name: the block above
// consumed a value the block below did not produce, and a silent drop is the defect this join
// exists to remove. `next` names the block below's records; null at the walk's bottom, where the
// list is the canary the caller checks for emptiness, as before.
inline std::vector<IncomingCotangent> boundary_cotangents(
    const std::vector<InputCotangent>& inputs, const std::vector<OpRecord>& ops,
    const Tape::Block* next, int64_t tokens, int64_t hidden) {
    std::vector<IncomingCotangent> out;
    for (const InputCotangent& in : inputs) {
        if (!residual_shape(in, tokens, hidden)) continue;
        if (next != nullptr && !block_wrote(ops, *next, in.buffer))
            throw std::runtime_error("r97: the join: the boundary buffer '" + in.buffer +
                                     "' the block above left was not written by the block below");
        out.emplace_back(in.buffer, 0, tokens, first_rows(in, tokens, hidden));
    }
    return out;
}

// A record names a whole engine buffer, but this step's op only touched `rows` of them, starting
// at `row_lo`. The block walk is over the step's rows, not the engine's capacity: hand every
// adjoint the slice it actually saw, with the row window at zero, so read and write buffers of
// different capacities line up and the snapshots hold no inactive rows. Parameters and other
// whole-buffer operands have no row window and are left as they are.
inline void tighten_operand(Operand& o) {
    if (o.rows <= 0 || o.shape.empty() || o.rows > o.shape[0]) return;
    o.shape[0] = o.rows;
    o.row_lo = 0;
    // A view's first dim is the same row axis, so it narrows with the buffer.
    if (!o.view_shape.empty() && o.rows <= o.view_shape[0]) o.view_shape[0] = o.rows;
}

// The attention family's adjoint reads q as [rows, heads, head_dim] and takes head_dim from its
// last axis. The engine hands the kernel the flattened [rows, heads * head_dim] buffer, so the
// walk gets the view the family expects. The kv operand carries the same head_dim as a pair.
inline void annotate_attention(OpRecord& rec) {
    if (rec.op != "attn_paged" || rec.operands.empty()) return;
    Operand& q = rec.operands[0];
    if (q.shape.size() != 2 || !q.view_shape.empty()) return;
    const Operand* kv = nullptr;
    for (const Operand& o : rec.operands)
        if (o.shape.size() == 4) {
            kv = &o;
            break;
        }
    if (kv == nullptr) return;
    const int64_t head_dim = kv->shape[3] / 2;
    if (head_dim <= 0 || q.shape[1] % head_dim != 0) return;
    q.view_shape = {q.shape[0], q.shape[1] / head_dim, head_dim};
    q.view_stride = {q.shape[1], head_dim, 1};
    q.view_off = 0;
}

inline void tighten_rows(OpRecord& rec) {
    for (Operand& o : rec.operands) tighten_operand(o);
    annotate_attention(rec);
    // The engine applies an adapter to the activation it already wrote: the expand's last
    // operand is both the value it replaces and the value it adds its branch to, but the tape
    // does not mark that write in place. Say so, so the walk can pass the direct path on.
    if (rec.op == "lora_expand" || rec.op == "lora_expand_mm")
        for (Operand& o : rec.operands)
            if (o.written) o.in_place = true;
}

// The family's shrink reads its activation as [rows][K]. The engine can hand it an activation
// split by head ([rows][heads][width]); that one record takes the row that knows the reference
// step's shape, and everything else stays on the family's own row.
inline void mark_join_lora(OpRecord& rec) {
    if (rec.op != "lora_shrink" && rec.op != "lora_shrink_mm") return;
    for (const Operand& o : rec.operands)
        if (!o.written && o.shape.size() > 2) {
            rec.params.push_back({"r97join", int64_t{1}});
            return;
        }
}

// The objective reads the logits boundary in the batch's row space, L-1 rows a sequence. The
// head writes the step's row space, all L rows a sequence, after the sampler's rows: its logits
// buffer row for the batch's row r is `hidden_rows()[i] + 1`, because row i of hidden() is the
// head's input row that predicts that scored token. A seed has to move to that space before the
// walk, or it lands on the wrong token (or on a sampler row that predicts nothing).
inline std::vector<Seed> head_seeds(const ForwardStep& step, const Batch& batch,
                                    const std::vector<OpRecord>& head,
                                    int64_t tokens) {
    int64_t rows = 0, vocab = 0;
    const std::vector<int64_t> lshape = step.shape("logits");
    if (lshape.size() != 2) throw std::runtime_error("host step: the logits boundary is not 2-D");
    rows = lshape[0];
    vocab = lshape[1];

    int64_t head_rows = 0;
    for (const OpRecord& rec : head)
        for (const Operand& o : rec.operands)
            if (o.written && o.buffer == "logits") head_rows = o.rows;

    // One entry a batch row, -1 where the row predicts nothing. hidden_rows() lists the scored
    // rows in the batch's order, so walking the scores in that order pairs them up.
    std::vector<int64_t> at(static_cast<size_t>(rows), -1);
    const std::vector<int64_t>& hidden_rows = step.hidden_rows();
    size_t scored = 0;
    int64_t off = 0;
    for (const TokenSeq& seq : batch) {
        const int64_t len = static_cast<int64_t>(seq.ids.size());
        for (int64_t p = 1; p < len; ++p) {
            if (p < static_cast<int64_t>(seq.score.size()) && !seq.score[static_cast<size_t>(p)]) continue;
            if (scored >= hidden_rows.size())
                throw std::runtime_error("host step: hidden_rows() is shorter than the scored rows");
            const int64_t batch_row = off + p - 1;
            if (batch_row < 0 || batch_row >= rows)
                throw std::runtime_error("host step: a scored row is outside the logits boundary");
            at[static_cast<size_t>(batch_row)] = hidden_rows[scored];
            ++scored;
        }
        off += len > 0 ? len - 1 : 0;
    }
    if (head_rows <= 0)
        throw std::runtime_error("host step: the head wrote no logits rows to seed");
    if (head_rows < tokens)
        throw std::runtime_error("host step: the head wrote fewer logits rows than tokens");

    std::vector<Seed> out;
    for (const Seed& seed : step.seeds()) {
        if (seed.buffer != "logits") {
            out.push_back(seed);
            continue;
        }
        const int64_t span = seed.row_hi - seed.row_lo;
        if (span <= 0) continue;
        if (static_cast<int64_t>(seed.cotangent.size()) != span * vocab)
            throw std::runtime_error("host step: a logits seed's cotangent does not match its rows");
        Seed moved;
        moved.buffer = seed.buffer;
        moved.row_lo = 0;
        moved.row_hi = head_rows;
        moved.cotangent.assign(static_cast<size_t>(head_rows * vocab), 0.0f);
        for (int64_t r = seed.row_lo; r < seed.row_hi; ++r) {
            const int64_t to = at[static_cast<size_t>(r)];
            if (to < 0) continue;
            if (to >= head_rows)
                throw std::runtime_error("host step: a seed maps past the head's logits rows");
            std::memcpy(moved.cotangent.data() + static_cast<size_t>(to * vocab),
                        seed.cotangent.data() + static_cast<size_t>((r - seed.row_lo) * vocab),
                        static_cast<size_t>(vocab) * sizeof(float));
        }
        out.push_back(std::move(moved));
    }
    return out;
}

// The observer names an unnamed operand after its position in the schema ("raw:2"), so two ops
// can give the same name to buffers of different shapes. A raw or absent operand is per-op by
// definition (a position table, a slot list, a token range), so give each its own name; nothing
// reads an unnamed operand by name.
inline OpRecord own_raws(OpRecord rec) {
    for (Operand& o : rec.operands)
        if (o.buffer.rfind("raw:", 0) == 0 || o.buffer.rfind("none:", 0) == 0)
            o.buffer += "@" + std::to_string(rec.position);
    return rec;
}

// A weight that several delta sites land in is one fused adapter on the engine's side: the
// shrink concatenates their A factors along the rank and the expand makes their B factors a
// block-diagonal matrix. The walk sees one gradient for the fused pair; split it back into the
// per-module pairs the delta's ParamSet names, using the order and shapes the sites gave.
struct SiteSlice {
    std::string module;
    int64_t rank = 0;
    int64_t in = 0;
    int64_t out = 0;
    int64_t rank_off = 0;
    int64_t out_off = 0;
};

using SiteGroups = std::map<std::string, std::vector<SiteSlice>>;

inline SiteGroups group_sites(const WireBuffers& wire) {
    SiteGroups groups;
    for (const WireBuffers::Site& site : wire.sites) {
        if (site.rank < 1) throw std::runtime_error("host step: a delta site has no rank");
        if (site.b.size() % static_cast<size_t>(site.rank) != 0)
            throw std::runtime_error("host step: delta site '" + site.module +
                                     "' has a B whose size is not its rank times its rows");
        if (site.a.size() % static_cast<size_t>(site.rank) != 0)
            throw std::runtime_error("host step: delta site '" + site.module +
                                     "' has an A whose size is not its rank times its columns");
        SiteSlice slice;
        slice.module = site.module;
        slice.rank = site.rank;
        slice.in = static_cast<int64_t>(site.a.size()) / site.rank;
        slice.out = static_cast<int64_t>(site.b.size()) / site.rank;
        std::vector<SiteSlice>& group = groups[site.weight_name];
        if (!group.empty()) {
            const SiteSlice& prev = group.back();
            slice.rank_off = prev.rank_off + prev.rank;
            slice.out_off = prev.out_off + prev.out;
        }
        group.push_back(std::move(slice));
    }
    return groups;
}

inline std::string fused_prefix(const std::string& weight) { return "@fused:" + weight; }

inline void add_grad(ParamSet* grads, const std::string& name, const std::vector<int64_t>& shape,
                     const std::vector<float>& values) {
    for (Param& p : *grads) {
        if (p.name != name) continue;
        if (p.shape != shape || p.value.size() != values.size())
            throw std::runtime_error("host step: gradient '" + name + "' arrived twice with shapes");
        for (size_t i = 0; i < values.size(); ++i) p.value[i] += values[i];
        return;
    }
    Param p;
    p.name = name;
    p.shape = shape;
    p.value = values;
    grads->push_back(std::move(p));
}

inline const Param* find_grad(const ParamSet& grads, const std::string& name) {
    for (const Param& p : grads)
        if (p.name == name) return &p;
    return nullptr;
}

// The live forward may skip an adapter path (a zero B is the usual init), so the walk sees no
// record for that factor. Every delta parameter still has a gradient: zero where the path did not
// run. Fill those before the split, which needs both factors' shapes.
inline void fill_missing_grads(ParamSet* grads, const ParamSet& params) {
    if (!grads) return;
    for (const Param& p : params) {
        if (find_grad(*grads, p.name)) continue;
        Param zero;
        zero.name = p.name;
        zero.shape = p.shape;
        zero.value.assign(p.value.size(), 0.0f);
        grads->push_back(std::move(zero));
    }
}

// The caller compares a step's gradients to the parameters position by position, and the split
// replaces entries (erase + append) rather than writing them in place. Put the set back in the
// parameters' order once everything is in it. A name this step did not produce is left for the
// caller's own check to report.
inline void order_grads_like(ParamSet* grads, const ParamSet& params) {
    if (!grads || grads->size() != params.size()) return;
    ParamSet ordered;
    ordered.reserve(params.size());
    for (const Param& p : params) {
        auto it = std::find_if(grads->begin(), grads->end(),
                               [&](const Param& g) { return g.name == p.name; });
        if (it == grads->end()) return;
        ordered.push_back(std::move(*it));
        grads->erase(it);
    }
    *grads = std::move(ordered);
}

// A delta parameter's leaf under its module, found by the shape the split is about to write. The
// adjoint's own leaf names are the engine's spelling, which the join must not repeat: a second
// adapter kind names its factors differently and this still finds and writes them.
inline std::string param_leaf(const ParamSet& params, const std::string& module,
                              const std::vector<int64_t>& shape, const std::string& fallback) {
    const std::string head = module + ".";
    for (const Param& p : params)
        if (p.name.size() > head.size() && p.name.compare(0, head.size(), head) == 0 &&
            p.shape == shape)
            return p.name.substr(head.size());
    return fallback;
}

inline std::string leaf_of(const std::string& name) {
    const size_t dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(dot + 1);
}

// The delta's prefix is what every gradient name starts with; a site's own module comes after it,
// so the split finds the walk's gradients under the same prefix and writes the delta's per-site
// ones. A delta need not use an empty prefix, and a step that names a gradient the delta never
// asked for leaves that parameter with no gradient at all.
//
// The walk's fused gradient carries whatever leaf its family's adjoint gave it; find it by the
// factor shapes the sites gave, not by an adapter's name, and write each module's factor under the
// leaf the delta's own ParamSet uses.
inline void split_adapter_grads(ParamSet* grads, const ParamSet& params, const SiteGroups& groups,
                                const std::string& grad_prefix) {
    if (!grads) return;
    ParamSet split;
    std::vector<std::string> replaced;
    for (const auto& [weight, slices] : groups) {
        if (slices.empty()) continue;
        const bool fused = slices.size() > 1;
        const std::string prefix =
            grad_prefix + (fused ? fused_prefix(weight) : slices.front().module);
        int64_t rank_total = 0, out_total = 0;
        const int64_t in = slices.front().in;
        for (const SiteSlice& slice : slices) {
            rank_total += slice.rank;
            out_total += slice.out;
            if (slice.in != in)
                throw std::runtime_error("host step: the delta sites of '" + weight +
                                         "' disagree about their input width");
        }
        if (rank_total <= 0 || in <= 0) continue;
        const Param* a = nullptr;
        const Param* b = nullptr;
        std::string a_name, b_name;
        for (const Param& p : *grads) {
            if (p.name.rfind(prefix + ".", 0) != 0) continue;
            const int64_t elems = static_cast<int64_t>(p.value.size());
            if (elems == rank_total * in) {
                a = &p;
                a_name = p.name;
            } else if (elems % rank_total == 0) {
                b = &p;
                b_name = p.name;
            }
        }
        // A live record can carry only one factor (the other path was skipped); the split then
        // has nothing to do for that weight and leaves the factor it does have in place.
        if (!a || !b) continue;
        const int64_t b_rows = static_cast<int64_t>(b->value.size()) / rank_total;
        // The kernel weight can have unadapted output rows after the adapter's own (a fused qkv
        // with a gate tail, say); the site's B is shorter than the weight and those rows are not
        // part of any module's gradient.
        if (b_rows < out_total)
            throw std::runtime_error("host step: the gradient of '" + weight +
                                     "' is shorter than its delta sites' B factors");
        for (const SiteSlice& slice : slices) {
            const std::vector<int64_t> a_shape{slice.rank, in};
            std::vector<float> av(static_cast<size_t>(slice.rank * in));
            for (int64_t j = 0; j < slice.rank; ++j)
                for (int64_t k = 0; k < in; ++k)
                    av[static_cast<size_t>(j * in + k)] =
                        a->value[static_cast<size_t>((slice.rank_off + j) * in + k)];
            add_grad(&split, grad_prefix + slice.module + "." +
                                  param_leaf(params, slice.module, a_shape, leaf_of(a_name)),
                     a_shape, av);

            const std::vector<int64_t> b_shape{slice.out, slice.rank};
            std::vector<float> bv(static_cast<size_t>(slice.out * slice.rank));
            for (int64_t n = 0; n < slice.out; ++n)
                for (int64_t j = 0; j < slice.rank; ++j)
                    bv[static_cast<size_t>(n * slice.rank + j)] =
                        b->value[static_cast<size_t>((slice.out_off + n) * rank_total +
                                                     slice.rank_off + j)];
            add_grad(&split, grad_prefix + slice.module + "." +
                                  param_leaf(params, slice.module, b_shape, leaf_of(b_name)),
                     b_shape, bv);
        }
        replaced.push_back(a_name);
        replaced.push_back(b_name);
    }
    if (split.empty()) return;
    // A parameter filled with zeros before the split may carry the same name the split writes;
    // replace it rather than keep both.
    for (const Param& p : split)
        grads->erase(std::remove_if(grads->begin(), grads->end(),
                                    [&](const Param& g) { return g.name == p.name; }),
                     grads->end());
    grads->erase(std::remove_if(grads->begin(), grads->end(),
                                [&](const Param& p) {
                                    return std::find(replaced.begin(), replaced.end(), p.name) !=
                                           replaced.end();
                                }),
                 grads->end());
    for (Param& p : split) grads->push_back(std::move(p));
}

inline std::string module_leaf(const std::string& module) {
    const size_t dot = module.find_last_of('.');
    return dot == std::string::npos ? module : module.substr(dot + 1);
}

// The family adjoints for the linear ops find their operands by the schema's names, while the
// live tape carries the engine's buffer names. They also take `M` from the schema's parameter,
// which the tape records as the engine's row capacity, not the rows this step used. Hand the
// adjoint the schema's names and the step's rows; the walk keeps the tape's record for versions,
// boundaries and snapshots.
inline std::string linear_schema_name(const OpRecord& rec, size_t j) {
    if (rec.op == "gemm_nt") return j == 0 ? "a" : j == 1 ? "b" : "";
    // gemm_nt_q's schema is a, a_scale, b, b_scale: the activation's quantisation sits between
    // the two matrix operands, unlike gemm_nt's plain a, b.
    if (rec.op == "gemm_nt_q")
        return j == 0 ? "a" : j == 1 ? "a_scale" : j == 2 ? "b" : j == 3 ? "b_scale" : "";
    if (rec.op == "logits_gemm") return j == 0 ? "x" : j == 1 ? "lm_head" : "";
    if (rec.op == "gather_rows") return j == 0 ? "x" : j == 1 ? "idx" : "";
    if (rec.op == "embed_lookup") return j == 0 ? "tokens" : j == 1 ? "wte" : "";
    return "";
}

inline void prepare_adjoint_record(const OpRecord& rec, OpRecord* out) {
    *out = rec;
    for (size_t j = 0; j < out->operands.size(); ++j) {
        std::string name = linear_schema_name(rec, j);
        if (!name.empty()) out->operands[j].buffer = std::move(name);
    }
    int64_t rows = 0;
    for (const Operand& o : rec.operands)
        if (o.written && !o.shape.empty()) {
            rows = o.shape[0];
            break;
        }
    if (rows <= 0)
        for (const Operand& o : rec.operands)
            if (!o.shape.empty()) {
                rows = o.shape[0];
                break;
            }
    if (rows > 0)
        for (auto& [key, value] : out->params)
            if (key == "M") {
                if (int64_t* m = std::get_if<int64_t>(&value)) *m = rows;
            }
}

// The weight a lora record belongs to, from the adapter factor it names: lora.<weight>.a or .b.
// The record names the kernel weight (`blk.3.ffn_down.weight`), while the delta's site names the
// parameter without the container's `.weight` leaf (`blk.3.ffn_down`); drop that leaf so both
// name the same site.
inline std::string record_weight(const OpRecord& rec) {
    for (const Operand& o : rec.operands) {
        if (o.buffer.rfind("lora.", 0) != 0) continue;
        const size_t dot = o.buffer.find_last_of('.');
        if (dot == std::string::npos) continue;
        const std::string leaf = o.buffer.substr(dot + 1);
        if (leaf != "a" && leaf != "b") continue;
        std::string weight = o.buffer.substr(5, dot - 5);
        const std::string suffix = ".weight";
        if (weight.size() > suffix.size() &&
            weight.compare(weight.size() - suffix.size(), suffix.size(), suffix) == 0)
            weight.resize(weight.size() - suffix.size());
        return weight;
    }
    return {};
}

inline void set_error(std::string* err, const char* what) {
    if (err) *err = what;
}

}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER && RAD_REPLAY_BLOCK
