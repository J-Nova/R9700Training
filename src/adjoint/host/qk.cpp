// src/adjoint/host/qk.cpp -- the qk family's f64 twins and adjoints. docs/adjoints/qk.md carries
// every formula; this file is the executable form of that note.
#include "qk.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace r97::adjoint::qk {
namespace {

void require(bool condition, const char* what) {
    if (!condition) throw std::invalid_argument(what);
}

int64_t checked_mul(int64_t a, int64_t b) {
    require(a >= 0 && b >= 0, "qk: negative extent");
    if (a != 0 && b > INT64_MAX / a) throw std::overflow_error("qk: extent overflow");
    return a * b;
}

void rope_geometry(const RopeShape& s, int64_t* row, int64_t* rotated_heads) {
    require(s.tokens >= 0 && s.n_head >= 0 && s.n_head_kv >= 0, "qk: negative rope extent");
    require(s.head_dim >= 0, "qk: negative head_dim");
    require(s.rotary_dim >= 0 && s.rotary_dim <= s.head_dim && (s.rotary_dim & 1) == 0,
            "qk: rotary_dim must be even and at most head_dim");
    *row = checked_mul(s.n_head + 2 * s.n_head_kv, s.head_dim);
    *rotated_heads = s.n_head + s.n_head_kv;
}

void kv_geometry(const KvStoreShape& s) {
    require(s.tokens >= 0 && s.kv_heads > 0 && s.head_dim > 0 && s.block_size > 0 &&
                s.n_blocks >= 0,
            "qk: bad kv_store extent");
}

int64_t kv_pool_base(const KvStoreShape& s, int64_t slot, int64_t head) {
    const int64_t block = slot / s.block_size;
    const int64_t offset = slot % s.block_size;
    return ((block * s.kv_heads + head) * s.block_size + offset) * 2 * s.head_dim;
}

}  // namespace

// ------------------------------------------------------------------ rope

RopeTable rope_table_f64(int64_t rows, int64_t rotary_dim, double theta, double scale) {
    require(rows >= 0, "qk: negative rope table rows");
    require(rotary_dim >= 0 && (rotary_dim & 1) == 0, "qk: rope table rotary_dim must be even");
    require(theta > 0.0 && std::isfinite(theta), "qk: rope theta must be positive and finite");
    require(scale > 0.0 && std::isfinite(scale), "qk: rope scale must be positive and finite");

    RopeTable table;
    table.rows = rows;
    table.rotary_dim = rotary_dim;
    table.cos_sin.assign((size_t)checked_mul(rows, rotary_dim), 0.0);
    const int64_t half = rotary_dim / 2;
    std::vector<double> inv((size_t)half);
    for (int64_t i = 0; i < half; ++i)
        inv[(size_t)i] = std::pow(theta, -2.0 * (double)i / (double)rotary_dim);
    for (int64_t p = 0; p < rows; ++p) {
        const double position = (double)p / scale;
        for (int64_t i = 0; i < half; ++i) {
            const double angle = position * inv[(size_t)i];
            table.cos_sin[(size_t)(p * rotary_dim + i)] = std::cos(angle);
            table.cos_sin[(size_t)(p * rotary_dim + half + i)] = std::sin(angle);
        }
    }
    return table;
}

void rope_forward_f64(const double* qkv, const int32_t* positions, const RopeShape& shape,
                      const RopeTable& table, double* out) {
    int64_t row = 0, rotated_heads = 0;
    rope_geometry(shape, &row, &rotated_heads);
    require(table.rotary_dim == shape.rotary_dim,
            "qk: rope table and operand disagree on rotary_dim");
    require(qkv && out && positions, "qk: null rope buffer");
    if (shape.tokens == 0 || row == 0) return;

    const int64_t half = shape.rotary_dim / 2;
    for (int64_t t = 0; t < shape.tokens; ++t) {
        const int64_t p = positions[t];
        require(p >= 0 && p < table.rows, "qk: rope position outside the table");
        const double* cs = table.at(p);
        const int64_t base = t * row;
        for (int64_t h = 0; h < rotated_heads; ++h) {
            const int64_t head = base + h * shape.head_dim;
            for (int64_t i = 0; i < half; ++i) {
                const int64_t lo = head + i;
                const int64_t hi = lo + half;
                const double x0 = qkv[lo], x1 = qkv[hi];
                const double c = cs[i], s = cs[half + i];
                out[lo] = x0 * c - x1 * s;
                out[hi] = x0 * s + x1 * c;
            }
            for (int64_t i = shape.rotary_dim; i < shape.head_dim; ++i)
                out[head + i] = qkv[head + i];
        }
        // Value heads are not rotated; copy them through.
        for (int64_t h = rotated_heads; h < shape.n_head + 2 * shape.n_head_kv; ++h)
            for (int64_t i = 0; i < shape.head_dim; ++i)
                out[base + h * shape.head_dim + i] = qkv[base + h * shape.head_dim + i];
    }
}

void rope_adjoint_f64(const double* d_out, const int32_t* positions, const RopeShape& shape,
                      const RopeTable& table, double* d_qkv) {
    int64_t row = 0, rotated_heads = 0;
    rope_geometry(shape, &row, &rotated_heads);
    require(table.rotary_dim == shape.rotary_dim,
            "qk: rope table and operand disagree on rotary_dim");
    require(d_out && d_qkv && positions, "qk: null rope buffer");
    if (shape.tokens == 0 || row == 0) return;

    const int64_t half = shape.rotary_dim / 2;
    for (int64_t t = 0; t < shape.tokens; ++t) {
        const int64_t p = positions[t];
        require(p >= 0 && p < table.rows, "qk: rope position outside the table");
        const double* cs = table.at(p);
        const int64_t base = t * row;
        for (int64_t h = 0; h < rotated_heads; ++h) {
            const int64_t head = base + h * shape.head_dim;
            for (int64_t i = 0; i < half; ++i) {
                const int64_t lo = head + i;
                const int64_t hi = lo + half;
                const double y0 = d_out[lo], y1 = d_out[hi];
                const double c = cs[i], s = cs[half + i];
                // The transpose of [[c, -s], [s, c]].
                d_qkv[lo] = c * y0 + s * y1;
                d_qkv[hi] = -s * y0 + c * y1;
            }
            for (int64_t i = shape.rotary_dim; i < shape.head_dim; ++i)
                d_qkv[head + i] = d_out[head + i];
        }
        for (int64_t h = rotated_heads; h < shape.n_head + 2 * shape.n_head_kv; ++h)
            for (int64_t i = 0; i < shape.head_dim; ++i)
                d_qkv[base + h * shape.head_dim + i] = d_out[base + h * shape.head_dim + i];
    }
}

// ------------------------------------------------------------------ kv_store

int64_t kv_slot(const int32_t* block_table, int64_t block_table_stride, int64_t block_size,
                int64_t sequence, int64_t position) {
    require(block_table && block_table_stride > 0 && block_size > 0 && sequence >= 0 &&
                position >= 0,
            "qk: bad pool address");
    const int64_t block = block_table[sequence * block_table_stride + position / block_size];
    return block < 0 ? -1 : block * block_size + position % block_size;
}

void kv_store_forward_f64(const double* k, const double* v, const int32_t* slot,
                          const KvStoreShape& shape, double* pool) {
    kv_geometry(shape);
    require(k && v && slot && pool, "qk: null kv_store buffer");
    const int64_t head_values = checked_mul(shape.kv_heads, shape.head_dim);
    const int64_t slots = checked_mul(shape.n_blocks, shape.block_size);
    for (int64_t t = 0; t < shape.tokens; ++t) {
        const int64_t s = slot[t];
        if (s < 0) continue;
        require(s < slots, "qk: kv_store slot outside the pool");
        const int64_t row = t * head_values;
        for (int64_t h = 0; h < shape.kv_heads; ++h) {
            const int64_t base = kv_pool_base(shape, s, h);
            const int64_t src = row + h * shape.head_dim;
            for (int64_t d = 0; d < shape.head_dim; ++d) {
                pool[base + d] = k[src + d];
                pool[base + shape.head_dim + d] = v[src + d];
            }
        }
    }
}

void kv_store_adjoint_f64(const double* d_pool, const int32_t* slot, const KvStoreShape& shape,
                          double* d_k, double* d_v) {
    kv_geometry(shape);
    require(d_pool && slot && d_k && d_v, "qk: null kv_store adjoint buffer");
    const int64_t head_values = checked_mul(shape.kv_heads, shape.head_dim);
    const int64_t slots = checked_mul(shape.n_blocks, shape.block_size);
    std::fill(d_k, d_k + checked_mul(shape.tokens, head_values), 0.0);
    std::fill(d_v, d_v + checked_mul(shape.tokens, head_values), 0.0);
    for (int64_t t = 0; t < shape.tokens; ++t) {
        const int64_t s = slot[t];
        if (s < 0) continue;
        require(s < slots, "qk: kv_store slot outside the pool");
        const int64_t row = t * head_values;
        for (int64_t h = 0; h < shape.kv_heads; ++h) {
            const int64_t base = kv_pool_base(shape, s, h);
            const int64_t dst = row + h * shape.head_dim;
            for (int64_t d = 0; d < shape.head_dim; ++d) {
                // A slot written by two tokens receives both cotangents.
                d_k[dst + d] += d_pool[base + d];
                d_v[dst + d] += d_pool[base + shape.head_dim + d];
            }
        }
    }
}

// ------------------------------------------------------------------ the gate

void sigmoid_forward_f64(const double* x, int64_t n, double* out) {
    require(n >= 0, "qk: negative sigmoid extent");
    require((n == 0) || (x && out), "qk: null sigmoid buffer");
    for (int64_t i = 0; i < n; ++i) {
        const double v = x[i];
        if (v >= 0.0) {
            out[i] = 1.0 / (1.0 + std::exp(-v));
        } else {
            const double e = std::exp(v);
            out[i] = e / (1.0 + e);
        }
    }
}

void sigmoid_adjoint_f64(const double* out, const double* d_out, int64_t n, double* d_x) {
    require(n >= 0, "qk: negative sigmoid extent");
    require((n == 0) || (out && d_out && d_x), "qk: null sigmoid buffer");
    for (int64_t i = 0; i < n; ++i) d_x[i] = d_out[i] * out[i] * (1.0 - out[i]);
}

void mul_forward_f64(const double* a, const double* b, int64_t rows, int64_t n,
                     bool b_broadcast, double* out) {
    require(rows >= 0 && n >= 0, "qk: negative mul extent");
    require((rows == 0 || n == 0) || (a && b && out), "qk: null mul buffer");
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t i = 0; i < n; ++i)
            out[r * n + i] = a[r * n + i] * (b_broadcast ? b[i] : b[r * n + i]);
}

void mul_adjoint_f64(const double* d_out, const double* a, const double* b, int64_t rows,
                     int64_t n, bool b_broadcast, double* d_a, double* d_b) {
    require(rows >= 0 && n >= 0, "qk: negative mul extent");
    require((rows == 0 || n == 0) || (d_out && a && b && (d_a || d_b)),
            "qk: null mul buffer");
    if (d_b && b_broadcast) std::fill(d_b, d_b + n, 0.0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < n; ++i) {
            const int64_t at = r * n + i;
            const double bv = b_broadcast ? b[i] : b[at];
            if (d_a) d_a[at] = d_out[at] * bv;
            if (d_b) {
                if (b_broadcast) d_b[i] += d_out[at] * a[at];
                else d_b[at] = d_out[at] * a[at];
            }
        }
    }
}

}  // namespace r97::adjoint::qk
