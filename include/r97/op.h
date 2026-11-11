// r97/op.h -- one record on the op tape.
//
// The forward driver appends an OpRecord per op it runs; the adjoint replays the step from these
// records. Operands appear in schema order, so position names the operand without a lookup.
#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace r97 {

// One operand a named buffer supplies, or records into. A read's rows are first row and row
// count; a write's are the rows it produces. Buffer identity is what the objective's Seed names.
// An in-place write (INOUT) also reads the value it replaces: the adjoint adds the cotangent that
// value takes to the write's slot, and the backward walk passes it on to the previous version.
struct Operand {
    std::string buffer;         // the buffer's name on the tape (e.g. "logits")
    std::string dtype;          // dtype name as the op reads it ("f32", "w8a8", ...)
    std::vector<int64_t> shape; // full logical shape, row-major
    int64_t row_lo = 0;         // first row this op touches
    int64_t rows = 0;           // how many rows this op touches; 0 = none/whole-buffer op
    bool written = false;       // read (false) or written (true)
    bool in_place = false;      // a write that also reads the value it replaces
};

struct OpRecord {
    std::string op;             // op name as the schema gives it
    std::vector<std::pair<std::string, std::variant<int64_t, std::string>>> params;
    std::vector<Operand> operands;  // in schema order
    int64_t block = -1;             // which transformer block, -1 outside one
    int64_t position = 0;           // position in the step, in append order
    int64_t n = 0;                  // the ranged count this issue resolved to
};

}  // namespace r97
