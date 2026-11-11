// tests/host/vector_tape.cpp -- VectorTape answers reads from its arrays and records them.
//
// The host tests are one binary with one main (tests/host/main.cpp); each test file registers its
// checks with R97_TEST instead of carrying an entry point of its own.
#include "support/test.h"
#include "support/vector_tape.h"

#include <cstdio>

using r97::Batch;
using r97::TokenSeq;

static Batch two_sequences() {
    Batch b;
    b.push_back(TokenSeq{{1, 2, 3, 4}, {0, 1, 0, 1}});
    b.push_back(TokenSeq{{5, 6, 7}, {0, 0, 1}});
    return b;
}

R97_TEST(vector_tape) {
    // A logits buffer of 5 rows x 3 cols: rows 0..2 for the first sequence, 3..4 for the second.
    std::vector<float> logits = {
        0, 1, 2,    // row 0
        3, 4, 5,    // row 1
        6, 7, 8,    // row 2
        9, 10, 11,  // row 3
        12, 13, 14, // row 4
    };
    r97test::VectorTape tape;
    tape.set_batch(two_sequences())
        .set_scored(4)
        .set_buffer("logits", {5, 3}, logits);

    // Rows [1, 3) come back whole, in order, as f32.
    std::vector<float> got = tape.read("logits", 1, 3, true);
    R97_CHECK(got == std::vector<float>({3, 4, 5, 6, 7, 8}));

    // The window is the whole buffer until a test restricts it.
    int64_t lo = -1, hi = -1;
    tape.window("logits", &lo, &hi);
    R97_CHECK(lo == 0 && hi == 5);

    tape.set_window("logits", 2, 5);
    tape.window("logits", &lo, &hi);
    R97_CHECK(lo == 2 && hi == 5);

    // Only rows inside the window read out of the restricted range, row counts per shape.
    R97_CHECK(tape.shape("logits") == std::vector<int64_t>({5, 3}));
    R97_CHECK(tape.boundaries() == std::vector<std::string>({"logits"}));
    R97_CHECK(tape.total_scored() == 4);
    R97_CHECK(tape.batch().size() == 2 && tape.batch()[1].ids[2] == 7);

    // The record holds every read, with its range and its differentiated flag; a read of another
    // buffer would not be folded into the first's record.
    R97_CHECK(tape.reads_of("logits").size() == 1);
    R97_CHECK(tape.reads_of("logits")[0].row_lo == 1 && tape.reads_of("logits")[0].row_hi == 3);
    R97_CHECK(tape.reads_of("logits")[0].differentiated == true);
    tape.read("logits", 0, 1, false);
    tape.read("logits", 1, 2, true);
    R97_CHECK(tape.reads().size() == 3);
    R97_CHECK(!tape.reads()[1].differentiated && tape.reads()[2].differentiated);
    R97_CHECK(tape.reads_of("other").empty());

    std::puts("vector_tape ok");
}
