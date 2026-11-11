// tests/support/self_check.h -- a host test file that runs itself from a static initializer.
//
// The host tests are one binary, collected from tests/host/*.cpp, and its entry point belongs to
// tests/host/main.cpp; a second main() would not link. A file that writes R97_SELF_CHECK instead
// of an R97_TEST registration runs its check from a static initializer, which the binary reaches
// before that main: the check runs once with its name printed, and a failure prints its expression
// and location and aborts, which is what makes the binary exit non-zero and ctest report it. New
// tests register with R97_TEST (tests/support/test.h), which the tier's runner runs before static
// teardown; converting a file that still says R97_SELF_CHECK is turning its line into that
// registration and nothing else.
//
// CHECK is a real check, not assert(): the project builds RelWithDebInfo, where -DNDEBUG makes
// assert() compile out and a check that never runs passes.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace r97test {

[[noreturn]] inline void check_failed(const char* expr, const char* file, int line) {
    std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expr);
    std::fflush(stderr);
    std::abort();
}

// Announce a test this build leaves out, by name, the way run() announces one it runs. A tier
// built without a patch then says which tests it skipped instead of passing silently or failing.
inline void skip(const char* name, const char* why) {
    std::printf("[host] %s ... skipped (%s)\n", name, why);
    std::fflush(stdout);
}

// Run one test. A test that throws is a failure too, reported by name, not a terminate() in
// another file's main.
inline void run(const char* name, void (*fn)()) {
    std::printf("[host] %s\n", name);
    std::fflush(stdout);
    try {
        fn();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[host] %s: threw: %s\n", name, e.what());
        std::fflush(stderr);
        std::abort();
    }
}

}  // namespace r97test

#define R97_CHECK(expr) ((expr) ? (void)0 : ::r97test::check_failed(#expr, __FILE__, __LINE__))
#define CHECK(expr) R97_CHECK(expr)

#define R97_SELF_CHECK(fn)                     \
    namespace {                                \
    struct fn##_self_check {                   \
        fn##_self_check() { ::r97test::run(#fn, fn); } \
    } fn##_self_check_instance;                \
    }
