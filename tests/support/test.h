// tests/support/test.h -- self-registering checks for the collected host test binary.
//
// Every source under tests/host/ becomes one executable, so only one of them may define
// main (today tests/host/main.cpp does). A test file therefore brings no main of its
// own: it writes its checks as
//
//   R97_TEST(name) { ... }
//
// and the runner runs every registered test before static teardown: the host and gpu binaries run
// theirs from their own main (`tests/host/main.cpp`, `tests/gpu/main.cpp`), which keeps the
// statics a test needs alive for the whole run, and the at-exit runner armed below is what the
// model tier uses. A failure makes the process exit non-zero, which is what ctest reads, so adding
// a test is adding one file and nothing else.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <vector>

namespace r97test {

struct Test {
    const char* name;
    void (*fn)();
};

inline std::vector<Test>& tests() {
    static std::vector<Test> t;
    return t;
}

struct Reg {
    Reg(const char* name, void (*fn)());
};

inline int run_all() {
    int failed = 0;
    for (const Test& t : tests()) {
        try {
            t.fn();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: threw: %s\n", t.name, e.what());
            ++failed;
        } catch (...) {
            std::fprintf(stderr, "%s: threw\n", t.name);
            ++failed;
        }
    }
    std::printf("%zu tests, %d failed\n", tests().size(), failed);
    return failed ? 1 : 0;
}

// Runs the registered tests from exit(). The host and gpu binaries call run_all() from their own
// main instead (`tests/host/main.cpp`, `tests/gpu/main.cpp`), because at exit the statics a
// test needs are gone; a binary whose main returns without running them -- the model tier -- lands
// here.
inline void run_at_exit() {
    const int failed = run_all();
    std::fflush(stdout);
    std::fflush(stderr);
    // The exit status is the only thing ctest sees from a main that is not ours. _Exit
    // skips the rest of exit(), so no other destructor can hide the failure.
    if (failed != 0) std::_Exit(1);
}

struct Runner {
    Runner() {
        // The handler runs after main returns, when every test has registered itself.
        if (std::atexit(&run_at_exit) != 0)
            std::fprintf(stderr, "r97test: could not register the runner\n");
    }
};

// Armed by the first registration rather than by a static of its own, so that the registry
// tests() is constructed first and therefore destroyed last: at exit the handler runs
// before the registry goes away, whatever order the translation units initialise in.
inline void arm_runner() {
    static Runner runner;
    (void)runner;
}

inline Reg::Reg(const char* name, void (*fn)()) {
    tests().push_back({name, fn});
    arm_runner();
}

}  // namespace r97test

// The default build is RelWithDebInfo, where <cassert> is compiled out; use R97_CHECK,
// which throws on failure and is caught by run_all(), for every real check.
#define R97_CHECK(cond) \
    do { \
        if (!(cond)) throw std::runtime_error("check failed: " #cond); \
    } while (0)

// Defines static void test_##name_() from the brace body that follows.
#define R97_TEST(name_) \
    static void test_##name_(); \
    static r97test::Reg reg_##name_(#name_, test_##name_); \
    static void test_##name_()
