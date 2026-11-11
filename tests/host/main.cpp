// tests/host/main.cpp -- the entry point of the collected host test binary.
//
// Like the gpu tier, the sources under tests/host/ build into one executable; each brings its
// checks as R97_TEST, and this file owns main so no test has to.
//
// The tests run here, from main, rather than from the at-exit runner tests/support/test.h arms
// from the first registration. That handler runs after the binary's static destructors, and the
// statics a host test needs are gone by then: chatdata_mask.cpp builds the fixture
// (support/fixture.h) from its own static initialiser, so the fixture, the temp directory behind
// it and radiance's device layer are destroyed before an armed handler would call a test.
// Running the tests inside main keeps every static alive for the whole run, whatever order the
// translation units initialise in; _Exit then skips the rest of exit(), so no destructor and no
// second run can hide the status. tests/gpu/main.cpp does the same, for the registry a plugin
// rebuilds as it loads.
#include "support/test.h"

#include <cstdio>
#include <cstdlib>

int main() {
    const int failed = r97test::run_all();
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(failed ? 1 : 0);
}
