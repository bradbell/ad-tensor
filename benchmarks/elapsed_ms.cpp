// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin elapsed_ms_benchmarks ben}
{xrst_spell
}

Computing Elapsed Time
######################

Syntax
******
{xrst_code cpp}
    ms = elapsed_ms();
{xrst_code}

ms
**
is the number of milliseconds since the previous call to elapsed_ms.

{xrst_end elapsed_ms_benchmarks}
*/
// BEGIN_SETUP
#include <chrono>
//
namespace {
    //
    // chrono
    namespace chrono = std::chrono;
    //
    // previous_time
    chrono::time_point previous_time = chrono::steady_clock::now();
}
//
// elapsed_ms
double elapsed_ms(void) {
    chrono::time_point current_time = chrono::steady_clock::now();
    auto microseconds = chrono::duration_cast<chrono::microseconds>(
        current_time - previous_time
    ).count();
    double ms = double(microseconds) / 1000.0;
    previous_time = current_time;
    return ms;
}
