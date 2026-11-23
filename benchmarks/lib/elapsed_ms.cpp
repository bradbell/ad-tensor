// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin elapsed_ms ben}
{xrst_spell
    hpp
}

The Elapsed Time Utility
########################

Syntax
******
{xrst_code cpp}
    #include "lib/elapsed.hpp"
    ms = elapsed_ms();
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_ELAPSED_MS, END_ELAPSED_MS
}

ms
**
is the number of milliseconds since the previous call,
by this thread, to elapsed_ms.
This value is unspecified for the first call to elapsed_ms by this thread.

{xrst_end elapsed_ms}
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
    thread_local chrono::time_point previous_time = chrono::steady_clock::now();
}
//
// BEGIN_ELAPSED_MS
double elapsed_ms(void)
{   // END_ELAPSED_MS
    chrono::time_point current_time = chrono::steady_clock::now();
    auto microseconds = chrono::duration_cast<chrono::microseconds>(
        current_time - previous_time
    ).count();
    double ms = double(microseconds) / 1000.0;
    previous_time = current_time;
    return ms;
}
