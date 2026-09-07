// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <filesystem>
#include <iostream>
#include <gtest/gtest.h>
#include <ad_tensor/configure.hpp>

// STRING
// Adding a second layer forces the preprocessor to expand the macro
// value before converting it into a quoted string .
#define QUOTE(x) #x
#define STRING(x) QUOTE(x)

TEST(benchmarks, version) {
    std::cout << "ad-tensor version = " << STRING(AD_TENSOR_VERSION) << "\n";
    std::cout << "cmake build type  = " << STRING(CMAKE_BUILD_TYPE) << "\n";
}
