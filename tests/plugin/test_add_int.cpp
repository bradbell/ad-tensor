// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include "get_plugin.hpp"

TEST(tests_plugin, add_int) {
    //
    auto add_int = get_plugin<int(int, int)>("add_int");
    EXPECT_EQ(add_int(2, 3), 5);
}
