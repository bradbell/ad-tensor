// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <ad_tensor/dev/op_enum.hpp>
//
TEST(tests_devel, op_enum)  {
    using ad_tensor::dev::op_enum_t;
    //
    EXPECT_EQ( op_enum_t::add,  op_enum_t::add );
    EXPECT_NE( op_enum_t::add,  op_enum_t::sub );
    EXPECT_NE( op_enum_t::add,  op_enum_t::mul );
    EXPECT_NE( op_enum_t::add,  op_enum_t::div );
}
