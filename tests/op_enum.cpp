// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <ad_tensor/op_enum.hpp>
//
TEST(tests, op_enum)  {
    using ad_tensor::op_enum;
    //
    EXPECT_EQ( op_enum::add,  op_enum::add );
    EXPECT_NE( op_enum::add,  op_enum::sub );
    EXPECT_NE( op_enum::add,  op_enum::mul );
    EXPECT_NE( op_enum::add,  op_enum::div );
}
