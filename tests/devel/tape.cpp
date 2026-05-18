// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/devel/tape.hpp>
//
TEST(tests_devel, tape)  {
    //
    // tapoe
    ad_tensor::devel::tape_t tape;
    //
    EXPECT_TRUE( tape.is_empty() );
    EXPECT_EQ( tape.tape_id(),     0 );
    EXPECT_FALSE( tape.recording() );
}
