// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adten, binary)  {
    using ad_tensor::adten_t;
    //
    // lhs, rhs
    at::Tensor lhs     = torch::tensor( {2.0, 4.0} );
    at::Tensor rhs     = torch::tensor( 2.0 );
    //
    // alhs, arhs
    adten_t alhs = adten_t( lhs );
    adten_t arhs = adten_t( rhs );
    //
    // plus
    at::Tensor  plus  = lhs + rhs;
    adten_t aplus = alhs + arhs;
    EXPECT_TRUE( plus.equal( aplus.at_ten() ) );
    //
    // minus
    at::Tensor  minus  = lhs - rhs;
    adten_t     aminus = alhs - arhs;
    EXPECT_TRUE( minus.equal( aminus.at_ten() ) );
    //
    // times
    at::Tensor  times  = lhs * rhs;
    adten_t     atimes = alhs * arhs;
    EXPECT_TRUE( times.equal( atimes.at_ten() ) );
    //
    // divide
    at::Tensor  divide  = lhs / rhs;
    adten_t     adivide = alhs / arhs;
    EXPECT_TRUE( divide.equal( adivide.at_ten() ) );
    //
    // less_than
    at::Tensor  less_than  = lhs < rhs;
    adten_t     aless_than = alhs < arhs;
    EXPECT_TRUE( less_than.equal( aless_than.at_ten() ) );
    //
    // less_equal
    at::Tensor  less_equal  = lhs <= rhs;
    adten_t     aless_equal = alhs <= arhs;
    EXPECT_TRUE( less_equal.equal( aless_equal.at_ten() ) );
    //
    // greater_than
    at::Tensor  greater_than  = lhs > rhs;
    adten_t     agreater_than = alhs > arhs;
    EXPECT_TRUE( greater_than.equal( agreater_than.at_ten() ) );
    //
    // greater_equal
    at::Tensor  greater_equal  = lhs >= rhs;
    adten_t     agreater_equal = alhs >= arhs;
    EXPECT_TRUE( greater_equal.equal( agreater_equal.at_ten() ) );
    //
    // equalty
    at::Tensor  equalty  = lhs == rhs;
    adten_t     aequalty = alhs == arhs;
    EXPECT_TRUE( equalty.equal( aequalty.at_ten() ) );
    //
    // not_equal
    at::Tensor  not_equal  = lhs != rhs;
    adten_t     anot_equal = alhs != arhs;
    EXPECT_TRUE( not_equal.equal( anot_equal.at_ten() ) );
}
// END_CPP
