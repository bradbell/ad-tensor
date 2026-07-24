// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adten, compound)  {
    using ad_tensor::adten_t;
    using at::Tensor;
    //
    // rhs, arhs
    Tensor  rhs  = torch::tensor( 2.0 );
    adten_t arhs = adten_t( rhs );
    //
    // plus_eq, aplus_eq
    Tensor  plus_eq  = torch::tensor( {-4.0, +4.0} );
    adten_t aplus_eq = adten_t( plus_eq.clone() );
    plus_eq  += rhs;
    aplus_eq += arhs;
    EXPECT_TRUE( plus_eq.equal( aplus_eq.at_ten() ) );
    //
    // minus_eq, aminus_eq
    Tensor  minus_eq  = torch::tensor( {-4.0, +4.0} );
    adten_t aminus_eq = adten_t( minus_eq.clone() );
    minus_eq  -= rhs;
    aminus_eq -= arhs;
    EXPECT_TRUE( minus_eq.equal( aminus_eq.at_ten() ) );
    //
    // times_eq, atimes_eq
    Tensor  times_eq  = torch::tensor( {-4.0, +4.0} );
    adten_t atimes_eq = adten_t( times_eq.clone() );
    times_eq  *= rhs;
    atimes_eq *= arhs;
    EXPECT_TRUE( times_eq.equal( atimes_eq.at_ten() ) );
    //
    // div_eq, adiv_eq
    Tensor  div_eq  = torch::tensor( {-4.0, +4.0} );
    adten_t adiv_eq = adten_t( div_eq.clone() );
    div_eq  /= rhs;
    adiv_eq /= arhs;
    EXPECT_TRUE( div_eq.equal( adiv_eq.at_ten() ) );
}
// END_CPP
