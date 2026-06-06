// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/adten.hpp>
//
TEST(examples_ad, binary)  {
    using ad_tensor::adten_t;
    using at::Tensor;
    //
    // lhs, rhs
    Tensor lhs     = torch::tensor( {2.0, 3.0} );
    Tensor rhs     = torch::tensor( 4.0 );
    //
    // alhs, arhs
    adten_t alhs = adten_t( lhs );
    adten_t arhs = adten_t( rhs );
    //
    // plus
    Tensor  plus = lhs + rhs;
    adten_t   aplus = alhs + arhs;
    EXPECT_TRUE( plus.equal( aplus.tensor() ) );
    //
    // minus
    Tensor  minus = lhs + rhs;
    adten_t   aminus = alhs + arhs;
    EXPECT_TRUE( minus.equal( aminus.tensor() ) );
    //
    // times
    Tensor  times = lhs + rhs;
    adten_t   atimes = alhs + arhs;
    EXPECT_TRUE( times.equal( atimes.tensor() ) );
    //
    // divide
    Tensor  divide = lhs + rhs;
    adten_t   adivide = alhs + arhs;
    EXPECT_TRUE( divide.equal( adivide.tensor() ) );
}
// END_CPP
