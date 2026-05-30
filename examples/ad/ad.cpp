// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/ad.hpp>
//
TEST(examples, ad)  {
    using ad_tensor::ad_t;
    using at::Tensor;
    //
    Tensor lhs     = torch::tensor( {2.0, 3.0} );
    Tensor rhs     = torch::tensor( {4.0, 5.0} );
    Tensor sum     = lhs + rhs;
    Tensor product = lhs * rhs;
    //
    ad_t alhs     = ad_t( lhs );
    ad_t arhs     = ad_t( rhs );
    ad_t asum     = alhs + arhs;
    ad_t aproduct = alhs * arhs;
    //
    EXPECT_TRUE( sum.equal( asum.tensor() ) );
    EXPECT_TRUE( product.equal( aproduct.tensor() ) );
}
// END_CPP
