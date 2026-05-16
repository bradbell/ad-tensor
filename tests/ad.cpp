// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/ad.hpp>
//
TEST(tests, ad)  {
    using ad_tensor::ad_t;
    using torch::Tensor;
    //
    Tensor lhs     = torch::tensor( {2.0, 3.0} );
    Tensor rhs     = torch::tensor( {4.0, 5.0} );
    Tensor sum     = lhs + rhs;
    Tensor product = lhs * rhs;
    //
    ad_t alhs     = ad_t( std::move(lhs) );
    ad_t arhs     = ad_t( std::move(rhs) );
    ad_t asum     = alhs + arhs;
    ad_t aproduct = alhs * arhs;
    //
    bool equal;
    equal = torch::all( sum == asum.tensor() ).item<bool>();
    EXPECT_TRUE(equal);
    //
    equal = torch::all( product == aproduct.tensor() ).item<bool>();
    EXPECT_TRUE(equal);
}
