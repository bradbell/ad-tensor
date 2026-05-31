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
TEST(examples, f_forward_var)  {
    using ad_tensor::ad_t;
    using ad_tensor::adfn_t;
    using ad_tensor::options_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // options
    options_t options;
    //
    // x
    // We use x for the domain variables
    vector<Tensor> x;
    x.push_back( torch::tensor( {2.0, 3.0} ) );
    x.push_back( torch::tensor( {4.0, 5.0} ) );
    //
    // ax
    vector<Tensor> p;
    auto [ap, ax] = ad_t::start_recording(p, x);
    //
    // acon
    // create a constant after start_recording so can use it in the recording
    ad_t acon( torch::tensor( {-1} ) );
    //
    // aprod
    ad_t aprod = ax[0] * ax[1];
    //
    // ay
    // We use y for the range space.
    vector<ad_t> ay;
    ay.push_back( acon + aprod);
    //
    // y = f(x)
    adfn_t f = ad_t::stop_recording(ay);
    //
    // x
    x.resize(0);
    x.push_back( torch::tensor( {6.0, 7.0} ) );
    x.push_back( torch::tensor( {8.0, 9.0} ) );
    //
    // all_var
    ad_tensor::vector<Tensor> all_par;
    ad_tensor::vector<Tensor> all_var = f.forward_var(all_par, x, options);
    //
    EXPECT_EQ( all_var.size(), 4 );
    //
    bool equal = all_var[0].equal( torch::tensor({6.0, 7.0}) );
    EXPECT_TRUE( equal );
    //
    equal = all_var[1].equal( torch::tensor({8.0, 9.0}) );
    EXPECT_TRUE( equal );
    //
    equal = all_var[2].equal( torch::tensor({48.0 , 63.0}) );
    EXPECT_TRUE( equal );
    //
    equal = all_var[3].equal( torch::tensor({47.0 , 62.0}) );
    EXPECT_TRUE( equal );
}
// END_CPP
