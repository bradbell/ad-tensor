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
TEST(examples_adfn, forward_par)  {
    using ad_tensor::ad_t;
    using ad_tensor::adfn_t;
    using ad_tensor::options_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // options
    options_t options;
    //
    // p
    // We use p for the domain parameters
    vector<Tensor> p;
    p.push_back( torch::tensor( {2.0, 3.0} ) );
    p.push_back( torch::tensor( {4.0, 5.0} ) );
    //
    // ap
    vector<Tensor> x;
    auto [ap, ax] = ad_t::start_recording(p, x);
    //
    // acon
    // create a constant after start_recording so can use it in the recording
    ad_t acon( torch::tensor( {-1} ) );
    //
    // aprod
    ad_t aprod = ap[0] * ap[1];
    //
    // ay
    // We use y for the range space.
    vector<ad_t> ay;
    ay.push_back( acon + aprod );
    //
    // y = f(p)
    adfn_t f = ad_t::stop_recording(ay);
    //
    // p
    p.resize(0);
    p.push_back( torch::tensor( {6.0, 7.0} ) );
    p.push_back( torch::tensor( {8.0, 9.0} ) );
    //
    // all_par
    ad_tensor::vector<Tensor> all_par = f.forward_par(p, options);
    //
    EXPECT_EQ( all_par.size(), 4 );
    //
    bool equal = all_par[0].equal( torch::tensor({6.0, 7.0}) );
    EXPECT_TRUE( equal );
    //
    equal = all_par[1].equal( torch::tensor({8.0, 9.0}) );
    EXPECT_TRUE( equal );
    //
    equal = all_par[2].equal( torch::tensor({48.0 , 63.0}) );
    EXPECT_TRUE( equal );
    //
    equal = all_par[3].equal( torch::tensor({47.0 , 62.0}) );
    EXPECT_TRUE( equal );
}
// END_CPP
