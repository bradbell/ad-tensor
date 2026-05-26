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
TEST(examples, forward_var)  {
    using ad_tensor::ad_t;
    using ad_tensor::adfn_t;
    using ad_tensor::options_t;
    using at::Tensor;
    using std::vector;
    //
    // dom_var
    vector<Tensor> dom_var;
    dom_var.push_back( torch::tensor( {2.0, 3.0} ) );
    dom_var.push_back( torch::tensor( {4.0, 5.0} ) );
    //
    // adom_var
    vector<Tensor> dom_par;
    auto [adom_par, adom_var] = ad_t::start_recording(
        dom_par, dom_var
    );
    //
    // acon
    // create a constant after start_recording so can use it in the recording
    ad_t acon( torch::tensor( {-1} ) );
    //
    // aprod
    ad_t aprod = adom_var.at(0) * adom_var.at(1);
    //
    // arange
    vector<ad_t> arange;
    arange.push_back(  acon + aprod);
    //
    // adfn
    adfn_t adfn = ad_t::stop_recording(arange);
    //
    // dom_var
    // dom_var is valid but unspecified before assignment
    dom_var = vector<Tensor>();
    dom_var.push_back( torch::tensor( {6.0, 7.0} ) );
    dom_var.push_back( torch::tensor( {8.0, 9.0} ) );
    //
    // options
    options_t options;
    //
    // all_var
    std::vector<Tensor> all_par;
    std::vector<Tensor> all_var = adfn.forward_var(
        all_par, dom_var, options
    );
    //
    EXPECT_EQ( all_var.size(), 4 );
    //
    Tensor equal = all_var.at(0) == torch::tensor( {6.0, 7.0} );
    EXPECT_TRUE( torch::all(equal).item<bool>() );
    //
    equal = all_var.at(1) == torch::tensor( {8.0, 9.0} );
    EXPECT_TRUE( torch::all(equal).item<bool>() );
    //
    equal = all_var.at(2) == torch::tensor( {48.0 , 63.0} );
    EXPECT_TRUE( torch::all(equal).item<bool>() );
    //
    equal = all_var.at(3) == torch::tensor( {47.0 , 62.0} );
    EXPECT_TRUE( torch::all(equal).item<bool>() );
}
// END_CPP
