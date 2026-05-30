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
TEST(examples, adfn_range)  {
    using ad_tensor::ad_t;
    using ad_tensor::adfn_t;
    using ad_tensor::options_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // dom_par
    vector<Tensor> dom_par;
    dom_par.push_back( torch::tensor( {2.0, 3.0} ) );
    //
    // dom_var
    vector<Tensor> dom_var;
    dom_var.push_back( torch::tensor( {4.0, 5.0} ) );
    //
    // adom_par, adom_var
    auto [adom_par, adom_var] = ad_t::start_recording(
        dom_par, dom_var
    );
    //
    // acon
    // create a constant after start_recording so can use it in the recording
    ad_t acon( torch::tensor( {6} ) );
    //
    // create a parameter and variable that are not used
    adom_par[0] - acon;
    adom_var[0] / acon;
    //
    // arange
    vector<ad_t> arange;
    arange.push_back( acon );                     // a constant
    arange.push_back( adom_par[0] + acon );    // a parameter
    arange.push_back( adom_var[0] * acon );    // a variable
    //
    // range = adfn(dom_par, dom_var)
    adfn_t adfn = ad_t::stop_recording(arange);
    //
    // dom_par
    // dom_par is valid but unspecified before assignment
    dom_par = vector<Tensor>();
    dom_par.push_back( torch::tensor( {7.0, 8.0} ) );
    //
    // dom_var
    // dom_var is valid but unspecified before assignment
    dom_var = vector<Tensor>();
    dom_var.push_back( torch::tensor( {9.0, 10.0} ) );
    //
    // options
    options_t options;
    //
    // range
    ad_tensor::vector<Tensor> all_par = adfn.forward_par(
        dom_par, options
    );
    ad_tensor::vector<Tensor> all_var = adfn.forward_var(
        all_par, dom_var, options
    );
    ad_tensor::vector<Tensor> range = adfn.get_range(all_par, all_var);
    //
    EXPECT_EQ( range.size(), 3 );
    //
    bool equal = range[0].equal( torch::tensor({6.0}) );
    EXPECT_TRUE( equal );
    //
    equal = range[1].equal( torch::tensor({13.0, 14.0}) );
    EXPECT_TRUE( equal );
    //
    equal = range[2].equal( torch::tensor({54.0 , 60.0}) );
    EXPECT_TRUE( equal );
}
// END_CPP
