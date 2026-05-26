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
TEST(examples, range)  {
    using ad_tensor::ad_t;
    using ad_tensor::adfn_t;
    using ad_tensor::options_t;
    using at::Tensor;
    using std::vector;
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
        std::move(dom_par), std::move(dom_var)
    );
    //
    // acon
    // create a constant after start_recording so can use it in the recording
    ad_t acon( torch::tensor( {6} ) );
    //
    // create a parameter and variable that are not used
    adom_par.at(0) - acon;
    adom_var.at(0) / acon;
    //
    // arange
    vector<ad_t> arange;
    arange.push_back( acon );                     // a constant
    arange.push_back( adom_par.at(0) + acon );    // a parameter
    arange.push_back( adom_var.at(0) * acon );    // a variable
    //
    // adfn
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
    // rng_vec
    std::vector<Tensor> all_par = adfn.forward_par(
        std::move(dom_par), options
    );
    std::vector<Tensor> all_var = adfn.forward_var(
        all_par, std::move(dom_var), options
    );
    std::vector<Tensor> rng_vec = adfn.range(all_par, all_var);
    //
    EXPECT_EQ( rng_vec.size(), 3 );
    //
    Tensor equal = rng_vec.at(0) == torch::tensor( {6.0} );
    EXPECT_TRUE( torch::all(equal).item<bool>() );
    //
    equal = rng_vec.at(1) == torch::tensor( {13.0, 14.0} );
    EXPECT_TRUE( torch::all(equal).item<bool>() );
    //
    equal = rng_vec.at(2) == torch::tensor( {54.0 , 60.0} );
    EXPECT_TRUE( torch::all(equal).item<bool>() );
}
// END_CPP
