// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <ad_tensor/adfn.hpp>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad.hpp>
//
TEST(examples, adfn_forward_der)  {
    using ad_tensor::ad_t;
    using ad_tensor::adfn_t;
    using ad_tensor::options_t;
    using at::Tensor;
    using std::vector;
    //
    // dom_var
    vector<Tensor> dom_var;
    dom_var.push_back( torch::tensor( {4.0, 8.0} ) );
    dom_var.push_back( torch::tensor( {2.0} ) );
    //
    // adom_var
    vector<Tensor> dom_par;
    auto [adom_par, adom_var] = ad_t::start_recording(
        dom_par, dom_var
    );
    //
    // arange
    vector<ad_t> arange;
    arange.push_back(  adom_var.at(0) + adom_var.at(1) );
    arange.push_back(  adom_var.at(0) - adom_var.at(1) );
    arange.push_back(  adom_var.at(0) * adom_var.at(1) );
    arange.push_back(  adom_var.at(0) / adom_var.at(1) );
    //
    // range = adfn(dom_par, dom_var)
    adfn_t adfn = ad_t::stop_recording(arange);
    //
    // options
    options_t options;
    //
    // all_var
    vector<Tensor> all_par;
    vector<Tensor> all_var = adfn.forward_var(all_par, dom_var, options);
    //
    // range
    vector<Tensor> range = adfn.get_range(all_par, all_var);
    //
    EXPECT_EQ( range.size(), arange.size() );
    //
    Tensor equal = range.at(0) == torch::tensor( {6.0, 10.0} );
    EXPECT_TRUE( torch::all(equal).item<bool>() );
    //
    // dom_der
    vector<Tensor> dom_der;
    dom_der.push_back( torch::tensor( {1.0, 1.0} ) );
    dom_der.push_back( torch::tensor( {1.0} ) );
    //
    // rng_der
    vector<Tensor> rng_der = adfn.forward_der(
        all_par, all_var, dom_der, options
    );
    //
    EXPECT_EQ( rng_der.size(), range.size() );
    //
    equal = rng_der.at(0) == torch::tensor( {2.0, 2.0} );
    EXPECT_TRUE( torch::all(equal).item<bool>() );
    //
    equal = rng_der.at(1) == torch::tensor( {0.0, 0.0} );
    EXPECT_TRUE( torch::all(equal).item<bool>() );
    //
    equal = rng_der.at(2) == torch::tensor( {2.0 + 4.0, 2.0 + 8.0} );
    EXPECT_TRUE( torch::all(equal).item<bool>() );
    //
    equal = rng_der.at(3) == torch::tensor({ (2.0-4.0)/4.0, (2.0-8.0)/4.0 });
    EXPECT_TRUE( torch::all(equal).item<bool>() );
}
// END_CPP
