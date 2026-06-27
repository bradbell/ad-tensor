// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <ad_tensor/adfn.hpp>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
//
TEST(examples_adfn, forward_der)  {
    using ad_tensor::adten_t;
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
    x.push_back( torch::tensor( {4.0, 8.0} ) );
    x.push_back( torch::tensor( {2.0} ) );
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // ay
    // We use y for the range space
    vector<adten_t> ay;
    ay.push_back(  ax[0] + ax[1] );
    ay.push_back(  ax[0] - ax[1] );
    ay.push_back(  ax[0] * ax[1] );
    ay.push_back(  ax[0] / ax[1] );
    //
    // y = f(x)
    // We use f for the adfn_t object.
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // var_all
    vector<Tensor> par_all;
    vector<Tensor> var_all = f.forward_var(par_all, x, options);
    //
    // y
    vector<Tensor> y = f.get_range(par_all, var_all);
    //
    EXPECT_EQ( y.size(), ay.size() );
    //
    bool equal = y[0].equal( torch::tensor({6.0, 10.0}) );
    EXPECT_TRUE( equal );
    //
    // dx
    vector<Tensor> dx;
    dx.push_back( torch::tensor( {1.0, 1.0} ) );
    dx.push_back( torch::tensor( {1.0} ) );
    //
    // dy
    vector<Tensor> dy = f.forward_der(
        par_all, var_all, dx, options
    );
    //
    EXPECT_EQ( dy.size(), y.size() );
    //
    equal = dy[0].equal( torch::tensor({2.0, 2.0}) );
    EXPECT_TRUE( equal );
    //
    equal = dy[1].equal( torch::tensor({0.0, 0.0}) );
    EXPECT_TRUE( equal );
    //
    equal = dy[2].equal( torch::tensor({2.0 + 4.0, 2.0 + 8.0}) );
    EXPECT_TRUE( equal );
    //
    equal = dy[3].equal( torch::tensor({ (2.0-4.0)/4.0, (2.0-8.0)/4.0 }) );
    EXPECT_TRUE( equal );
}
// END_CPP
