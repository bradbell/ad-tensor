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
TEST(tests, adfn_reverse_der)  {
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
    vector<Tensor> x;
    x.push_back( torch::tensor( {4.0, 5.0} ) );
    x.push_back( torch::tensor( {2.0, 3.0} ) );
    //
    // ax
    vector<Tensor> p;
    auto [ap, ax] = ad_t::start_recording(p, x);
    //
    // ay
    vector<ad_t> ay;
    ay.push_back(  ax[0] + ax[1] );
    ay.push_back(  ax[0] - ax[1] );
    ay.push_back(  ax[0] * ax[1] );
    //
    // y = f(x)
    adfn_t f = ad_t::stop_recording(ay);
    //
    // all_var
    vector<Tensor> all_par;
    vector<Tensor> all_var = f.forward_var(all_par, x, options);
    //
    // y
    vector<Tensor> y = f.get_range(all_par, all_var);
    //
    EXPECT_EQ( y.size(), ay.size() );
    //
    bool equal = y[0].equal( x[0] + x[1] );
    EXPECT_TRUE( equal );
    //
    equal = y[1].equal( x[0] - x[1] );
    EXPECT_TRUE( equal );
    //
    equal = y[2].equal( x[0] * x[1] );
    EXPECT_TRUE( equal );
    //
    // dy
    vector<Tensor> dy;
    dy.push_back( torch::tensor( {1.0, 2.0} ) );
    dy.push_back( torch::tensor( {3.0, 4.0} ) );
    dy.push_back( torch::tensor( {5.0, 6.0} ) );
    //
    // dx
    vector<Tensor> dx = f.reverse_der(all_par, all_var, dy, options);
    //
    EXPECT_EQ( dx.size(), x.size() );
    //
    equal = dx[0].equal( dy[0] + dy[1] + dy[2] * x[1] );
    EXPECT_TRUE( equal );
    //
    equal = dx[1].equal( dy[0] - dy[1] + dy[2] * x[0] );
    EXPECT_TRUE( equal );
}
// END_CPP
