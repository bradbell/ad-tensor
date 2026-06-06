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
TEST(examples_adfn, reverse_der)  {
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
    // We use x for the domain parameters
    vector<Tensor> x;
    x.push_back( torch::tensor( {4.0, 8.0} ) );
    x.push_back( torch::tensor( {2.0} ) );
    //
    // ax
    vector<Tensor> p;
    auto [ap, ax] = adten_t::start_recording(p, x);
    //
    // ay
    // we use y for the range space
    vector<adten_t> ay;
    ay.push_back(  ax[0] + ax[1] );
    //
    // y = f(x)
    adfn_t f = adten_t::stop_recording(ay);
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
    bool equal = y[0].equal( x[0] + x[1] );
    EXPECT_TRUE( equal );
    //
    // dy
    vector<Tensor> dy;
    dy.push_back( torch::tensor( {1.0, 2.0} ) );
    //
    // dx
    vector<Tensor> dx = f.reverse_der(par_all, var_all, dy, options);
    //
    EXPECT_EQ( dx.size(), x.size() );
    //
    equal = dx[0].equal( dy[0] );
    EXPECT_TRUE( equal );
    //
    equal = dx[1].equal( dy[0].sum() );
    EXPECT_TRUE( equal );
}
// END_CPP
