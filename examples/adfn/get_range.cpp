// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/adten.hpp>
//
TEST(examples_adfn, get_range)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::options_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // options
    options_t options;
    //
    // p
    vector<Tensor> p;
    p.push_back( torch::tensor( {2.0, 3.0} ) );
    //
    // x
    vector<Tensor> x;
    x.push_back( torch::tensor( {4.0, 5.0} ) );
    //
    // ap, ax
    auto [ax, ap] = adten_t::start_recording(x, p);
    //
    // acon
    // create a constant after start_recording so can use it in the recording
    adten_t acon( torch::tensor( {6} ) );
    //
    // create a parameter and variable that are not used
    ap[0] - acon;
    ax[0] / acon;
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( acon );            // a constant
    ay.push_back( ap[0] + acon );    // a parameter
    ay.push_back( ax[0] * acon );    // a variable
    //
    // y = f(p, x)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // p
    p.resize(0);
    p.push_back( torch::tensor( {7.0, 8.0} ) );
    //
    // x
    x.resize(0);
    x.push_back( torch::tensor( {9.0, 10.0} ) );
    //
    // y
    vector<Tensor> par_all = f.forward_par(p);
    vector<Tensor> var_all = f.forward_var(x, par_all);
    vector<Tensor> y       = f.get_range(par_all, var_all);
    //
    EXPECT_EQ( y.size(), 3 );
    //
    bool equal = y[0].equal( torch::tensor({6.0}) );
    EXPECT_TRUE( equal );
    //
    equal = y[1].equal( torch::tensor({13.0, 14.0}) );
    EXPECT_TRUE( equal );
    //
    equal = y[2].equal( torch::tensor({54.0 , 60.0}) );
    EXPECT_TRUE( equal );
}
// END_CPP
