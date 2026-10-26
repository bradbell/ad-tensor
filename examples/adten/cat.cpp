// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <torch/torch.h>
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adten, cat)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // x
    vector<at::Tensor> x;
    x.push_back( torch::tensor( {
        {0.0, 1.0, 2.0},
        {3.0, 4.0, 5.0}
    } ) );
    x.push_back( torch::tensor( {
        {6.0,  7.0,  8.0},
        {9.0, 10.0, 11.0}
    } ) );
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    // ay
    vector<adten_t> ay;
    ay.push_back( ad_tensor::cat({ax[0], ax[1]}, 0) );
    ay.push_back( ad_tensor::cat({ax[0], ax[1]}, 1) );
    //
    // y = f(x)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // v_all
    vector<at::Tensor> v_all = f.forward_var(x);
    //
    // check
    at::Tensor check;
    //
    // y
    vector<at::Tensor> y = f.get_range(v_all);
    check = torch::tensor( {
        {0.0,  1.0,  2.0},
        {3.0,  4.0,  5.0},
        {6.0,  7.0,  8.0},
        {9.0, 10.0, 11.0}
    } );
    EXPECT_TRUE( y[0].equal(check) );
    check = torch::tensor( {
        {0.0, 1.0, 2.0, 6.0,  7.0,  8.0},
        {3.0, 4.0, 5.0, 9.0, 10.0, 11.0}
    } );
    EXPECT_TRUE( y[1].equal(check) );
}
// END_CPP
