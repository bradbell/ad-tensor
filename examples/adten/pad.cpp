// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
#include <torch/torch.h>
//
TEST(examples_adten, pad)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    namespace functional = torch::nn::functional;
    //
    // p
    vector<at::Tensor> p;
    p.push_back( torch::tensor( {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    } ) );
    //
    // v
    vector<at::Tensor> v;
    v.push_back( torch::tensor( {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    } ) );
    //
    auto options_p = functional::PadFuncOptions( {1, 2} );
    options_p.mode(torch::kConstant);
    options_p.value(7.0);
    //
    auto options_v = functional::PadFuncOptions( {0, 0, 2, 1} );
    options_v.mode(torch::kConstant);
    options_v.value(7.0);
    //
    // ap, av
    adten_t ap = adten_t( p[0] );
    adten_t av = adten_t( v[0] );
    //
    // ap_pad, av_pad
    adten_t ap_pad = ad_tensor::pad(ap, options_p);
    adten_t av_pad = ad_tensor::pad(av, options_v);
    //
    // check ap
    at::Tensor check_p = torch::tensor( {
        {7.0, 1.0, 2.0, 3.0, 7.0, 7.0},
        {7.0, 4.0, 5.0, 6.0, 7.0, 7.0}
    } );
    EXPECT_TRUE( ap_pad.at_ten().equal( check_p ) );
    //
    // check av
    at::Tensor check_v = torch::tensor( {
        {7.0, 7.0},
        {7.0, 7.0},
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0},
        {7.0, 7.0}
    } );
    EXPECT_TRUE( av_pad.at_ten().equal( check_v ) );

}
// END_CPP
