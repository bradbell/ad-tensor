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
    // av, ap
    auto [av, ap] = adten_t::start_recording(v, p);
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( ad_tensor::pad(ap[0], options_p) );
    ay.push_back( ad_tensor::pad(av[0], options_v) );
    //
    // f
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // y
    vector<at::Tensor> par_all = f.forward_par(p);
    vector<at::Tensor> var_all = f.forward_var(v, par_all);
    vector<at::Tensor> y = f.get_range(var_all, par_all);
    //
    // check y[0]
    at::Tensor check_p = torch::tensor( {
        {7.0, 1.0, 2.0, 3.0, 7.0, 7.0},
        {7.0, 4.0, 5.0, 6.0, 7.0, 7.0}
    } );
    EXPECT_TRUE( y[0].equal( check_p ) );
    //
    // check y[1]
    at::Tensor check_v = torch::tensor( {
        {7.0, 7.0},
        {7.0, 7.0},
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0},
        {7.0, 7.0}
    } );
    EXPECT_TRUE( y[1].equal( check_v ) );

}
// END_CPP
