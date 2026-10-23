// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
#include <torch/torch.h>
//
TEST(examples_adten, slice)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // x
    vector<at::Tensor> x;
    x.push_back( torch::tensor( {
        {1.0,  2.0,  3.0,  4.0},
        {5.0,  6.0,  7.0,  8.0},
        {9.0, 10.0, 11.0, 12.0},
    } ) );
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // ay
    vector<adten_t> ay;
    int64_t dim   = 1;
    int64_t start = 1;
    int64_t end   = 3;
    ay.push_back( ax[0].slice(dim, start, end) );
    //
    // y = f(x)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // var_all
    vector<at::Tensor> var_all = f.forward_var(x);
    //
    // y
    vector<at::Tensor> y = f.get_range(var_all);
    EXPECT_EQ( y.size(), ay.size() );
    //
    at::Tensor check = torch::tensor( {
        {2.0,  3.0},
        {6.0,  7.0},
        {10.0, 11.0}
    } );
    bool equal = y[0].equal( check );
    EXPECT_TRUE( equal );
    //
    // dx
    vector<at::Tensor> dx;
    dx.push_back( torch::tensor( {
        {9.0, 10.0, 11.0, 12.0},
        {5.0,  6.0,  7.0,  8.0},
        {1.0,  2.0,  3.0,  4.0},
    } ) );
    //
    // dy
    vector<at::Tensor> dy = f.forward_der(dx, var_all);
    check = torch::tensor( {
        {10.0, 11.0},
        { 6.0,  7.0},
        { 2.0,  3.0}
    } );
    equal = dy[0].equal( check );
    EXPECT_TRUE( equal );
#if 0
    //
    // dx
    dx    = f.reverse_der(dy, var_all, par_all);
    check = torch::tensor( {
        {5.0, 6.0},
        {7.0, 8.0}
    } );
    equal = dx[0].equal( check );
    EXPECT_TRUE( equal );
#endif
}
// END_CPP
