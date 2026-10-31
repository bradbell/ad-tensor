// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
#include <torch/torch.h>
//
TEST(examples_adten, split)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // x
    vector<at::Tensor> x = { torch::tensor( {
        {1, 2, 3,  4,  5,  6},
        {7, 8, 9, 10, 11, 12},
    } ) };
    //
    // ap, ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // split_size, dim
    vector<int64_t> split_size( {1, 2, 3} );
    int64_t         dim = 1;
    //
    // ay
    vector<adten_t> ay = ax[0].split(split_size, dim);
    //
    // y = f(x, p)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // par_all, var_all
    vector<at::Tensor> var_all = f.forward_var(x);
    //
    // y
    vector<at::Tensor> y = f.get_range(var_all);
    //
    EXPECT_EQ( y.size(), split_size.size() );
    //
    bool equal = y[0].equal( torch::tensor( {
        {1},
        {7},
    } ) );
    EXPECT_TRUE( equal );
    //
    equal = y[1].equal( torch::tensor( {
        {2, 3},
        {8, 9},
    } ) );
    EXPECT_TRUE( equal );
    //
    equal = y[2].equal( torch::tensor( {
        { 4,  5,  6},
        {10, 11, 12},
    } ) );
    EXPECT_TRUE( equal );
}
// END_CPP
