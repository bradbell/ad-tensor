// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <torch/torch.h>
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adten, bucketize)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // input
    at::Tensor input =  torch::tensor({
        {3, 6, 9},
        {0, 6, 10}
    });
    //
    // boundaries
    at::Tensor boundaries =  torch::tensor({1, 3, 5, 6, 9});
    //
    // x
    vector<at::Tensor> x = {input, boundaries};
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( bucketize(ax[0], ax[1]) );
    //
    // y = f(x)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // v_all
    vector<at::Tensor> v_all = f.forward_var(x);
    //
    // y
    vector<at::Tensor> y = f.get_range(v_all);
    EXPECT_TRUE( y[0].equal( torch::tensor({
        {1, 3, 4},
        {0, 3, 5}
    }) ) );
}
// END_CPP
