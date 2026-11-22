// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <torch/torch.h>
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(tests_adten, bucketize)  {
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
    // p
    vector<at::Tensor> p = {boundaries};
    //
    // x
    vector<at::Tensor> x = {input};
    //
    // ax, ap
    auto [ax, ap] = adten_t::start_recording(x, p);
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( bucketize(ax[0], ap[0]) );
    //
    // y = f(x)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // p_all, v_all
    vector<at::Tensor> p_all = f.forward_par(p);
    vector<at::Tensor> v_all = f.forward_var(x, p_all);
    //
    // y
    vector<at::Tensor> y = f.get_range(v_all);
    EXPECT_TRUE( y[0].equal( torch::tensor({
        {1, 3, 4},
        {0, 3, 5}
    }) ) );
    //
    // dx, dy
    vector<at::Tensor> dx = { torch::tensor({
        {1, 2, 3},
        {4, 5, 6}
    }) };
    vector<at::Tensor> dy = f.forward_der(dx, v_all, p_all);
    EXPECT_TRUE( ad_tensor::not_defined( dy[0] ) );
    //
    // py, px
    vector<at::Tensor> py = { torch::tensor({
        {1, 2, 3},
        {4, 5, 6}
    }) };
    vector<at::Tensor> px = f.reverse_der(py, v_all, p_all);
    EXPECT_TRUE( ad_tensor::not_defined( px[0] ) );
}
// END_CPP
