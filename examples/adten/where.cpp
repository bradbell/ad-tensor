// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <torch/torch.h>
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adten, where)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // x
    vector<Tensor> x;
    x.push_back( torch::tensor( {-4.0, 4.0} ) );
    /* TODO: get this test to work
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    // ax_pos
    adten_t ax_pos = ax[0] > 0;
    //
    // ax_sqrt
    adten_t ax_sqrt = ax[0].sqrt();
    //
    // ax_sq
    adten_t ax_sq   = ax[0] * ax[0];
    //
    // ay
    vector<adten_t> ay;
    y.push_back( adten::where(ax_pos, ax_sqrt, ax_sq);
    //
    // y = f(x)
    adfn_t y = adten_t::stop_recording(ay, "f");
    //
    // v_all
    vector<Tensor> v_all = f.forward_var(x);
    //
    // y
    vector<Tensor> y = f.get_range(v_all);
    //
    bool equal = y.equal( torch::tensor( { 16.0, 2.0 } )
    EXPECT_TRUE(equal);
    */
}
// END_CPP
