// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
#include <torch/torch.h>
//
TEST(examples_adten, expand)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // x
    vector<at::Tensor> x;
    x.push_back( torch::tensor( {1.0, 2.0, 3.0} ) );
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // shape
    vector<int64_t> shape = {2, -1};
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( ax[0].expand( shape ) );
    //
    // y = f(x)
    adfn_t f = adten_t::stop_recording(ay, "f");
    f.set_trace(true);
    //
    // var_all
    vector<at::Tensor> var_all = f.forward_var(x);
    //
    // y
    vector<at::Tensor> y = f.get_range(var_all);
    //
    EXPECT_EQ( y.size(), ay.size() );
    //
    bool equal = y[0].equal( torch::tensor( {
        {1.0, 2.0, 3.0},
        {1.0, 2.0, 3.0}
    } ) );
    EXPECT_TRUE( equal );
}
// END_CPP
