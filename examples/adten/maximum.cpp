// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adten, maximum)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // first
    at::Tensor first  = torch::tensor( {1.0, 2.0, 3.0} );
    at::Tensor second = torch::tensor( {4.0, 2.0, 0.0} );
    // x
    vector<at::Tensor> x = {first, second};
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( ax[0].maximum( ax[1] ) );
    //
    // y = f(x)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // v_all
    vector<at::Tensor> v_all = f.forward_var(x);
    //
    // y
    vector<at::Tensor> y = f.get_range(v_all);
    //
    EXPECT_EQ( y.size(), ay.size() );
    //
    at::Tensor check = torch::tensor( {4.0, 2.0, 3.0} );
    EXPECT_TRUE( y[0].equal( check ) );
}
// END_CPP
