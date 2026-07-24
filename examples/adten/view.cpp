// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
#include <torch/torch.h>
//
TEST(examples_adten, view)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // x
    vector<Tensor> x;
    x.push_back( torch::tensor( {1.0, 2.0, 3.0, 4.0} ) );
    //
    // ap, ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // shape
    vector<int64_t> shape = {2, 2};
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( ax[0].view( shape ) );
    //
    // y = f(x)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // var_all
    vector<Tensor> var_all = f.forward_var(x);
    //
    // y
    vector<Tensor> y = f.get_range(var_all);
    //
    EXPECT_EQ( y.size(), ay.size() );
    //
    bool equal = y[0].equal( torch::tensor( { {1.0, 2.0}, {3.0, 4.0} } ) );
    EXPECT_TRUE( equal );
    //
    // dx
    vector<Tensor> dx;
    dx.push_back( torch::tensor( {5.0, 6.0, 7.0, 8.0} ) );
    //
    // dy
    vector<Tensor> dy = f.forward_der(dx, var_all);
    equal = dy[0].equal( torch::tensor( { {5.0, 6.0}, {7.0, 8.0} } ) );
    EXPECT_TRUE( equal );
    //
    // dy, dx
    dy[0] = torch::tensor( { {1.0, 2.0}, {3.0, 4.0} } );
    dx    = f.reverse_der(dy, var_all);
    equal = dx[0].equal( torch::tensor( {1.0, 2.0, 3.0, 4.0} ) );
    EXPECT_TRUE( equal );
}
// END_CPP
