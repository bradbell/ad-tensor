// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adten, minus)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // p
    vector<at::Tensor> p;
    p.push_back( torch::tensor( {2.0, 3.0} ) );
    //
    // x
    vector<at::Tensor> x;
    x.push_back( torch::tensor( { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} } ) );
    //
    // ap, ax
    auto [ax, ap] = adten_t::start_recording(x, p);
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( - ap[0] );
    ay.push_back( - ax[0] );
    //
    // y = f(x, p)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // par_all, var_all
    vector<at::Tensor> par_all = f.forward_par(p);
    vector<at::Tensor> var_all = f.forward_var(x, par_all);
    //
    // y
    vector<at::Tensor> y = f.get_range(var_all, par_all);
    //
    EXPECT_EQ( y.size(), ay.size() );
    //
    bool equal = y[0].equal( - p[0] );
    EXPECT_TRUE( equal );
    //
    equal = y[1].equal( - x[0] );
    EXPECT_TRUE( equal );
}
// END_CPP
