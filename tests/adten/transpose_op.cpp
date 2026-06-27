// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <ad_tensor/adfn.hpp>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/to_string.hpp>
//
TEST(tests, adten_transpose_op)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::options_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // options
    options_t options;
    //
    // p
    vector<Tensor> p;
    //
    // x
    vector<Tensor> x;
    x.push_back( torch::tensor( {
        {1.0, 2.0},
        {3.0, 4.0}
    } ) );
    //
    // ap, ax
    auto [ax, ap] = adten_t::start_recording(x, p);
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( ax[0].transpose(0, 1) );
    //
    // y = f(p)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // par_all, var_all
    vector<Tensor> par_all = f.forward_par(p);
    vector<Tensor> var_all = f.forward_var(x, par_all);
    //
    // y
    vector<Tensor> y = f.get_range(par_all, var_all);
    //
    EXPECT_EQ( y.size(), ay.size() );
    //
    Tensor check = torch::tensor( {
        {1.0, 3.0},
        {2.0, 4.0}
    } );
    bool equal = y[0].equal( check );
    EXPECT_TRUE( equal );
    //
    // dx
    vector<Tensor> dx;
    dx.push_back( torch::tensor( {
        {5.0, 6.0},
        {7.0, 8.0}
    } ) );
    //
    // dy
    vector<Tensor> dy = f.forward_der(par_all, var_all, dx, options);
    check = torch::tensor( {
        {5.0, 7.0},
        {6.0, 8.0}
    } );
    equal = dy[0].equal( check );
    EXPECT_TRUE( equal );
    //
    // dx
    dx    = f.reverse_der(par_all, var_all, dy, options);
    check = torch::tensor( {
        {5.0, 6.0},
        {7.0, 8.0}
    } );
    equal = dx[0].equal( check );
    EXPECT_TRUE( equal );
}
// END_CPP
