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
TEST(tests, adten_matmul_op)  {
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
    double x00 = 1.0, x01 = 2.0, x10 = 3.0, x11 = 4.0;
    //
    // x
    vector<Tensor> x;
    x.push_back( torch::tensor( {
        {x00, x01},
        {x10, x11}
    } ) );
    //
    // ap, ax
    auto [ap, ax] = adten_t::start_recording(p, x);
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( ax[0].matmul( ax[0] ) );
    //
    // y = f(p)
    adfn_t f = adten_t::stop_recording(ay);
    //
    // par_all, var_all
    vector<Tensor> par_all = f.forward_par(p, options);
    vector<Tensor> var_all = f.forward_var(par_all, x, options);
    //
    // y
    vector<Tensor> y = f.get_range(par_all, var_all);
    //
    EXPECT_EQ( y.size(), ay.size() );
    //
    Tensor check = torch::tensor( {
        {x00*x00 + x01*x10, x00*x01 + x01*x11 },
        {x10*x00 + x11*x10, x10*x01 + x11*x11 }
    } );
    bool equal = y[0].equal( check );
    EXPECT_TRUE( equal );
    //
    // dx
    vector<Tensor> dx;
    dx.push_back( torch::tensor( {
        {0.0, 0.0},
        {0.0, 1.0}
    } ) );
    //
    // dy
    vector<Tensor> dy = f.forward_der(par_all, var_all, dx, options);
    check = torch::tensor( {
            {0.0,       x01},
            {x10, 2.0 * x11}
    } );
    equal = dy[0].equal( check );
    EXPECT_TRUE( equal );
    // dy, dx
    dy[0] = torch::tensor( {
        {1.0, 0.0}, {0.0, 0.0}
    } );
    dx    = f.reverse_der(par_all, var_all, dy, options);
    check = torch::tensor( {
        {2.0 * x00, x10},
        {x01,       0.0}
    } );
    equal = dx[0].equal( check );
    EXPECT_TRUE( equal );
}
// END_CPP
