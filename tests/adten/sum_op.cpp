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
TEST(tests, adten_sum_op)  {
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
    p.push_back( torch::tensor( {2.0, 3.0} ) );
    //
    // x
    vector<Tensor> x;
    x.push_back( torch::tensor( { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} } ) );
    //
    // ap, ax
    auto [ax, ap] = adten_t::start_recording(x, p);
    //
    // dim
    vector<int64_t> dim_array( { 1 } );
    c10::IntArrayRef dim(dim_array);
    //
    // ay
    vector<adten_t> ay;
    ay.push_back(  ap[0].sum() );
    ay.push_back( ax[0].sum(dim) );
    //
    // y = f(p)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // par_all, var_all
    vector<Tensor> par_all = f.forward_par(p);
    vector<Tensor> var_all = f.forward_var(x, par_all);
    //
    // y
    vector<Tensor> y = f.get_range(var_all, par_all);
    //
    EXPECT_EQ( y.size(), ay.size() );
    //
    bool equal = y[0].equal( p[0].sum() );
    EXPECT_TRUE( equal );
    //
    equal = y[1].equal( x[0].sum(dim) );
    EXPECT_TRUE( equal );
    //
    // dx
    vector<Tensor> dx;
    dx.push_back(torch::tensor( { {7.0, 8.0, 9.0}, {10.0, 11.0, 12.0} } ));
    //
    // dy
    vector<Tensor> dy = f.forward_der(dx, var_all, par_all);
    //
    EXPECT_EQ( dy.size(), y.size() );
    //
    equal = dy[0].equal( torch::tensor( 0.0 ) );
    EXPECT_TRUE( equal );
    //
    equal = dy[1].equal( dx[0].sum(dim) );
    EXPECT_TRUE( equal );
    //
    // dy, dx
    dy[0] = torch::tensor( 2.0 );
    dy[1] = torch::tensor( {3.0, 4.0} );
    dx    = f.reverse_der(par_all, var_all, dy, options);
    //
    EXPECT_EQ( x.size(), dx.size() );
    //
    equal = dx[0].equal(torch::tensor( { {3.0, 3.0, 3.0}, {4.0, 4.0, 4.0} } ));
    EXPECT_TRUE( equal );
}
// END_CPP
