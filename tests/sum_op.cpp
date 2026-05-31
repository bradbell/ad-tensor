// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <ad_tensor/adfn.hpp>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad.hpp>
#include <ad_tensor/dev/to_string.hpp>
//
TEST(tests, sum_op)  {
    using ad_tensor::ad_t;
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
    x.push_back( torch::tensor( {4.0, 5.0, 6.0} ) );
    //
    // ap, ax
    auto [ap, ax] = ad_t::start_recording(p, x);
    //
    // dim
    vector<long> dim_array( { 0 } );
    c10::ArrayRef<long> dim(dim_array);
    //
    // ay
    vector<ad_t> ay;
    ay.push_back(  ap[0].sum() );
    ay.push_back( ax[0].sum(dim) );
    //
    // y = f(p)
    adfn_t f = ad_t::stop_recording(ay);
    //
    // all_par, all_var
    vector<Tensor> all_par = f.forward_par(p, options);
    vector<Tensor> all_var = f.forward_var(all_par, x, options);
    //
    // y
    vector<Tensor> y = f.get_range(all_par, all_var);
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
    dx.push_back( torch::tensor( {7.0, 8.0} ) );
    //
    // dy
    vector<Tensor> dy = f.forward_der(all_par, all_var, dx, options);
    //
    EXPECT_EQ( dy.size(), y.size() );
    //
    std::cout << "dy[0] = " << ad_tensor::dev::to_string(dy[0]) << "\n";
    equal = dy[0].equal( torch::tensor( 0.0 ) );
    EXPECT_TRUE( equal );
    //
    std::cout << "dy[1] = " << ad_tensor::dev::to_string(dy[1]) << "\n";
    equal = dy[1].equal( dx[0].sum(dim) );
    EXPECT_TRUE( equal );
}
// END_CPP
