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
TEST(tests, view_op)  {
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
    //
    // x
    vector<Tensor> x;
    x.push_back( torch::tensor( {1.0, 2.0, 3.0, 4.0} ) );
    //
    // ap, ax
    auto [ap, ax] = ad_t::start_recording(p, x);
    //
    // shape
    vector<long> shape = {2, 2};
    //
    // ay
    vector<ad_t> ay;
    ay.push_back( ax[0].view( shape ) );
    //
    // y = f(p)
    adfn_t f = ad_t::stop_recording(ay);
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
    bool equal = y[0].equal( torch::tensor( { {1.0, 2.0}, {3.0, 4.0} } ) );
    EXPECT_TRUE( equal );
    //
    // dx
    vector<Tensor> dx;
    dx.push_back( torch::tensor( {5.0, 6.0, 7.0, 8.0} ) );
    //
    // dy
    vector<Tensor> dy = f.forward_der(par_all, var_all, dx, options);
    equal = dy[0].equal( torch::tensor( { {5.0, 6.0}, {7.0, 8.0} } ) );
    EXPECT_TRUE( equal );
    //
    // dy, dx
    dy[0] = torch::tensor( { {1.0, 2.0}, {3.0, 4.0} } );
    dx    = f.reverse_der(par_all, var_all, dy, options);
    equal = dx[0].equal( torch::tensor( {1.0, 2.0, 3.0, 4.0} ) );
    EXPECT_TRUE( equal );
}
// END_CPP
