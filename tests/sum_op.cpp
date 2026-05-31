// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <ad_tensor/adfn.hpp>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad.hpp>
//
TEST(tests, sum_op)  {
    using ad_tensor::ad_t;
    using ad_tensor::adfn_t;
    using ad_tensor::options_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // x
    vector<Tensor> p;
    p.push_back( torch::tensor( {2.0, 3.0} ) );
    //
    // ap
    vector<Tensor> x;
    auto [ap, ax] = ad_t::start_recording(p, x);
    //
    // ay
    vector<ad_t> ay;
    ay.push_back(  ap[0].sum() );
    //
    // y = f(p)
    adfn_t f = ad_t::stop_recording(ay);
    //
    // options
    options_t options;
    //
    // all_par
    vector<Tensor> all_par = f.forward_par(p, options);
    //
    // y
    vector<Tensor> all_var;
    vector<Tensor> y = f.get_range(all_par, all_var);
    //
    EXPECT_EQ( y.size(), ay.size() );
    //
    bool equal = y[0].equal( p[0].sum() );
    EXPECT_TRUE( equal );
}
// END_CPP
