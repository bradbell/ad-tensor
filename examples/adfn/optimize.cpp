// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adfn, optimize)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // p
    // We use p for the domain parameters
    vector<Tensor> p;
    p.push_back( torch::tensor( {1.0, 1.0} ) );
    p.push_back( torch::tensor( {1.0, 1.0} ) );
    //
    // v
    // We use v for the domain variables
    vector<Tensor> v;
    v.push_back( torch::tensor( {1.0, 1.0} ) );
    //
    // ap, av
    auto [av, ap] = adten_t::start_recording(v, p);
    //
    // not_used_con
    adten_t not_used_con = adten_t( torch::tensor(2.0) );
    adten_t used_con     = adten_t( torch::tensor(3.0) );
    //
    // ar
    vector<adten_t> ar;
    ar.push_back( used_con );
    //
    // r = f(v, p)
    adfn_t f = adten_t::stop_recording(ar, "f");
    adfn_t g = adfn_t::optimize(f);
    //
    /* TODO: get this test to pass
    vector<Tensor> p_all = g.forward_par(p);
    vector<Tensor> v_all = g.forward_var(p);
    vector<Tensor> range = g.get_range(v_all, p_all);
    //
    EXPECT_EQ( range[0].item<float>(), used_con.at_ten().item<float>() );
    */
}
// END_CPP
