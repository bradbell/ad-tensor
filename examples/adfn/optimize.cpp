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
    // not_used_con, used_con
    adten_t not_used_con = adten_t( torch::tensor(2.0) );
    adten_t used_con     = adten_t( torch::tensor(3.0) );
    //
    // not_used_par, used_par
    adten_t not_used_par = ap[0] + ap[1];
    adten_t used_par     = ap[0] * ap[1];
    //
    // not_used_var, used_var
    adten_t not_used_var = ap[0] + av[0];
    adten_t used_var     = ap[0] * av[0];
    //
    // ar
    vector<adten_t> ar;
    ar.push_back( used_con );
    ar.push_back( used_par );
    ar.push_back( used_var );
    //
    // r = f(v, p)
    adfn_t f = adten_t::stop_recording(ar, "f");
    adfn_t g = f.optimize();
    //
    // n_con
    EXPECT_EQ( f.n_con(), 3 );
    EXPECT_EQ( g.n_con(), 2 );
    //
    // n_par
    EXPECT_EQ( f.n_par(), 4);
    EXPECT_EQ( g.n_par(), 3);
    //
    // n_var
    EXPECT_EQ( f.n_var(), 3);
    EXPECT_EQ( g.n_var(), 2);
    //
    vector<Tensor> p_all = g.forward_par(p);
    vector<Tensor> v_all = g.forward_var(v, p_all);
    vector<Tensor> range = g.get_range(v_all, p_all);
    //
    bool equal = range[0].equal( used_con.at_ten() );
    EXPECT_TRUE( equal );
    //
    equal = range[1].equal( used_par.at_ten() );
    EXPECT_TRUE( equal );
    //
    equal = range[2].equal( used_var.at_ten() );
    EXPECT_TRUE( equal );
}
// END_CPP
