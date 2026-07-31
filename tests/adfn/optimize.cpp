// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
// BEGIN_ADFN_OPTIMIZE_CON
TEST(tests_adfn, optimize_con) {
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
    // not_used_con, used_con, repeat_con
    adten_t not_used_con = adten_t( torch::tensor(2.0) );
    adten_t used_con     = adten_t( torch::tensor(3.0) );
    adten_t repeat_con   = adten_t( torch::tensor(3.0) );
    //
    vector<adten_t> ar;
    ar.push_back( used_con );
    ar.push_back( repeat_con );
    //
    // r = f(v, p)
    adfn_t f = adten_t::stop_recording(ar, "f");
    EXPECT_EQ( f.n_con(), 4 );
    //
    // g = f
    adfn_t g = f.optimize();
    EXPECT_EQ( g.n_con(), 2 );
    //
    vector<Tensor> p_all = g.forward_par(p);
    vector<Tensor> v_all = g.forward_var(v, p_all);
    vector<Tensor> range = g.get_range(v_all, p_all);
    //
    bool equal = range[0].equal( used_con.at_ten() );
    EXPECT_TRUE( equal );
    //
    equal = range[1].equal( repeat_con.at_ten() );
    EXPECT_TRUE( equal );
}
// END_ADFN_OPTIMIZE_CON
//
// BEGIN_OPTIMIZE_AGRAPH
TEST(tests_adfn, optimize_agraph) {
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
    // not_used_par, used_par, repeat_par
    adten_t not_used_par = ap[0] + ap[0];
    adten_t used_par     = ap[0] * ap[1];
    adten_t repeat_par   = ap[0] * ap[1];
    //
    // not_used_var, used_var, repeat_var
    adten_t not_used_var = av[0] + ap[0];
    adten_t used_var     = av[0] * ap[1];
    adten_t repeat_var   = av[0] * ap[1];
    //
    vector<adten_t> ar;
    ar.push_back( used_par );
    ar.push_back( repeat_par );
    ar.push_back( used_var );
    ar.push_back( repeat_var );
    //
    //
    // r = f(v, p)
    adfn_t f = adten_t::stop_recording(ar, "f");
    EXPECT_EQ( f.n_par(), 5 );
    EXPECT_EQ( f.n_var(), 4 );
    //
    // g = f
    adfn_t g = f.optimize();
    EXPECT_EQ( g.n_par(), 3 );
    EXPECT_EQ( g.n_var(), 2 );
    //
    vector<Tensor> p_all = g.forward_par(p);
    vector<Tensor> v_all = g.forward_var(v, p_all);
    vector<Tensor> range = g.get_range(v_all, p_all);
    //
    bool equal = range[0].equal( used_par.at_ten() );
    EXPECT_TRUE( equal );
    //
    equal = range[1].equal( repeat_par.at_ten() );
    EXPECT_TRUE( equal );
    //
    equal = range[2].equal( used_var.at_ten() );
    EXPECT_TRUE( equal );
    //
    equal = range[3].equal( repeat_var.at_ten() );
    EXPECT_TRUE( equal );
}
// END_OPTIMIZE_AGRAPH
//
// BEGIN_OPTIMIZE_CALL
TEST(tests_adfn, optimize_call)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // v
    // We use v for the domain variables
    vector<Tensor> v;
    v.push_back( torch::tensor( {1.0, 1.0} ) );
    v.push_back( torch::tensor( {2.0, 2.0} ) );
    v.push_back( torch::tensor( {3.0, 3.0} ) );
    //
    // av
    vector<adten_t> av = adten_t::start_recording(v);
    //
    // ar
    vector<adten_t> ar;
    ar.push_back( av[0] );  // v_all[0]
    ar.push_back( av[1] );  // v_all[1]
    ar.push_back( av[2] );  // v_all[2]
    //
    // r = f(v)
    adfn_t f           = adten_t::stop_recording(ar, "f");
    size_t f_chkpnt_id = ad_tensor::make_chkpnt(f, v);
    //
    // p
    vector<Tensor> p;
    p.push_back( torch::tensor( {2.0, 2.0} ) );  // p_all[0]
    //
    // av, ap
    vector<adten_t> ap;
    std::tie(av, ap) = adten_t::start_recording(v, p);
    //
    // z = g(v, p)
    adten_t used_par = ap[0] + ap[0];  // p_all[1]
    adten_t used_var = av[0] * av[1];  // v_all[3]
    adten_t not_used_var = av[1] + av[2];  // v_all[4]
    vector<adten_t> ax = { used_par, used_var, not_used_var };
    //
    // ay = p_all[2], v_all[5], v_all[6] (v_all[6] is not used)
    vector<adten_t> au = ad_tensor::call_chkpnt(f_chkpnt_id, ax);
    //
    // This is a repeated call and so optimization removes it
    // au = p_all[3], v_all[7], v_all[8] (v_all[8] is not used)
    vector<adten_t> ay = ad_tensor::call_chkpnt(f_chkpnt_id, ax);
    //
    used_par = ay[0] + ay[0]; // p_all[4]
    used_var = ay[1] + ay[1]; // v_all[9]
    vector<adten_t> az = { used_par, used_var };
    adfn_t          g  = adten_t::stop_recording(az, "g");
    g.set_trace(true);
    EXPECT_EQ(g.n_con(), 1);
    EXPECT_EQ(g.n_par(), 5);
    EXPECT_EQ(g.n_var(), 10);
    //
    // z
    vector<Tensor> p_all = g.forward_par(p);
    vector<Tensor> v_all = g.forward_var(v, p_all);
    vector<Tensor> z     = g.get_range(v_all, p_all);
    //
    bool equal = z[0].equal( p[0] + p[0] + p[0] + p[0] );
    EXPECT_TRUE( equal );
    //
    equal = z[1].equal( v[0] * v[1] + v[0] * v[1]);
    EXPECT_TRUE( equal );
    //
    // h = g
    adfn_t h  = g.optimize();
    EXPECT_EQ(h.n_con(), 1);
    EXPECT_EQ(h.n_par(), 4);
    EXPECT_EQ(h.n_var(), 6);
    //
    // z
    p_all = h.forward_par(p);
    v_all = h.forward_var(v, p_all);
    z     = h.get_range(v_all, p_all);
    //
    equal = z[0].equal( p[0] + p[0] + p[0] + p[0] );
    EXPECT_TRUE( equal );
    //
    equal = z[1].equal( v[0] * v[1] + v[0] * v[1]);
    EXPECT_TRUE( equal );
    //
    // dv
    vector<Tensor> dv;
    dv.push_back( torch::tensor( {1.0, 2.0} ) );
    dv.push_back( torch::tensor( {3.0, 4.0} ) );
    dv.push_back( torch::tensor( {5.0, 6.0} ) );
    //
    // dz
    vector<Tensor> dz = h.forward_der(dv, v_all, p_all);
    //
    equal = dz[0].equal( torch::empty( {0} ) );
    EXPECT_TRUE(equal);
    //
    equal = dz[1].equal( dv[0] * (v[1] + v[1]) + (v[0] + v[0]) * dv[1] );
    EXPECT_TRUE(equal);
}
// END_OPTIMIZE_CALL
