// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <ad_tensor/dev/optimize.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/chkpnt.hpp>
#include <ad_tensor/dev/to_string.hpp>
//
TEST(tests_optimize, rng_depend_simple) {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    using ad_tensor::dev::to_string;
    //
    // c_used
    Tensor c_used = torch::tensor( {1.0, 2.0} );
    //
    // p
    vector<Tensor> p;
    p.push_back( torch::tensor( {0.0, 0.0} ) );
    //
    // v
    vector<Tensor> v;
    v.push_back( torch::tensor( {0.0, 0.0} ) );
    v.push_back( torch::tensor( {0.0, 0.0} ) );
    //
    // av, ap
    auto [av, ap] = adten_t::start_recording(v, p);  // p[0], v[0], v[1]
    //
    // ac_used, ac_not_used
    // c[0] is always the constant empty_at_ten()
    adten_t ac_used     = adten_t( c_used );                        // c[1]
    adten_t ac_not_used = adten_t( torch::tensor( {3.0, 4.0} ) );   // c[2]
    //
    // ap_used, ap_not_used
    adten_t ap_used     = ap[0] + ac_used;      // p[1]
    adten_t ap_not_used = ap[0] + ac_not_used;  // p[2]
    //
    // av_used, av_not_used
    adten_t av_used     = av[1] + ap_used;      // v[2]
    adten_t av_not_used = av[1] + ap_not_used;  // v[3]
    //
    // ar
    vector<adten_t> ar;
    ar.push_back( av_used );
    //
    // r = f(v, p)
    adfn_t f = adten_t::stop_recording(ar, "f");
    //
    // p, v
    p[0] = torch::tensor( {5.0, 6.0} );
    v[0] = torch::tensor( {7.0, 8.0} );
    v[1] = torch::tensor( {9.0, 10.0} );
    //
    // r
    vector<Tensor> p_all = f.forward_par(p);
    vector<Tensor> v_all = f.forward_var(v, p_all);
    vector<Tensor> r     = f.get_range(v_all, p_all);
    //
    EXPECT_EQ( r.size(), 1 );
    //
    bool equal = r[0].equal( v[1] + p[0] + c_used );
    EXPECT_TRUE( equal );
    //
    auto [depend_con, depend_par, depend_var] = ad_tensor::dev::rng_depend(&f);
    //
    // depend_con
    EXPECT_EQ( depend_con[0], false);
    EXPECT_EQ( depend_con[1], true);
    EXPECT_EQ( depend_con[2], false);
    //
    // depend_par
    EXPECT_EQ( depend_par[0], true);
    EXPECT_EQ( depend_par[1], true);
    EXPECT_EQ( depend_par[2], false);
    //
    // depend_var
    EXPECT_EQ( depend_var[0], false);
    EXPECT_EQ( depend_var[1], true);
    EXPECT_EQ( depend_var[2], true);
    EXPECT_EQ( depend_var[3], false);
}
//
TEST(tests_optimize, rng_depend_chkpnt) {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    using ad_tensor::dev::to_string;
    //
    // x, ax
    vector<Tensor> x;
    x.push_back( torch::tensor( {0.0, 0.0} ) );
    x.push_back( torch::tensor( {0.0, 0.0} ) );
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // y = f(x) = [ x_1, x_0 ] ^T
    vector<adten_t> ay;
    ay.push_back( ax[1] );
    ay.push_back( ax[0] );
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // atom_id_f
    size_t atom_id_f = ad_tensor::make_chkpnt(f);
    //
    // c_used
    Tensor c_used = torch::tensor( {1.0, 2.0} );
    //
    // p
    vector<Tensor> p;
    p.push_back( torch::tensor( {0.0, 0.0} ) );
    //
    // v, av, ap
    vector<Tensor> v;
    v.push_back( torch::tensor( {0.0, 0.0} ) );
    v.push_back( torch::tensor( {0.0, 0.0} ) );
    auto [av, ap] = adten_t::start_recording(v, p);  // p[0], v[0], v[1]
    //
    // ac_used, ac_not_used
    // c[0] is always the constant empty_at_ten()
    adten_t ac_used     = adten_t( c_used );                        // c[1]
    adten_t ac_not_used = adten_t( torch::tensor( {3.0, 4.0} ) );   // c[2]
    //
    // ap_used, ap_not_used
    adten_t ap_used     = ap[0] + ac_used;      // p[1]
    adten_t ap_not_used = ap[0] + ac_not_used;  // p[2]
    //
    // av_used, av_not_used
    adten_t av_used     = av[0] + ap_used;      // v[2]
    adten_t av_not_used = av[1] + ap_not_used;  // v[3]
    //
    // ay
    ax[0] = av_used;
    ax[1] = av_not_used;
    ay    = ad_tensor::call_chkpnt(atom_id_f, ax); // v[4], v[5]
    //
    // ar
    vector<adten_t> ar;
    ar.push_back( ay[1] ); // v[5]
    //
    // r = g(v, p)
    adfn_t g = adten_t::stop_recording(ar, "g");
    //
    // p, v
    p[0] = torch::tensor( {5.0, 6.0} );
    v[0] = torch::tensor( {7.0, 8.0} );
    v[1] = torch::tensor( {9.0, 10.0} );
    //
    // r
    vector<Tensor> p_all = g.forward_par(p);
    vector<Tensor> v_all = g.forward_var(v, p_all);
    vector<Tensor> r     = g.get_range(v_all, p_all);
    //
    EXPECT_EQ( r.size(), 1 );
    //
    bool equal = r[0].equal( v[0] + p[0] + c_used );
    EXPECT_TRUE( equal );
    //
    auto [depend_con, depend_par, depend_var] = ad_tensor::dev::rng_depend(&g);
    //
    // depend_var
    EXPECT_EQ( depend_var.size(), 6);
    EXPECT_EQ( depend_var[5], true);
    EXPECT_EQ( depend_var[4], false);
    EXPECT_EQ( depend_var[3], false);
    EXPECT_EQ( depend_var[2], true);
    EXPECT_EQ( depend_var[1], false);
    EXPECT_EQ( depend_var[0], true);
    //
    // depend_par
    EXPECT_EQ( depend_par.size(), 3);
    EXPECT_EQ( depend_par[2], false);
    EXPECT_EQ( depend_par[1], true);
    EXPECT_EQ( depend_par[0], true);
    //
    // depend_con
    EXPECT_EQ( depend_con.size(), 3);
    EXPECT_EQ( depend_con[2], false);
    EXPECT_EQ( depend_con[1], true);
    EXPECT_EQ( depend_con[0], false);
}
