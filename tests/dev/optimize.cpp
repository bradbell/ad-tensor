// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <ad_tensor/dev/optimize.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/to_string.hpp>
//
TEST(tests_optimize, rev_depend) {
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
    adten_t ac_used     = adten_t( c_used );                        // c[0]
    adten_t ac_not_used = adten_t( torch::tensor( {3.0, 4.0} ) );   // c[1]
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
    f.set_trace(true);
    f.print_con();
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
    auto [con_depend, par_depend, var_depend] = ad_tensor::dev::rev_depend(&f);
    //
    // con_depend
    EXPECT_EQ( con_depend[0], true);
    EXPECT_EQ( con_depend[1], false);
    //
    // par_depend
    EXPECT_EQ( par_depend[0], true);
    EXPECT_EQ( par_depend[1], true);
    EXPECT_EQ( par_depend[2], false);
    //
    // var_depend
    EXPECT_EQ( var_depend[0], false);
    EXPECT_EQ( var_depend[1], true);
    EXPECT_EQ( var_depend[2], true);
    EXPECT_EQ( var_depend[3], false);
}
