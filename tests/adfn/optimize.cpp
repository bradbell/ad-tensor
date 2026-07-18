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
    // v
    // We use v for the domain variables
    vector<Tensor> v;
    v.push_back( torch::tensor( {1.0, 1.0} ) );
    v.push_back( torch::tensor( {1.0, 1.0} ) );
    v.push_back( torch::tensor( {1.0, 1.0} ) );
    //
    // av
    vector<adten_t> av = adten_t::start_recording(v);
    //
    // ar
    vector<adten_t> ar;
    ar.push_back( av[0] );
    ar.push_back( av[1] );
    ar.push_back( av[2] );
    //
    // r = f(v)
    adfn_t f           = adten_t::stop_recording(ar, "f");
    size_t f_chkpnt_id = ad_tensor::make_chkpnt(f, v);
    //
    // p
    vector<Tensor> p;
    p.push_back( torch::tensor( {0.0, 0.0} ) );
    //
    // av, ap
    vector<adten_t> ap;
    std::tie(av, ap) = adten_t::start_recording(v, p);
    //
    // z = g(v, p)
    vector<adten_t> ax = { ap[0], av[0] + av[1], av[1] + av[2] };
    vector<adten_t> ay = ad_tensor::call_chkpnt(f_chkpnt_id, ax);
    vector<adten_t> az = { ay[0], ay[1] };
    adfn_t          g  = adten_t::stop_recording(az, "g");
    g.set_trace(true);
    //
    // z
    vector<Tensor> p_all = g.forward_par(p);
    vector<Tensor> v_all = g.forward_var(v, p_all);
    vector<Tensor> z     = g.get_range(v_all, p_all);
    //
    bool equal = z[0].equal( p[0] );
    EXPECT_TRUE( equal );
    //
    equal = z[1].equal( v[0] + v[1] );
    EXPECT_TRUE( equal );
    //
    // z = g(v, p)
    /* TODO: get the code below to work
    g                  = adfn_t::optimize(g);
    //
    // z
    p_all = g.forward_par(p);
    v_all = g.forward_var(v, p_all);
    z     = g.get_range(v_all, p_all);
    //
    equal = z[0].equal( p[0] );
    EXPECT_TRUE( equal );
    //
    equal = z[1].equal( v[0] + v[1] );
    EXPECT_TRUE( equal );
    */
}
// END_CPP
