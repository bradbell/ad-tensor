// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin chkpnt_ad_reverse_der usr}

Recording Reverse Derivatives That Use Checkpoints
##################################################

Discussion
**********
For this example

.. math::

    f(v)  &= v^2 \\
    g(v)  &= f(v) * v &= v^3  \\
    h(v)  &= g'(v)    &= 3 * v^2 \\
    h'(v) &= 6 * v


Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}
{xrst_end chkpnt_ad_reverse_der}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
namespace {
    using at::Tensor;
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::direction_t;
}
TEST(examples_chkpnt, ad_reverse_der)  {
    //
    // v
    vector<Tensor> v;
    v.push_back( torch::tensor( {0.0, 0.0} ) );
    //
    // av
    vector<adten_t> av = adten_t::start_recording(v);
    //
    // ar
    vector<adten_t> ar;
    ar.push_back( av[0] * av[0] );
    //
    // r = f(v) = v * v
    adfn_t f = adten_t::stop_recording(ar, "f");
    //
    // chkpnt_id
    vector<direction_t> directions = { direction_t::reverse };
    size_t chkpnt_id  = ad_tensor::make_chkpnt(f, v, directions);
    //
    // s = g(v) = f(v) * v = v * v * v
    av = adten_t::start_recording(v);
    ar = ad_tensor::call_chkpnt(chkpnt_id, av);
    vector<adten_t> as;
    as.push_back( ar[0] * av[0] );
    adfn_t g  = adten_t::stop_recording(as, "g");
    //
    // av, av_all;;
    av = adten_t::start_recording(v);
    vector<adten_t> av_all = g.forward_var(av);
    //
    // gp = h(v) = g'(v) = 3.0 * v * v
    vector<adten_t> ads;
    ads.push_back( adten_t( torch::tensor( {1.0, 1.0} ) ) );
    vector<adten_t> agp = g.reverse_der(ads, av_all);
    adfn_t h = adten_t::stop_recording(agp, "h");
    //
    // v, v_all, gp
    v[0]                 = torch::tensor( {3.0, 4.0} );
    vector<Tensor> v_all = h.forward_var(v);
    vector<Tensor> gp    = h.get_range(v_all);
    //
    // equal
    bool equal =  gp[0].equal( 3.0 * v[0] * v[0] );
    EXPECT_TRUE(equal);
    //
    // dds
    vector<Tensor> dv;
    dv.push_back( torch::tensor( {1.0, 1.0} ) );
    vector<Tensor> hp = h.forward_der(dv, v_all);
    //
    // equal
    equal =  hp[0].equal( 6.0 * v[0] );
    EXPECT_TRUE(equal);
}
// END_CPP
