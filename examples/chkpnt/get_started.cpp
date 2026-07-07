// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
/*
{xrst_begin chkpnt_get_started usr}
{xrst_spell
    dv
    ds
}

Getting Started Using Checkpoints
#################################

Discussion
**********
For this example

.. math::

    f(v) &= v^2 \\
    g(v) &= f(v) * v = v^3

It follows that the derivative g'(v) is given by

.. math::
    g'(v) = 3 * v^2

The domain derivative in the dv direction is

.. math::
    g'(v) * dv = 3 * v^2 * dv

The range derivative in the ds direction is

.. math::
    ds * g'(v) = 3 * v^2 * ds

There is no real difference between the domain and range direction derivatives
for g because both its domain and range have only one tensor.

Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}
{xrst_end chkpnt_get_started}
*/
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_chkpnt, get_started)  {
    using at::Tensor;
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    //
    // v
    vector<Tensor> v;
    v.push_back( torch::tensor( {1.0, 1.0} ) );
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
    size_t chkpnt_id  = ad_tensor::make_chkpnt(f);
    //
    // s = g(v) = f(v) * v = v * v * v
    av = adten_t::start_recording(v);
    ar = ad_tensor::call_chkpnt(chkpnt_id, av);
    vector<adten_t> as;
    as.push_back( ar[0] * av[0] );
    adfn_t g  = adten_t::stop_recording(as, "g");
    //
    // v, v_all, s
    v[0] = torch::tensor( {3.0, 4.0} );
    vector<Tensor> v_all = g.forward_var(v);
    vector<Tensor> s     = g.get_range(v_all);
    //
    // equal
    bool equal =  s[0].equal( v[0] * v[0] * v[0] );
    EXPECT_TRUE(equal);
    //
    // dv, ds
    vector<Tensor> dv;
    dv.push_back( torch::tensor( {1.0, 2.0} ) );
    vector<Tensor> ds = g.forward_der(dv, v_all);
    //
    // equal
    equal =  ds[0].equal( 3.0 * v[0] * v[0] * dv[0] );
    EXPECT_TRUE(equal);
    //
    // ds, dv
    ds[0] = torch::tensor( {3.0, 4.0} );
    dv = g.reverse_der(ds, v_all);
    //
    // equal
    equal =  dv[0].equal( 3.0 * v[0] * v[0] * ds[0] );
    EXPECT_TRUE(equal);
}
// END_CPP
