// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn_forward_der_example usr}
{xrst_spell
    cc
    dv
}

Domain Direction Derivative Example
###################################

Discussion
**********
For this example

.. math::

    f(v) = \left[ \begin{array}{c}
        v_0 + v_1 \\
        v_0 - v_1 \\
        v_0 * v_1 \\
        v_0 / v_1 \\
    \end{array} \right ]

It follows that the derivative f'(v) is given by

.. math::

    f'(v) = \left[ \begin{array}{cc}
        1 & 1  \\
        1 & - 1 \\
        v_1 & v_0 \\
        1 / v_1 & - v_0 / v_1^2 \\
    \end{array} \right ]

The domain derivative in the direction dv is given by

.. math::

    f'(v) * dv = \left[ \begin{array}{cc}
        dv_0 + dv_1  \\
        dv_0 - dv_1   \\
        v_1 * dv_0 + v_0 * dv_1 \\
        dv_0 / v_1  - v_0 * dv_1 / v_1^2 \\
    \end{array} \right ]

Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}

{xrst_end adfn_forward_der_example}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adfn, forward_der)  {
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
    //
    // av
    vector<adten_t> av = adten_t::start_recording(v);
    //
    // ar
    // We use r for the range space
    vector<adten_t> ar;
    ar.push_back(  av[0] + av[1] );
    ar.push_back(  av[0] - av[1] );
    ar.push_back(  av[0] * av[1] );
    ar.push_back(  av[0] / av[1] );
    //
    // r = f(v)
    // We use f for the adfn_t object.
    adfn_t f = adten_t::stop_recording(ar, "f");
    //
    // v, v_all
    v[0] = torch::tensor( {1.0, 2.0} );
    v[1] = torch::tensor( {3.0, 4.0} );
    vector<Tensor> v_all = f.forward_var(v);
    //
    // dv
    vector<Tensor> dv;
    dv.push_back( torch::tensor( {5.0, 6.0} ) );
    dv.push_back( torch::tensor( {7.0, 8.0} ) );
    //
    // dr
    vector<Tensor> dr = f.forward_der(dv, v_all);
    //
    // equal, close
    bool equal, close;
    //
    equal = dr[0].equal( dv[0] + dv[1] );
    EXPECT_TRUE( equal );
    //
    equal = dr[1].equal( dv[0] - dv[1] );
    EXPECT_TRUE( equal );
    //
    equal = dr[2].equal( v[1] * dv[0] + v[0] * dv[1] );
    EXPECT_TRUE( equal );
    //
    close = dr[3].allclose( dv[0] / v[1] - v[0] * dv[1] /(v[1] * v[1]) );
    EXPECT_TRUE( close );
}
// END_CPP
