// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn_reverse_der_example usr}
{xrst_spell
    cc
}

Range Direction Derivative Example
##################################

Discussion
**********
For this example

.. math::

    f(v) = \left[ \begin{array}{c} \\
        v_0 + v_1 \\
        v_0 - v_1 \\
        v_0 * v_1 \\
        v_0 / v_1 \\
    \end{array} \right ]

It follows that the derivative f'(v) is given by

.. math::

    f'(v) = \left[ \begin{array}{cc} \\
        1 & 1  \\
        1 & - 1 \\
        v_1 & v_0 \\
        1 / v_1 & - v_0 / v_1^2 \\
    \end{array} \right ]

So the range derivative in the direction dr is given by

.. math::

    dr * f'(v) = \left[ \begin{array}{c}
        dr_0 + dr_1 + dr_2 * v_1 + dr_3 / v_1  \\
        dr_0 - dr_1 + dr_2 * v_0 - dr_3 * v_0 / v_1^2
    \end{array} \right ]^T

Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}

{xrst_end adfn_reverse_der_example}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adfn, reverse_der)  {
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
    // dr
    vector<Tensor> dr;
    dr.push_back( torch::tensor( {5.0, 6.0} ) );
    dr.push_back( torch::tensor( {7.0, 8.0} ) );
    dr.push_back( torch::tensor( {9.0, 10.0} ) );
    dr.push_back( torch::tensor( {11.0, 12.0} ) );
    //
    // dv
    vector<Tensor> dv = f.reverse_der(dr, v_all);
    //
    // close
    bool close;
    //
    close = dv[0].allclose(
        dr[0] + dr[1] + dr[2] * v[1] + dr[3] / v[1]
    );
    EXPECT_TRUE( close );
    //
    close = dv[1].allclose(
        dr[0] - dr[1] + dr[2] * v[0] - dr[3] * v[0] / ( v[1] * v[1] )
    );
    EXPECT_TRUE( close );
}
// END_CPP
