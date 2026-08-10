// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
-------------------------------------------------------------------------------
{xrst_begin unary_inverse usr}
{xrst_spell
    cc
    cccc
    inv
    dv
}

Example Matrix Inverse
######################

Inverse
*******
.. math::

    v & = \left( \begin{array}{cc}
        v_0 & v_1 \\
        v_2 & v_3
    \end{array} \right)
    \\
    \det(v) &= v_0 v_3 - v_1 v_2
    \\
    {\rm inv} (v) & =
    \frac{1}{ \det(v) } \left( \begin{array}{cc}
        v_3  & - v_1 \\
        - v_2 & v_0
    \end{array} \right)
    \\
    f(v) & =
    \frac{1}{ \det(v)} \left( \begin{array}{c}
        v_3 \\
        -v_1 \\
        -v_2 \\
        v_0
    \end{array} \right)

Derivative
**********
.. math::
    \\
    f^{(1)} (v) =
    \frac{1}{ \det(v)} \left( \begin{array}{cccc}
        0  & 0  & 0  & 1   \\
        0  & -1 & 0  & 0   \\
        0  & 0  & -1 & 0  \\
        1  & 0  & 0  & 0  \\
    \end{array} \right)
    -
    \frac{1}{ \det(v)^2 } \left( \begin{array}{c}
        v_3 \\
        -v_1 \\
        -v_2 \\
        v_0
    \end{array} \right)
    \left( \begin{array}{c}
        v_3  \\
        -v_2 \\
        -v_1 \\
        v_0
    \end{array} \right)^T

Forward Direction Derivative
****************************

.. math::

    f^{(1)} (v) \left( \begin{array}{c}
        dv_0 \\
         dv_1 \\
         dv_2 \\
         dv_3
    \end{array} \right)
    =
    \frac{1}{ \det(v) }
    \left( \begin{array}{c}
        dv_3   \\
        - dv_1 \\
        - dv_2 \\
        dv_0
    \end{array} \right)
    -
    \frac{ v_3 dv_0 - v_2 dv_1 - v_1 dv_2 + v_0 dv_3}{ \det(v)^2 }
    \left( \begin{array}{c}
        v_3 \\
        -v_1 \\
        -v_2 \\
        v_0
    \end{array} \right)

Reverse Direction Derivative
****************************

.. math::

    \left( \begin{array}{c}
        dr_0 \\
        dr_1 \\
        dr_2 \\
        dr_3
    \end{array} \right)^T
    f^{(1)} (v)
    =
    \frac{1}{ \det(v) }
    \left( \begin{array}{c}
        dr_3   \\
        - dr_1 \\
        - dr_2 \\
        dr_0
    \end{array} \right)^T
    -
    \frac{ v_3 dr_0 - v_1 dr_1 - v_2 dr_2 + v_0 dr_3}{ \det(v)^2 }
    \left( \begin{array}{c}
        v_3 \\
        -v_2 \\
        -v_1 \\
        v_0
    \end{array} \right)^T


Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}

{xrst_end unary_inverse}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adten, unary_inverse)  {
    using ad_tensor::adten_t;
    using ad_tensor::vector;
    using at::Tensor;
    //
    // options
    at::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64);
    //
    // v
    double v0 = 1.0, v1 = 2.0, v2 = 3.0, v3 = 4.0;
    vector<Tensor> v;
    v.push_back( torch::tensor( {
        {v0, v1},
        {v2, v3}
    }, options ) );
    //
    // det
    double det = v0 * v3 - v1 * v2;
    //
    // inverse
    /* inv(v) = {
        {v3, -v1},
        {-v2, v0}
    } / det(v0, v1, v2, v3)
    */
    Tensor inverse = torch::tensor(
        { {v3, -v1}, {-v2, v0} }, options
    ) / det;
    //
    // f
    vector<adten_t> av = adten_t::start_recording(v);
    vector<adten_t>  ar;
    ar.push_back( av[0].inverse() );
    ad_tensor::adfn_t f  = adten_t::stop_recording(ar, "f");
    //
    // v_all, r
    vector<Tensor> v_all = f.forward_var(v);
    vector<Tensor> r     = f.get_range(v_all);
    EXPECT_TRUE( r[0].allclose( inverse ) );
    //
    // dv, dr
    vector<Tensor> dv;
    double dv0 = 5.0, dv1 = 6.0, dv2 = 7.0, dv3 = 8.0;
    dv.push_back( torch::tensor(
        { {dv0, dv1}, {dv2, dv3} }, options
    ) );
    vector<Tensor> dr = f.forward_der(dv, v_all);
    //
    // check
    double term   = v3 * dv0 - v2 * dv1 - v1 * dv2 + v0 * dv3;
    term          = term / (det * det );
    double dr0  = + dv3 / det - v3 * term;
    double dr1  = - dv1 / det + v1 * term;
    double dr2  = - dv2 / det + v2 * term;
    double dr3  = + dv0 / det - v0 * term;
    Tensor check = torch::tensor({
        {dr0, dr1},
        {dr2, dr3}
    }, options );
    EXPECT_TRUE( dr[0].allclose( check ) );
    //
    // dr, dv
    dr0 = 5.0, dr1 = 6.0, dr2 = 7.0, dr3 = 8.0;
    dr[0] = torch::tensor(
        { {dr0, dr1}, {dr2, dr3} }, options
    );
    dv = f.reverse_der(dr, v_all);
    //
    // check
    term   = v3 * dr0 - v1 * dr1 - v2 * dr2 + v0 * dv3;
    term          = term / (det * det );
    dv0    = + dr3 / det - v3 * term;
    dv1    = - dr1 / det + v2 * term;
    dv2    = - dr2 / det + v1 * term;
    dv3    = + dr0 / det - v0 * term;
    check = torch::tensor({
        {dv0, dv1},
        {dv2, dv3}
    }, options );
    EXPECT_TRUE( dv[0].allclose( check ) );
}
// END_CPP
