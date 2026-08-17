// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin example_solve usr}
{xrst_spell
    cc
    ll
}

Example Solving linear Equations
################################

Problem
*******
Given :math:`v \in \mathbb{R}^3` and :math:`p \in \mathbb{R}^4`
solve for :math:`r \in \mathbb{R}^4` where

.. math::

    \left( \begin{array}{cc}
        p_0 & p_1 \\
        p_2 & p_3
    \end{array} \right)
    =
    \left( \begin{array}{cc}
        v_0 & 0   \\
        v_1 & v_2
    \end{array} \right)
    \left( \begin{array}{cc}
        r_0 & r_1 \\
        r_2 & r_3
    \end{array} \right)

.. math::

    \left( \begin{array}{cc}
        p_0 & p_1 \\
        p_2 & p_3
    \end{array} \right)
    =
    \left( \begin{array}{cc}
        v_0 r_0            & v_0 r_1           \\
        v_1 r_0 + v_2 r_2  & v_1 r_1 + v_2 r_3 
    \end{array} \right)

.. math::

    \begin{array}{ll}
    r_0 = p_0 / v_0              & r_1 = p_1 / v_0  \\
    r_2 = (p_2 - v_1 r_0) / v_2  & r_3 = (p_3 - v_1 r_1) / v_2 
    \end{array}

It follows that :math:`r = f(p, v)` where

.. math::

    f(v, p) & = \left( \begin{array}{c}
        p_0 / v_0 \\
        p_1 / v_0 \\
        (p_2 - v_1 p_0 / v_0) / v_2 \\
        (p_3 - v_1 p_1 / v_0) / v_2
    \end{array} \right)
    
Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}

{xrst_end example_solve}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
#include <torch/torch.h>
TEST(examples_adten, solve)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // p
    double p0 = 1.0, p1 = 2.0, p2 = 3.0, p3 = 4.0;
    vector<Tensor> p;
    p.push_back( torch::tensor( {p0, p1, p2, p3} ) );
    //
    // v
    double v0 = 5.0, v1 = 6.0, v2 = 7.0;
    vector<Tensor> v;
    v.push_back( torch::tensor( {v0, v1, v2} ) );
    //
    // ap, av
    auto [av, ap] = adten_t::start_recording(v, p);
    //
    // aP
    adten_t aP     = ap[0].view( {2, 2} );
    //
    // aV
    adten_t azeros = adten_t( torch::zeros( {2, 2} ) );
    std::optional<at::Tensor> row_index = torch::tensor( {0, 1, 1} );
    std::optional<at::Tensor> col_index = torch::tensor( {0, 0, 1} );
    c10::List< std::optional<at::Tensor> > index_list = {row_index, col_index};
    adten_t aV  = azeros.index_put(index_list, av[0]);
    //
    // aR
    bool left = true;
    adten_t aR = ad_tensor::linalg_solve(aV, aP, left);
    //
    // r = f(v, p)
    vector<adten_t> ar = {aR};
    adfn_t f           = adten_t::stop_recording(ar, "f");
    //
    // par_all, var_all
    vector<Tensor> par_all = f.forward_par(p);
    vector<Tensor> var_all = f.forward_var(v, par_all);
    //
    // r
    vector<Tensor> r = f.get_range(var_all, par_all);
    Tensor check = torch::tensor({
        {p0 / v0,                     p1 / v0},
        {(p2 - v1 * p0 / v0) / v2,   (p3 - v1 * p1 / v0) / v2}
    });
    bool close  = r[0].allclose( check );
    EXPECT_TRUE( close );
}
// END_CPP
