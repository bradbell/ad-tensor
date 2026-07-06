// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn_forward_dep_example usr}
{xrst_spell
    ccc
}

Range Dependencies Example
##########################

Discussion
**********
For this example

.. math::

    f(v, p) = \left[ \begin{array}{l}
        p_0 \\
        p_0 + p_1 v_0^1 \\
        p_0 + p_1 v_0^1 + p_2 v_0^2 \\
    \end{array} \right]

The (row,col) indices for the possibly non-zeros,
in the sparsity pattern for the dependency of f with respect to the
parameter vector p:

.. math::

    \left[ \begin{array}{ccc}
        (0,0) &       & \\
        (1,0) & (1,1) & \\
        (2,0) & (2,1) & (2,2) \\
    \end{array} \right]

The indices for the dependency of f  with respect to the variable vector v:

.. math::

    \left[ \begin{array}{c}
        \\
        (1,0) \\
        (2,0) \\
    \end{array} \right]

Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}

{xrst_end adfn_forward_dep_example}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
namespace {
    std::array<size_t, 2> pair(size_t row, size_t col) {
        return std::array<size_t, 2>({row, col});
    }

}
TEST(examples_adfn, forward_dep)  {
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
    // ar
    vector<adten_t> ar;
    ar.push_back( ap[0] );
    ar.push_back( ap[0] + ap[1] * av[0] );
    ar.push_back( ap[0] + ap[1] * av[0] + ap[2] * av[0] * av[0]);
    //
    // r = f(v, p)
    adfn_t f = adten_t::stop_recording(ar, "f");
    //
    // depend_par, depend_var
    auto [depend_par, depend_var] = f.forward_dep();
    depend_par.sort();
    depend_var.sort();
    //
    // depend_par
    EXPECT_EQ( depend_par.size(), 6 );
    EXPECT_EQ( depend_par[0], pair(0, 0) );
    EXPECT_EQ( depend_par[1], pair(1, 0) );
    EXPECT_EQ( depend_par[2], pair(1, 1) );
    EXPECT_EQ( depend_par[3], pair(2, 0) );
    EXPECT_EQ( depend_par[4], pair(2, 1) );
    EXPECT_EQ( depend_par[5], pair(2, 2) );
    //
    // depend_var
    depend_var.sort();
    EXPECT_EQ( depend_var.size(), 2 );
    EXPECT_EQ( depend_var[0], pair(1, 0) );
    EXPECT_EQ( depend_var[1], pair(2, 0) );
}
// END_CPP
