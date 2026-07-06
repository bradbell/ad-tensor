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

    f(x, p) = \left[ \begin{array}{l}
        p_0 \\
        p_0 + p_1 x_0^1 \\
        p_0 + p_1 x_0^1 + p_2 x_0^2 \\
    \end{array} \right]

Using 1 for possibly non-zeros,
the sparsity pattern with respect to the parameter vector p is

.. math::

    \left[ \begin{array}{ccc}
        1 & 0 & 0 \\
        1 & 1 & 0 \\
        1 & 1 & 1 \\
    \end{array} \right]

The sparsity pattern with respect to the variable vector x is

.. math::

    \left[ \begin{array}{ccc}
        0 \\
        1 \\
        1 \\
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
    // x
    // We use x for the domain variables
    vector<Tensor> x;
    x.push_back( torch::tensor( {1.0, 1.0} ) );
    //
    // ap, ax
    auto [ax, ap] = adten_t::start_recording(x, p);
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( ap[0] );
    ay.push_back( ap[0] + ap[1] * ax[0] );
    ay.push_back( ap[0] + ap[1] * ax[0] + ap[2] * ax[0] * ax[0]);
    //
    // y = f(x, p)
    adfn_t f = adten_t::stop_recording(ay, "f");
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
