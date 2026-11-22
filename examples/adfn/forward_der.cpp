// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn_forward_der_example usr}
{xrst_spell
    cc
    dx
}

Domain Direction Derivative Example
###################################

Discussion
**********
For this example

.. math::

    f(x) = \left[ \begin{array}{c}
        x_0 + x_1 \\
        x_0 - x_1 \\
        x_0 * x_1 \\
        x_0 / x_1 \\
    \end{array} \right ]

It follows that the derivative f'(x) is given by

.. math::

    f'(x) = \left[ \begin{array}{cc}
        1 & 1  \\
        1 & - 1 \\
        x_1 & x_0 \\
        1 / x_1 & - x_0 / x_1^2 \\
    \end{array} \right ]

The domain derivative in the direction dx is given by

.. math::

    f'(x) * dx = \left[ \begin{array}{cc}
        dx_0 + dx_1  \\
        dx_0 - dx_1   \\
        x_1 * dx_0 + x_0 * dx_1 \\
        dx_0 / x_1  - x_0 * dx_1 / x_1^2 \\
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
    using ad_tensor::vector;
    //
    // x
    // We use x for the domain variables
    vector<at::Tensor> x;
    x.push_back( torch::tensor( {1.0, 1.0} ) );
    x.push_back( torch::tensor( {1.0, 1.0} ) );
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // ay
    // We use y for the range space
    vector<adten_t> ay;
    ay.push_back(  ax[0] + ax[1] );
    ay.push_back(  ax[0] - ax[1] );
    ay.push_back(  ax[0] * ax[1] );
    ay.push_back(  ax[0] / ax[1] );
    //
    // y = f(x)
    // We use f for the adfn_t object.
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // x, v_all
    x[0] = torch::tensor( {1.0, 2.0} );
    x[1] = torch::tensor( {3.0, 4.0} );
    vector<at::Tensor> v_all = f.forward_var(x);
    //
    // dx
    vector<at::Tensor> dx;
    dx.push_back( torch::tensor( {5.0, 6.0} ) );
    dx.push_back( torch::tensor( {7.0, 8.0} ) );
    //
    // dy
    vector<at::Tensor> dy = f.forward_der(dx, v_all);
    //
    // equal, close
    bool equal, close;
    //
    equal = dy[0].equal( dx[0] + dx[1] );
    EXPECT_TRUE( equal );
    //
    equal = dy[1].equal( dx[0] - dx[1] );
    EXPECT_TRUE( equal );
    //
    equal = dy[2].equal( x[1] * dx[0] + x[0] * dx[1] );
    EXPECT_TRUE( equal );
    //
    close = dy[3].allclose( dx[0] / x[1] - x[0] * dx[1] /(x[1] * x[1]) );
    EXPECT_TRUE( close );
}
// END_CPP
