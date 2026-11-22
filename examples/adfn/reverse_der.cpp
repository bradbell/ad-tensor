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

    f(x) = \left[ \begin{array}{c} \\
        x_0 + x_1 \\
        x_0 - x_1 \\
        x_0 * x_1 \\
        x_0 / x_1 \\
    \end{array} \right ]

It follows that the derivative f'(x) is given by

.. math::

    f'(x) = \left[ \begin{array}{cc} \\
        1 & 1  \\
        1 & - 1 \\
        x_1 & x_0 \\
        1 / x_1 & - x_0 / x_1^2 \\
    \end{array} \right ]

The range derivative in the direction dy is given by

.. math::

    dy * f'(x) = \left[ \begin{array}{c}
        dy_0 + dy_1 + dy_2 * x_1 + dy_3 / x_1  \\
        dy_0 - dy_1 + dy_2 * x_0 - dy_3 * x_0 / x_1^2
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
    // dy
    vector<at::Tensor> dy;
    dy.push_back( torch::tensor( {5.0, 6.0} ) );
    dy.push_back( torch::tensor( {7.0, 8.0} ) );
    dy.push_back( torch::tensor( {9.0, 10.0} ) );
    dy.push_back( torch::tensor( {11.0, 12.0} ) );
    //
    // dx
    vector<at::Tensor> dx = f.reverse_der(dy, v_all);
    //
    // close
    bool close;
    //
    close = dx[0].allclose(
        dy[0] + dy[1] + dy[2] * x[1] + dy[3] / x[1]
    );
    EXPECT_TRUE( close );
    //
    close = dx[1].allclose(
        dy[0] - dy[1] + dy[2] * x[0] - dy[3] * x[0] / ( x[1] * x[1] )
    );
    EXPECT_TRUE( close );
}
// END_CPP
