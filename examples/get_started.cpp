// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin get_started usr}

Get Started Using ad-tensor
###########################

Function
********
This example function is

.. math::

    f_0 (x) = c_0 + c_1 \cdot x_0 + c_2 \cdot x_0 \cdot x_0

c
=
is a vector with three elements.
Each element of this vector is a tensor with one element.

x
=
is a vector with one element.
The element of x is a one dimensional tensor with four elements.

f(x)
====
is a vector with one element.
The element of f(x) is a one dimensional tensor with four elements.

Derivative
**********
The derivative of this function is given by

.. math::

    f_0 '(x) = c_1 + 2 \cdot c_2 \cdot x_0

f'(x)
=====
is a vector with one element.
The element of f'(x) is a one dimensional tensor with four elements.

Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}

{xrst_end get_started}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>

TEST(examples, get_started) {
    //
    // vector, adten_t
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    //
    // c
    vector<at::Tensor> c = {
        torch::tensor( 1.0 ),
        torch::tensor( 2.0 ),
        torch::tensor( 3.0 ),
    };
    //
    // x
    // This value is used during recording, but not during evaluation.
    vector<at::Tensor> x = { torch::zeros( {4} ) };
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // asum, ax_i
    adten_t asum( torch::tensor( 0.0 ) );
    adten_t ax_i( torch::tensor( 1.0 ) );
    //
    // asum, ax_i
    for(size_t i = 0; i < c.size(); ++i) {
        asum += adten_t( c[i] ) * ax_i;
        ax_i *= ax[0];
    }
    //
    // ay
    vector<adten_t> ay = { asum };
    //
    // y = f(x)
    ad_tensor::adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // x, var_all
    // New value for x used during evaluation.
    x[0] = torch::tensor( {4.0, 5.0, 6.0, 7.0} );
    vector<at::Tensor> var_all = f.forward_var(x);
    //
    // y
    vector<at::Tensor> y = f.get_range(var_all);
    EXPECT_TRUE( y[0].equal(
        c[0] + c[1] * x[0] + c[2] * x[0] * x[0]
    ) );
    //
    // dy
    vector<at::Tensor> dx = { torch::ones( x[0].sizes() ) };
    vector<at::Tensor> dy = f.forward_der(dx, var_all);
    EXPECT_TRUE( dy[0].equal(
        c[1] + 2.0 * c[2] * x[0]
    ) );
    //
    // px
    vector<at::Tensor> py = { torch::ones( x[0].sizes() ) };
    vector<at::Tensor> px = f.reverse_der(py, var_all);
    EXPECT_TRUE( px[0].equal(
        c[1] + 2.0 * c[2] * x[0]
    ) );
}
// END_CPP
