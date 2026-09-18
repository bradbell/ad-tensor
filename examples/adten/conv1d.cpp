// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin example_conv1d usr}
{xrst_spell
    nx
    nw
    ny
    dx
    dw
}

Examples One Dimensional Cross Correlation
##########################################

Function Value
**************
Given the input vector :math:`x \in \mathbb{R}^{nx}`,
the weight vector :math:`w \in \mathbb{R}^{nw}` ,
and the bias tensor :math:`b \in \mathbb{R}` ,
compute the cross correlation vector :math:`y \in \mathbb{R}^{ny}`
defined by :math:`ny = nx - nw + 1` and

.. math::

    y_i ( x, w, b ) = b + \sum_{j=0}^{nw-1} x_{i + j} \cdot w_j

Partial Derivatives
*******************

For :math:`0 <= i < ny, \; 0 <= j < nw`,
the partial of :math:`y_i` w.r.t. :math:`w_j` is

.. math::

    \partial y_i / \partial w_j = x_{i + j}

For :math:`0 <= i < ny, \; 0 <= k < nx`,
the partial of :math:`y_i` w.r.t :math:`x_k` is

.. math::

    \partial y_i / \partial x_k = \begin{cases}
        w_{k - i} & \text{if} \; i <= k < i + nw \\
        0         & \text{otherwise}
    \end{cases}

For :math:`0 <= i < ny`,
the partial of :math:`y_i` w.r.t math:`b` is one.

Forward Derivatives
*******************
Given the forward derivatives
:math:`dx \in \mathbb{R}^{nx}, dw \in \mathbb{R}^{nw} , db \in \mathbb{R}` ,
the forward derivative of :math:`dy \in \mathbb{R}^{ny}` is given by

.. math::

    dy_i = db + \sum_{j=0}^{nw-1} x_{i+j} \cdot dw_j + dx_{i+j} \cdot w_j


Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}

{xrst_end example_conv1d}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad_tensor.hpp>
TEST(examples_adten, conv1d) {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // x_vec, w_vec, b_vec
    vector<float> x_vec = {1.0, 2.0, 3.0, 4.0};
    vector<float> w_vec = {0.25, 0.5, 0.25};
    vector<float> b_vec = {-1.0};
    //
    // x, w, b
    at::Tensor x = torch::tensor(x_vec).view( {1, 1, 4} );
    at::Tensor w = torch::tensor(w_vec).view( {1, 1, 3} );
    at::Tensor b = torch::tensor(b_vec);
    //
    // v
    vector<at::Tensor> v;
    v.push_back( x );
    v.push_back( w );
    v.push_back( b );
    //
    // adom_var
    vector<adten_t> av = adten_t::start_recording(v);
    //
    //
    // ax, aw, ab
    adten_t ax  = av[0];
    adten_t aw = av[1];
    adten_t ab   = av[2];
    //
    // ay
    auto options   = torch::nn::functional::Conv1dFuncOptions();
    adten_t   ay = ad_tensor::conv1d(ax, aw, ab, options);
    //
    // r = f(v)
    vector<adten_t> ar = {ay};
    adfn_t f           = adten_t::stop_recording(ar, "f");
    //
    // var_all
    vector<at::Tensor> var_all = f.forward_var(v);
    //
    // r
    vector<at::Tensor> r = f.get_range(var_all);
    //
    // y
    at::Tensor y = r[0].contiguous();
    //
    // nx, nw, ny
    int64_t nx = x.sizes()[2];
    int64_t nw = w.sizes()[2];
    int64_t ny = y.sizes()[2];
    EXPECT_EQ( ny,  nx - nw + 1 );
    //
    // check
    vector<float> y_vec(
        y.data_ptr<float>(),
        y.data_ptr<float>() + y.numel()
    );
    for(int64_t i = 0; i < ny; ++i) {
        float sum = b_vec[0];
        for(int64_t j = 0; j < nw; ++j) {
            sum += x_vec[i + j] * w_vec[j];
        }
        EXPECT_EQ( y_vec[i], sum );
    }
}
// END_CPP
