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
    pw
    py
    px
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

For :math:`0 <= i < ny,
the partial of :math:`y_i` w.r.t :math:`x_k` is

.. math::

    \partial y_i / \partial x_k = \begin{cases}
        w_{k - i} & \text{if} \; i <= k < i + nw \\
        0         & \text{otherwise}
    \end{cases}

For :math:`0 <= i < ny`,
the partial of :math:`y_i` w.r.t :math:`b` is one.

Forward Derivatives
*******************
Given the forward derivatives
:math:`dx \in \mathbb{R}^{nx}, dw \in \mathbb{R}^{nw} , db \in \mathbb{R}` ,
the forward derivative of :math:`dy \in \mathbb{R}^{ny}` is given by

.. math::

    dy_i = db + \sum_{j=0}^{nw-1} x_{i+j} \cdot dw_j + dx_{i+j} \cdot w_j

Reverse Partials
****************
We are given the partial derivative of the objective w.r.t the output
:math:`py \in \mathbb{R}^{ny}` .
Reverse mode removes the output from the expression of the objective.
If we interpret :math:`w_{k-i}` as zero when :math:`k < i`,
the contributions to the partial w.r.t the input,
:math:`px \in \mathbb{R}^{nx}` ,
is given by

.. math::

    px_k \text{ +=} \sum_{i=0}^{ny-1} py_i \cdot w_{k-i}

The contributions to the partial w.r.t the weight and bias,
:math:`pw \in \mathbb{R}^{nw} , pb \in \mathbb{R}` ,
are given by

.. math::

    pw_j & \text{ +=} \sum_{i=0}^{ny-1} py_i \cdot x_{i+j} \\
    pb   & \text{ +=} \sum_{i=0}^{ny-1} py_i



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
    adten_t ab = av[2];
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
    //
    // zero_x, zero_w, zero_b, dv
    at::Tensor zero_x = torch::zeros( x.sizes() );
    at::Tensor zero_w = torch::zeros( w.sizes() );
    at::Tensor zero_b = torch::zeros( b.sizes() );
    vector<at::Tensor> dv = { zero_x, zero_w, zero_b };
    //
    // check partal of y w.r.t x[k]
    for(int64_t k = 0; k < nx; ++k) {
        //
        // dy
        dv[0]                 = torch::eye(nx).select(0, k).view( x.sizes() );
        vector<at::Tensor> dr = f.forward_der(dv, var_all);
        at::Tensor         dy = dr[0].contiguous();
        //
        // check
        vector<float> dy_vec(
            dy.data_ptr<float>(),
            dy.data_ptr<float>() + y.numel()
        );
        for(int64_t i = 0; i < ny; ++i) {
            if( i <= k && k < i + nw ) {
                EXPECT_EQ( dy_vec[i], w_vec[k - i] );
            } else {
                EXPECT_EQ( dy_vec[i], float( 0.0 ) );
            }
        }
    }
    dv[0] = zero_x;
    //
    // check partal of y w.r.t w[j]
    for(int64_t j = 0; j < nw; ++j) {
        //
        // dy
        dv[1]                 = torch::eye(nw).select(0, j).view( w.sizes() );
        vector<at::Tensor> dr = f.forward_der(dv, var_all);
        at::Tensor         dy = dr[0].contiguous();
        //
        // check
        vector<float> dy_vec(
            dy.data_ptr<float>(),
            dy.data_ptr<float>() + y.numel()
        );
        for(int64_t i = 0; i < ny; ++i) {
            EXPECT_EQ( dy_vec[i], x_vec[i + j] );
        }
    }
    dv[1] = zero_w;
    //
    // check derivative of y w.r.t. b
    {
        dv[2] = torch::tensor( {1.0} );
        vector<at::Tensor> dr = f.forward_der(dv, var_all);
        at::Tensor         dy = dr[0].contiguous();
        //
        // check
        vector<float> dy_vec(
            dy.data_ptr<float>(),
            dy.data_ptr<float>() + y.numel()
        );
        for(int64_t i = 0; i < ny; ++i) {
            EXPECT_EQ( dy_vec[i], float(1.0) );
        }
    }
}
// END_CPP
