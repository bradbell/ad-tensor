// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin example_conv1d usr}
{xrst_spell
    nw
    nc
}

Examples One Dimensional Cross Correlation
##########################################

Function Value
**************
Given the input vector :math:`x \in \mathbb{R}^{ni}`,
the weight vector :math:`w \in \mathbb{R}^{nw}` ,
and the bias tensor :math:`b \in \mathbb{R}` ,
compute the cross correlation vector :math:`c \in \mathbb{R}^{nc}
defined by :math:`nc = ni - nw + 1` and

.. math::

    c_i ( x, w, b ) = b + \sum_{j=0}^{nw-1} x_{i + j} \cdot w_j

Partial Derivatives
*******************

The partial of :math:`c_i` w.r.t. :math:`w_j` is

.. math::

    \partial c_i / \partial w_j = x_{i + j}

The partial of :math:`c_i` w.r.t math:`x_k` is

.. math::

    \partial c_i / \partial x_k = w_{k - i}

The partial of :math:`c_i` w.r.t math:`b` is one.


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
    // x_vec, w_vec, bias_vec
    vector<float> x_vec  = {1.0, 2.0, 3.0, 4.0};
    vector<float> w_vec = {0.25, 0.5, 0.25};
    vector<float> bias_vec   = {-1.0};
    //
    // x, w, bias
    at::Tensor x  = torch::tensor(x_vec).view( {1, 1, 4} );
    at::Tensor w = torch::tensor(w_vec).view( {1, 1, 3} );
    at::Tensor bias   = torch::tensor(bias_vec);
    //
    // v
    vector<at::Tensor> v;
    v.push_back( x );
    v.push_back( w );
    v.push_back( bias );
    //
    // adom_var
    vector<adten_t> av = adten_t::start_recording(v);
    //
    //
    // ax, aw, abias
    adten_t ax  = av[0];
    adten_t aw = av[1];
    adten_t abias   = av[2];
    //
    // ac
    auto options   = torch::nn::functional::Conv1dFuncOptions();
    adten_t   ac = ad_tensor::conv1d(ax, aw, abias, options);
    //
    // r = f(v)
    vector<adten_t> ar = {ac};
    adfn_t f           = adten_t::stop_recording(ar, "f");
    //
    // var_all
    vector<at::Tensor> var_all = f.forward_var(v);
    //
    // r
    vector<at::Tensor> r = f.get_range(var_all);
    //
    // c
    at::Tensor c = r[0].contiguous();
    //
    // ni, nw, nc
    int64_t ni = x.sizes()[2];
    int64_t nw = w.sizes()[2];
    int64_t nc = c.sizes()[2];
    EXPECT_EQ( nc,  ni - nw + 1 );
    //
    // c_vec
    vector<float> c_vec(
        c.data_ptr<float>(),
        c.data_ptr<float>() + c.numel()
    );
    //
    for(int64_t i = 0; i < nc; ++i) {
        float sum = bias_vec[0];
        for(int64_t j = 0; j < nw; ++j) {
            sum += x_vec[i + j] * w_vec[j];
        }
        EXPECT_EQ( c_vec[i], sum );
    }
}
// END_CPP
