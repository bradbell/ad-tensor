// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin fit_poly_benchmark ben}
{xrst_spell
    pytorch
    autograd
}

Fitting A Polynomial Benchmark
##############################

Loss Function
*************
Let :math:`x \in {\rm R}^m` be a uniform grid on points on [-1,+1] .
The loss function :math:`f : {\rm R}^4 \rightarrow {\rm R}`
for this example is defined by

.. math::

    f(c) = \sum_{i=0}^{m-1} \left(
        \exp( x_i ) - \sum_{j=0}^3 c_j x_i^j
    \right)^2

This objective comes from the pytorch tutorial
`Pytorch Tensors and autograd
<https://docs.pytorch.org/tutorials/beginner/ pytorch_with_examples.html#pytorch-tensors-and-autograd>`_ ,

Common Code
***********
{xrst_literal ,
    BEGIN_COMMON, END_COMMON
}

Autograd Benchmark
******************
{xrst_literal ,
    BEGIN_AUTOGRAD, END_AUTOGRAD
}

AD Tensor Benchmark
*******************
{xrst_literal ,
    BEGIN_AD_TENSOR, END_AD_TENSOR
}

{xrst_end fit_poly_benchmark}
*/
// BEGIN_COMMON
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad_tensor.hpp>
//
namespace {
    //
    // vector
    using ad_tensor::vector;
    //
    // double_nan
    const double double_nan = std::numeric_limits<double>::quiet_NaN();
    //
    // number_coefficients, number_grid_points
    const size_t number_coefficients   = 4;
    const size_t number_grid_points    = 2000;
    //
    // expected_relative_loss
    const double expected_relative_loss = 1e-3;
    //
    // learning_rate, number_learning_steps
    const double learning_rate         = 1e-5;
    const size_t number_learning_steps = 5000;
    //
    // loss
    template<class Tensor>
    Tensor loss(const vector<Tensor>& c, const Tensor& x, const Tensor& y) {
        assert( size_t( x.numel() ) == number_grid_points );
        assert( size_t( y.numel() ) == number_grid_points );
        assert( size_t( c.size() )  == number_coefficients );
        int64_t n_y = y.numel();
        Tensor xp      = Tensor( torch::ones( {n_y} ) );
        Tensor predict = c[0] * xp;
        for(size_t j = 1; j < number_coefficients; ++j) {
            xp       = xp * x;
            predict  = predict + c[j] * xp;
        }
        Tensor residual = (y - predict);
        return (residual * residual).sum();
    }
}
// END_COMMON
//
// BEGIN_AUTOGRAD
TEST(benchmarks, fit_poly_autograd) {
    //
    // Tensor
    using torch::Tensor;
    //
    // x, y
    Tensor x = torch::linspace(-1.0, 1.0, number_grid_points);
    Tensor y = x.exp();
    //
    // c
    vector<Tensor> c;
    for(size_t j = 0; j < number_coefficients; ++j) {
        c.push_back( torch::randn( {1}, torch::requires_grad() ) );
    }
    //
    // learning_rage, dloss, initial_loss, relative_loss
    double initial_loss      = double_nan;
    double relative_loss     = double_nan;
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // loss_t
        Tensor loss_t = loss(c, x, y);
        //
        // initial_loss
        if( t == 0 ) {
            initial_loss = loss_t.item<double>();
        }
        // relative_loss
        if( t == number_learning_steps - 1 ) {
            relative_loss = loss_t.item<double>() / initial_loss;
        }
        //
        // c
        loss_t.backward();
        {   torch::NoGradGuard no_grad;
            //
            for(size_t j = 0; j < number_coefficients; ++j) {
                c[j] -= learning_rate * c[j].grad();
                c[j].grad().zero_();
            }
        }
    }
    //
    EXPECT_LT(relative_loss, expected_relative_loss);
}
// END_AUTOGRAD
//
// BEGIN_AD_TENSOR
TEST(benchmarks, fit_poly_ad_tensor) {
    //
    // adten_t, adfn_t
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    //
    // x, y
    at::Tensor x = torch::linspace(-1.0, 1.0, number_grid_points);
    at::Tensor y = x.exp();
    //
    // c
    vector<at::Tensor> c;
    for(size_t j = 0; j < number_coefficients; ++j) {
        c.push_back( torch::randn( {1} ) );
    }
    //
    // ac
    vector<adten_t> ac = adten_t::start_recording(c);
    //
    // ax, ay
    adten_t ax(x);
    adten_t ay(y);
    //
    // adfn
    vector<adten_t> aloss = { loss(ac, ax, ay) };
    adfn_t adfn = adten_t::stop_recording(aloss, "adfn");
    //
    // learning_rage, dloss, initial_loss, relative_loss
    vector<at::Tensor> dloss = { torch::tensor(1.0) };
    double initial_loss      = double_nan;
    double relative_loss     = double_nan;
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // var_all
        vector<at::Tensor> var_all = adfn.forward_var(c);
        //
        // grad
        vector<at::Tensor> grad  = adfn.reverse_der(dloss, var_all);
        //
        // initial_loss
        if( t == 0 ) {
            vector<at::Tensor> loss_t  = adfn.get_range(var_all);
            initial_loss = loss_t[0].item<double>();
        }
        // relative_loss
        if( t == number_learning_steps - 1 ) {
            vector<at::Tensor> loss_t  = adfn.get_range(var_all);
            relative_loss = loss_t[0].item<double>() / initial_loss;
        }
        //
        // c
        for(size_t j = 0; j < number_coefficients; ++j) {
            c[j] -= learning_rate * grad[j];
        }
    }
    //
    EXPECT_LT(relative_loss, expected_relative_loss);
}
// END_AD_TENSOR
