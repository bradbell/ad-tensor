// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <chrono>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad_tensor.hpp>

namespace {
    //
    // Tensor
    using torch::Tensor;
    using ad_tensor::vector;
    //
    // loss
    Tensor loss(const vector<Tensor>& c, const Tensor& y, const Tensor& x) {
        assert( y.numel() == x.numel() );
        int64_t n_y = y.numel();
        size_t n_c  = c.size();
        Tensor xp      = torch::ones( {n_y} );
        Tensor predict = c[0] * xp;
        for(size_t j = 1; j < n_c; ++j) {
            xp       = xp * x;
            predict  = predict + c[j] * xp;
        }
        Tensor residual = (y - predict);
        return (residual * residual).sum();
    }
}

// https://docs.pytorch.org/tutorials/beginner/
//  pytorch_with_examples.html#pytorch-tensors-and-autograd
TEST(benchmarks, fit_poly_autograd) {
    //
    // x, y
    Tensor x = torch::linspace(-1.0, 1.0, 2000);
    Tensor y = x.exp();
    //
    // c
    size_t n_c = 4;
    vector<Tensor> c;
    for(size_t j = 0; j < n_c; ++j) {
        c.push_back( torch::randn( {1}, torch::requires_grad() ) );
    }
    //
    // learning_rage, t
    double initial_loss  = 1.0;
    double relative_loss = 1.0;
    double learning_rate = 1e-5;
    for(size_t t = 0; t < 5000; ++t) {
        Tensor loss_t = loss(c, y, x);
        if( t == 0 ) {
            initial_loss = loss_t.item<double>();
        }
        relative_loss = loss_t.item<double>() / initial_loss;
        loss_t.backward();
        {   torch::NoGradGuard no_grad;
            //
            for(size_t j = 0; j < n_c; ++j) {
                c[j] -= learning_rate * c[j].grad();
                c[j].grad().zero_();
            }
        }
    }
    //
    EXPECT_LT(relative_loss, 2e-4);
}
