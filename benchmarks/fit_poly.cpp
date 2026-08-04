// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <chrono>
#include <gtest/gtest.h>
#include <torch/torch.h>

// https://docs.pytorch.org/tutorials/beginner/
//  pytorch_with_examples.html#pytorch-tensors-and-autograd
TEST(benchmarks, fit_poly_autograd) {
    using torch::Tensor;
    //
    // x, y
    Tensor x = torch::linspace(-1.0, 1.0, 2000);
    Tensor y = x.exp();
    //
    // a, b, c, d
    Tensor a = torch::randn( {1}, torch::requires_grad() );
    Tensor b = torch::randn( {1}, torch::requires_grad() );
    Tensor c = torch::randn( {1}, torch::requires_grad() );
    Tensor d = torch::randn( {1}, torch::requires_grad() );
    //
    // x2, x3
    Tensor x2 = x * x;
    Tensor x3 = x * x2;
    //
    // learning_rage, t
    double initial_loss  = 1.0;
    double relative_loss = 1.0;
    double learning_rate = 1e-5;
    for(size_t t = 0; t < 5000; ++t) {
        Tensor y_pred = a + b * x + c * x2 + d * x3;
        Tensor loss   = ( (y_pred - y) * (y_pred - y) ).sum();
        if( t == 0 ) {
            initial_loss = loss.item<double>();
        }
        relative_loss = loss.item<double>() / initial_loss;
        loss.backward();
        {   torch::NoGradGuard no_grad;
            //
            a -= learning_rate * a.grad();
            b -= learning_rate * b.grad();
            c -= learning_rate * c.grad();
            d -= learning_rate * d.grad();
            //
            a.grad().zero_();
            b.grad().zero_();
            c.grad().zero_();
            d.grad().zero_();
        }
    }
    //
    EXPECT_LT(relative_loss, 2e-4);
}
