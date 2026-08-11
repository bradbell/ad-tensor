// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
//
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad_tensor.hpp>
//
namespace {
    //
    // vector
    using ad_tensor::vector;
    //
    // inf
    const double inf = std::numeric_limits<double>::infinity();
    //
    // number_coefficients, number_grid_points
    const size_t number_data_points   = 1000;
    const size_t matrix_size          = 10;
    //
    // expected_relative_loss
    const double expected_relative_loss = 0.2;
    //
    // learning_rate, number_learning_steps
    const double learning_rate         = 1e-3;
    const size_t number_learning_steps = 1000;
    //
    // options
    at::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64);
    //
    // y, mu_hat
    const at::Tensor y = torch::randn(
        {matrix_size, number_data_points}, options
    );
    const at::Tensor mu_hat = y.mean( /* dim = */ 1, /* keepdim = */ true);
    //
    // loss
    template<class TensorType>
    TensorType loss(const TensorType& L ) {
        //
        // n
        TensorType n = TensorType(
            torch::tensor( double(number_data_points), options)
        );
        //
        TensorType Lt        = L.transpose(0, 1);
        TensorType Sigma_inv = L.matmul(Lt);
        TensorType det_term  = Sigma_inv.logdet();
        //
        TensorType res       = TensorType(y - mu_hat);
        TensorType Lt_res    = Lt.matmul(res);
        TensorType res_term  = (Lt_res * Lt_res).sum() / n;
        //
        return res_term - det_term;
    }
    //
    // minimum_L
    at::Tensor compute_minimum_L(void) {
        at::Tensor minimum_L = torch::full(
            {matrix_size, matrix_size}, -inf, options
        );
        for(size_t j = 0; j < matrix_size; ++j) {
            minimum_L[j][j] = 1e-1;
        }
        minimum_L = torch::tril( minimum_L );
        return minimum_L;
    }
    const at::Tensor minimum_L = compute_minimum_L();
    //
    // initial_L
    at::Tensor compute_initial_L(void) {
        at::Tensor initial_L = torch::randn( {matrix_size, matrix_size} );
        initial_L            = torch::tril( initial_L );
        initial_L            = torch::maximum( initial_L, minimum_L );
        return initial_L;
    }
    const at::Tensor initial_L = compute_initial_L();
}
//
TEST(benchmarks, multi_normal_autograd) {
    //
    // L, initial_loss
    torch::Tensor L      = initial_L.clone();
    double initial_loss  = loss(L).item<double>();
    //
    // t
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // L, loss_t
        L.set_requires_grad(true);
        torch::Tensor loss_t = loss(L);
        loss_t.backward();
        //
        // L
        {   torch::NoGradGuard no_grad;
            //
            L = L - learning_rate * L.grad();
            L = torch::tril( L );
            L = torch::maximum(L, minimum_L);
        }
    }
    //
    // relative_loss
    double relative_loss = loss(L).item<double>() / initial_loss;
    EXPECT_LT(relative_loss, expected_relative_loss);
}
//
TEST(benchmarks, multi_normal_ad_tensor) {
    //
    // adten_t, adfn_t
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    //
    // L
    vector<at::Tensor> L     = { initial_L.clone() };
    //
    // aL
    vector<adten_t> aL    = adten_t::start_recording(L);
    vector<adten_t> aloss = { loss( aL[0] ) };
    adfn_t          adfn  = adten_t::stop_recording(aloss, "adfn");
    //
    // dloss, initial_loss, t
    vector<at::Tensor> dloss = { torch::tensor(1.0) };
    double initial_loss      = loss(L[0]).item<double>();
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // var_all
        vector<at::Tensor> var_all = adfn.forward_var(L);
        //
        // grad
        vector<at::Tensor> grad = adfn.reverse_der(dloss, var_all);
        //
        // L
        {   torch::NoGradGuard no_grad;
            //
            L[0] = L[0] - learning_rate * grad[0];
            L[0] = torch::tril( L[0] );
            L[0] = torch::maximum(L[0], minimum_L);
        }
    }
    //
    // relative_loss
    double relative_loss = loss(L[0]).item<double>() / initial_loss;
    EXPECT_LT(relative_loss, expected_relative_loss);
}
