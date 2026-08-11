// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin multi_normal_benchmark ben}
{xrst_spell
    cholesky
    autograd
}

Fitting a Multivariate Normal Distribution
##########################################

Loss Function
*************
Let :math:`\Sigma \in \mathbb{R}^{m \times m}` be a
symmetric positive definite matrix and :math:`\mu \in \mathbb{R}^m` .
Suppose :math:`y_i \in \mathbb{R}^m` for :math:`i = 0 , \ldots , n-1`
is distributed normally with mean :math:`\mu` and variance :math:`\Sigma` .
The probability density for the sequence :math:`\{ y_i \}` given
its mean and variance is

.. math::

    \mathbb{p} ( \{ y_i \} | \mu , \Sigma )
    =
    ( 2 \pi )^{- n m / 2 } \det( \Sigma )^{ - n / 2 }
    \cdot
    \exp \left[ -\frac{1}{2}
        \sum_{i=0}^{n-1} ( y_i - \mu)^T \Sigma^{-1} (y_i - \mu)
    \right]

The negative log density is

.. math::

    - \log [ \mathbb{p} ( \{ y_i \} | \mu , \Sigma ) ]
    =
    \frac{n m}{2} \log(2 \pi) + \frac{n}{2} \log \det( \Sigma )
    +
    \frac{1}{2} \sum_{i=0}^{n-1} ( y_i - \mu)^T \Sigma^{-1} (y_i - \mu)

We express :math:`\Sigma^{-1}` in terms of its Cholesky factor
:math:`L` ; i.e., :math:`L` is lower triangular,
:math:`\Sigma^{-1} = L L^T` , and

.. math::

    - \log [ \mathbb{p} ( \{ y_i \} | \mu , L ) ]
    =
    \frac{n m}{2} \log(2 \pi) - \frac{n}{2} \log \det( L L^T  )
    +
    \frac{1}{2} \sum_{i=0}^{n-1} ( y_i - \mu)^T L L^T (y_i - \mu)

We drop the constant :math:`n m \log(2 \pi)` and the factor of one half.
In addition, we scale the objective by dividing by the number of data points
:math:`n`.
The resulting objective is
:math:`f : \mathbb{R}^m \times \mathbb{R}^{m \times m} \rightarrow \mathbb{R}`
where

.. math::
    f( \mu, L )
    =
    - \log \det( L L^T )
    +
    \frac{1}{n} \sum_{i=0}^{n-1} ( y_i - \mu)^T L L^T (y_i - \mu)

and :math:`L` is restricted to the lower triangular matrices
with positive entries on the diagonal.
The minimizer of this function with respect to :math:`\mu` is

.. math::

    \hat{\mu} = \frac{1}{n} \sum_{i=0}^{n-1} y_i

see the `proof <https://statproofbook.github.io/P/mvn-mle.html>`_ .
We define our loss function as

.. math::

    g(L) = f( \hat{\mu} , L ) =
    - \log \det( L L^T )
    +
    \frac{1}{n} \sum_{i=0}^{n-1} ( y_i - \hat{\mu} )^T L L^T (y_i - \hat{\mu} )


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


{xrst_end multi_normal_benchmark}
*/
//
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad_tensor.hpp>
//
// BEGIN_COMMON
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
// END_COMMON
//
// BEGIN_AUTOGRAD
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
// END_AUTOGRAD
//
// BEGIN_AD_TENSOR
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
// END_AD_TENSOR
