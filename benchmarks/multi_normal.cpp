// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin multi_normal_benchmark ben}
{xrst_spell
    cholesky
    autograd
    gtest
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

learn_ms
********
In the code below, gtest reports the total time for each test.
Each test also prints the value learn_ms, which is the time
in milliseconds for the learning loop; i.e., it does not include
the time to setup the calculation of the gradients in the learning loop.

Common Code
***********
{xrst_literal ,
    BEGIN_COMMON, END_COMMON
}

Autograd Code
*************
{xrst_literal ,
    BEGIN_AUTOGRAD, END_AUTOGRAD
}

AD Tensor Code
**************
{xrst_literal ,
    BEGIN_AD_TENSOR, END_AD_TENSOR
}

AD Tensor With Optimization
***************************
{xrst_literal ,
    BEGIN_OPTIMIZE, END_OPTIMIZE
}

AD Tensor With Source Generation
********************************
{xrst_literal ,
    BEGIN_SRC_GEN, END_SRC_GEN
}

{xrst_end multi_normal_benchmark}
*/
//
//
// BEGIN_COMMON
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad_tensor.hpp>
namespace {
    //
    // chrono, vector, adten_t, adfn_t
    namespace chrono = std::chrono;
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    //
    // previous_time, elapsed_ms
    chrono::time_point previous_time = chrono::steady_clock::now();
    double elapsed_ms(void) {
        chrono::time_point current_time = chrono::steady_clock::now();
        auto microseconds = chrono::duration_cast<chrono::microseconds>(
            current_time - previous_time
        ).count();
        double ms = double(microseconds) / 1000.0;
        previous_time = current_time;
        return ms;
    }
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
    // previous_time
    elapsed_ms();
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
    // learn_ms
    double learn_ms = elapsed_ms();
    std::cout << "learn_ms = " << learn_ms << "\n";
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
    // L
    vector<at::Tensor> L     = { initial_L.clone() };
    //
    // adfn
    vector<adten_t> aL    = adten_t::start_recording(L);
    vector<adten_t> aloss = { loss( aL[0] ) };
    adfn_t          adfn  = adten_t::stop_recording(aloss, "adfn");
    //
    // previous_time
    elapsed_ms();
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
    // learn_ms
    double learn_ms = elapsed_ms();
    std::cout << "learn_ms = " << learn_ms << "\n";
    //
    // relative_loss
    double relative_loss = loss(L[0]).item<double>() / initial_loss;
    EXPECT_LT(relative_loss, expected_relative_loss);
}
// END_AD_TENSOR
//
// BEGIN_OPTIMIZE
TEST(benchmarks, multi_normal_optimize) {
    //
    // L
    vector<at::Tensor> L     = { initial_L.clone() };
    //
    // f_loss
    vector<adten_t> aL     = adten_t::start_recording(L);
    vector<adten_t> aloss  = { loss( aL[0] ) };
    adfn_t          f_loss = adten_t::stop_recording(aloss, "f_loss");
    //
    // aL
    aL     = adten_t::start_recording(L);
    //
    // avar_all
    vector<adten_t> avar_all = f_loss.forward_var(aL);
    //
    // agrad
    vector<adten_t> adloss = { adten_t( torch::tensor(1.0) ) };
    vector<adten_t> agrad  = f_loss.reverse_der(adloss, avar_all);
    //
    // f_grad
    adfn_t f_grad = adten_t::stop_recording(agrad, "f_grad");
    f_grad.optimize();
    //
    // previous_time
    elapsed_ms();
    //
    // initial_loss, t
    double initial_loss      = loss(L[0]).item<double>();
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // var_all
        vector<at::Tensor> var_all = f_grad.forward_var(L);
        //
        // grad
        vector<at::Tensor> grad = f_grad.get_range(var_all);
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
    // learn_ms
    double learn_ms = elapsed_ms();
    std::cout << "learn_ms = " << learn_ms << "\n";
    //
    // relative_loss
    double relative_loss = loss(L[0]).item<double>() / initial_loss;
    EXPECT_LT(relative_loss, expected_relative_loss);
}
// END_OPTIMIZE
//
// BEGIN_SRC_GEN
TEST(benchmarks, multi_normal_src_gen) {
    //
    // fs, plugin
    namespace fs     = std::filesystem;
    namespace plugin = ad_tensor::plugin;
    //
    // L
    vector<at::Tensor> L     = { initial_L.clone() };
    //
    // f_loss
    vector<adten_t> aL     = adten_t::start_recording(L);
    vector<adten_t> aloss  = { loss( aL[0] ) };
    adfn_t          f_loss = adten_t::stop_recording(aloss, "f_loss");
    //
    // aL
    aL     = adten_t::start_recording(L);
    //
    // avar_all
    vector<adten_t> avar_all = f_loss.forward_var(aL);
    //
    // agrad
    vector<adten_t> adloss = { adten_t( torch::tensor(1.0) ) };
    vector<adten_t> agrad  = f_loss.reverse_der(adloss, avar_all);
    //
    // f_grad
    adfn_t f_grad = adten_t::stop_recording(agrad, "f_grad");
    f_grad.optimize();
    //
    // source_path
    fs::path source_path  = fs::temp_directory_path() / "multi_normal";
    if( ! fs::is_directory(source_path) ) {
        fs::create_directory( source_path );
    }
    //
    // source_path: f_grad.cpp, f_grad.con
    f_grad.src_gen(source_path.string());
    //
    // src_gen_path/build
    bool quiet                   = true;
    bool use_installed_ad_tensor = false;
    plugin::build_lib(
        source_path, quiet, use_installed_ad_tensor
    );
    //
    // f_grad_plugin
    fs::path build_path        = source_path / "build";
    std::string plugin_lib     = "plugin_lib";
    std::string function_name  = f_grad.get_name();
    auto f_plugin = plugin::function_object(
        build_path, plugin_lib, function_name
    );
    //
    // previous_time
    elapsed_ms();
    //
    // dom_par, initial_loss, t
   vector<at::Tensor> dom_par;
    double initial_loss      = loss(L[0]).item<double>();
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // var_all
        vector<at::Tensor> var_all = f_grad.forward_var(L);
        //
        // grad
        vector<at::Tensor> grad = f_plugin(L, dom_par);
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
    // learn_ms
    double learn_ms = elapsed_ms();
    std::cout << "learn_ms = " << learn_ms << "\n";
    //
    // relative_loss
    double relative_loss = loss(L[0]).item<double>() / initial_loss;
    EXPECT_LT(relative_loss, expected_relative_loss);
}
// END_SRC_GEN
