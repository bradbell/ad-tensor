// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin fit_poly_benchmark ben}
{xrst_spell
    pytorch
    autograd
    gtest
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

{xrst_end fit_poly_benchmark}
*/
// BEGIN_COMMON
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad_tensor.hpp>
//
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
    template<class TensorType>
    TensorType loss(
        const vector<TensorType>& c ,
        const TensorType&         grid ,
        const TensorType&         data ) {
        assert( size_t( grid.numel() ) == number_grid_points );
        assert( size_t( data.numel() ) == number_grid_points );
        assert( size_t( c.size() )  == number_coefficients );
        int64_t n_data = data.numel();
        TensorType grid_p      = TensorType( torch::ones( {n_data} ) );
        TensorType predict = c[0] * grid_p;
        for(size_t j = 1; j < number_coefficients; ++j) {
            grid_p       = grid_p * grid;
            predict  = predict + c[j] * grid_p;
        }
        TensorType residual = (data - predict);
        return (residual * residual).sum();
    }
    //
    // grid, data
    torch::Tensor grid = torch::linspace(-1.0, 1.0, number_grid_points);
    torch::Tensor data = grid.exp();
}
// END_COMMON
//
// BEGIN_AUTOGRAD
TEST(benchmarks, fit_poly_autograd) {
    //
    // c
    vector<torch::Tensor> c;
    for(size_t j = 0; j < number_coefficients; ++j) {
        c.push_back( torch::randn( {1}, torch::requires_grad() ) );
    }
    //
    // previous_time
    elapsed_ms();
    //
    // initial_loss, t
    double initial_loss      = loss(c, grid, data).item<double>();
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // loss_t
        torch::Tensor loss_t = loss(c, grid, data);
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
    // learn_ms
    // For this case learn_ms is near equal the total time reported by gtest
    // double learn_ms = elapsed_ms();
    // std::cout << "learn_ms = " << learn_ms << "\n";
    //
    // relative_loss
    double relative_loss = loss(c, grid, data).item<double>() / initial_loss;
    EXPECT_LT(relative_loss, expected_relative_loss);
}
// END_AUTOGRAD
//
// BEGIN_AD_TENSOR
TEST(benchmarks, fit_poly_ad_tensor) {
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
    // agrid, adtat
    adten_t agrid(grid);
    adten_t adata(data);
    //
    // adfn
    vector<adten_t> aloss = { loss(ac, agrid, adata) };
    adfn_t adfn = adten_t::stop_recording(aloss, "adfn");
    //
    // previous_time
    elapsed_ms();
    //
    // dloss, initial_loss, t
    vector<at::Tensor> dloss = { torch::tensor(1.0) };
    double initial_loss      = loss(c, grid, data).item<double>();
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // var_all
        vector<at::Tensor> var_all = adfn.forward_var(c);
        //
        // grad
        vector<at::Tensor> grad  = adfn.reverse_der(dloss, var_all);
        //
        // c
        for(size_t j = 0; j < number_coefficients; ++j) {
            c[j] -= learning_rate * grad[j];
        }
    }
    //
    // learn_ms
    // For this case learn_ms is near equal the total time reported by gtest
    // double learn_ms = elapsed_ms();
    //std::cout << "learn_ms = " << learn_ms << "\n";
    //
    // relative_loss
    double relative_loss = loss(c, grid, data).item<double>() / initial_loss;
    EXPECT_LT(relative_loss, expected_relative_loss);
}
// END_AD_TENSOR
//
// BEGIN_OPTIMIZE
TEST(benchmarks, fit_poly_optimize) {
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
    // agrid, adata
    adten_t agrid(grid);
    adten_t adata(data);
    //
    // f_loss
    vector<adten_t> aloss = { loss(ac, agrid, adata) };
    adfn_t f_loss = adten_t::stop_recording(aloss, "f_loss");
    //
    // ac
    ac = adten_t::start_recording(c);
    //
    // adloss
    vector<adten_t> adloss = { adten_t( torch::tensor(1.0) ) };
    //
    // avar_all
    vector<adten_t> avar_all = f_loss.forward_var(ac);
    //
    // agrad
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
    double initial_loss      = loss(c, grid, data).item<double>();
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // var_all
        vector<at::Tensor> var_all = f_grad.forward_var(c);
        //
        // grad
        vector<at::Tensor> grad  = f_grad.get_range(var_all);
        //
        // c
        for(size_t j = 0; j < number_coefficients; ++j) {
            c[j] -= learning_rate * grad[j];
        }
    }
    //
    // learn_ms
    // For this case learn_ms is near equal the total time reported by gtest
    // double learn_ms = elapsed_ms();
    // std::cout << "learn_ms = " << learn_ms << "\n";
    //
    // relative_loss
    double relative_loss = loss(c, grid, data).item<double>() / initial_loss;
    EXPECT_LT(relative_loss, expected_relative_loss);
}
// END_OPTIMIZE
//
// BEGIN_SRC_GEN
#if INCLUDE_PLUGIN
TEST(benchmarks, fit_poly_src_gen) {
    //
    // fs, plugin
    namespace fs     = std::filesystem;
    namespace plugin = ad_tensor::plugin;
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
    // agrid, adata
    adten_t agrid(grid);
    adten_t adata(data);
    //
    // f_loss
    vector<adten_t> aloss = { loss(ac, agrid, adata) };
    adfn_t f_loss = adten_t::stop_recording(aloss, "f_loss");
    //
    // ac
    ac = adten_t::start_recording(c);
    //
    // adloss
    vector<adten_t> adloss = { adten_t( torch::tensor(1.0) ) };
    //
    // avar_all
    vector<adten_t> avar_all = f_loss.forward_var(ac);
    //
    // agrad
    vector<adten_t> agrad  = f_loss.reverse_der(adloss, avar_all);
    //
    // f_grad
    adfn_t f_grad = adten_t::stop_recording(agrad, "f_grad");
    f_grad.optimize();
    //
    // source_path
    fs::path source_path  = fs::temp_directory_path() / "fit_poly";
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
    double initial_loss      = loss(c, grid, data).item<double>();
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // var_all
        vector<at::Tensor> var_all = f_grad.forward_var(c);
        //
        // grad
        vector<at::Tensor> grad  = f_plugin(c, dom_par);
        //
        // c
        for(size_t j = 0; j < number_coefficients; ++j) {
            c[j] -= learning_rate * grad[j];
        }
    }
    //
    // learn_ms
    double learn_ms = elapsed_ms();
    std::cout << "learn_ms = " << learn_ms << "\n";
    //
    // relative_loss
    double relative_loss = loss(c, grid, data).item<double>() / initial_loss;
    EXPECT_LT(relative_loss, expected_relative_loss);
}
#endif
// END_SRC_GEN
