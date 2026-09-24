// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin smooth_spline_benchmark ben}
{xrst_spell
    autograd
    gtest
    libtorch
}

Fitting A Cubic Smoothing Spline Benchmark
##########################################

Loss Function
*************
Let :math:`x \in {\rm R}^m` be a uniform grid of points in
:math:`[- \pi , + \pi ]` .
The loss function :math:`f : {\rm R}^m \rightarrow {\rm R}`
for this example is defined by

.. math::

    f(y) = \sum_{i=0}^{m-1} \left( \sin( x_i ) - y_i \right)^2
         + \sum_{i=0)^{m-3) \left( y_i - 2 y_{i+1} + y_{i+2} \right)^2

This objective is a finite difference approximation for the
second derivative in a cubic
`Smoothing spline <https://en.wikipedia.org/wiki/Smoothing_spline>`_ ,

learn_ms
********
In the code below, gtest reports the total time for each test.
The plugin test also prints the value learn_ms, which is the time
in milliseconds for the learning loop; i.e., it does not include
the time to compile and link the plugin.

Common Code
***********
{xrst_literal ,
    BEGIN_COMMON, END_COMMON
}

Autograd Code
*************
This case uses Libtorch's autograd to compute gradients of the loss function.
{xrst_literal ,
    BEGIN_AUTOGRAD, END_AUTOGRAD
}

AD Tensor Code
**************
This case records the loss function and
uses reverse mode to compute gradients.
{xrst_literal ,
    BEGIN_AD_TENSOR, END_AD_TENSOR
}

AD Tensor Record Gradient
*************************
This case records the reverse mode gradient and uses it to compute gradients,
{xrst_literal ,
    BEGIN_RECORD_GRADIENT, END_RECORD_GRADIENT
}

AD Tensor With Plugin
*********************
This case records the reverse mode gradient,
generates its source code, compiles and links its source code,
and then uses its plugin to compute gradients,
{xrst_literal ,
    BEGIN_PLUGIN, END_PLUGIN
}

{xrst_end smooth_spline_benchmark}
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
    // chrono, vector, adten_t, adfn_t, Conv1dFuncOptions
    namespace chrono = std::chrono;
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using torch::nn::functional::Conv1dFuncOptions;
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
    // number_grid_points
    const size_t number_grid_points    = 2000;
    //
    // expected_relative_loss
    const double expected_relative_loss = 1e-3;
    //
    // learning_rate, number_learning_steps
    const double learning_rate         = 1e-5;
    const size_t number_learning_steps = 5000;
    //
    // conv1d
    adten_t conv1d(
        const adten_t&           input   ,
        const adten_t&           weight  ,
        const Conv1dFuncOptions& options ) {
        adten_t bias = adten_t( at::Tensor() );
        return ad_tensor::conv1d(input, weight, bias, options);
    }
    //
    // data, weight
    double pi    = 3.1412592653;
    double dx    = 2.0 * pi / double(number_grid_points - 1);
    double dx_sq = dx * dx;
    double scale = 1.0 / ( dx_sq * double( number_grid_points) );
    at::Tensor grid      = torch::linspace(-pi, pi, number_grid_points);
    at::Tensor data      = grid.sin();
    auto       options   = Conv1dFuncOptions();
    at::Tensor weight_at =
        torch::tensor({1.0, -2.0, 1.0}).view({1, 1, 3}) * torch::tensor(scale);
    //
    // loss
    template<class TensorType>
    TensorType loss(const TensorType y ) {
        TensorType weight      = TensorType( weight_at );
        TensorType finite_diff = conv1d(y, weight, options);
        TensorType data_res    = (TensorType(data) - y);
        return (finite_diff*finite_diff).sum() + (data_res*data_res).sum();
    }
}
// END_COMMON
//
// BEGIN_AUTOGRAD
TEST(benchmarks, smooth_spline_autograd) {
    //
    // y
    torch::Tensor y = torch::rand(
        {1, 1, number_grid_points}, torch::requires_grad()
    );
    //
    // previous_time
    elapsed_ms();
    //
    // initial_loss, t
    double initial_loss = loss(y).item<double>();
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // loss_t
        torch::Tensor loss_t = loss(y);
        //
        // y
        loss_t.backward();
        {   torch::NoGradGuard no_grad;
            //
            y -= learning_rate * y.grad();
            y.grad().zero_();
        }
    }
    //
    // learn_ms
    // For this case learn_ms is near equal the total time reported by gtest
    // double learn_ms = elapsed_ms();
    // std::cout << "learn_ms = " << learn_ms << "\n";
    //
    // relative_loss
    double relative_loss = loss(y).item<double>() / initial_loss;
    EXPECT_LT(relative_loss, expected_relative_loss);
}
// END_AUTOGRAD
//
// BEGIN_AD_TENSOR
TEST(benchmarks, smooth_spline_ad_tensor) {
    //
    // y
    vector<at::Tensor> y;
    y.push_back( torch::rand( {1, 1, number_grid_points} ) );
    //
    // ay
    vector<adten_t> ay = adten_t::start_recording(y);
    //
    // adfn
    vector<adten_t> aloss = { loss(ay[0]) };
    adfn_t adfn = adten_t::stop_recording(aloss, "adfn");
    adfn.optimize();
    //
    // previous_time
    elapsed_ms();
    //
    // dloss, initial_loss, t
    vector<at::Tensor> dloss = { torch::tensor(1.0) };
    double initial_loss      = loss(y[0]).item<double>();
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // var_all
        vector<at::Tensor> var_all = adfn.forward_var(y);
        //
        // grad
        vector<at::Tensor> grad  = adfn.reverse_der(dloss, var_all);
        //
        // y
        y[0] -= learning_rate * grad[0];
    }
    //
    // learn_ms
    // For this case learn_ms is near equal the total time reported by gtest
    // double learn_ms = elapsed_ms();
    //std::cout << "learn_ms = " << learn_ms << "\n";
    //
    // relative_loss
    double relative_loss = loss(y[0]).item<double>() / initial_loss;
    EXPECT_LT(relative_loss, expected_relative_loss);
}
// END_AD_TENSOR
//
/* BEGIN_TODO: run other cases once conv1d is implemented for AD tensors
//
// BEGIN_RECORD_GRADIENT
TEST(benchmarks, smooth_spline_record_gradient) {
    //
    // y
    vector<at::Tensor> y;
    y.push_back( torch::rand( {1, 1, number_grid_points} ) );
    //
    // ay
    vector<adten_t> ay = adten_t::start_recording(y);
    //
    // f_loss
    vector<adten_t> aloss = { loss(ay[0]) };
    adfn_t f_loss = adten_t::stop_recording(aloss, "f_loss");
    //
    // ay
    ay = adten_t::start_recording(y);
    //
    // adloss
    vector<adten_t> adloss = { adten_t( torch::tensor(1.0) ) };
    //
    // avar_all
    vector<adten_t> avar_all = f_loss.forward_var(ay);
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
    double initial_loss      = loss(y[0]).item<double>();
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // var_all
        vector<at::Tensor> var_all = f_grad.forward_var(y);
        //
        // grad
        vector<at::Tensor> grad  = f_grad.get_range(var_all);
        //
        // y
        y[0] -= learning_rate * grad[0];
    }
    //
    // learn_ms
    // For this case learn_ms is near equal the total time reported by gtest
    // double learn_ms = elapsed_ms();
    // std::cout << "learn_ms = " << learn_ms << "\n";
    //
    // relative_loss
    double relative_loss = loss(y[0]).item<double>() / initial_loss;
    EXPECT_LT(relative_loss, expected_relative_loss);
}
// END_RECORD_GRADIENT
//
// BEGIN_PLUGIN
#if INCLUDE_PLUGIN
TEST(benchmarks, smooth_spline_plugin) {
    //
    // fs, plugin
    namespace fs     = std::filesystem;
    namespace plugin = ad_tensor::plugin;
    //
    // y
    vector<at::Tensor> y;
    y.push_back( torch::rand( {1, 1, number_grid_points} ) );
    //
    // ay
    vector<adten_t> ay = adten_t::start_recording(y);
    //
    // f_loss
    vector<adten_t> aloss = { loss(ay[0]) };
    adfn_t f_loss = adten_t::stop_recording(aloss, "f_loss");
    //
    // ay
    ay = adten_t::start_recording(y);
    //
    // adloss
    vector<adten_t> adloss = { adten_t( torch::tensor(1.0) ) };
    //
    // avar_all
    vector<adten_t> avar_all = f_loss.forward_var(ay);
    //
    // agrad
    vector<adten_t> agrad  = f_loss.reverse_der(adloss, avar_all);
    //
    // f_grad
    adfn_t f_grad = adten_t::stop_recording(agrad, "f_grad");
    f_grad.optimize();
    //
    // source_path
    fs::path source_path  = fs::temp_directory_path() / "smooth_spline";
    if( ! fs::is_directory(source_path) ) {
        fs::create_directory( source_path );
    }
    try { fs::permissions(
        source_path, fs::perms::owner_all, fs::perm_options::replace
    ); } catch (...) {
        // cannot change permissions for source_path directory
        EXPECT_TRUE(false);
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
    auto f_plugin = plugin::src_gen_fun(
        build_path, plugin_lib, function_name
    );
    //
    // previous_time
    elapsed_ms();
    //
    // dom_par, initial_loss, t
    vector<at::Tensor> dom_par;
    double initial_loss      = loss(y[0]).item<double>();
    for(size_t t = 0; t < number_learning_steps; ++t) {
        //
        // var_all
        vector<at::Tensor> var_all = f_grad.forward_var(y);
        //
        // grad
        vector<at::Tensor> grad  = f_plugin(y, dom_par);
        //
        // y
        y[0] -= learning_rate * grad[0];
    }
    //
    // learn_ms
    double learn_ms = elapsed_ms();
    std::cout << "learn_ms = " << learn_ms << "\n";
    //
    // relative_loss
    double relative_loss = loss(y[0]).item<double>() / initial_loss;
    EXPECT_LT(relative_loss, expected_relative_loss);
}
#endif
END_TODO */
// END_PLUGIN
