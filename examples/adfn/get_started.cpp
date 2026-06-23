// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/adten.hpp>
//
namespace {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::options_t;
    using ad_tensor::vector;
    using at::Tensor;
    //
    adfn_t objective(size_t n_data) {
        //
        // options
        options_t options;
        //
        // dom_par
        Tensor x               = torch::zeros(n_data);
        Tensor y               = torch::zeros(n_data);
        vector<Tensor> dom_par = {x, y};
        //
        // dom_var
        Tensor slope           = torch::tensor(0.0);
        Tensor intercept       = torch::tensor(0.0);
        vector<Tensor> dom_var = {slope, intercept};
        //
        // adom_par, adom_var
        auto [adom_par, adom_var] = adten_t::start_recording(dom_par, dom_var);
        //
        // ax, ay, aslope, aintercept
        adten_t ax         = adom_par[0];
        adten_t ay         = adom_par[1];
        adten_t aslope     = adom_var[0];
        adten_t aintercept = adom_var[1];
        //
        // asumsq
        adten_t amodel     = aslope * ax  + aintercept;
        adten_t aresidual  = ay - amodel;
        adten_t ares_sq    = aresidual * aresidual;
        adten_t asumsq     = ares_sq.sum();
        //
        // ares_sq = objective( (x,y), (slope,intercept) )
        vector<adten_t> arange = {asumsq};
        adfn_t f = adten_t::stop_recording(arange, "f");
        //
        return f;
    }
}
//
// get_started_first_derivative
TEST(examples_adfn, get_started_first_derivative)  {
    //
    // options
    options_t options;
    //
    // n_data
    size_t n_data = 3;
    //
    // sumsq = f( (x,y), (slope, intercept) )
    adfn_t f = objective(n_data);
    //
    // x, y, dom_par
    Tensor x               = torch::tensor( {0.0, 1.0, 2.0}  );
    Tensor y               = torch::tensor( {2.0, 5.0, 8.0} );
    vector<Tensor> dom_par = {x, y};
    //
    // dom_var
    Tensor slope           = torch::tensor(1.0);
    Tensor intercept       = torch::tensor(1.0);
    vector<Tensor> dom_var = {slope, intercept};
    //
    // par_all, var_all
    vector<Tensor> par_all = f.forward_par(dom_par, options);
    vector<Tensor> var_all = f.forward_var(par_all, dom_var, options);
    //
    // sumsq
    vector<Tensor> range = f.get_range(par_all, var_all);
    Tensor sumsq         = range[0];
    //
    // residual
    Tensor model    = slope * x + intercept;
    Tensor residual = y - model;
    //
    // check
    Tensor res_sq   = residual * residual;
    Tensor check    = res_sq.sum();
    EXPECT_EQ( sumsq.item<float>(), check.item<float>() );
    //
    // sumsq_intercept, sumsq_slope
    vector<Tensor> rng_der = { torch::tensor(1.0) };
    vector<Tensor> dom_der = f.reverse_der(par_all, var_all, rng_der, options);
    Tensor sumsq_slope     = dom_der[0];
    Tensor sumsq_intercept = dom_der[1];
    //
    // sumsq_slope
    check = - 2.0 * (residual * x).sum();
    EXPECT_EQ( sumsq_slope.item<float>(), check.item<float>() );
    //
    // sumsq_intercept
    check = - 2.0 * residual.sum();
    EXPECT_EQ( sumsq_intercept.item<float>(), check.item<float>() );
}
//
// get_started_second_derivative
TEST(examples_adfn, get_started_second_derivative)  {
    //
    // options
    options_t options;
    //
    // n_data
    size_t n_data = 3;
    //
    // sumsq = f( (x,y), (slope, intercept) )
    adfn_t f = objective(n_data);
    //
    // dom_par
    Tensor x               = torch::zeros(n_data);
    Tensor y               = torch::zeros(n_data);
    vector<Tensor> dom_par = {x, y};
    //
    // dom_var
    Tensor slope           = torch::tensor(0.0);
    Tensor intercept       = torch::tensor(0.0);
    vector<Tensor> dom_var = {slope, intercept};
    //
    // adom_par, adom_var
    auto [adom_par, adom_var] = adten_t::start_recording(dom_par, dom_var);
    //
    // g_0  = f_slope ( (x, y), (slope, intercept) )
    // g_1  = f_interceopt ( (x, y), (slope, intercept) )
    vector<adten_t>   apar_all = f.forward_par(adom_par, options);
    vector<adten_t>   avar_all = f.forward_var(apar_all, adom_var, options);
    vector<adten_t> arng_der   = { adten_t( torch::tensor(1.0) ) };
    vector<adten_t> adom_der   = f.reverse_der(
        apar_all, avar_all, arng_der, options
    );
    adfn_t g = adten_t::stop_recording(adom_der, "g");
    //
    // x, y, dom_par
    x       = torch::tensor( {0.0, 1.0, 2.0}  );
    y       = torch::tensor( {2.0, 5.0, 8.0} );
    dom_par = {x, y};
    //
    // dom_var
    slope     = torch::tensor(1.0);
    intercept = torch::tensor(1.0);
    dom_var   = {slope, intercept};
    //
    // par_all, var_all
    vector<Tensor> par_all = g.forward_par(dom_par, options);
    vector<Tensor> var_all = g.forward_var(par_all, dom_var, options);
    //
    // sumsq_slope_slope, sumsq_intercept_slope
    vector<Tensor> dom_der  = { torch::tensor(1.0), torch::tensor(0.0) };
    vector<Tensor> rng_der  = g.forward_der(par_all, var_all, dom_der, options);
    Tensor sumsq_slope_slope     = rng_der[0];
    Tensor sumsq_intercept_slope = rng_der[1];
    //
    // sumsq_slope_slope
    Tensor check = 2.0 * (x * x).sum();
    EXPECT_EQ( sumsq_slope_slope.item<float>(), check.item<float>() );
    //
    // sumsq_intercept_slope
    check = 2.0 * x.sum();
    EXPECT_EQ( sumsq_intercept_slope.item<float>(), check.item<float>() );
    //
    // sumsq_slope_intercept, sumsq_intercept_intercept
    dom_der  = { torch::tensor(0.0), torch::tensor(1.0) };
    rng_der  = g.forward_der(par_all, var_all, dom_der, options);
    Tensor sumsq_slope_intercept     = rng_der[0];
    Tensor sumsq_intercept_intercept = rng_der[1];
    //
    // sumsq_slope_intercept
    check = 2.0 * x.sum();
    EXPECT_EQ( sumsq_slope_intercept.item<float>(), check.item<float>() );
    //
    // sumsq_intercept_intercept
    check = 2.0 * torch::ones(n_data).sum();
    EXPECT_EQ( sumsq_slope_intercept.item<float>(), check.item<float>() );

}
// END_CPP
