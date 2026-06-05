// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/ad.hpp>
//
namespace {
    using ad_tensor::ad_t;
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
        Tensor slope           = torch::zeros(1);
        Tensor intercept       = torch::zeros(1);
        vector<Tensor> dom_var = {slope, intercept};
        //
        // adom_par, adom_var
        auto [adom_par, adom_var] = ad_t::start_recording(dom_par, dom_var);
        //
        // ax, ay, aslope, aintercept
        ad_t ax         = adom_par[0];
        ad_t ay         = adom_par[1];
        ad_t aslope     = adom_var[0];
        ad_t aintercept = adom_var[1];
        //
        // asumsq
        ad_t amodel     = aslope * ax  + aintercept;
        ad_t aresidual  = ay - amodel;
        ad_t ares_sq    = aresidual * aresidual;
        ad_t asumsq     = ares_sq.sum();
        //
        // ares_sq = objective( (x,y), (slope,intercept) )
        vector<ad_t> arange = {asumsq};
        adfn_t f = ad_t::stop_recording(arange);
        //
        return f;
    }
}
//
TEST(examples_adfn, get_started)  {
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
    Tensor slope           = torch::tensor( 1.0 );
    Tensor intercept       = torch::tensor( 1.0 );
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
    // sumsq_slope_der
    vector<Tensor> dom_der = { torch::ones(1), torch::zeros(1) };
    vector<Tensor> rng_der = f.forward_der(par_all, var_all, dom_der, options);
    Tensor sumsq_slope_der = rng_der[0];
    //
    // sumsq_intercept_der
    dom_der = { torch::zeros(1), torch::ones(1) };
    rng_der = f.forward_der(par_all, var_all, dom_der, options);
    Tensor sumsq_intercept_der = rng_der[0];
    //
    // check
    check = - 2.0 * (residual * x).sum();
    EXPECT_EQ( sumsq_slope_der.item<float>(), check.item<float>() );
    check = - 2.0 * residual.sum();
    EXPECT_EQ( sumsq_intercept_der.item<float>(), check.item<float>() );
    //
    // sumsq_intercept_der, sumsq_slope_der
    rng_der   = { torch::ones(1) };
    dom_der   = f.reverse_der(par_all, var_all, rng_der, options);
    sumsq_slope_der     = dom_der[0];
    sumsq_intercept_der = dom_der[1];
    //
    // check
    check = - 2.0 * (residual * x).sum();
    EXPECT_EQ( sumsq_slope_der.item<float>(), check.item<float>() );
    check = - 2.0 * residual.sum();
    EXPECT_EQ( sumsq_intercept_der.item<float>(), check.item<float>() );
}
// END_CPP
