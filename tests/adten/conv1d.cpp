// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad_tensor.hpp>
TEST(tests_adten, conv1d) {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // x_vec, w_vec, b_vec
    vector<float> x_vec = {1.0, 2.0, 3.0, 4.0};
    vector<float> w_vec = {0.25, 0.5, 0.25};
    vector<float> b_vec = {-1.0};
    //
    // x, w, b
    at::Tensor x = torch::tensor(x_vec).view( {1, 1, 4} );
    at::Tensor w = torch::tensor(w_vec).view( {1, 1, 3} );
    at::Tensor b = torch::tensor(b_vec);
    //
    // v
    vector<at::Tensor> v;
    v.push_back( x );
    v.push_back( w );
    v.push_back( b );
    //
    // adom_var
    vector<adten_t> av = adten_t::start_recording(v);
    //
    //
    // ax, aw, ab
    adten_t ax  = av[0];
    adten_t aw = av[1];
    adten_t ab = av[2];
    //
    // ay
    auto options   = torch::nn::functional::Conv1dFuncOptions();
    adten_t   ay = ad_tensor::conv1d(ax, aw, ab, options);
    //
    // r = f(v)
    vector<adten_t> ar = {ay};
    adfn_t f           = adten_t::stop_recording(ar, "f");
    //
    // var_all
    vector<at::Tensor> var_all = f.forward_var(v);
    //
    // r
    vector<at::Tensor> r = f.get_range(var_all);
    //
    // y
    at::Tensor y = r[0].contiguous();
    //
    // nx, nw, ny
    int64_t nx = x.sizes()[2];
    int64_t nw = w.sizes()[2];
    int64_t ny = y.sizes()[2];
    EXPECT_EQ( ny,  nx - nw + 1 );
    // ------------------------------------------------------------------------`
    // check function value
    // ------------------------------------------------------------------------`
    vector<float> y_vec(
        y.data_ptr<float>(),
        y.data_ptr<float>() + y.numel()
    );
    for(int64_t k = 0; k < ny; ++k) {
        float sum = b_vec[0];
        for(int64_t j = 0; j < nw; ++j) {
            sum += x_vec[k + j] * w_vec[j];
        }
        EXPECT_EQ( y_vec[k], sum );
    }
    // ------------------------------------------------------------------------`
    // check forward derivativre
    // ------------------------------------------------------------------------`
    //
    // zero_x, zero_w, zero_b, dv
    at::Tensor zero_x = torch::zeros( x.sizes() );
    at::Tensor zero_w = torch::zeros( w.sizes() );
    at::Tensor zero_b = torch::zeros( b.sizes() );
    vector<at::Tensor> dv = { zero_x, zero_w, zero_b };
    //
    // check partal of y w.r.t x[i]
    for(int64_t i = 0; i < nx; ++i) {
        //
        // dy
        dv[0]                 = torch::eye(nx).select(0, i).view( x.sizes() );
        vector<at::Tensor> dr = f.forward_der(dv, var_all);
        at::Tensor         dy = dr[0].contiguous();
        //
        // check
        vector<float> dy_vec(
            dy.data_ptr<float>(),
            dy.data_ptr<float>() + y.numel()
        );
        for(int64_t k = 0; k < ny; ++k) {
            if( k <= i && i < k + nw ) {
                EXPECT_EQ( dy_vec[k], w_vec[i - k] );
            } else {
                EXPECT_EQ( dy_vec[k], float( 0.0 ) );
            }
        }
    }
    dv[0] = zero_x;
    //
    // check partal of y w.r.t w[j]
    for(int64_t j = 0; j < nw; ++j) {
        //
        // dy
        dv[1]                 = torch::eye(nw).select(0, j).view( w.sizes() );
        vector<at::Tensor> dr = f.forward_der(dv, var_all);
        at::Tensor         dy = dr[0].contiguous();
        //
        // check
        vector<float> dy_vec(
            dy.data_ptr<float>(),
            dy.data_ptr<float>() + y.numel()
        );
        for(int64_t k = 0; k < ny; ++k) {
            EXPECT_EQ( dy_vec[k], x_vec[k + j] );
        }
    }
    dv[1] = zero_w;
    //
    // check derivative of y w.r.t. b
    {
        dv[2] = torch::tensor( {1.0} );
        vector<at::Tensor> dr = f.forward_der(dv, var_all);
        at::Tensor         dy = dr[0].contiguous();
        //
        // check
        vector<float> dy_vec(
            dy.data_ptr<float>(),
            dy.data_ptr<float>() + y.numel()
        );
        for(int64_t k = 0; k < ny; ++k) {
            EXPECT_EQ( dy_vec[k], float(1.0) );
        }
    }
    // ------------------------------------------------------------------------`
    // check reverse derivativre
    // ------------------------------------------------------------------------`
    //
    vector<at::Tensor> pr = { at::Tensor() };
    for(int64_t k = 0; k < ny; ++k) {
        //
        // pv
        pr[0]                 = torch::eye(ny).select(0, k).view( y.sizes() );
        vector<at::Tensor> pv = f.reverse_der(pr, var_all);
        //
        // check pb
        at::Tensor  pb = pv[2].contiguous();
        vector<float> pb_vec(
            pb.data_ptr<float>(),
            pb.data_ptr<float>() + 1
        );
        EXPECT_EQ( pb_vec[0], 1.0 );
        //
        // check pw
        at::Tensor  pw = pv[1].contiguous();
        vector<float> pw_vec(
            pw.data_ptr<float>(),
            pw.data_ptr<float>() + w.numel()
        );
        for(int64_t j = 0; j < nw; ++j) {
            EXPECT_EQ( pw_vec[j], x_vec[k+j] );
        }
        //
        // check px
        at::Tensor  px = pv[0].contiguous();
        vector<float> px_vec(
            px.data_ptr<float>(),
            px.data_ptr<float>() + x.numel()
        );
        for(int64_t i = 0; i < nw; ++i) {
            if( k <= i && i < k + nw ) {
                EXPECT_EQ( px_vec[i], w_vec[i-k] );
            } else {
                EXPECT_EQ( px_vec[i], float(0.0) );
            }
        }
    }
}
