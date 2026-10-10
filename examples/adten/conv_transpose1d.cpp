// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad_tensor.hpp>
TEST(examples_adten, conv_transpose1d) {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    namespace functional = torch::nn::functional;
    //
    // y_vec, w_vec, b_vec
    vector<float> y_vec = {2.0};
    vector<float> w_vec = {0.15, 0.5, 0.35};
    vector<float> b_vec = {-1.0};
    //
    // y, w, b
    at::Tensor y = torch::tensor(y_vec).view( {1, 1, 1} );
    at::Tensor w = torch::tensor(w_vec).view( {1, 1, 3} );
    at::Tensor b = torch::tensor(b_vec);
    //
    // v
    vector<at::Tensor> v;
    v.push_back( y );
    v.push_back( w );
    v.push_back( b );
    //
    // adom_var
    vector<adten_t> av = adten_t::start_recording(v);
    //
    // ay, aw, ab
    adten_t ay = av[0];
    adten_t aw = av[1];
    adten_t ab = av[2];
    //
    // ax
    auto  tran_options = functional::ConvTranspose1dFuncOptions();
    adten_t ax         = ad_tensor::conv_transpose1d(ay, aw, ab, tran_options);
    //
    // r = f(v)
    vector<adten_t> ar = {ax};
    adfn_t          f  = adten_t::stop_recording(ar, "f");
    //
    // var_all
    vector<at::Tensor> var_all = f.forward_var(v);
    //
    // r
    vector<at::Tensor> r = f.get_range(var_all);
    //
    // y
    at::Tensor x = r[0].contiguous();
    //
    // ny, nw, nx
    int64_t ny = y.sizes()[2];
    int64_t nw = w.sizes()[2];
    int64_t nx = x.sizes()[2];
    EXPECT_EQ( nx,  ny + nw - 1 );
    // ------------------------------------------------------------------------`
    // check function value
    // ------------------------------------------------------------------------`
    vector<float> x_vec(
        x.data_ptr<float>(),
        x.data_ptr<float>() + x.numel()
    );
    for(int64_t i = 0; i < nx; ++i) {
        float sum = b_vec[0];
        int64_t k_begin = 0;
        if( i + 1 >= nw ) {
            k_begin = (i + 1) - nw;
        }
        int64_t k_end = i + 1;
        if( k_end > ny ) {
            k_end = ny;
        }
        for(int64_t k = k_begin; k < k_end ; ++k) {
            sum += y_vec[k] * w_vec[i - k];
        }
        EXPECT_EQ( x_vec[i], sum );
    }
    // ------------------------------------------------------------------------`
    // Alternative Method
    // ------------------------------------------------------------------------`
    auto pad_options = functional::PadFuncOptions(
        {nw-1, nw-1, 0, 0, 0, 0}
    );
    auto  conv_options = functional::Conv1dFuncOptions();
    at::Tensor y_hat = functional::pad(y, pad_options);
    at::Tensor w_hat = w.flip( {2} );
    at::Tensor x_hat = ad_tensor::conv1d( y_hat, w_hat, b, conv_options );
    EXPECT_TRUE( x.equal( x_hat ) );
}
// END_CPP
