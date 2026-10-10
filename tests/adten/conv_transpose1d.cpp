// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad_tensor.hpp>
TEST(tests_adten, conv_transpose1d) {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // x_vec, w_vec, b_vec
    vector<float> x_vec = {2.0};
    vector<float> w_vec = {0.25, 0.5, 0.25};
    vector<float> b_vec = {-1.0};
    //
    // y, w, b
    at::Tensor x = torch::tensor(x_vec).view( {1, 1, 1} );
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
    // ax, aw, ab
    adten_t ax = av[0];
    adten_t aw = av[1];
    adten_t ab = av[2];
    //
    // ay
    auto    options  = torch::nn::functional::ConvTranspose1dFuncOptions();
    adten_t ay       = ad_tensor::conv_transpose1d(ax, aw, ab, options);
    //
    // r = f(v)
    vector<adten_t> ar = {ay};
    adfn_t          f  = adten_t::stop_recording(ar, "f");
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
    EXPECT_EQ( ny,  nx + nw - 1 );
    // ------------------------------------------------------------------------`
    // check function value
    // ------------------------------------------------------------------------`
    vector<float> y_vec(
        y.data_ptr<float>(),
        y.data_ptr<float>() + y.numel()
    );
    for(int64_t i = 0; i < ny; ++i) {
        float sum = b_vec[0];
        int64_t k_begin = 0;
        if( i + 1 >= nw ) {
            k_begin = (i + 1) - nw;
        }
        int64_t k_end = i + 1;
        if( k_end > nx ) {
            k_end = nx;
        }
        for(int64_t k = k_begin; k < k_end ; ++k) {
            sum += x_vec[k] * w_vec[i - k];
        }
        EXPECT_EQ( y_vec[i], sum );
    }
}
// END_CPP
