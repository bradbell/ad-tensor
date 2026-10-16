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
    // x_vec, w_vec, b_vec
    vector<float> x_vec = {2.0f};
    vector<float> w_vec = {0.15f, 0.5f, 0.35f};
    vector<float> b_vec = {-1.0f};
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
    auto  tran_options = functional::ConvTranspose1dFuncOptions();
    adten_t ay         = ad_tensor::conv_transpose1d(ax, aw, ab, tran_options);
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
    size_t  ny = y.sizes()[2];
    EXPECT_EQ( ny,  nx + nw - 1 );
    // ------------------------------------------------------------------------`
    // check function value
    // ------------------------------------------------------------------------`
    vector<float> y_vec(
        y.data_ptr<float>(),
        y.data_ptr<float>() + y.numel()
    );
    for(size_t i = 0; i < ny; ++i) {
        float sum = b_vec[0];
        size_t k_first = size_t( std::max( int64_t(0), int64_t(i) - nw + 1 ) );
        size_t k_last  = size_t( std::min( int64_t(i), nx -1 ) );
        for(size_t k = k_first; k <= k_last ; ++k) {
            sum += x_vec[k] * w_vec[i - k];
        }
        EXPECT_EQ( y_vec[i], sum );
    }
    // ------------------------------------------------------------------------`
    // Alternative Method
    // ------------------------------------------------------------------------`
    auto pad_options = functional::PadFuncOptions(
        {nw-1, nw-1, 0, 0, 0, 0}
    );
    auto  conv_options = functional::Conv1dFuncOptions();
    at::Tensor x_hat = functional::pad(x, pad_options);
    at::Tensor w_hat = w.flip( {2} );
    at::Tensor y_hat = ad_tensor::conv1d( x_hat, w_hat, b, conv_options );
    EXPECT_TRUE( y.equal( y_hat ) );
}
// END_CPP
