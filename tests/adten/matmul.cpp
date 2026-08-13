// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/empty_at_ten.hpp>
//
#define SCALAR_EQ(x, y) \
    EXPECT_EQ( (x).item<double>(), (y).item<double>() )
//
TEST(tests_adten, matmul_mat_vec)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // vec
    Tensor vec  = torch::tensor( {2.0, 3.0} );
    //
    // mat
    Tensor mat = torch::tensor( { {4.0, 5.0}, {6.0, 7.0} } );
    //
    // v
    vector<Tensor> v = {vec, mat};
    //
    // av
    vector<adten_t> av = adten_t::start_recording(v);
    //
    // avec, amat
    adten_t& avec = av[0];
    adten_t& amat = av[1];
    //
    // avec
    // matmul does not support one dimensional arguments
    int64_t raw_array_two[] = {2,1};
    c10::IntArrayRef shape_two(raw_array_two, 2);
    avec = avec.view(shape_two);
    //
    // prod
    // convert to a one dimensional result
    adten_t prod = amat.matmul( avec );
    int64_t raw_array_one[] = {2};
    c10::IntArrayRef shape_one(raw_array_one, 1);
    prod = prod.view(shape_one);
    //
    vector<adten_t> ay;
    ay.push_back( prod );  // y[0] = mat * vec
    //
    // y = f(x)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // var_all
    vector<Tensor> var_all = f.forward_var(v);
    //
    // y
    vector<Tensor> y = f.get_range(var_all);
    //
    // empty
    Tensor empty = ad_tensor::empty_at_ten();
    //
    //
    // y[0]
    EXPECT_EQ( y[0].sizes().size(), 1);
    SCALAR_EQ( y[0][0], mat[0][0] * vec[0] + mat[0][1] * vec[1] );
    SCALAR_EQ( y[0][1], mat[1][0] * vec[0] + mat[1][1] * vec[1] );
    //
    // dy[0]
    vector<Tensor> dv( v.size(), empty );
    dv[0] = torch::tensor( {8.0, 9.0} );
    vector<Tensor> dy = f.forward_der(dv, var_all);
    EXPECT_EQ( dy[0].sizes().size(), 1);
    SCALAR_EQ( dy[0][0], mat[0][0] * dv[0][0] + mat[0][1] * dv[0][1] );
    SCALAR_EQ( dy[0][1], mat[1][0] * dv[0][0] + mat[1][1] * dv[0][1] );
    //
    // dv[0]
    dy[0] = torch::tensor( {8.0, 9.0} );
    dv    = f.reverse_der(dy, var_all);
    EXPECT_EQ( dv[0].sizes().size(), 1);
    SCALAR_EQ( dv[0][0], dy[0][0] * mat[0][0] + dy[0][1] * mat[1][0] );
    SCALAR_EQ( dv[0][1], dy[0][0] * mat[0][1] + dy[0][1] * mat[1][1] );
}
//
TEST(tests_adten, matmul_batch)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // batch_vec, batch_shape
    Tensor batch_vec  = torch::tensor({
        {{ 2.0, 3.0 }}, {{ 4.0, 5.0 }}, {{ 6.0, 7.0 }}
    });
    c10::IntArrayRef batch_shape = batch_vec.sizes();
    EXPECT_EQ( batch_shape.size(), 3);
    EXPECT_EQ( batch_shape[0], 3 );
    EXPECT_EQ( batch_shape[1], 1 );
    EXPECT_EQ( batch_shape[2], 2);
    //
    // mat
    Tensor mat = torch::tensor( { {10.0, 11.0}, {12.0, 13.0} } );
    //
    // v
    vector<Tensor> v = {batch_vec, mat};
    //
    // av
    vector<adten_t> av = adten_t::start_recording(v);
    //
    // abatch_vec, amat
    adten_t& abatch_vec = av[0];
    adten_t& amat       = av[1];
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( abatch_vec.matmul( amat ) );  // y[0] = batch_vec * mat
    //
    // y = f(x)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // var_all
    vector<Tensor> var_all = f.forward_var(v);
    //
    // y
    vector<Tensor> y = f.get_range(var_all);
    //
    // empty
    Tensor empty = ad_tensor::empty_at_ten();
    //
    //
    // y[0]
    EXPECT_EQ( y[0].sizes().size(), 3);
    EXPECT_EQ( y[0].sizes()[0], 3 );
    EXPECT_EQ( y[0].sizes()[1], 1 );
    EXPECT_EQ( y[0].sizes()[2], 2);
    for(size_t k = 0; k < 3; ++k) {
        Tensor yk = y[0][k];
        Tensor vk = batch_vec[k];
        SCALAR_EQ( yk[0][0], vk[0][0] * mat[0][0] + vk[0][1] * mat[1][0] );
        SCALAR_EQ( yk[0][1], vk[0][0] * mat[0][1] + vk[0][1] * mat[1][1] );
    }
    //
    // dy[0]
    vector<Tensor> dv( v.size(), empty );
    Tensor dmat = torch::tensor( { {20.0, 21.0}, {22.0, 23.0} } );
    dv[1]       = dmat;
    vector<Tensor> dy = f.forward_der(dv, var_all);
    for(size_t k = 0; k < 3; ++k) {
        Tensor dyk = dy[0][k];
        Tensor vk  = batch_vec[k];
        SCALAR_EQ( dyk[0][0], vk[0][0] * dmat[0][0] + vk[0][1] * dmat[1][0] );
        SCALAR_EQ( dyk[0][1], vk[0][0] * dmat[0][1] + vk[0][1] * dmat[1][1] );
    }
    //
    // dv
    dy[0] = torch::tensor({
        {{ 12.0, 13.0 }}, {{ 14.0, 15.0 }}, {{ 16.0, 17.0 }}
    });
    dv         = f.reverse_der(dy, var_all);
    //
    // dv[0]
    for(size_t k = 0; k < 3; ++k) {
        Tensor dyk   = dy[0][k];
        Tensor dvk   = dv[0][k];
        //
        SCALAR_EQ( dvk[0][0], dyk[0][0] * mat[0][0] + dyk[0][1] * mat[0][1] );
        SCALAR_EQ( dvk[0][1], dyk[0][0] * mat[1][0] + dyk[0][1] * mat[1][1] );
    }
    //
    // dv[1]
    Tensor sum = torch::zeros( {2, 2} );
    for(size_t k = 0; k < 3; ++k) {
        Tensor dyk   = dy[0][k];
        Tensor vk    = batch_vec[k];
        //
        sum[0][0] = sum[0][0] + dyk[0][0] * vk[0][0];
        sum[1][0] = sum[1][0] + dyk[0][0] * vk[0][1];
        sum[0][1] = sum[0][1] + dyk[0][1] * vk[0][0];
        sum[1][1] = sum[1][1] + dyk[0][1] * vk[0][1];
    }
    EXPECT_TRUE( dv[1].equal( sum ) );
}
// END_CPP
