// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <ad_tensor/defined.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/adfn.hpp>
#include <torch/torch.h>
#include <gtest/gtest.h>
//
TEST(tests_adten, split)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    using ad_tensor::not_defined;
    //
    // split_size, n_split, dim
    vector<int64_t> split_size( {1, 2, 3} );
    size_t          n_split = split_size.size();
    int64_t         dim = 1;
    //
    // p
    vector<at::Tensor> p = { torch::tensor( {
        {1, 2, 3,  4,  5,  6},
        {7, 8, 9, 10, 11, 12},
    } ) };
    //
    // x
    vector<at::Tensor> x = { torch::tensor( {
        {1, 2, 3,  4,  5,  6},
        {7, 8, 9, 10, 11, 12},
    } ) };
    //
    // ap, ax
    auto [ax, ap] = adten_t::start_recording(x, p);
    //
    // ay
    vector<adten_t> ap_split = ap[0].split(split_size, dim);
    vector<adten_t> ax_split = ax[0].split(split_size, dim);
    vector<adten_t> ay;
    for(size_t i = 0; i < n_split; ++i) {
        ay.push_back( ax_split[i] );
    }
    for(size_t i = 0; i < n_split; ++i) {
        ay.push_back( ap_split[i] );
    }
    //
    // y = f(x, p)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // par_all, var_all
    vector<at::Tensor> par_all = f.forward_var(p);
    vector<at::Tensor> var_all = f.forward_var(x, par_all);
    // ---------------------------------------------------------------------
    // check function value
    // ---------------------------------------------------------------------
    //
    // y
    vector<at::Tensor> y = f.get_range(var_all, par_all);
    //
    EXPECT_EQ( y.size(), 2 * n_split );
    //
    EXPECT_TRUE( y[0].equal( torch::tensor( {
        {1},
        {7},
    } ) ) );
    //
    EXPECT_TRUE(y[1].equal( torch::tensor( {
        {2, 3},
        {8, 9},
    } ) ) );
    //
    EXPECT_TRUE( y[2].equal( torch::tensor( {
        { 4,  5,  6},
        {10, 11, 12},
    } ) ) );
    //
    for(size_t i = 0; i < n_split; ++i) {
        EXPECT_TRUE( y[n_split + i].equal( y[i] ) );
    }
    // ----------------------------------------------------------------------
    // check forward_der
    // ----------------------------------------------------------------------
    // dx
    vector<at::Tensor> dx = { - x[0] };
    //
    // dy
    vector<at::Tensor> dy = f.forward_der(dx, var_all, par_all);
    for(size_t i = 0; i < n_split; ++i) {
        EXPECT_TRUE( dy[i].equal( -y[i]  ) );
    }
    for(size_t i = 0; i < n_split; ++i) {
        EXPECT_TRUE( not_defined( dy[n_split + i] ) );
    }
    // ----------------------------------------------------------------------
    // check reverse_der
    // ----------------------------------------------------------------------
    // py, px
    vector<at::Tensor> py = {
        - y[0], -y[1], -y[2], not_defined(), not_defined(), not_defined()
    };
    vector<at::Tensor> px = f.reverse_der(py, var_all, par_all);
    //
    EXPECT_TRUE( px[0].equal( - x[0] ) );
}
// END_CPP
