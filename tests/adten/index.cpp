// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/defined.hpp>
#include <gtest/gtest.h>
#include <ad_tensor/adten.hpp>
#include <torch/torch.h>
//
TEST(tests_adten, index)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // from, row_index, col_index
    at::Tensor from = torch::tensor( { {1.0, 2.0}, {3.0, 4.0} } );
    at::Tensor row_index  = torch::tensor( {0, 1} );
    at::Tensor col_index  = torch::tensor( {0, 1} );
    //
    // x
    vector<at::Tensor> x = {from, row_index, col_index};
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // aextract
    adten_t&         afrom      = ax[0];
    vector<adten_t>  aindex_vec = {ax[1], ax[2]};
    adten_t aextract = afrom.index(aindex_vec);
    //
    // ay
    vector<adten_t> ay = { aextract };
    //
    // f(from, row_index, col_index)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // var_all
    vector<at::Tensor> var_all = f.forward_var(x);
    //
    // y
    vector<at::Tensor> y = f.get_range(var_all);
    //
    bool equal = y[0].equal( torch::tensor( {1.0, 4.0} ) );
    EXPECT_TRUE( equal );
    //
    // dx
    at::Tensor         dfrom      = torch::tensor( {{5.0, 6.0},{7.0, 8.0}} );
    at::Tensor         drow_index = torch::tensor( {9.0, 9.0} );
    at::Tensor         dcol_index = torch::tensor( {10.0, 11.0} );
    vector<at::Tensor> dx         = {dfrom, drow_index, dcol_index};
    //
    // dy
    vector<at::Tensor> dy = f.forward_der(dx, var_all);
    equal = dy[0].equal( torch::tensor( {5.0, 8.0} ) );
    EXPECT_TRUE( equal );
    //
    // py
    vector<at::Tensor> py = { torch::tensor( {12.0, 13.0} ) };
    vector<at::Tensor> px = f.reverse_der(py, var_all);
    EXPECT_TRUE( px[0].equal( torch::tensor( {{12.0, 0.0}, {0.0, 13.0}} ) ) );
    EXPECT_TRUE( ad_tensor::not_defined( px[1] ) );
    EXPECT_TRUE( ad_tensor::not_defined( px[2] ) );
}
