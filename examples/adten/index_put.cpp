// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
#include <torch/torch.h>
//
TEST(examples_adten, index_put)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // index_list_t
    //
    // before
    at::Tensor before;
    before = torch::tensor( { {1.0, 2.0}, {3.0, 4.0} } );
    //
    // replace
    at::Tensor replace;
    replace = torch::tensor( { 0.0 } );
    //
    // row_index, col_index
    at::Tensor  row_index  = torch::tensor( {0} );
    at::Tensor  col_index  = torch::tensor( {1} );
    //
    // x
    vector<at::Tensor> x = {before, replace};
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // a_after
    adten_t& abefore   =  ax[0];
    adten_t& areplace   = ax[1];
    adten_t  arow_index = adten_t(row_index);
    adten_t  acol_index = adten_t(col_index);
    adten_t a_after = abefore.index_put({arow_index, acol_index}, areplace);
    //
    // ay
    vector<adten_t> ay = { a_after };
    //
    // f(before, replace) = before.index_put({row_index, col_index}, replace)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // replace
    replace = torch::tensor( {5.0} );
    x[1]    = replace;
    //
    // var_all
    vector<at::Tensor> var_all = f.forward_var(x);
    //
    // y
    vector<at::Tensor> y = f.get_range(var_all);
    //
    bool equal = y[0].equal( torch::tensor( { {1.0, 5.0}, {3.0, 4.0} } ) );
    EXPECT_TRUE( equal );
}
// END_CPP
