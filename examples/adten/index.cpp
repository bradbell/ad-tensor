// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
#include <torch/torch.h>
//
TEST(examples_adten, index)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // from, row_index, col_index
    at::Tensor from = torch::tensor( { {1.0, 2.0}, {3.0, 4.0} } );
    at::Tensor row_index  = torch::tensor( {1, 1} );
    at::Tensor col_index  = torch::tensor( {0, 1} );
    // x
    vector<at::Tensor> x = {from, row_index, col_index};
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // aextract
    adten_t& afrom  = ax[0];
    adten_t aextract = afrom.index( {ax[1], ax[2]} );
    //
    // ay
    vector<adten_t> ay = { aextract };
    //
    // f(from) = from.index(index_list)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // from
    from      = torch::tensor( { {6.0, 7.0}, {8.0, 9.0} } );
    row_index = torch::tensor( {0, 0} );
    col_index = torch::tensor( {1, 0} );
    x[0] = from;
    x[1] = row_index;
    x[2] = col_index;
    //
    // var_all
    vector<at::Tensor> var_all = f.forward_var(x);
    //
    // y
    vector<at::Tensor> y = f.get_range(var_all);
    //
    bool equal = y[0].equal( torch::tensor( {7.0, 6.0} ) );
    EXPECT_TRUE( equal );
}
// END_CPP
