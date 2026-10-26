// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
#include <torch/torch.h>
//
TEST(tests_adten, index)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // index_list_t
    //
    // from
    at::Tensor from = torch::tensor( { {1.0, 2.0}, {3.0, 4.0} } );
    //
    // index_list
    std::optional<at::Tensor>  col_index  = torch::tensor( {0, 1} );
    std::optional<at::Tensor>  row_index  = torch::tensor( {0, 1} );
    c10::List< std::optional<at::Tensor> > index_list = { col_index, row_index };
    //
    // x
    vector<at::Tensor> x = {from};
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // aextract
    adten_t& afrom   = ax[0];
    adten_t aextract = afrom.index(index_list);
    //
    // ay
    vector<adten_t> ay = { aextract };
    //
    // f(from) = from.index(index_list)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // from
    from = torch::tensor( { {6.0, 7.0}, {8.0, 9.0} } );
    x[0] = from;
    //
    // var_all
    vector<at::Tensor> var_all = f.forward_var(x);
    //
    // y
    vector<at::Tensor> y = f.get_range(var_all);
    //
    bool equal = y[0].equal( torch::tensor( {6.0, 9.0} ) );
    EXPECT_TRUE( equal );
    //
    // dx
    at::Tensor         dfrom  = torch::tensor( {{10.0, 11.0}, {12.0, 13.0} } );
    vector<at::Tensor> dx     = {dfrom};
    //
    // dy
    vector<at::Tensor> dy = f.forward_der(dx, var_all);
    equal = dy[0].equal( torch::tensor( {10.0, 13.0} ) );
    EXPECT_TRUE( equal );
    //
    // dy
    dy[0] = torch::tensor( {14.0, 15.0} );
    //
    // dx
    dx = f.reverse_der(dy, var_all);
    equal = dx[0].equal( torch::tensor( { {14.0, 0.0}, {0.0, 15.0} } ) );
    EXPECT_TRUE( equal );
}
// END_CPP
