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
    using at::Tensor;
    using ad_tensor::vector;
    //
    // index_list_t
    //
    // before
    Tensor before;
    before = torch::tensor( { {1.0, 2.0}, {3.0, 4.0} } );
    //
    // replace
    Tensor replace;
    replace = torch::tensor( { 0.0 } );
    //
    // index_list
    std::optional<Tensor>  col_index  = torch::tensor( {0} );
    std::optional<Tensor>  row_index  = torch::tensor( {1} );
    c10::List< std::optional<Tensor> > index_list = { col_index, row_index };
    //
    // x
    vector<Tensor> x = {before, replace};
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // a_after
    adten_t& a_before  = ax[0];
    adten_t& a_replace = ax[1];
    adten_t a_after    = a_before.index_put(index_list, a_replace);
    //
    // ay
    vector<adten_t> ay = { a_after };
    //
    // after = before.index_put(index_list, replace)
    adfn_t f = adten_t::stop_recording(ay, "f");
    f.set_trace(true);
    f.print_con();
    //
    // replace
    replace = torch::tensor( {5.0} );
    x[1]    = replace;
    //
    // var_all
    vector<Tensor> var_all = f.forward_var(x);
    //
    // y
    vector<Tensor> y = f.get_range(var_all);
    //
    bool equal = y[0].equal( torch::tensor( { {1.0, 5.0}, {3.0, 4.0} } ) );
    EXPECT_TRUE( equal );
    //
    // dx
    Tensor         dbefore    = torch::tensor( {{6.0, 7.0}, {8.0, 9.0} } );
    Tensor         dreplace   = torch::tensor( { 10.0 } );
    vector<Tensor> dx         = {dbefore, dreplace};
    //
    // dy
    vector<Tensor> dy = f.forward_der(dx, var_all);
    equal = dy[0].equal( torch::tensor( { {6.0, 10.0}, {8.0, 9.0} } ) );
    EXPECT_TRUE( equal );
    //
    // TODO: Implement and and add reverse_der example for this operator.
    // (This requires the index operator.)
}
// END_CPP
