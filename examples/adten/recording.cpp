// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/adten.hpp>
//
TEST(examples_ad, record)  {
    using ad_tensor::adten_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // dom_par, dom_var
    vector<Tensor> dom_par = { torch::tensor( {2.0, 3.0} ) };
    vector<Tensor> dom_var = { torch::tensor( {4.0, 5.0} ) };
    //
    // adom_var, adom_par
    auto [ adom_var, adom_par ] = adten_t::start_recording(
        dom_var, dom_par
    );
    //
    // dom_par, dom_var
    // change from unspecified state
    dom_par = vector<Tensor>();
    dom_var = vector<Tensor>();
    //
    // arange
    vector<adten_t> arange = { adom_par[0], adom_var[0] };
    //
    // stop_recording
    adten_t::stop_recording(arange, "" );
    //
    bool equal = adom_par[0].tensor().equal( torch::tensor( {2.0, 3.0} ) );
    EXPECT_TRUE(equal);
    //
    equal = adom_var[0].tensor().equal( torch::tensor( {4.0, 5.0} ) );
    EXPECT_TRUE(equal);
}
// END_CPP
