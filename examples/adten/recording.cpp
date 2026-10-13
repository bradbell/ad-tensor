// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adten, record)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // dom_par, dom_var
    vector<at::Tensor> dom_par = { torch::tensor( {2.0, 3.0} ) };
    vector<at::Tensor> dom_var = { torch::tensor( {4.0, 5.0} ) };
    //
    // adom_var, adom_par
    auto [ adom_var, adom_par ] = adten_t::start_recording(
        dom_var, dom_par
    );
    //
    // arange
    vector<adten_t> arange = { adom_par[0] + adom_var[0] };
    //
    // stop_recording
    const char* name = "f";
    adfn_t f = adten_t::stop_recording(arange, name );
    //
    // range
    dom_par[0] = torch::tensor( {5.0, 6.0} );
    vector<at::Tensor> par_all = f.forward_par(dom_par);
    vector<at::Tensor> var_all = f.forward_var(dom_var, par_all);
    vector<at::Tensor> range   = f.get_range(var_all, par_all);
    //
    EXPECT_EQ( range.size(), 1 );
    EXPECT_TRUE( range[0].equal( torch::tensor( {9.0, 11.0} ) ) );
}
// END_CPP
