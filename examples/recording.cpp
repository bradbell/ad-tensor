// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/ad.hpp>
//
TEST(examples, record)  {
    using ad_tensor::ad_t;
    using at::Tensor;
    using std::vector;
    //
    // dom_par, dom_var
    vector<Tensor> dom_par = { torch::tensor( {2.0, 3.0} ) };
    vector<Tensor> dom_var = { torch::tensor( {4.0, 5.0} ) };
    //
    // adom_par, adom_var
    auto [ adom_par, adom_var ] = ad_t::start_recording(
        std::move(dom_par), std::move(dom_var)
    );
    //
    // dom_par, dom_var
    // change from unspecified state
    dom_par = vector<Tensor>();
    dom_var = vector<Tensor>();
    //
    // arange
    std::vector<ad_t> arange = { adom_par.at(0), adom_var.at(0) };
    //
    // stop_recording
    ad_t::stop_recording( arange  );
    //
    {   Tensor check =  adom_par.at(0).tensor() == torch::tensor( {2.0, 3.0} );
        bool equal   = torch::all(check).item<bool>();
        EXPECT_TRUE(equal);
    }
    {   Tensor check =  adom_var.at(0).tensor() == torch::tensor( {4.0, 5.0} );
        bool equal   = torch::all(check).item<bool>();
        EXPECT_TRUE(equal);
    }
}
// END_CPP
