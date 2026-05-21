// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/record.hpp>
//
TEST(examples, record)  {
    using ad_tensor::record_t;
    using ad_tensor::ad_t;
    using at::Tensor;
    using std::vector;
    //
    vector<Tensor> dom_par = { torch::tensor( {2.0, 3.0} ) };
    vector<Tensor> dom_var = { torch::tensor( {4.0, 5.0} ) };
    //
    // record_start
    auto [ adom_par, adom_var ] = record_t::start(dom_par, dom_var);
    //
    // record_stop
    record_t::stop( adom_var[0]  );
    //
    {   Tensor check =  adom_par[0].tensor() == dom_par[0];
        bool equal   = torch::all(check).item<bool>();
        EXPECT_TRUE(equal);
    }
    {   Tensor check =  adom_var[0].tensor() == dom_var[0];
        bool equal   = torch::all(check).item<bool>();
        EXPECT_TRUE(equal);
    }
}
// END_CPP
