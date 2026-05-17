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
    using ad_tensor::start_recording;
    using ad_tensor::ad_t;
    using torch::Tensor;
    //
    Tensor dom_par = torch::tensor( {2.0, 3.0} );
    Tensor dom_var = torch::tensor( {4.0, 5.0} );
    //
    auto [ adom_par, adom_var ] = start_recording(
        dom_par.clone(), dom_var.clone()
    );
    bool equal;
    equal = torch::all( adom_par.tensor() == dom_par ).item<bool>();
    EXPECT_TRUE(equal);
    //
    equal = torch::all( adom_var.tensor() == dom_var ).item<bool>();
    EXPECT_TRUE(equal);
}
// END_CPP
