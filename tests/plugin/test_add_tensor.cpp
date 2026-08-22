// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
#include <plugin/get_plugin.hpp> // see ./get_plugin.hpp.in

TEST(tests_plugin, add_tensor) {
    //
    // at_tensor_vector_t, return_t
    typedef ad_tensor::vector<at::Tensor>               at_tensor_vector_t;
    typedef std::tuple<at_tensor_vector_t, std::string> return_t;
    //
    // dom_par, dom_var
    ad_tensor::vector<at::Tensor> dom_par = { torch::tensor( {1, 2} ) };
    ad_tensor::vector<at::Tensor> dom_var = { torch::tensor( {3, 4} ) };
    //
    auto add_tensor = get_plugin<
        return_t(const at_tensor_vector_t&, const at_tensor_vector_t&)
    >( "add_tensor" );
    //
    // range
    auto [range, message] = add_tensor(dom_par, dom_var);
    //
    EXPECT_EQ(message, "");
    EXPECT_EQ(range.size(), 1);
    //
    // check
    at::Tensor check = dom_par[0] + dom_var[0];
    EXPECT_TRUE( check.equal( range[0] ) );
}
