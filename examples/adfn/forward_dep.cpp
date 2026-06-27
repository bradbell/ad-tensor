// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/adten.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/ad_type.hpp>
namespace {
    std::array<size_t, 2> pair(size_t row, size_t col) {
        return std::array<size_t, 2>({row, col});
    }

}
TEST(examples_adfn, forward_dep)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // c
    // We use c for the domain parameters
    vector<Tensor> c;
    c.push_back( torch::tensor( {1.0, 1.0} ) );
    c.push_back( torch::tensor( {1.0, 1.0} ) );
    c.push_back( torch::tensor( {1.0, 1.0} ) );
    //
    // x
    // We use x for the domain variables
    vector<Tensor> x;
    x.push_back( torch::tensor( {1.0, 1.0} ) );
    //
    // ac, ax
    auto [ac, ax] = adten_t::start_recording(c, x);
    //
    // ap
    vector<adten_t> ap;
    ap.push_back( ac[0] );
    ap.push_back( ac[0] + ac[1] * ax[0] );
    ap.push_back( ac[0] + ac[1] * ax[0] + ac[2] * ax[0] * ax[0]);
    //
    // p = f(c, x)
    adfn_t f = adten_t::stop_recording(ap, "f");
    //
    // depend_par, depend_var
    auto [depend_par, depend_var] = f.forward_dep();
    depend_par.sort();
    depend_var.sort();
    //
    // depend_par
    EXPECT_EQ( depend_par.size(), 6 );
    EXPECT_EQ( depend_par[0], pair(0, 0) );
    EXPECT_EQ( depend_par[1], pair(1, 0) );
    EXPECT_EQ( depend_par[2], pair(1, 1) );
    EXPECT_EQ( depend_par[3], pair(2, 0) );
    EXPECT_EQ( depend_par[4], pair(2, 1) );
    EXPECT_EQ( depend_par[5], pair(2, 2) );
    //
    // depend_var
    depend_var.sort();
    EXPECT_EQ( depend_var.size(), 2 );
    EXPECT_EQ( depend_var[0], pair(1, 0) );
    EXPECT_EQ( depend_var[1], pair(2, 0) );
}
// END_CPP
