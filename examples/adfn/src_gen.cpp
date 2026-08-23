// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adfn, src_gen)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // p
    // We use p for the domain parameters
    vector<Tensor> p;
    p.push_back( torch::tensor( {1.0, 1.0} ) );
    //
    // v
    // We use v for the domain variables
    vector<Tensor> v;
    v.push_back( torch::tensor( {1.0, 1.0} ) );
    //
    // ap, av
    auto [av, ap] = adten_t::start_recording(v, p);
    //
    // ar
    vector<adten_t> ar;
    ar.push_back( av[0] + ap[0] );
    //
    // r = f(v, p)
    adfn_t f = adten_t::stop_recording(ar, "f");
    //
    // src
    std::string src = f.src_gen();
    //
    EXPECT_TRUE(src.starts_with("// ad_tensor::adfn::src_gen output") );
    //
    // std::cout << "src = " << src << "\n";
}
// END_CPP
