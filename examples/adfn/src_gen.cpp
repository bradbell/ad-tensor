// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <filesystem>
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adfn, src_gen)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    namespace plugin = ad_tensor::plugin;
    namespace fs     =  std::filesystem;
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
    // src_gen_path
    fs::path src_gen_path  = fs::temp_directory_path() / "src_gen";
    if( ! fs::is_directory(src_gen_path) ) {
        fs::create_directory( src_gen_path );
    }
    //
    // src_gen_path: f.cpp, f.con
    f.src_gen(src_gen_path.string());
    //
    // src_gen_path/build
    bool quiet                   = true;
    bool use_installed_ad_tensor = false;
    plugin::build_lib(
        src_gen_path, quiet, use_installed_ad_tensor
    );
    //
    // f_plugin
    fs::path plugin_path       = src_gen_path / "build/plugin_lib";
    std::string plugin_lib     = plugin_path.string();
    std::string function_name  = f.get_name();
    auto f_plugin = plugin::function_object(plugin_lib, function_name);
    //
    // r
    vector<Tensor> r = f_plugin(v, p);
    EXPECT_EQ( r.size(), 1 );
    EXPECT_TRUE( r[0].equal( v[0] + p[0] ) );
    //
}
// END_CPP
