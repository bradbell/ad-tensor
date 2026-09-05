// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <filesystem>
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(tests_adfn, src_gen)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    namespace plugin = ad_tensor::plugin;
    namespace fs     =  std::filesystem;
    //
    // dim
    vector<int64_t> dim = { 1 };
    //
    // p
    // We use p for the domain parameters
    vector<Tensor> p;
    p.push_back( torch::tensor( {
        {2.0, 1.0},
        {3.0, 4.0}
    } ) );
    //
    // v
    // We use v for the domain variables
    vector<Tensor> v;
    v.push_back( torch::tensor( {
        {4.0, 3.0},
        {5.0, 6.0}
    } ) );
    //
    // ap, av
    auto [av, ap] = adten_t::start_recording(v, p);
    //
    // ar
    vector<adten_t> ar;
    ar.push_back( (-av[0])               + (-ap[0]) );
    ar.push_back( av[0].exp()            + ap[0].exp() );
    ar.push_back( av[0].logdet()         + ap[0].logdet() );
    ar.push_back( av[0].inverse()        + ap[0].inverse() );
    ar.push_back( av[0].sum(dim)         + ap[0].sum(dim) );
    ar.push_back( av[0].transpose(0, 1)  + ap[0].transpose(0, 1) );
    ar.push_back( av[0].matmul( ap[0] ) );
    //
    // r = f(v, p)
    adfn_t f = adten_t::stop_recording(ar, "f");
    //
    // source_path
    fs::path source_path  = fs::temp_directory_path() / "test_src_gen";
    if( ! fs::is_directory(source_path) ) {
        fs::create_directory( source_path );
    }
    //
    // source_path: f.cpp, f.con
    f.src_gen(source_path.string());
    //
    // src_gen_path/build
    bool quiet                   = true;
    bool use_installed_ad_tensor = false;
    plugin::build_lib(
        source_path, quiet, use_installed_ad_tensor
    );
    //
    // f_plugin
    fs::path build_path        = source_path / "build";
    std::string plugin_lib     = "plugin_lib";
    std::string function_name  = f.get_name();
    auto f_plugin = plugin::function_object(
        build_path, plugin_lib, function_name
    );
    //
    // r
    vector<Tensor> r = f_plugin(v, p);
    EXPECT_EQ( r.size(), 7 );
    EXPECT_TRUE( r[0].equal( (-v[0])                + (-p[0]) ) );
    EXPECT_TRUE( r[1].equal( v[0].exp()             + p[0].exp() ) );
    EXPECT_TRUE( r[2].equal( v[0].logdet()          + p[0].logdet() ) );
    EXPECT_TRUE( r[3].equal( v[0].inverse()         + p[0].inverse() ) );
    EXPECT_TRUE( r[4].equal( v[0].sum(dim)          + p[0].sum(dim) ) );
    EXPECT_TRUE( r[5].equal( v[0].transpose(0, 1)   + p[0].transpose(0, 1) ) );
    EXPECT_TRUE( r[6].equal( v[0].matmul( p[0] ) ) );
    //
}
