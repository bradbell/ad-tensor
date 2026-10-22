// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <filesystem>
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
namespace {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    namespace plugin = ad_tensor::plugin;
    namespace fs     =  std::filesystem;
    //
    // src_gen_path
    std::string setup_src_gen_path(void) {
        //
        // src_gen_path
        fs::path src_gen_path  = fs::temp_directory_path() / "test_src_gen";
        if( fs::is_directory(src_gen_path) ) {
            try {
                fs::remove_all(src_gen_path);
            } catch(...) {
                EXPECT_TRUE(false);
            }
        }
        fs::create_directory( src_gen_path );
        //
        try { fs::permissions(
            src_gen_path, fs::perms::owner_all, fs::perm_options::replace
        ); } catch (...) {
            // cannot change permissions for src_gen_path directory
            EXPECT_TRUE(false);
        }
        return src_gen_path.string();
    }
    fs::path src_gen_path = setup_src_gen_path();
    //
    // create_chkpnt
    // g(v) = { v[0] + v[1], v[0] * v[1] }
    size_t create_chkpnt(void) {
        //
        // av
        vector<at::Tensor> v;
        v.push_back( torch::zeros({5}) );
        v.push_back( torch::zeros({5}) );
        vector<adten_t> av = adten_t::start_recording(v);
        //
        // ar
        vector<adten_t> ar;
        ar.push_back( av[0] + av[1] );
        ar.push_back( av[0] * av[1] );
        //
        // r = g(v)
        adfn_t g = adten_t::stop_recording(ar, "g");
        //
        // src_gen_path : g.cpp, g.con
        g.src_gen(src_gen_path.string());
        //
        // chkpnt_id
        size_t chkpnt_id = ad_tensor::make_chkpnt(g);
        //
        return chkpnt_id;
    }
}
//
TEST(tests_adfn, src_gen)  {
    //
    // chkpnt_id
    size_t chkpnt_id = create_chkpnt();
    //
    // index_list
    std::optional<at::Tensor>  col_index  = torch::tensor( {0, 1} );
    std::optional<at::Tensor>  row_index  = torch::tensor( {0, 1} );
    c10::List< std::optional<at::Tensor> >
        index_list = { col_index, row_index };
    //
    // replace
    at::Tensor replace = torch::tensor( { 10.0, 11.0 } );
    //
    // bias, conv1d_options
    at::Tensor bias = torch::zeros( {1} );
    auto conv1d_options    = torch::nn::functional::Conv1dFuncOptions();
    //
    // pad_options
    auto pad_options = torch::nn::functional::PadFuncOptions( {1, 0, 0, 1} );
    pad_options.value( 7.0 );
    //
    // dim
    vector<int64_t> dim = { 1 };
    //
    // p
    // We use p for the domain parameters
    vector<at::Tensor> p;
    p.push_back( torch::tensor( {
        {2.0, 1.0},
        {3.0, 4.0}
    } ) );
    p.push_back( torch::tensor( {1.0, 2.0, 3.0, 4.0} ).view( {1,1,4} ) );
    p.push_back( torch::tensor( {1.0, 3.0, 5.0, 7.0, 9.0} ) );
    //
    // v
    // We use v for the domain variables
    vector<at::Tensor> v;
    v.push_back( torch::tensor( {
        {4.0, 3.0},
        {5.0, 6.0}
    } ) );
    v.push_back( torch::tensor( {5.0, 6.0, 7.0} ).view( {1,1,3} ) );
    v.push_back( torch::tensor( {2.0, 3.0, 4.0, 5.0, 6.0} ) );
    //
    // ap, av
    auto [av, ap] = adten_t::start_recording(v, p);
    //
    // ar
    vector<adten_t> ar;
    //
    // unary operators
    ar.push_back( (-av[0])               + (-ap[0]) );
    ar.push_back( av[0].exp()            + ap[0].exp() );
    ar.push_back( av[0].logdet()         + ap[0].logdet() );
    ar.push_back( av[0].inverse()        + ap[0].inverse() );
    ar.push_back( av[0].sum(dim)         + ap[0].sum(dim) );
    ar.push_back( av[0].flip(dim)        + ap[0].flip(dim) );
    ar.push_back( av[0].transpose(0, 1)  + ap[0].transpose(0, 1) );
    ar.push_back( av[0].view({4})        + ap[0].view({4}) );
    //
    // binary operators
    ar.push_back( av[0].matmul( ap[0] ) );
    ar.push_back( av[0] <= ap[0] );
    ar.push_back( linalg_solve( av[0], ap[0], true ) );
    //
    // conv1d
    adten_t ainput  = ap[1];
    adten_t aweight = av[1];
    adten_t abias   = adten_t( bias );
    ar.push_back( ad_tensor::conv1d(ainput, aweight, abias, conv1d_options) );
    //
    // pad
    ar.push_back( ad_tensor::pad(av[0], pad_options) );
    //
    // index
    ar.push_back( av[0].index( index_list ) );
    //
    // index_put
    ar.push_back( ap[0].index_put( index_list , adten_t(replace) ) );
    //
    // where
    ar.push_back( where( ap[2] > av[2], ap[2], av[2] ) );
    //
    // call
    vector<adten_t> ax;
    ax.push_back( av[2] );
    ax.push_back( ap[2] );
    vector<adten_t> ay = call_chkpnt(chkpnt_id, ax);
    ar.push_back( ay[0] );
    ar.push_back( ay[1] );
    //
    // r = f(v, p)
    adfn_t f = adten_t::stop_recording(ar, "f");
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
    fs::path build_path        = src_gen_path / "build";
    std::string plugin_lib     = "plugin_lib";
    std::string function_name  = f.get_name();
    auto f_plugin = plugin::src_gen_fun(
        build_path, plugin_lib, function_name
    );
    //
    // r
    size_t i = 0;
    vector<at::Tensor> r = f_plugin(v, p);
    EXPECT_EQ( r.size(), f.n_range() );
    EXPECT_TRUE( r[i++].equal( (-v[0])              + (-p[0]) ) );
    EXPECT_TRUE( r[i++].equal( v[0].exp()           + p[0].exp() ) );
    EXPECT_TRUE( r[i++].equal( v[0].logdet()        + p[0].logdet() ) );
    EXPECT_TRUE( r[i++].equal( v[0].inverse()       + p[0].inverse() ) );
    EXPECT_TRUE( r[i++].equal( v[0].sum(dim)        + p[0].sum(dim) ) );
    EXPECT_TRUE( r[i++].equal( v[0].flip(dim)       + p[0].flip(dim) ) );
    EXPECT_TRUE( r[i++].equal( v[0].transpose(0, 1) + p[0].transpose(0, 1) ) );
    EXPECT_TRUE( r[i++].equal( v[0].view({4})       + p[0].view({4}) ) );
    //
    EXPECT_TRUE( r[i++].equal( v[0].matmul( p[0] ) ) );
    EXPECT_TRUE( r[i++].equal( v[0] <= p[0] ) );
    EXPECT_TRUE( r[i++].equal( linalg_solve( v[0], p[0], true ) ) );
    //
    // conv1d
    EXPECT_TRUE( r[i++].equal(
        ad_tensor::conv1d(p[1], v[1], bias, conv1d_options)
    ));
    //
    // pad
    EXPECT_TRUE( r[i++].equal(
        torch::nn::functional::pad(v[0], pad_options)
    ));
    //
    // index
    EXPECT_TRUE( r[i++].equal( v[0].index( index_list ) ) );
    //
    // index_put
    EXPECT_TRUE( r[i++].equal( p[0].index_put( index_list , replace ) ) );
    //
    // where
    EXPECT_TRUE( r[i++].equal( where( p[2] > v[2], p[2], v[2] ) ) );
    //
    // call
    EXPECT_TRUE( r[i++].equal( v[2] + p[2] ) );
    EXPECT_TRUE( r[i++].equal( v[2] * p[2] ) );
}
