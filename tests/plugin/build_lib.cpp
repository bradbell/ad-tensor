// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// -----------------------------------------------------------------------------
//
#include <filesystem>
#include <source_location>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/dev/user_assert.hpp>
#include <ad_tensor/plugin.hpp>
#include "get_plugin.hpp"
//
// --------------------------------------------------------------------------
TEST(tests_plugin, build_lib) {
    //
    // fs, string, vector, user_assert
    namespace fs = std::filesystem;
    using std::string;
    using ad_tensor::vector;
    using ad_tensor::dev::user_assert;
    //
    // plugin_path
    fs::path plugin_path  = fs::temp_directory_path() / "plugin";
    if( ! fs::is_directory(plugin_path) ) {
        fs::create_directory( plugin_path );
    }
    //
    // plugin_path
    std::source_location location      = std::source_location::current();
    fs::path location_path             = location.file_name();
    fs::path cmake_current_source_path = location_path.parent_path();
    vector<string> source_name = { "add_int.cpp", "add_tensor.cpp" };
    for(size_t i = 0; i < source_name.size(); ++i) {
        fs::path  from_path  = cmake_current_source_path / source_name[i];
        fs::path  to_path    = plugin_path / source_name[i];
        fs::copy(from_path, to_path, fs::copy_options::overwrite_existing);
    }
    //
    // quiet
    bool quiet = true;
    //
    // use_installed_ad_tensor
    bool use_installed_ad_tensor = false;
    //
    ad_tensor::plugin::build_lib(
        plugin_path.string(), quiet, use_installed_ad_tensor
    );
    //
    auto add_int = get_plugin<int(int, int)>( "./plugin_lib", "add_int");
    EXPECT_EQ(add_int(2, 3), 5);
    //
    // at_tensor_vector_t, return_t
    typedef ad_tensor::vector<at::Tensor>               at_tensor_vector_t;
    typedef std::tuple<at_tensor_vector_t, string> return_t;
    //
    // dom_par, dom_var
    ad_tensor::vector<at::Tensor> dom_par = { torch::tensor( {1, 2} ) };
    ad_tensor::vector<at::Tensor> dom_var = { torch::tensor( {3, 4} ) };
    //
    auto add_tensor = get_plugin<
        return_t(const at_tensor_vector_t&, const at_tensor_vector_t&)
    >( "./plugin_lib", "add_tensor" );
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
