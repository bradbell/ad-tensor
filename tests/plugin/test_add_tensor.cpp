// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <iostream>
#include <boost/dll/import.hpp>
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>

namespace dll = boost::dll;

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
    // plugin_dirs
    std::vector< std::string > plugin_dirs = {
        "build/tests/plugin",
        "tests/plugin",
        "plugin",
        ".",
    };
    //
    // found, index, dir
    bool   found              = false;
    size_t index              = 0;
    std::string error_message = "";
    while( ! found && index < plugin_dirs.size() ) {
        std::string dir = plugin_dirs[index++];
        //
        // plugin_path
        boost::filesystem::path plugin_path = dir + "/plugin_lib";
        //
        //  Define the path to the compiled plugin file
        // On Windows this looks for "plugin_lib.dll",
        // on Linux "libplugin_lib.so"
        try {
            // Import the function by specifying its signature <int(int, int)>
            // and alias name "add_tensor".
            // Automatically adds .dll/.so extension
            auto add_tensor = dll::import_alias<
                return_t(const at_tensor_vector_t&, const at_tensor_vector_t&)
            >(
                plugin_path,
                "add_tensor",
                dll::load_mode::append_decorations
            );
            //
            // range
            return_t return_tuple  = add_tensor(dom_par, dom_var);
            auto [range, message]  = std::move( return_tuple );
            //
            EXPECT_EQ(message, "");
            EXPECT_EQ(range.size(), 1);
            //
            // check
            at::Tensor check = dom_par[0] + dom_var[0];
            EXPECT_TRUE( check.equal( range[0] ) );
            //
            // found
            found = true;
        } catch (const std::exception& e) {
            error_message += "Failed to load " + plugin_path.string() + "\n";
        }
    }
    if( ! found ) {
        std::cerr << error_message;
    }
    EXPECT_TRUE(found);
}
