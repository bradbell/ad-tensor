// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <iostream>
#include <boost/dll/import.hpp>
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace dll = boost::dll;

TEST(tests_plugin, add_tensor) {
    //
    // dll_name
    std::string dll_ext = boost::dll::shared_library::suffix().string();
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
                at::Tensor(const at::Tensor&, const at::Tensor&)
            >(
                plugin_path,
                "add_tensor",
                dll::load_mode::append_decorations
            );
            //
            // Execute the function
            at::Tensor x     = torch::tensor( {1, 2} );
            at::Tensor y     = torch::tensor( {3, 4} );
            at::Tensor check = torch::tensor( {4, 6} );
            EXPECT_TRUE( add_tensor(x, y).equal( check ) );
            //
            // found
            found = true;
        } catch (const std::exception& e) {
            error_message += "Failed to load " + plugin_path.string() + "\n";
        }
    }
    if( ! found ) {
        std::cout << error_message;
    }
    EXPECT_TRUE(found);
}
