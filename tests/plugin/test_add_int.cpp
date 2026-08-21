// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <iostream>
#include <boost/dll/import.hpp>
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>

namespace dll = boost::dll;

TEST(tests_plugin, add_int) {
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
            // and alias name "add_int".
            // Automatically adds .dll/.so extension
            auto add_int = dll::import_alias<int(int, int)>(
                plugin_path,
                "add_int",
                dll::load_mode::append_decorations
            );
            //
            // Execute the function
            EXPECT_EQ(add_int(2, 3), 5);
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
