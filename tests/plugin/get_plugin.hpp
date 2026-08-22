// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <iostream>
#include <boost/dll/import.hpp>
#include <boost/function.hpp>
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>

template <class FunctionType>
boost::function<FunctionType> get_plugin(std::string plugin_name) {
    //
    // plugin_dirs
    std::vector< std::string > plugin_dirs = {
        "build/tests/plugin",
        "tests/plugin",
        "plugin",
        ".",
    };
    //
    std::string error_message = "get_plugin:\n";
    for( std::string dir : plugin_dirs ) {
        //
        // plugin_path
        boost::filesystem::path plugin_path = dir + "/plugin_lib";
        //
        try {
            // plugin
            auto plugin = boost::dll::import_alias<FunctionType>(
                plugin_path,
                plugin_name,
                boost::dll::load_mode::append_decorations
            );
            return plugin;
        } catch (const std::exception& e) {
            error_message += "Failed to load " + plugin_path.string() + "\n";
        }
    }
    std::cerr << error_message << "\n";
    //
    // This already failed above so it will fail again
    return  boost::dll::import_alias<FunctionType>(
        "./plugin_lib",
        plugin_name,
        boost::dll::load_mode::append_decorations
    );
}
