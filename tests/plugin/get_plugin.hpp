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
boost::function<FunctionType> get_plugin(
    std::string plugin_dir     ,
    std::string library_name   ,
    std::string function_alias ) {
    //
    // fs
    namespace fs = boost::filesystem;
    //
    // prefix
#if defined(_WIN32) || defined(_WIN64)
    std::string prefix    = "";
    std::string extension = ".dll";
#else
    // CMake builds shared modules as *.so on Linux and Mac
    std::string prefix    = "lib";
    std::string extension = ".so";
#endif
    //
    // plugin_path
    fs::path plugin_path = fs::path(plugin_dir) / (prefix + library_name + extension);
    //
    try {
        // plugin
        auto plugin = boost::dll::import_alias<FunctionType>(
            plugin_path,
            function_alias,
            boost::dll::load_mode::append_decorations
         );
         return plugin;
    } catch(const std::exception& e) {
        std::cerr <<
            "get_plugin: Failed to load " + plugin_path.string() + "\n";
    }
    //
    // This already failed above so it will fail again
    return  boost::dll::import_alias<FunctionType>(
        plugin_path,
        function_alias,
        boost::dll::load_mode::append_decorations
    );
}
