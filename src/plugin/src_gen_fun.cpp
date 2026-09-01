// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/plugin.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace {
    //
    // src_gen_fun_t
    using src_gen_fun_t = ad_tensor::vector<at::Tensor>(
        const ad_tensor::vector<at::Tensor>& dom_par ,
        const ad_tensor::vector<at::Tensor>& dom_var
    );
    //
    // function_object
    template <class FunctionType>
    boost::function<FunctionType> function_object(
        std::string plugin_lib     ,
        std::string function_alias ) {
        //
        // plugin_path
        boost::filesystem::path plugin_path(plugin_lib);
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
            std::string msg = "link_plugin: failed to link ";
            msg +=  function_alias + " in library " + plugin_lib;
            ad_tensor::dev::user_assert(false, msg);
        }
        //
        // This already failed above so it will fail again
        return  boost::dll::import_alias<FunctionType>(
            plugin_path,
            function_alias,
            boost::dll::load_mode::append_decorations
        );
    }
}

namespace ad_tensor { namespace plugin {
    boost::function<src_gen_fun_t> src_gen_fun(
        const std::string& plugin_lib     ,
        const std::string& function_alias ) {
            return function_object<src_gen_fun_t>(plugin_lib, function_alias);
    }
} }
