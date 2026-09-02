// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin function_object usr}
{xrst_spell
    obj
    dll
    libmy
}

Linking Plugin Functions to Current Program
###########################################

Syntax
******
{xrst_code cpp}
    fun_obj = ad_tensor::plugin::function_object(plugin_lib. function_alias)
{xrst_code}

Prototype
*********
{xrst_literal ,
    // BEGIN_FUNCTION_OBJECT, END_FUNCTION_OBJECT
}

function_type
*************
{xrst_literal ,
    // BEGIN_FUNCTION_TYPE, END_FUNCTION_TYPE
}

plugin_lib
**********
This is the location of the shared library that has the function
we are linking.
The path ends with the system independent part of the library file name.
For example, if on Linux the library file is ``libmy_lib.so``,
plugin_lib will end with ``my_lib`` .

function_alias
**************
This is the Boost dll alias for this function.

fun_obj
*******
is a function object that can evaluate the function corresponding
to function_alias using the syntax
{xrst_code cpp}
    range = fun_obj(dom_par, dom_var)
{xrst_code}
The prototype for dom_par, dom_var, and range are given under
function_type above.


{xrst_end function_object}
*/
#include <ad_tensor/plugin.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace {
    //
    // BEGIN_FUNCTION_TYPE
    using function_type = ad_tensor::vector<at::Tensor>(
        const ad_tensor::vector<at::Tensor>& dom_par ,
        const ad_tensor::vector<at::Tensor>& dom_var
    );
    // END_FUNCTION_TYPE
    //
    // link_function
    template <class FunctionType>
    boost::function<FunctionType> link_function(
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
    // BEGIN_FUNCTION_OBJECT
    boost::function<function_type> function_object(
        const std::string& plugin_lib     ,
        const std::string& function_alias )
    {   // END_FUNCTION_OBJECT
        return link_function<function_type>(plugin_lib, function_alias);
    }
} }
