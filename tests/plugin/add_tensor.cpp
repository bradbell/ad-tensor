// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <boost/config.hpp>
#include <boost/dll/alias.hpp>
#include <iostream>
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>

namespace {
    //
    // at_tensor_vector_t, return_t
    typedef ad_tensor::vector<at::Tensor>               at_tensor_vector_t;
    typedef std::tuple<at_tensor_vector_t, std::string> return_t;
}

// add_tensor_plugin
return_t add_tensor_plugin(
    const at_tensor_vector_t& dom_par ,
    const at_tensor_vector_t& dom_var ) {
    //
    // range, message
    at_tensor_vector_t range;
    std::string        message;
    //
    if( dom_par.size() != 1 ) {
        message = "add_tensor_plugin: dom_par.size() != 1";
        return return_t(range, message);
    }
    if( dom_var.size() != 1 ) {
        message = "add_tensor_plugin: dom_var.size() != 1";
        return return_t(range, message);
    }
    // range
    range.push_back( dom_par[0] + dom_var[0] );
    //
    return return_t(range, message);
}

// add_tensor
BOOST_DLL_ALIAS(add_tensor_plugin, add_tensor)
