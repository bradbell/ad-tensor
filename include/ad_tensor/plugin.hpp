#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
//
#include <string>
#include <boost/dll/import.hpp>
#include <boost/function.hpp>
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
//
namespace ad_tensor { namespace plugin {
    //
    // BEGIN_BUILD_LIB
    void build_lib(
        const std::string& plugin_dir                     ,
        bool               quiet                          ,
        bool               use_installed_ad_tensor = true
    );
    // END_BUILD_LIB
    //
    // src_gen_fun
    boost::function< ad_tensor::vector<at::Tensor>(
        const ad_tensor::vector<at::Tensor>& ,
        const ad_tensor::vector<at::Tensor>&
    ) > src_gen_fun(
        const std::string& plugin_lib     ,
        const std::string& function_alias
    );
} }
