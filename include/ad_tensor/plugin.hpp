#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
//
namespace ad_tensor { namespace plugin {
    //
    // build_lib
    void build_lib(
        const std::string& plugin_dir          ,
        bool               quiet               ,
        bool               use_ad_tensor_repo
    );
} }
