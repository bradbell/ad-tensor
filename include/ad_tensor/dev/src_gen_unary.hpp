#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
namespace ad_tensor { namespace dev {
    std::string src_gen_unary(
        size_t                                                op_index        ,
        adtype_t                                              op_adtype       ,
        const agraph_t&                                       agraph          ,
        const std::function< std::string(size_t, adtype_t) >& tensor_src
    );
} }
