// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
//
namespace ad_tensor { namespace dev {
    //
    // op_enum
    op_enum_t dom_op_t::op_enum(void) const {
        return op_enum_t::dom;
    }
    //
    // op_string
    std::string dom_op_t::op_string(void) const {
        return std::string("dom");
    }
    //
    // forward_par
    void dom_op_t::forward_par(
        size_t                            op_index    ,
        const agraph_t&                   agraph      ,
        const std::vector<at::Tensor>&    con_vec     ,
        std::vector<at::Tensor>&          par_vec
    ) const
    { }
} }
