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
    // forward_par
    void dom_op_t::forward_par(
        size_t                            op_index    ,
        const agraph_t&                   agraph      ,
        const std::vector<at::Tensor>&    con_vec     ,
        std::vector<at::Tensor>&          par_vec
    ) const
    { }
    //
    // forward_var
    void dom_op_t::forward_var(
        size_t                            op_index    ,
        const agraph_t&                   agraph      ,
        const std::vector<at::Tensor>&    con_vec     ,
        const std::vector<at::Tensor>&    par_vec     ,
        std::vector<at::Tensor>&          var_vec
    ) const
    { }
    //
    // forward_der
    void dom_op_t::forward_der(
        size_t                            op_index    ,
        const agraph_t&                   agraph      ,
        const std::vector<at::Tensor>&    con_vec     ,
        const std::vector<at::Tensor>&    par_vec     ,
        const std::vector<at::Tensor>&    var_vec     ,
        std::vector<at::Tensor>&          for_der
    ) const
    { }
    //
    // reverse_der
    void dom_op_t::reverse_der(
        size_t                            op_index    ,
        const agraph_t&                   agraph      ,
        const std::vector<at::Tensor>&    con_vec     ,
        const std::vector<at::Tensor>&    par_vec     ,
        const std::vector<at::Tensor>&    var_vec     ,
        std::vector<at::Tensor>&          rev_der
    ) const
    { }
} }
