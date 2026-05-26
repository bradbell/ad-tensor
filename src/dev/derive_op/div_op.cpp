// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
//
namespace ad_tensor { namespace dev {
    //
    // op_enum
    op_enum_t div_op_t::op_enum(void) const {
        return op_enum_t::div;
    }
    //
    // forward_par
    void div_op_t::forward_par(
        size_t                            op_index    ,
        const agraph_t&                   agraph      ,
        const std::vector<at::Tensor>&    con_vec     ,
        std::vector<at::Tensor>&          par_vec
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start.at(op_index);
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start.at(op_index+1) - arg_index;
        assert( n_arg == 2 && "div_t: n_arg != 2" );
# endif
        //
        // lhs_tensor, rhs_tensor
        at::Tensor lhs_tensor  = tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec
        );
        at::Tensor rhs_tensor  = tensor_at_arg_index(
            arg_index + 1, agraph, con_vec, par_vec
        );
        //
        // par_vec
        par_vec.at(op_index) = lhs_tensor / rhs_tensor;
    }
    //
    // forward_var
    void div_op_t::forward_var(
        size_t                            op_index    ,
        const agraph_t&                   agraph      ,
        const std::vector<at::Tensor>&    con_vec     ,
        const std::vector<at::Tensor>&    par_vec     ,
        std::vector<at::Tensor>&          var_vec
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start.at(op_index);
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start.at(op_index+1) - arg_index;
        assert( n_arg == 2 && "div_t: n_arg != 2" );
# endif
        //
        // lhs_tensor, rhs_tensor
        at::Tensor lhs_tensor  = tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec, var_vec
        );
        at::Tensor rhs_tensor  = tensor_at_arg_index(
            arg_index + 1, agraph, con_vec, par_vec, var_vec
        );
        //
        // var_vec
        var_vec.at(op_index) = lhs_tensor / rhs_tensor;
    }
} }
