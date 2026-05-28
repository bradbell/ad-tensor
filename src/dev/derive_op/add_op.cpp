// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
//
namespace ad_tensor { namespace dev {
    // ------------------------------------------------------------------------
    // op_enum
    op_enum_t add_op_t::op_enum(void) const {
        return op_enum_t::add;
    }
    // ------------------------------------------------------------------------
    // forward_par
    void add_op_t::forward_par(
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
        assert( n_arg == 2 && "add_t: n_arg != 2" );
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
        par_vec.at(op_index) = lhs_tensor + rhs_tensor;
    }
    // ------------------------------------------------------------------------
    // forward_var
    void add_op_t::forward_var(
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
        assert( n_arg == 2 && "add_t: n_arg != 2" );
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
        var_vec.at(op_index) = lhs_tensor + rhs_tensor;
    }
    // ------------------------------------------------------------------------
    // forward_der
    void add_op_t::forward_der(
        size_t                            op_index    ,
        const agraph_t&                   agraph      ,
        const std::vector<at::Tensor>&    con_vec     ,
        const std::vector<at::Tensor>&    par_vec     ,
        const std::vector<at::Tensor>&    var_vec     ,
        std::vector<at::Tensor>&          for_der
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start.at(op_index);
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start.at(op_index+1) - arg_index;
        assert( n_arg == 2 && "add_t: n_arg != 2" );
# endif
        //
        // lhs_type, rhs_type
        ad_type_t lhs_type = agraph.m_arg_type.at(arg_index);
        ad_type_t rhs_type = agraph.m_arg_type.at(arg_index + 1);
        //
        if( lhs_type != ad_type_t::variable ) {
            assert( rhs_type == ad_type_t::variable );
            //
            // drhs_tensor
            at::Tensor drhs_tensor  = tensor_at_arg_index(
                arg_index + 1, agraph, con_vec, par_vec, for_der
            );
            //
            // for_der
            for_der[op_index] = drhs_tensor;
            //
        } else if( rhs_type != ad_type_t::variable ) {
            assert( lhs_type == ad_type_t::variable );
            //
            // dlhs_tensor
            at::Tensor dlhs_tensor  = tensor_at_arg_index(
                arg_index, agraph, con_vec, par_vec, for_der
            );
            //
            // for_der
            for_der[op_index] = dlhs_tensor;
            //
        } else {
            //
            // dlhs_tensor
            at::Tensor dlhs_tensor  = tensor_at_arg_index(
                arg_index, agraph, con_vec, par_vec, for_der
            );
            // drhs_tensor
            at::Tensor drhs_tensor  = tensor_at_arg_index(
                arg_index + 1, agraph, con_vec, par_vec, for_der
            );
            //
            // for_der
            for_der[op_index] = dlhs_tensor + drhs_tensor;
        };
    }
    // ------------------------------------------------------------------------
    // reverse_der
    void add_op_t::reverse_der(
        size_t                            op_index    ,
        const agraph_t&                   agraph      ,
        const std::vector<at::Tensor>&    con_vec     ,
        const std::vector<at::Tensor>&    par_vec     ,
        const std::vector<at::Tensor>&    var_vec     ,
        std::vector<at::Tensor>&          rev_der
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start.at(op_index);
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start.at(op_index+1) - arg_index;
        assert( n_arg == 2 && "add_t: n_arg != 2" );
# endif
        //
        // lhs_type, rhs_type
        ad_type_t lhs_type = agraph.m_arg_type.at(arg_index);
        ad_type_t rhs_type = agraph.m_arg_type.at(arg_index + 1);
        //
        // rev_der[lhs_index]
        if( lhs_type == ad_type_t::variable ) {
            //
            // lhs_index
            size_t lhs_index = agraph.m_arg_value.at(arg_index);
            //
            // rev_der
            rev_der.at(lhs_index) =
                rev_der.at(lhs_index) + rev_der.at(op_index);
        }
        //
        // rev_der[rhs_index]
        if( rhs_type == ad_type_t::variable ) {
            //
            // rhs_index
            size_t rhs_index = agraph.m_arg_value.at(arg_index + 1);
            //
            // rev_der
            rev_der.at(rhs_index) =
                rev_der.at(rhs_index) + rev_der.at(op_index);
        }
    }
} }
