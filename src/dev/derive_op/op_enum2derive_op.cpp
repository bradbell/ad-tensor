// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/derive_op.hpp>
//
namespace ad_tensor { namespace dev {
    //
    template <class TensorType>
    const base_op_t<TensorType>& op_enum2derive_op(op_enum_t op_enum) {
        // BEGIN_SORT_THIS_LINE_PLUS_1
        static const add_op_t<TensorType>         add_op;
        static const call_op_t<TensorType>        call_op;
        static const call_result_op_t<TensorType> call_result_op;
        static const div_op_t<TensorType>         div_op;
        static const dom_op_t<TensorType>         dom_op;
        static const matmul_op_t<TensorType>      matmul_op;
        static const minus_op_t<TensorType>       minus_op;
        static const mul_op_t<TensorType>         mul_op;
        static const solve_op_t<TensorType>       solve_op;
        static const sub_op_t<TensorType>         sub_op;
        static const sum_op_t<TensorType>         sum_op;
        static const transpose_op_t<TensorType>   transpose_op;
        static const view_op_t<TensorType>        view_op;
        // END_SORT_THIS_LINE_MINUS_1
        //
        switch(op_enum) {
            // BEGIN_SORT_THIS_LINE_PLUS_1
            case op_enum_t::add:         return add_op;
            case op_enum_t::call:        return call_op;
            case op_enum_t::call_result: return call_result_op;
            case op_enum_t::div:         return div_op;
            case op_enum_t::dom:         return dom_op;
            case op_enum_t::matmul:      return matmul_op;
            case op_enum_t::minus:       return minus_op;
            case op_enum_t::mul:         return mul_op;
            case op_enum_t::solve:       return solve_op;
            case op_enum_t::sub:         return sub_op;
            case op_enum_t::sum:         return sum_op;
            case op_enum_t::transpose:   return transpose_op;
            case op_enum_t::view:        return view_op;
            // END_SORT_THIS_LINE_MINUS_1
            //
            default:
            assert( false && "op_enum2derive_op: invalid op_enum value" );
        }
        // should not get here
        return dom_op;
    }
    template const base_op_t<adten_t>& op_enum2derive_op<adten_t>(
        op_enum_t op_enum
    );
    template const base_op_t<at::Tensor>& op_enum2derive_op<at::Tensor>(
        op_enum_t op_enum
    );
} }
