// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
//
namespace ad_tensor { namespace dev {
    //
    template <class TensorType>
    const base_op_t<TensorType>& op_enum2derive_op(op_enum_t op_enum) {
        // BEGIN_SORT_THIS_LINE_PLUS_1
        static const add_op_t add_op;
        static const div_op_t div_op;
        static const dom_op_t dom_op;
        static const mul_op_t mul_op;
        static const sub_op_t sub_op;
        static const sum_op_t sum_op;
        // END_SORT_THIS_LINE_MINUS_1
        //
        switch(op_enum) {
            // BEGIN_SORT_THIS_LINE_PLUS_1
            case op_enum_t::add: return add_op;
            case op_enum_t::div: return div_op;
            case op_enum_t::dom: return dom_op;
            case op_enum_t::mul: return mul_op;
            case op_enum_t::sub: return sub_op;
            case op_enum_t::sum: return sum_op;
            // END_SORT_THIS_LINE_MINUS_1
            //
            default:
            assert( false && "op_enum2derive_op: invalid op_enum value" );
        }
        // should not get here
        return dom_op;
    }
    template const base_op_t<at::Tensor>& op_enum2derive_op<at::Tensor>(
        op_enum_t op_enum
    );
} }
