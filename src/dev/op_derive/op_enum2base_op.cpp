// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
//
namespace ad_tensor { namespace dev {
    //
    const base_op_t& op_enum2base_op(op_enum_t op_enum) {
        switch(op_enum) {
            // BEGIN_SORT_THIS_LINE_PLUS_1
            case op_enum_t::add: return add_op;
            case op_enum_t::div: return div_op;
            case op_enum_t::dom: return dom_op;
            case op_enum_t::mul: return mul_op;
            case op_enum_t::sub: return sub_op;
            // END_SORT_THIS_LINE_MINUS_1
            //
            default:
            assert( false && "op_enum2base_op: invalid op_enum value" );
        }
        // should not get here
        return add_op;
    }
} }
