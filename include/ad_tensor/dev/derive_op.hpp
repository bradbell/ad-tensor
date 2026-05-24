#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/base_op.hpp>
//
#define OP_DERIVE(op_name) \
    struct op_name ## _t : public base_op_t { \
        \
        op_enum_t op_enum(void) const override; \
        \
        void forward_par( \
            size_t                            op_index    , \
            const agraph_t&                   agraph      , \
            const std::vector<at::Tensor>&    con_vec     , \
            std::vector<at::Tensor>&          par_vec     \
        ) const override; \
    } const op_name;
//
namespace ad_tensor { namespace dev {
    const base_op_t& op_enum2base_op(op_enum_t op_enum);
    // BEGIN_SORT_THIS_LINE_PLUS_1
    OP_DERIVE(add_op)
    OP_DERIVE(sub_op)
    OP_DERIVE(mul_op)
    OP_DERIVE(div_op)
    // END_SORT_THIS_LINE_MINUS_1
} }
