#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin derive_op dev}
{xrst_spell
    enum
    namespace
    dev
}

The Derived Operators Classes
#############################

OP_DERIVE(op_name)
******************
This macro call creates the class op_name_t,
and corresponding object op_name .
This is done inside the ``ad_tensor::dev`` namespace .
{xrst_literal ,
    BEGIN_SORT_THIS_LINE, END_SORT_THIS_LINE
}

op_enum2base_op
***************
This maps op_enum_t values to the corresponding base_op_t value:
{xrst_literal ,
    BEGIN_OP_ENUM2BASE_OP, END_OP_ENUM2BASE_OP
}


{xrst_end derive_op}
*/
#include <ad_tensor/dev/base_op.hpp>
//
// This is included so that derived operators that do not implement a virtual
// routine can report the routine and operator this to the user.
#include <ad_tensor/dev/user_assert.hpp>
//
#define OP_DERIVE(op_name) \
    struct op_name ## _t : public base_op_t { \
        \
        op_enum_t op_enum(void) const override; \
        \
        void forward_par( \
            size_t                            op_index    , \
            const agraph_t&                   agraph      , \
            const ad_tensor::vector<at::Tensor>&    con_vec     , \
            ad_tensor::vector<at::Tensor>&          par_vec \
        ) const override; \
        \
        void forward_var( \
            size_t                            op_index    , \
            const agraph_t&                   agraph      , \
            const ad_tensor::vector<at::Tensor>&    con_vec     , \
            const ad_tensor::vector<at::Tensor>&    par_vec     , \
            ad_tensor::vector<at::Tensor>&          var_vec \
        ) const override; \
        \
        void forward_der( \
            size_t                            op_index    , \
            const agraph_t&                   agraph      , \
            const ad_tensor::vector<at::Tensor>&    con_vec     , \
            const ad_tensor::vector<at::Tensor>&    par_vec     , \
            const ad_tensor::vector<at::Tensor>&    var_vec     , \
            ad_tensor::vector<at::Tensor>&          for_der \
        ) const override; \
        \
        void reverse_der( \
            size_t                            op_index    , \
            const agraph_t&                   agraph      , \
            const ad_tensor::vector<at::Tensor>&    con_vec     , \
            const ad_tensor::vector<at::Tensor>&    par_vec     , \
            const ad_tensor::vector<at::Tensor>&    var_vec     , \
            ad_tensor::vector<at::Tensor>&          rev_der \
        ) const override; \
    } const op_name;
//
namespace ad_tensor { namespace dev {
    // BEGIN_OP_ENUM2BASE_OP
    // base_op = ad_tensor::dev::op_enum2base_op(op_enum)
    const base_op_t& op_enum2base_op(op_enum_t op_enum);
    // END_OP_ENUM2BASE_OP
    //
    // BEGIN_SORT_THIS_LINE_PLUS_1
    OP_DERIVE(add_op)
    OP_DERIVE(div_op)
    OP_DERIVE(dom_op)
    OP_DERIVE(mul_op)
    OP_DERIVE(sub_op)
    // END_SORT_THIS_LINE_MINUS_1
} }
