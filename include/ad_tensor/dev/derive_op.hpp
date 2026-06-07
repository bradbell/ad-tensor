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

AD_TENSOR_DERIVE_OP(op_name)
****************************
This macro call creates the class op_name_t,
and corresponding object op_name .
This is done inside the ``ad_tensor::dev`` namespace .
{xrst_literal ,
    BEGIN_SORT_THIS_LINE, END_SORT_THIS_LINE
}

op_enum2derive_op
*****************
This maps op_enum_t values to the corresponding base_op_t<TensorType> value:
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
#define AD_TENSOR_DERIVE_OP(op_name) \
    template<class TensorType> \
    struct op_name ## _t : public base_op_t<TensorType> { \
        \
        void forward_par( \
            size_t                                  op_index    , \
            const agraph_t&                         agraph      , \
            const ad_tensor::vector<at::Tensor>&    con_vec     , \
            ad_tensor::vector<TensorType>&          par_vec \
        ) const override; \
        \
        void forward_var( \
            size_t                                  op_index    , \
            const agraph_t&                         agraph      , \
            const ad_tensor::vector<at::Tensor>&    con_vec     , \
            const ad_tensor::vector<TensorType>&    par_vec     , \
            ad_tensor::vector<TensorType>&          var_vec \
        ) const override; \
        \
        void forward_der( \
            size_t                                  op_index    , \
            const agraph_t&                         agraph      , \
            const ad_tensor::vector<at::Tensor>&    con_vec     , \
            const ad_tensor::vector<TensorType>&    par_vec     , \
            const ad_tensor::vector<TensorType>&    var_vec     , \
            ad_tensor::vector<TensorType>&          for_der \
        ) const override; \
        \
        void reverse_der( \
            size_t                                  op_index    , \
            const agraph_t&                         agraph      , \
            const ad_tensor::vector<at::Tensor>&    con_vec     , \
            const ad_tensor::vector<TensorType>&    par_vec     , \
            const ad_tensor::vector<TensorType>&    var_vec     , \
            ad_tensor::vector<TensorType>&          rev_der \
        ) const override; \
    };
//
namespace ad_tensor { namespace dev {
    // BEGIN_OP_ENUM2BASE_OP
    // base_op = ad_tensor::dev::op_enum2derive_op(op_enum)
    template <class TensorType>
    const base_op_t<TensorType>& op_enum2derive_op(op_enum_t op_enum);
    // END_OP_ENUM2BASE_OP
    //
    // BEGIN_SORT_THIS_LINE_PLUS_1
    AD_TENSOR_DERIVE_OP(add_op)
    AD_TENSOR_DERIVE_OP(div_op)
    AD_TENSOR_DERIVE_OP(dom_op)
    AD_TENSOR_DERIVE_OP(matmul_op)
    AD_TENSOR_DERIVE_OP(minus_op)
    AD_TENSOR_DERIVE_OP(mul_op)
    AD_TENSOR_DERIVE_OP(sub_op)
    AD_TENSOR_DERIVE_OP(sum_op)
    AD_TENSOR_DERIVE_OP(transpose_op)
    AD_TENSOR_DERIVE_OP(view_op)
    // END_SORT_THIS_LINE_MINUS_1
} }
