// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/ad.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/dev/broadcast.hpp>
#include <ad_tensor/dev/tensor_at_index.hpp>
//
namespace ad_tensor { namespace dev {
    // ------------------------------------------------------------------------
    // forward_par
    template<class TensorType>
    void minus_op_t<TensorType>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<TensorType>&          par_vec
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 1 && "minus: n_arg != 1" );
# endif
        //
        // operand_tensor
        TensorType operand_tensor  = tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec
        );
        //
        // par_vec
        par_vec[op_index] = - operand_tensor;
    }
    template void minus_op_t<ad_t>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<ad_t>&                par_vec
    ) const;
    template void minus_op_t<at::Tensor>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<at::Tensor>&          par_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_var
    template<class TensorType>
    void minus_op_t<TensorType>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        ad_tensor::vector<TensorType>&          var_vec
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 1 && "minus: n_arg != 1" );
# endif
        //
        // operand_tensor
        TensorType operand_tensor  = tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec, var_vec
        );
        //
        // var_vec
        var_vec[op_index] = - operand_tensor;
    }
    template void minus_op_t<ad_t>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<ad_t>&          par_vec     ,
        ad_tensor::vector<ad_t>&                var_vec
    ) const;
    template void minus_op_t<at::Tensor>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        ad_tensor::vector<at::Tensor>&          var_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_der
    template<class TensorType>
    void minus_op_t<TensorType>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        const ad_tensor::vector<TensorType>&    var_vec     ,
        ad_tensor::vector<TensorType>&          for_der
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 1 && "minus: n_arg != 1" );
        ad_type_t operand_type  = agraph.m_arg_type[arg_index];
        assert( operand_type == ad_type_t::variable && "minus::forward_der: "
            "operand is not a variable"
        );
# endif
        //
        // operand_index
        size_t    operand_index = agraph.m_arg_value[arg_index];
        //
        // for_der
        for_der[op_index] = - for_der[operand_index];
    }
    template void minus_op_t<ad_t>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<ad_t>&          par_vec     ,
        const ad_tensor::vector<ad_t>&          var_vec     ,
        ad_tensor::vector<ad_t>&                for_der
    ) const;
    template void minus_op_t<at::Tensor>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        const ad_tensor::vector<at::Tensor>&    var_vec     ,
        ad_tensor::vector<at::Tensor>&          for_der
    ) const;
    // ------------------------------------------------------------------------
    // reverse_der
    template<class TensorType>
    void minus_op_t<TensorType>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        const ad_tensor::vector<TensorType>&    var_vec     ,
        ad_tensor::vector<TensorType>&          rev_der
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 1 && "minus: n_arg != 1" );
        ad_type_t operand_type  = agraph.m_arg_type[arg_index];
        assert( operand_type == ad_type_t::variable &&
            "minus::forward_der: operand is not a variable"
        );
# endif
        //
        // operand_index
        size_t    operand_index = agraph.m_arg_value[arg_index];
        //
        // rev_der[operand_index]
        if( rev_der[operand_index].numel() == 0 ) {
            rev_der[operand_index] = - rev_der[op_index];
        } else {
            rev_der[operand_index] -= rev_der[op_index];
        }
    }
    template void minus_op_t<ad_t>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<ad_t>&          par_vec     ,
        const ad_tensor::vector<ad_t>&          var_vec     ,
        ad_tensor::vector<ad_t>&                rev_der
    ) const;
    template void minus_op_t<at::Tensor>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        const ad_tensor::vector<at::Tensor>&    var_vec     ,
        ad_tensor::vector<at::Tensor>&          rev_der
    ) const;
} }
