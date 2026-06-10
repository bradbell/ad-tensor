// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/broadcast.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/dev/tensor_at_index.hpp>
//
namespace ad_tensor { namespace dev {
    // ------------------------------------------------------------------------
    // forward_par
    template<class TensorType>
    void mul_op_t<TensorType>::forward_par(
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
        assert( n_arg == 2 && "mul: n_arg != 2" );
# endif
        //
        // lhs_tensor, rhs_tensor
        TensorType lhs_tensor  = tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec
        );
        TensorType rhs_tensor  = tensor_at_arg_index(
            arg_index + 1, agraph, con_vec, par_vec
        );
        //
        // par_vec
        par_vec[op_index] = lhs_tensor * rhs_tensor;
    }
    template void mul_op_t<adten_t>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<adten_t>&             par_vec
    ) const;
    template void mul_op_t<at::Tensor>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<at::Tensor>&          par_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_var
    template<class TensorType>
    void mul_op_t<TensorType>::forward_var(
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
        assert( n_arg == 2 && "mul: n_arg != 2" );
# endif
        //
        // lhs_tensor, rhs_tensor
        TensorType lhs_tensor  = tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec, var_vec
        );
        TensorType rhs_tensor  = tensor_at_arg_index(
            arg_index + 1, agraph, con_vec, par_vec, var_vec
        );
        //
        // var_vec
        var_vec[op_index] = lhs_tensor * rhs_tensor;
    }
    template void mul_op_t<adten_t>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<adten_t>&       par_vec     ,
        ad_tensor::vector<adten_t>&             var_vec
    ) const;
    template void mul_op_t<at::Tensor>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        ad_tensor::vector<at::Tensor>&          var_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_der
    template<class TensorType>
    void mul_op_t<TensorType>::forward_der(
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
        assert( n_arg == 2 && "mul: n_arg != 2" );
# endif
        //
        // lhs_type, rhs_type
        ad_type_t lhs_type = agraph.m_arg_type[arg_index];
        ad_type_t rhs_type = agraph.m_arg_type[arg_index + 1];
        //
        // lhs_index, rhs_index
        size_t lhs_index = agraph.m_arg_value[arg_index];
        size_t rhs_index = agraph.m_arg_value[arg_index + 1];
        //
        if( lhs_type != ad_type_t::variable ) {
            assert( rhs_type == ad_type_t::variable );
            //
            // lhs_tensor
            TensorType lhs_tensor  = tensor_at_arg_index(
                arg_index, agraph, con_vec, par_vec, var_vec
            );
            //
            // for_der
            for_der[op_index] = lhs_tensor * for_der[rhs_index];
            //
        } else if( rhs_type != ad_type_t::variable ) {
            assert( lhs_type == ad_type_t::variable );
            //
            // rhs_tensor
            TensorType rhs_tensor  = tensor_at_arg_index(
                arg_index + 1, agraph, con_vec, par_vec, var_vec
            );
            //
            // for_der
            for_der[op_index] = for_der[lhs_index] * rhs_tensor;
            //
        } else {
            //
            // for_der
            for_der[op_index] =
                var_vec[lhs_index] * for_der[rhs_index] +
                for_der[lhs_index] * var_vec[rhs_index];
        };
    }
    template void mul_op_t<adten_t>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<adten_t>&       par_vec     ,
        const ad_tensor::vector<adten_t>&       var_vec     ,
        ad_tensor::vector<adten_t>&             for_der
    ) const;
    template void mul_op_t<at::Tensor>::forward_der(
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
    void mul_op_t<TensorType>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        const ad_tensor::vector<TensorType>&    var_vec     ,
        ad_tensor::vector<TensorType>&          rev_der
    ) const {
        //
        // lock
        bool lock;
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 2 && "mul: n_arg != 2" );
# endif
        //
        // lhs_type, rhs_type
        ad_type_t lhs_type = agraph.m_arg_type[arg_index];
        ad_type_t rhs_type = agraph.m_arg_type[arg_index + 1];
        //
        // lhs_index, rhs_index
        size_t lhs_index = agraph.m_arg_value[arg_index];
        size_t rhs_index = agraph.m_arg_value[arg_index + 1];
        //
        // rev_der[lhs_index]
        if( lhs_type == ad_type_t::variable ) {
            //
            // rhs_tensor
            TensorType rhs_tensor  = tensor_at_arg_index(
                arg_index + 1, agraph, con_vec, par_vec, var_vec
            );
            //
            // dim
            lock = true;
            c10::IntArrayRef dim = broadcast(
                lock, var_vec[op_index].sizes(), var_vec[lhs_index].sizes()
            );
            //
            // prod
            TensorType prod = rev_der[op_index] * rhs_tensor;
            //
            // rev_der[lhs_index] += prod
            plus_equal(rev_der[lhs_index], prod, dim);
            //
            // dim
            lock = false;
            broadcast(lock);
        }
        //
        // rev_der[rhs_index]
        if( rhs_type == ad_type_t::variable ) {
            //
            // lhs_tensor
            TensorType lhs_tensor  = tensor_at_arg_index(
                arg_index, agraph, con_vec, par_vec, var_vec
            );
            //
            // dim
            lock = true;
            c10::IntArrayRef dim = broadcast(
                lock, var_vec[op_index].sizes(), var_vec[rhs_index].sizes()
            );
            //
            // prod
            TensorType prod = rev_der[op_index] * lhs_tensor;
            //
            // rev_der[rhs_index] += prod
            plus_equal(rev_der[rhs_index], prod, dim);
            //
            // dim
            lock = false;
            broadcast(lock);
        }
    }
    template void mul_op_t<adten_t>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<adten_t>&       par_vec     ,
        const ad_tensor::vector<adten_t>&       var_vec     ,
        ad_tensor::vector<adten_t>&             rev_der
    ) const;
    template void mul_op_t<at::Tensor>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        const ad_tensor::vector<at::Tensor>&    var_vec     ,
        ad_tensor::vector<at::Tensor>&          rev_der
    ) const;
} }
