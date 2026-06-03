// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/ad.hpp>
#include <ad_tensor/dev/broadcast.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/dev/tensor_at_index.hpp>
//
namespace ad_tensor { namespace dev {
    // ------------------------------------------------------------------------
    // forward_par
    template<class TensorType>
    void div_op_t<TensorType>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<TensorType>&    con_vec     ,
        ad_tensor::vector<TensorType>&          par_vec
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 2 && "div: n_arg != 2" );
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
        par_vec[op_index] = lhs_tensor / rhs_tensor;
    }
    template void div_op_t<ad_t>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<ad_t>&          con_vec     ,
        ad_tensor::vector<ad_t>&                par_vec
    ) const;
    template void div_op_t<at::Tensor>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<at::Tensor>&          par_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_var
    template<class TensorType>
    void div_op_t<TensorType>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<TensorType>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        ad_tensor::vector<TensorType>&          var_vec
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 2 && "div: n_arg != 2" );
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
        var_vec[op_index] = lhs_tensor / rhs_tensor;
    }
    template void div_op_t<ad_t>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<ad_t>&          con_vec     ,
        const ad_tensor::vector<ad_t>&          par_vec     ,
        ad_tensor::vector<ad_t>&                var_vec
    ) const;
    template void div_op_t<at::Tensor>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        ad_tensor::vector<at::Tensor>&          var_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_der
    template<class TensorType>
    void div_op_t<TensorType>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<TensorType>&    con_vec     ,
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
        assert( n_arg == 2 && "div: n_arg != 2" );
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
            // for_der
            for_der[op_index] =
                - var_vec[op_index] * for_der[lhs_index] / var_vec[rhs_index];
            //
        } else if( rhs_type != ad_type_t::variable ) {
            assert( lhs_type == ad_type_t::variable );
            //
            // for_der
            for_der[op_index] = for_der[lhs_index] / var_vec[rhs_index];;
            //
        } else {
            //
            // for_der
            for_der[op_index] = for_der[lhs_index] / var_vec[rhs_index]
                - var_vec[op_index] * for_der[rhs_index] / var_vec[rhs_index];
        };
    }
    template void div_op_t<ad_t>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<ad_t>&          con_vec     ,
        const ad_tensor::vector<ad_t>&          par_vec     ,
        const ad_tensor::vector<ad_t>&          var_vec     ,
        ad_tensor::vector<ad_t>&                for_der
    ) const;
    template void div_op_t<at::Tensor>::forward_der(
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
    void div_op_t<TensorType>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<TensorType>&    con_vec     ,
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
        assert( n_arg == 2 && "div: n_arg != 2" );
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
        // rhs_tensor
        TensorType rhs_tensor = tensor_at_arg_index(
            arg_index + 1, agraph, con_vec, par_vec, var_vec
        );
        //
        // quotient
        TensorType quotient = rev_der[op_index] / rhs_tensor;
        //
        // rev_der[lhs_index]
        if( lhs_type == ad_type_t::variable ) {
            //
            // dim
            lock = true;
            c10::ArrayRef<long int> dim = broadcast(
                lock, var_vec[op_index].sizes(), var_vec[lhs_index].sizes()
            );
            //
            // rev_der[lhs_index] += quotient
            rev_plus_equal(dim, quotient, rev_der[lhs_index]);
            //
            // dim
            lock = false;
            broadcast(lock);
        }
        //
        // rev_der[rhs_index]
        if( rhs_type == ad_type_t::variable ) {
            //
            // prod
            TensorType prod = quotient * var_vec[op_index];
            //
            // dim
            lock = true;
            c10::ArrayRef<long int> dim = broadcast(
                lock, var_vec[op_index].sizes(), var_vec[rhs_index].sizes()
            );
            //
            // rev_der[rhs_index] -= prod
            rev_minus_equal(dim, prod, rev_der[rhs_index]);
            //
            // dim
            lock = false;
            broadcast(lock);
        }
    }
    template void div_op_t<at::Tensor>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        const ad_tensor::vector<at::Tensor>&    var_vec     ,
        ad_tensor::vector<at::Tensor>&          rev_der
    ) const;
} }
