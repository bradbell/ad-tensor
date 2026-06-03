// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/ad.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/dev/size_ptr2array_ref.hpp>
//
namespace ad_tensor { namespace dev {
    // ------------------------------------------------------------------------
    // forward_par
    template<class TensorType>
    void view_op_t<TensorType>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<TensorType>&    con_vec     ,
        ad_tensor::vector<TensorType>&          par_vec
    ) const {
        //
        // lock
        bool lock;
        //
        // arg_index
        size_t    arg_index = agraph.m_arg_start[op_index];
        //
        // n_dim
        size_t n_dim = agraph.m_arg_value[arg_index + 1];
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_index + 1] );
        //
#ifndef NDEBUG
        //
        // ad_type
        ad_type_t ad_type   = agraph.m_arg_type[arg_index];
        assert( ad_type  == ad_type_t::parameter );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 2 + n_dim );
#endif
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_index];
        //
        // shape
        lock = true;
        c10::ArrayRef<long> shape = size_ptr2array_ref(
                lock, agraph.m_arg_value.data() + arg_index + 1
        );
        assert( shape.size() == n_dim );
        //
        // par_vec
        par_vec[op_index] = par_vec[operand_index].view(shape);
        //
        // shape
        lock = false;
        size_ptr2array_ref(lock);
    }
    template void view_op_t<ad_t>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<ad_t>&          con_vec     ,
        ad_tensor::vector<ad_t>&                par_vec
    ) const;
    template void view_op_t<at::Tensor>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<at::Tensor>&          par_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_var
    template<class TensorType>
    void view_op_t<TensorType>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<TensorType>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        ad_tensor::vector<TensorType>&          var_vec
    ) const {
        //
        // lock
        bool lock;
        //
        // arg_index
        size_t    arg_index = agraph.m_arg_start[op_index];
        //
        // n_dim
        size_t n_dim = agraph.m_arg_value[arg_index + 1];
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_index + 1] );
        //
#ifndef NDEBUG
        //
        // ad_type
        ad_type_t ad_type   = agraph.m_arg_type[arg_index];
        assert( ad_type  == ad_type_t::variable );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 2 + n_dim );
#endif
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_index];
        //
        // shape
        lock = true;
        c10::ArrayRef<long> shape = size_ptr2array_ref(
                lock, agraph.m_arg_value.data() + arg_index + 1
        );
        assert( shape.size() == n_dim );
        //
        // var_vec
        var_vec[op_index] = var_vec[operand_index].view(shape);
        //
        // shape
        lock = false;
        size_ptr2array_ref(lock);
    }
    template void view_op_t<ad_t>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<ad_t>&          con_vec     ,
        const ad_tensor::vector<ad_t>&          par_vec     ,
        ad_tensor::vector<ad_t>&                var_vec
    ) const;
    template void view_op_t<at::Tensor>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        ad_tensor::vector<at::Tensor>&          var_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_der
    template<class TensorType>
    void view_op_t<TensorType>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<TensorType>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        const ad_tensor::vector<TensorType>&    var_vec     ,
        ad_tensor::vector<TensorType>&          for_der
    ) const {
        //
        // arg_index
        size_t    arg_index = agraph.m_arg_start[op_index];
        //
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_index];
        //
        // for_der
        c10::ArrayRef<long> shape = var_vec[op_index].sizes();
        //
        // var_vec
        for_der[op_index] = for_der[operand_index].view(shape);
    }
    template void view_op_t<ad_t>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<ad_t>&          con_vec     ,
        const ad_tensor::vector<ad_t>&          par_vec     ,
        const ad_tensor::vector<ad_t>&          var_vec     ,
        ad_tensor::vector<ad_t>&                for_der
    ) const;
    template void view_op_t<at::Tensor>::forward_der(
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
    void view_op_t<TensorType>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<TensorType>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        const ad_tensor::vector<TensorType>&    var_vec     ,
        ad_tensor::vector<TensorType>&          rev_der
    ) const {
        //
        // arg_index
        size_t    arg_index = agraph.m_arg_start[op_index];
        //
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_index];
        //
        // shape
        c10::ArrayRef<long> shape = var_vec[operand_index].sizes();
        //
        // rev_der
        if( rev_der[operand_index].numel() == 0 ) {
            rev_der[operand_index] = rev_der[op_index].view(shape);
        } else {
            rev_der[operand_index] += rev_der[op_index].view(shape);
        }
    }
    template void view_op_t<ad_t>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<ad_t>&          con_vec     ,
        const ad_tensor::vector<ad_t>&          par_vec     ,
        const ad_tensor::vector<ad_t>&          var_vec     ,
        ad_tensor::vector<ad_t>&                rev_der
    ) const;
    template void view_op_t<at::Tensor>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        const ad_tensor::vector<at::Tensor>&    var_vec     ,
        ad_tensor::vector<at::Tensor>&          rev_der
    ) const;
} }
