// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
//
namespace ad_tensor { namespace dev {
    // ------------------------------------------------------------------------
    // forward_par
    template<class TensorType>
    void logdet_op_t<TensorType>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<TensorType>&          par_vec
    ) const {
        //
        // arg_start
        size_t    arg_start = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        //
        // adtype
        adtype_t adtype   = agraph.m_arg_type[arg_start];
        assert( adtype  == adtype_t::parameter );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 1 && "logdet_op: n_arg != 1" );
#endif
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_start];
        //
        // par_vec
        par_vec[op_index] = par_vec[operand_index].logdet();
    }
    template void logdet_op_t<adten_t>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<adten_t>&             par_vec
    ) const;
    template void logdet_op_t<at::Tensor>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<at::Tensor>&          par_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_var
    template<class TensorType>
    void logdet_op_t<TensorType>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        vector<TensorType>&          var_vec
    ) const {
        //
        // arg_start
        size_t    arg_start = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        //
        // adtype
        adtype_t adtype   = agraph.m_arg_type[arg_start];
        assert( adtype  == adtype_t::variable );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 1 && "logdet_op: n_arg != 1" );
#endif
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_start];
        //
        // var_vec
        var_vec[op_index] = var_vec[operand_index].logdet();
    }
    template void logdet_op_t<adten_t>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        vector<adten_t>&             var_vec
    ) const;
    template void logdet_op_t<at::Tensor>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        vector<at::Tensor>&          var_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_der
    template<class TensorType>
    void logdet_op_t<TensorType>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          for_der
    ) const {
        //
        // arg_start
        size_t    arg_start = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        //
        // adtype
        adtype_t adtype   = agraph.m_arg_type[arg_start];
        assert( adtype  == adtype_t::variable );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 1 && "logdet_op: n_arg != 1" );
#endif
        // operand_index, operand
        size_t operand_index  = agraph.m_arg_value[arg_start];
        if( ! for_der[operand_index].defined() ) {
            return;
        }
        const TensorType& operand = var_vec[operand_index];
        //
        // for_der
        size_t n_dim = operand.sizes().size();
        user_assert(
            2 <= n_dim,
            "logdet: operand must have 2 or more dimensions"
        );
        vector<int64_t> dim = { int64_t(n_dim - 2), int64_t(n_dim-1) };
        TensorType  inv_tran = operand.inverse().transpose(n_dim-2, n_dim-1);
        for_der[op_index] = (
            for_der[operand_index] * inv_tran
        ).sum(dim);
    }
    template void logdet_op_t<adten_t>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             for_der
    ) const;
    template void logdet_op_t<at::Tensor>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        const vector<at::Tensor>&    var_vec     ,
        vector<at::Tensor>&          for_der
    ) const;
    // ------------------------------------------------------------------------
    // reverse_der
    template<class TensorType>
    void logdet_op_t<TensorType>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          rev_der
    ) const {
        //
        // array
        thread_local vector<int64_t> array;
        //
        // check for case where this operation is not connected to the range
        if( ! rev_der[op_index].defined() ) {
            return;
        }
        //
        // arg_start
        size_t    arg_start = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        //
        // adtype
        adtype_t adtype   = agraph.m_arg_type[arg_start];
        assert( adtype  == adtype_t::variable );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 1 && "logdet_op: n_arg != 1" );
#endif
        // operand_index, operand
        size_t            operand_index = agraph.m_arg_value[arg_start];
        const TensorType& operand       = var_vec[operand_index];
        //
        // rev_der[operand_index]
        c10::IntArrayRef  ref = operand.sizes();
        vector<int64_t> shape( ref.begin(), ref.end() );
        size_t n_dim = shape.size();
        assert( 2 <= n_dim );
        shape[n_dim - 1] = 1;
        shape[n_dim - 2] = 1;
        TensorType inv_tran = operand.inverse().transpose(n_dim-1, n_dim-2);
        if( ! rev_der[operand_index].defined() ) {
            rev_der[operand_index]  =
                inv_tran * rev_der[op_index].view(shape);
        } else {
            rev_der[operand_index] +=
                inv_tran * rev_der[op_index].view(shape);
        }
    }
    template void logdet_op_t<adten_t>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             rev_der
    ) const;
    template void logdet_op_t<at::Tensor>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        const vector<at::Tensor>&    var_vec     ,
        vector<at::Tensor>&          rev_der
    ) const;
} }
