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
namespace {
    //
    // where
    using at::Tensor;
    using ad_tensor::adten_t;
    /*
    inline Tensor where(
        const Tensor& cond, const Tensor& false_case, const Tensor& true_case
    ) {
        return at::where(cond, false_case, true_case);
    }
    */
    inline adten_t where(
        const adten_t& cond, const adten_t& false_case, const adten_t& true_case
    ) {
        return adten_t::where(cond, false_case, true_case);
    }
}
//
namespace ad_tensor { namespace dev {
    // ------------------------------------------------------------------------
    // forward_par
    template<class TensorType>
    void where_op_t<TensorType>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<TensorType>&          par_vec
    ) const {
        //
        // arg_start
        size_t arg_start = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3 && "add: n_arg != 3" );
# endif
        //
        // cond, true_case, false_case
        TensorType cond  = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_vec
        );
        TensorType true_case  = tensor_at_arg_index(
            arg_start + 1, agraph, con_vec, par_vec
        );
        TensorType false_case  = tensor_at_arg_index(
            arg_start + 2, agraph, con_vec, par_vec
        );
        //
        // par_vec
        par_vec[op_index] = where(cond, true_case, false_case);
    }
    template void where_op_t<adten_t>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<adten_t>&             par_vec
    ) const;
    template void where_op_t<at::Tensor>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<at::Tensor>&          par_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_var
    template<class TensorType>
    void where_op_t<TensorType>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        vector<TensorType>&          var_vec
    ) const {
        //
        // arg_start
        size_t arg_start = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3 && "add: n_arg != 3" );
# endif
        //
        // cond, true_case, false_case
        TensorType cond  = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_vec, var_vec
        );
        TensorType true_case  = tensor_at_arg_index(
            arg_start + 1, agraph, con_vec, par_vec, var_vec
        );
        TensorType false_case  = tensor_at_arg_index(
            arg_start + 2, agraph, con_vec, par_vec, var_vec
        );
        //
        // var_vec
        var_vec[op_index] = where(cond, true_case, false_case);
        //
    }
    template void where_op_t<adten_t>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        vector<adten_t>&             var_vec
    ) const;
    template void where_op_t<at::Tensor>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        vector<at::Tensor>&          var_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_der
    template<class TensorType>
    void where_op_t<TensorType>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          for_der
    ) const {
        //
        // arg_start
        size_t arg_start = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3 && "add: n_arg != 3" );
# endif
        // cond
        TensorType cond  = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_vec, var_vec
        );
        //
        // true_type, false_type
        adtype_t true_type  = agraph.m_arg_type[arg_start + 1];
        adtype_t false_type = agraph.m_arg_type[arg_start + 2];
        //
        // true_index. false_index
        size_t true_index  = agraph.m_arg_value[arg_start + 1];
        size_t false_index = agraph.m_arg_value[arg_start + 2];
        //
        // zero, variable
        TensorType zero = TensorType( torch::zeros( {1} ) );
        adtype_t  variable = adtype_t::variable;
        //
        if( true_type == variable ) {
            if( false_type == variable ) {
                for_der[op_index] =
                    where(cond, for_der[true_index], for_der[false_index] );
            } else {
                for_der[op_index] = where(cond, for_der[true_index], zero );
            }
        } else if( false_type == variable ) {
            for_der[op_index] = where(cond, zero, for_der[false_index] );
        }
    }
    template void where_op_t<adten_t>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             for_der
    ) const;
    template void where_op_t<at::Tensor>::forward_der(
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
    void where_op_t<TensorType>::reverse_der(
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
        // arg_start
        size_t arg_start = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3 && "add: n_arg != 3" );
# endif
        // cond
        TensorType cond  = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_vec, var_vec
        );
        //
        // true_type, false_type
        adtype_t true_type  = agraph.m_arg_type[arg_start + 1];
        adtype_t false_type = agraph.m_arg_type[arg_start + 2];
        //
        // rev_der[true_index]
        if( true_type  == adtype_t::variable ) {
            //
            // true_index
            size_t true_index  = agraph.m_arg_value[arg_start + 1];
            //
            // dim
            broadcast(
                var_vec[op_index].sizes(), var_vec[true_index].sizes(), array
            );
            c10::IntArrayRef dim(array);
            //
            // rev_der[true_index] += rev_der[op_index]
            plus_equal(rev_der[true_index], rev_der[op_index], dim);
            //
        }
        //
        // rev_der[false_index]
        if( false_type  == adtype_t::variable ) {
            //
            // false_index
            size_t false_index  = agraph.m_arg_value[arg_start];
            //
            // dim
            broadcast(
                var_vec[op_index].sizes(), var_vec[false_index].sizes(), array
            );
            c10::IntArrayRef dim(array);
            //
            // rev_der[false_index] += rev_der[op_index]
            plus_equal(rev_der[false_index], rev_der[op_index], dim);
            //
        }
    }
    template void where_op_t<adten_t>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             rev_der
    ) const;
    template void where_op_t<at::Tensor>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        const vector<at::Tensor>&    var_vec     ,
        vector<at::Tensor>&          rev_der
    ) const;
} }
