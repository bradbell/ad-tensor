// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/tensor_at_index.hpp>
#include <ad_tensor/dev/shape_at_index.hpp>
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/broadcast.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
//
namespace ad_tensor { namespace dev {
    // ------------------------------------------------------------------------
    // forward_par
    template<class TensorType>
    void index_put_op_t<TensorType>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<TensorType>&          par_vec
    ) const {
        //
        // index_list
        thread_local c10::List< std::optional<at::Tensor> > index_list;
        //
        // arg_start, n_idx
        size_t arg_start = agraph.m_arg_start[op_index];
        size_t n_idx     = agraph.m_arg_value[arg_start + 2];
        //
#ifndef NDEBUG
        size_t arg_end   = agraph.m_arg_start[op_index + 1];
        size_t n_arg     = arg_end - arg_start;
        assert( n_arg  == 3 + n_idx );
#endif
        //
        // index_list
        index_list.resize(0);
        for(size_t i = 0; i < n_idx; ++i) {
            size_t con_index = agraph.m_arg_value[arg_start + 3 + i];
            std::optional<at::Tensor> index = con_vec[con_index];
            index_list.push_back( index );
        }
        //
        // before, replace
        TensorType before  = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_vec
        );
        TensorType replace  = tensor_at_arg_index(
            arg_start + 1, agraph, con_vec, par_vec
        );
        //
        // par_vec
        par_vec[op_index] = before.index_put( index_list, replace );
    }
    template void index_put_op_t<adten_t>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<adten_t>&             par_vec
    ) const;
    template void index_put_op_t<at::Tensor>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<at::Tensor>&          par_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_var
    template<class TensorType>
    void index_put_op_t<TensorType>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        vector<TensorType>&          var_vec
    ) const {
        //
        // index_list
        thread_local c10::List< std::optional<at::Tensor> > index_list;
        //
        // arg_start, n_idx
        size_t arg_start = agraph.m_arg_start[op_index];
        size_t n_idx     = agraph.m_arg_value[arg_start + 2];
        //
#ifndef NDEBUG
        size_t arg_end   = agraph.m_arg_start[op_index + 1];
        size_t n_arg     = arg_end - arg_start;
        assert( n_arg  == 3 + n_idx );
#endif
        //
        // index_list
        index_list.resize(0);
        for(size_t i = 0; i < n_idx; ++i) {
            size_t con_index = agraph.m_arg_value[arg_start + 3 + i];
            std::optional<at::Tensor> index = con_vec[con_index];
            index_list.push_back( index );
        }
        //
        // before, replace
        TensorType before  = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_vec, var_vec
        );
        TensorType replace  = tensor_at_arg_index(
            arg_start + 1, agraph, con_vec, par_vec, var_vec
        );
        //
        // par_vec
        var_vec[op_index] = before.index_put( index_list, replace );
    }
    template void index_put_op_t<adten_t>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        vector<adten_t>&             var_vec
    ) const;
    template void index_put_op_t<at::Tensor>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        vector<at::Tensor>&          var_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_der
    template<class TensorType>
    void index_put_op_t<TensorType>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          for_der
    ) const {
        //
        // index_list
        thread_local c10::List< std::optional<at::Tensor> > index_list;
        //
        // arg_start, n_idx
        size_t arg_start = agraph.m_arg_start[op_index];
        size_t n_idx     = agraph.m_arg_value[arg_start + 2];
        //
#ifndef NDEBUG
        size_t arg_end   = agraph.m_arg_start[op_index + 1];
        size_t n_arg     = arg_end - arg_start;
        assert( n_arg  == 3 + n_idx );
#endif
        //
        // index_list
        index_list.resize(0);
        for(size_t i = 0; i < n_idx; ++i) {
            size_t con_index = agraph.m_arg_value[arg_start + 3 + i];
            std::optional<at::Tensor> index = con_vec[con_index];
            index_list.push_back( index );
        }
        //
        // before_shape, replace_shape
        c10::IntArrayRef before_shape  = shape_at_arg_index(
            arg_start, agraph, con_vec, par_vec, var_vec
        );
        c10::IntArrayRef replace_shape  = shape_at_arg_index(
            arg_start+1, agraph, con_vec, par_vec, var_vec
        );
        //
        // before_index, replace_index
        size_t before_index  = agraph.m_arg_value[arg_start];
        size_t replace_index = agraph.m_arg_value[arg_start + 1];
        //
        // before_type, replace_type
        adtype_t before_type  = agraph.m_arg_type[arg_start];
        adtype_t replace_type = agraph.m_arg_type[arg_start + 1];
        //
        // for_der[op_index]
        if( before_type != adtype_t::variable ) {
            assert( replace_type == adtype_t::variable );
            //
            TensorType zeros  = TensorType( torch::zeros(before_shape) );
            for_der[op_index] = zeros.index_put(
                index_list, for_der[replace_index]
            );
        } else if( replace_type != adtype_t::variable ) {
            assert( before_type == adtype_t::variable );
            //
            TensorType zeros  = TensorType( torch::zeros(replace_shape) );
            for_der[op_index] = for_der[before_index].index_put(
                index_list, zeros
            );
        } else {
            for_der[op_index] = for_der[before_index].index_put(
                index_list, for_der[replace_index]
            );
        };
    }
    template void index_put_op_t<adten_t>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             for_der
    ) const;
    template void index_put_op_t<at::Tensor>::forward_der(
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
    void index_put_op_t<TensorType>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          rev_der
    ) const {
        // TODO: Need to first implement
        // replace =  output.index( index_list )
        assert( false );
    }
    template void index_put_op_t<adten_t>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             rev_der
    ) const;
    template void index_put_op_t<at::Tensor>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        const vector<at::Tensor>&    var_vec     ,
        vector<at::Tensor>&          rev_der
    ) const;
} }
