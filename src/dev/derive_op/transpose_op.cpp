// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
//
namespace ad_tensor { namespace dev {
    // ------------------------------------------------------------------------
    // forward_par
    template<class TensorType>
    void transpose_op_t<TensorType>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<TensorType>&          par_vec
    ) const {
        //
        // arg_index
        size_t    arg_start = agraph.m_arg_start[op_index];
        //
        // dim1
        int64_t dim1 = int64_t( agraph.m_arg_value[arg_start + 1] );
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_start + 1] );
        //
        // dim2
        int64_t dim2 = int64_t( agraph.m_arg_value[arg_start + 2] );
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_start + 2] );
        //
#ifndef NDEBUG
        //
        // ad_type
        ad_type_t ad_type   = agraph.m_arg_type[arg_start];
        assert( ad_type  == ad_type_t::parameter );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3 );
#endif
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_start];
        //
        // par_vec
        par_vec[op_index] = par_vec[operand_index].transpose(dim1, dim2);
    }
    template void transpose_op_t<adten_t>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<adten_t>&             par_vec
    ) const;
    template void transpose_op_t<at::Tensor>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<at::Tensor>&          par_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_var
    template<class TensorType>
    void transpose_op_t<TensorType>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        vector<TensorType>&          var_vec
    ) const {
        //
        // arg_index
        size_t    arg_start = agraph.m_arg_start[op_index];
        //
        // dim1
        int64_t dim1 = int64_t( agraph.m_arg_value[arg_start + 1] );
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_start + 1] );
        //
        // dim2
        int64_t dim2 = int64_t( agraph.m_arg_value[arg_start + 2] );
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_start + 2] );
        //
#ifndef NDEBUG
        //
        // ad_type
        ad_type_t ad_type   = agraph.m_arg_type[arg_start];
        assert( ad_type  == ad_type_t::variable );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3 );
#endif
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_start];
        //
        // var_vec
        var_vec[op_index] = var_vec[operand_index].transpose(dim1, dim2);
    }
    template void transpose_op_t<adten_t>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        vector<adten_t>&             var_vec
    ) const;
    template void transpose_op_t<at::Tensor>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        vector<at::Tensor>&          var_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_der
    template<class TensorType>
    void transpose_op_t<TensorType>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          for_der
    ) const {
        //
        // arg_index
        size_t    arg_start = agraph.m_arg_start[op_index];
        //
        // dim1
        int64_t dim1 = int64_t( agraph.m_arg_value[arg_start + 1] );
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_start + 1] );
        //
        // dim2
        int64_t dim2 = int64_t( agraph.m_arg_value[arg_start + 2] );
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_start + 2] );
        //
#ifndef NDEBUG
        //
        // ad_type
        ad_type_t ad_type   = agraph.m_arg_type[arg_start];
        assert( ad_type  == ad_type_t::variable );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3 );
#endif
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_start];
        //
        // for_der
        for_der[op_index] = for_der[operand_index].transpose(dim1, dim2);
    }
    template void transpose_op_t<adten_t>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             for_der
    ) const;
    template void transpose_op_t<at::Tensor>::forward_der(
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
    void transpose_op_t<TensorType>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          rev_der
    ) const {
        //
        // arg_index
        size_t    arg_start = agraph.m_arg_start[op_index];
        //
        // dim1
        int64_t dim1 = int64_t( agraph.m_arg_value[arg_start + 1] );
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_start + 1] );
        //
        // dim2
        int64_t dim2 = int64_t( agraph.m_arg_value[arg_start + 2] );
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_start + 2] );
        //
#ifndef NDEBUG
        //
        // ad_type
        ad_type_t ad_type   = agraph.m_arg_type[arg_start];
        assert( ad_type  == ad_type_t::variable );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3 );
#endif
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_start];
        //
        // rev_der
        if( rev_der[operand_index].numel() == 0 ) {
            rev_der[operand_index] = rev_der[op_index].transpose(dim2, dim1);
        } else {
            rev_der[operand_index] += rev_der[op_index].transpose(dim2, dim1);
        }
    }
    template void transpose_op_t<adten_t>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             rev_der
    ) const;
    template void transpose_op_t<at::Tensor>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        const vector<at::Tensor>&    var_vec     ,
        vector<at::Tensor>&          rev_der
    ) const;
} }
