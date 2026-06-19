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
    void view_op_t<TensorType>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<TensorType>&          par_vec
    ) const {
        //
        // arg_start
        size_t    arg_start = agraph.m_arg_start[op_index];
        //
        // n_dim
        size_t n_dim = agraph.m_arg_value[arg_start + 1];
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_start + 1] );
        //
#ifndef NDEBUG
        //
        // ad_type
        ad_type_t ad_type   = agraph.m_arg_type[arg_start];
        assert( ad_type  == ad_type_t::parameter );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3 && "view_op: n_arg != 3");
#endif
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_start];
        //
        // shape
        size_t start_int64 = agraph.m_arg_value[arg_start + 2];
        const int64_t* begin = agraph.m_int64.data() + start_int64;
        const int64_t* end   = begin + n_dim;
        c10::IntArrayRef shape(begin, end);
        //
        // par_vec
        par_vec[op_index] = par_vec[operand_index].view(shape);
    }
    template void view_op_t<adten_t>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<adten_t>&             par_vec
    ) const;
    template void view_op_t<at::Tensor>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<at::Tensor>&          par_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_var
    template<class TensorType>
    void view_op_t<TensorType>::forward_var(
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
        // n_dim
        size_t n_dim = agraph.m_arg_value[arg_start + 1];
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_start + 1] );
        //
#ifndef NDEBUG
        //
        // ad_type
        ad_type_t ad_type   = agraph.m_arg_type[arg_start];
        assert( ad_type  == ad_type_t::variable );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3 && "view_op: n_arg != 3");
#endif
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_start];
        //
        // shape
        size_t start_int64 = agraph.m_arg_value[arg_start + 2];
        const int64_t* begin = agraph.m_int64.data() + start_int64;
        const int64_t* end   = begin + n_dim;
        c10::IntArrayRef shape(begin, end);
        //
        // var_vec
        var_vec[op_index] = var_vec[operand_index].view(shape);
    }
    template void view_op_t<adten_t>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        vector<adten_t>&             var_vec
    ) const;
    template void view_op_t<at::Tensor>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        vector<at::Tensor>&          var_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_der
    template<class TensorType>
    void view_op_t<TensorType>::forward_der(
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
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_start];
        //
        // for_der
        c10::IntArrayRef shape = var_vec[op_index].sizes();
        //
        // var_vec
        for_der[op_index] = for_der[operand_index].view(shape);
    }
    template void view_op_t<adten_t>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             for_der
    ) const;
    template void view_op_t<at::Tensor>::forward_der(
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
    void view_op_t<TensorType>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          rev_der
    ) const {
        //
        // arg_start
        size_t    arg_start = agraph.m_arg_start[op_index];
        //
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_start];
        //
        // shape
        c10::IntArrayRef shape = var_vec[operand_index].sizes();
        //
        // rev_der
        if( rev_der[operand_index].numel() == 0 ) {
            rev_der[operand_index] = rev_der[op_index].view(shape);
        } else {
            rev_der[operand_index] += rev_der[op_index].view(shape);
        }
    }
    template void view_op_t<adten_t>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             rev_der
    ) const;
    template void view_op_t<at::Tensor>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        const vector<at::Tensor>&    var_vec     ,
        vector<at::Tensor>&          rev_der
    ) const;
} }
