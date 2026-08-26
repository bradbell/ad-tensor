// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/no_elements.hpp>
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void inverse_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<TensorType>&          par_all
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
    assert( n_arg == 1 && "inverse_op: n_arg != 1" );
#endif
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    // par_all
    par_all[op_index] = par_all[operand_index].inverse();
}
template void inverse_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const;
template void inverse_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void inverse_op_t<TensorType>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    vector<TensorType>&          var_all
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
    assert( n_arg == 1 && "inverse_op: n_arg != 1" );
#endif
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    // var_all
    var_all[op_index] = var_all[operand_index].inverse();
}
template void inverse_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
template void inverse_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
// ------------------------------------------------------------------------
// forward_der
template<class TensorType>
void inverse_op_t<TensorType>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
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
    assert( n_arg == 1 && "inverse_op: n_arg != 1" );
#endif
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    if( no_elements(for_der[operand_index]) ) {
        return;
    }
    // for_der
    for_der[op_index] = - var_all[op_index].matmul(
        for_der[operand_index].matmul( var_all[op_index] )
    );
}
template void inverse_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             for_der
) const;
template void inverse_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
// ------------------------------------------------------------------------
// reverse_der
template<class TensorType>
void inverse_op_t<TensorType>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          rev_der
) const {
    //
    // array
    thread_local vector<int64_t> array;
    //
    // check for case where this operation is not connected to the range
    if( no_elements(rev_der[op_index]) ) {
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
    assert( n_arg == 1 && "inverse_op: n_arg != 1" );
#endif
    // operand_index, operand_shape
    size_t           operand_index  = agraph.m_arg_value[arg_start];
    //
    // inv_tran
    size_t n_dim = var_all[op_index].sizes().size();
    TensorType inv_tran = var_all[op_index].transpose(n_dim-2, n_dim-1);
    TensorType prod     = inv_tran.matmul(
        rev_der[op_index].matmul( inv_tran )
    );
    //
    // rev_der[operand_index]
    if( no_elements(rev_der[operand_index]) ) {
        rev_der[operand_index]  = - prod;
    } else {
        rev_der[operand_index] -= prod;
    }
}
template void inverse_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const;
template void inverse_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const;
// ---------------------------------------------------------------------------
// src_gen
template <class TensorType>
std::string inverse_op_t<TensorType>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    user_assert(false, "src_gen not yet implemented for inverse operator" );
    return "";
}
template std::string inverse_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const;
template std::string inverse_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const;
} } // End ad_tensor::dev
