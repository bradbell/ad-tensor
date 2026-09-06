// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/no_elements.hpp>
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void transpose_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<TensorType>&          par_all
) const {
    //
    // arg_start
    size_t    arg_start = agraph.m_arg_start[op_index];
    //
    // dim1
    int64_t dim1 = int64_t( agraph.m_arg_value[arg_start + 1] );
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 1] );
    //
    // dim2
    int64_t dim2 = int64_t( agraph.m_arg_value[arg_start + 2] );
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 2] );
    //
#ifndef NDEBUG
    //
    // adtype
    adtype_t adtype   = agraph.m_arg_type[arg_start];
    assert( adtype  == adtype_t::parameter );
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 3 );
#endif
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    // par_all
    par_all[op_index] = par_all[operand_index].transpose(dim1, dim2);
}
template void transpose_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const;
template void transpose_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void transpose_op_t<TensorType>::forward_var(
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
    // dim1
    int64_t dim1 = int64_t( agraph.m_arg_value[arg_start + 1] );
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 1] );
    //
    // dim2
    int64_t dim2 = int64_t( agraph.m_arg_value[arg_start + 2] );
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 2] );
    //
#ifndef NDEBUG
    //
    // adtype
    adtype_t adtype   = agraph.m_arg_type[arg_start];
    assert( adtype  == adtype_t::variable );
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 3 );
#endif
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    // var_all
    var_all[op_index] = var_all[operand_index].transpose(dim1, dim2);
}
template void transpose_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
template void transpose_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
// ------------------------------------------------------------------------
// forward_der
template<class TensorType>
void transpose_op_t<TensorType>::forward_der(
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
    // dim1
    int64_t dim1 = int64_t( agraph.m_arg_value[arg_start + 1] );
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 1] );
    //
    // dim2
    int64_t dim2 = int64_t( agraph.m_arg_value[arg_start + 2] );
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 2] );
    //
#ifndef NDEBUG
    //
    // adtype
    adtype_t adtype   = agraph.m_arg_type[arg_start];
    assert( adtype  == adtype_t::variable );
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 3 );
#endif
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    // for_der[op_index]
    if( has_elements(for_der[operand_index]) ) {
        for_der[op_index] = for_der[operand_index].transpose(dim1, dim2);
    }
}
template void transpose_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             for_der
) const;
template void transpose_op_t<at::Tensor>::forward_der(
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
void transpose_op_t<TensorType>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          rev_der
) const {
    //
    // arg_start
    size_t    arg_start = agraph.m_arg_start[op_index];
    //
    // dim1
    int64_t dim1 = int64_t( agraph.m_arg_value[arg_start + 1] );
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 1] );
    //
    // dim2
    int64_t dim2 = int64_t( agraph.m_arg_value[arg_start + 2] );
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 2] );
    //
#ifndef NDEBUG
    //
    // adtype
    adtype_t adtype   = agraph.m_arg_type[arg_start];
    assert( adtype  == adtype_t::variable );
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 3 );
#endif
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    // rev_der
    if( no_elements(rev_der[operand_index]) ) {
        rev_der[operand_index] = rev_der[op_index].transpose(dim2, dim1);
    } else {
        rev_der[operand_index] += rev_der[op_index].transpose(dim2, dim1);
    }
}
template void transpose_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const;
template void transpose_op_t<at::Tensor>::reverse_der(
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
std::string transpose_op_t<TensorType>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    //
    // string
    using std::string;
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 3 && "add: n_arg != 1" );
# endif
    //
    // operand_src
    size_t   operand_index  = agraph.m_arg_value[arg_start];
    adtype_t operand_adtype = agraph.m_arg_type[arg_start];
    string   operand_src    = tensor_src(operand_index, operand_adtype);
    //
    // target_src
    size_t   target_index  = op_index;
    adtype_t target_adtype =
        variable_agraph ? adtype_t::variable : adtype_t::parameter;
    string   target_src    = tensor_src(target_index, target_adtype);
    //
    // dim1, dim_2
    assert( agraph.m_arg_type[arg_start + 1] == adtype_t::none );
    size_t dim1 = agraph.m_arg_value[arg_start + 1];
    assert( agraph.m_arg_type[arg_start + 2] == adtype_t::none );
    size_t dim2 = agraph.m_arg_value[arg_start + 2];
    //
    // src
    constexpr const char* fmt = "{} = {}.transpose({}, {});";
    string src = std::format(fmt, target_src, operand_src, dim1, dim2);
    //
    return src;
}
template std::string transpose_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const;
template std::string transpose_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const;
} } // End ad_tensor::dev
