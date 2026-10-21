// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/defined.hpp>
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void view_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<TensorType>&          par_all
) const {
    //
    // arg_start
    size_t    arg_start = agraph.m_arg_start[op_index];
    //
    // n_dim
    size_t n_dim = agraph.m_arg_value[arg_start + 1];
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 1] );
    //
#ifndef NDEBUG
    //
    // adtype
    adtype_t adtype   = agraph.m_arg_type[arg_start];
    assert( adtype  == adtype_t::parameter );
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 + n_dim && "view_op: n_arg != 2 + n_dim");
#endif
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    // shape
    const size_t* begin = agraph.m_arg_value.data() + arg_start + 2;
    const size_t* end   = begin + n_dim;
    const vector<int64_t> shape(begin, end);
    //
    // par_all
    par_all[op_index] = par_all[operand_index].view(shape);
}
template void view_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const;
template void view_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void view_op_t<TensorType>::forward_var(
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
    // n_dim
    size_t n_dim = agraph.m_arg_value[arg_start + 1];
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 1] );
    //
#ifndef NDEBUG
    //
    // adtype
    adtype_t adtype   = agraph.m_arg_type[arg_start];
    assert( adtype  == adtype_t::variable );
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 + n_dim  && "view_op: n_arg != 2 + n_dim ");
#endif
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    // shape
    const size_t* begin = agraph.m_arg_value.data() + arg_start + 2;
    const size_t* end   = begin + n_dim;
    const vector<int64_t> shape(begin, end);
    //
    // var_all
    var_all[op_index] = var_all[operand_index].view(shape);
}
template void view_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
template void view_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
// ------------------------------------------------------------------------
// forward_der
template<class TensorType>
void view_op_t<TensorType>::forward_der(
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
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    // for_der
    c10::IntArrayRef shape = var_all[op_index].sizes();
    //
    // for_der[op_index]
    if( is_defined(for_der[operand_index]) ) {
        for_der[op_index] = for_der[operand_index].view(shape);
    }
}
template void view_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
template void view_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             for_der
) const;
// ------------------------------------------------------------------------
// reverse_der
template<class TensorType>
void view_op_t<TensorType>::reverse_der(
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
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    // shape
    c10::IntArrayRef shape = var_all[operand_index].sizes();
    //
    // rev_der
    if( not_defined(rev_der[operand_index]) ) {
        rev_der[operand_index] = rev_der[op_index].view(shape);
    } else {
        rev_der[operand_index] += rev_der[op_index].view(shape);
    }
}
template void view_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const;
template void view_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const;
// ---------------------------------------------------------------------------
// src_gen
template <> std::string view_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    adtype_t                                              op_adtype       ,
    const agraph_t&                                       agraph          ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    //
    // string
    using std::string;
    //
    // arg_start
    size_t    arg_start = agraph.m_arg_start[op_index];
    assert( adtype_t::none !=  agraph.m_arg_type[arg_start] );
    //
    // n_dim
    size_t n_dim = agraph.m_arg_value[arg_start + 1];
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 1] );
    //
#ifndef NDEBUG
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 + n_dim && "view_op: n_arg != 2 + n_dim");
    for(size_t i = 0; i < n_dim; ++i) {
        assert( agraph.m_arg_type[arg_start + 2 + i] == adtype_t::none );
    }
#endif
    // operand_src
    size_t   operand_index  = agraph.m_arg_value[arg_start];
    adtype_t operand_adtype = agraph.m_arg_type[arg_start];
    string   operand_src    = tensor_src(operand_index, operand_adtype);
    //
    // target_src
    string target_src = tensor_src(op_index, op_adtype);
    //
    // src
    string src = std::format("{} = {}.view({{", target_src, operand_src);
    for(size_t i = 0; i < n_dim; ++i) {
        size_t size = agraph.m_arg_value[arg_start + 2 + i];
        if( i == 0 ) {
            src += std::format("{}", size);
        } else {
            src += std::format(", {}", size);
        }
    }
    src += "});";
    //
    return src;
}
template <> std::string view_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    adtype_t                                              op_adtype       ,
    const agraph_t&                                       agraph          ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    assert(false && "adten_t version of src_gen called for view operator");
    return "";
}
} } // End ad_tensor::dev
