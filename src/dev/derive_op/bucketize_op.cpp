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
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void bucketize_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    vector<TensorType>&          par_all
) const {
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 4 && "bucketize: n_arg != 4" );
    assert( agraph.m_arg_type[arg_start + 2] == vitype_t::none );
    assert( agraph.m_arg_type[arg_start + 3] == vitype_t::none );
# endif
    //
    // input, boundaries, out_int32, right
    TensorType input  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all
    );
    TensorType boundaries  = tensor_at_arg_index(
        arg_start + 1, agraph, con_vec, par_all
    );
    bool out_int32 = bool( agraph.m_arg_value[arg_start + 2] );
    bool right     = bool( agraph.m_arg_value[arg_start + 3] );
    //
    // par_all
    par_all[op_index] = bucketize(input, boundaries, out_int32, right);
}
template void bucketize_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    vector<at::Tensor>&          par_all
) const;
template void bucketize_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    vector<adten_t>&             par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void bucketize_op_t<TensorType>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<TensorType>&    par_all     ,
    vector<TensorType>&          var_all
) const {
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 4 && "bucketize: n_arg != 4" );
    assert( agraph.m_arg_type[arg_start + 2] == vitype_t::none );
    assert( agraph.m_arg_type[arg_start + 3] == vitype_t::none );
# endif
    //
    // input, boundaries, out_int32, right
    TensorType input  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all, var_all
    );
    TensorType boundaries  = tensor_at_arg_index(
        arg_start + 1, agraph, con_vec, par_all, var_all
    );
    bool out_int32 = bool( agraph.m_arg_value[arg_start + 2] );
    bool right     = bool( agraph.m_arg_value[arg_start + 3] );
    //
    // var_all
    var_all[op_index] = bucketize(input, boundaries, out_int32, right);
}
template void bucketize_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
template void bucketize_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
// ------------------------------------------------------------------------
// forward_der
template<class TensorType>
void bucketize_op_t<TensorType>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          for_der
) const {
}
template void bucketize_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
template void bucketize_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             for_der
) const;
// ------------------------------------------------------------------------
// reverse_der
template<class TensorType>
void bucketize_op_t<TensorType>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          rev_der
) const {
}
template void bucketize_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const;
template void bucketize_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const;
// ---------------------------------------------------------------------------
// src_gen
template <> std::string bucketize_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    vitype_t                                              op_vitype       ,
    const agraph_t&                                       agraph          ,
    const vector<int64_t>&                                int64_vec       ,
    const std::function< std::string(size_t, vitype_t) >& tensor_src
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
    assert( n_arg == 4 && "bucketize: n_arg != 4" );
# endif
    //
    // input_src
    size_t   input_index  = agraph.m_arg_value[arg_start];
    vitype_t input_vitype = agraph.m_arg_type[arg_start];
    string   input_src    = tensor_src(input_index, input_vitype);
    //
    // boundaries_src
    size_t   bnd_index      = agraph.m_arg_value[arg_start + 1];
    vitype_t bnd_vitype     = agraph.m_arg_type[arg_start + 1];
    string   boundaries_src = tensor_src(bnd_index, bnd_vitype);
    //
    // out_int32_src
    bool out_int32 = bool( agraph.m_arg_value[arg_start + 2] );
    string out_int32_src = "false";
    if( out_int32 ) {
        out_int32_src = "true";
    }
    //
    // right_src
    bool right = bool( agraph.m_arg_value[arg_start + 3] );
    string right_src = "false";
    if( right ) {
        right_src = "true";
    }
    //
    // target_src
    string target_src = tensor_src(op_index, op_vitype);
    //
    // src
    constexpr const char* fmt = "{} = bucketize({}, {}, {}, {});";
    string src = std::format(fmt,
        target_src, input_src, boundaries_src, out_int32_src, right_src
    );
    //
    return src;
}
template <> std::string bucketize_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    vitype_t                                              op_vitype       ,
    const agraph_t&                                       agraph          ,
    const vector<int64_t>&                                int64_vec       ,
    const std::function< std::string(size_t, vitype_t) >& tensor_src
) const {
    assert(false && "adten_t version of src_gen called for bucketize operator");
    return "";
}
} } // End ad_tensor::dev
