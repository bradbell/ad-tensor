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
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void index_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    vector<TensorType>&          par_all
) const {
    //
    // index_vec
    thread_local vector<TensorType> index_vec;
    //
    // arg_start, n_dim
    size_t arg_start = agraph.m_arg_start[op_index];
    size_t n_dim     = agraph.m_arg_value[arg_start + 1];
    //
#ifndef NDEBUG
    size_t arg_end   = agraph.m_arg_start[op_index + 1];
    size_t n_arg     = arg_end - arg_start;
    assert( n_arg  == 2 + n_dim );
#endif
    //
    // index_vec
    index_vec.resize(0);
    for(size_t i = 0; i < n_dim; ++i) {
        TensorType index = tensor_at_arg_index(
            arg_start + 2 + i, agraph, con_vec, par_all
        );
        index_vec.push_back( index );
    }
    //
    // from
    TensorType from  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all
    );
    // par_all
    par_all[op_index] = ad_tensor::index(from, index_vec);
}
template void index_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    vector<at::Tensor>&          par_all
) const;
template void index_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    vector<adten_t>&             par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void index_op_t<TensorType>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<TensorType>&    par_all     ,
    vector<TensorType>&          var_all
) const {
    //
    // index_vec
    thread_local vector<TensorType> index_vec;
    //
    // arg_start, n_dim
    size_t arg_start = agraph.m_arg_start[op_index];
    size_t n_dim     = agraph.m_arg_value[arg_start + 1];
    //
#ifndef NDEBUG
    size_t arg_end   = agraph.m_arg_start[op_index + 1];
    size_t n_arg     = arg_end - arg_start;
    assert( n_arg  == 2 + n_dim );
#endif
    //
    // index_vec
    index_vec.resize(0);
    for(size_t i = 0; i < n_dim; ++i) {
        TensorType index = tensor_at_arg_index(
            arg_start + 2 + i, agraph, con_vec, par_all, var_all
        );
        index_vec.push_back( index );
    }
    //
    // from
    TensorType from  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all, var_all
    );
    //
    // par_all
    var_all[op_index] = ad_tensor::index(from, index_vec);
}
template void index_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
template void index_op_t<adten_t>::forward_var(
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
void index_op_t<TensorType>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          for_der
) const {
    //
    // index_vec
    thread_local vector<TensorType> index_vec;
    //
    // arg_start, n_dim
    size_t arg_start = agraph.m_arg_start[op_index];
    size_t n_dim     = agraph.m_arg_value[arg_start + 1];
    //
#ifndef NDEBUG
    size_t arg_end   = agraph.m_arg_start[op_index + 1];
    size_t n_arg     = arg_end - arg_start;
    assert( n_arg  == 2 + n_dim );
#endif
    // from_index
    size_t   from_index  = agraph.m_arg_value[arg_start];
    vitype_t from_vitype = agraph.m_arg_type[arg_start];
    if( from_vitype != vitype_t::variable ) {
        return;
    }
    //
    // index_vec
    index_vec.resize(0);
    for(size_t i = 0; i < n_dim; ++i) {
        TensorType index = tensor_at_arg_index(
            arg_start + 2 + i, agraph, con_vec, par_all, var_all
        );
        index_vec.push_back( index );
    }
    //
    // for_der[op_index]
    for_der[op_index] = ad_tensor::index( for_der[from_index], index_vec);
}
template void index_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
template void index_op_t<adten_t>::forward_der(
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
void index_op_t<TensorType>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          rev_der
) const {
    //
    // index_vec
    thread_local vector<TensorType> index_vec;
    //
    // arg_start, n_dim
    size_t arg_start = agraph.m_arg_start[op_index];
    size_t n_dim     = agraph.m_arg_value[arg_start + 1];
    //
#ifndef NDEBUG
    size_t arg_end   = agraph.m_arg_start[op_index + 1];
    size_t n_arg     = arg_end - arg_start;
    assert( n_arg  == 2 + n_dim );
#endif
    //
    // index_vec
    index_vec.resize(0);
    for(size_t i = 0; i < n_dim; ++i) {
        TensorType index = tensor_at_arg_index(
            arg_start + 2 + i, agraph, con_vec, par_all, var_all
        );
        index_vec.push_back( index );
    }
    //
    // from_shape, from_index
    c10::IntArrayRef from_shape  = shape_at_arg_index(
        arg_start, agraph, con_vec, par_all, var_all
    );
    size_t from_index  = agraph.m_arg_value[arg_start];
    //
    // extract_der
    TensorType zeros       = TensorType( torch::zeros( from_shape ) );
    TensorType extract_der = rev_der[op_index];
    extract_der             = zeros.index_put(index_vec, extract_der);
    //
    // rev_der[from_index] += extract_der
    c10::IntArrayRef shape = var_all[from_index].sizes();
    plus_equal(
        rev_der[from_index], shape, extract_der
    );
}
//
// TODO: Change this to use routine above once index_put is changed to
// use vecoor of index types
template <> void index_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const {
    assert(false && "adten_t version of rev_der called for index operator");
};
template <> void index_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const {
    assert(false && "adten_t version of rev_der called for index operator");
};
// ---------------------------------------------------------------------------
// src_gen
template <> std::string index_op_t<at::Tensor>::src_gen(
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
    // arg_start, n_dim
    size_t arg_start = agraph.m_arg_start[op_index];
    size_t n_dim     = agraph.m_arg_value[arg_start + 1];
    //
#ifndef NDEBUG
    size_t arg_end   = agraph.m_arg_start[op_index + 1];
    size_t n_arg     = arg_end - arg_start;
    assert( n_arg  == 2 + n_dim );
#endif
    //
    // src
    string src = "{    vector<at::indexing::TensorIndex> index_vec;\n";
    for(size_t i = 0; i < n_dim; ++i) {
        size_t   index   = agraph.m_arg_value[arg_start + 2 + i];
        vitype_t vitype  = agraph.m_arg_type[arg_start + 2 + 1];
        string index_src = tensor_src(index, vitype);
        src += "    index_vec.push_back( " + index_src + ");\n";
    }
    //
    // from_src
    size_t   from_index  = agraph.m_arg_value[arg_start];
    vitype_t from_vitype = agraph.m_arg_type[arg_start];
    string from_src      = tensor_src(from_index, from_vitype);
    //
    // target_src
    string target_src = tensor_src(op_index, op_vitype);
    //
    // src
    constexpr const char* fmt = "    {} = {}.index( index_vec );\n";
    src += std::format(fmt, target_src, from_src);
    src += "}";
    //
    return src;
}
template <> std::string index_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    vitype_t                                              op_vitype       ,
    const agraph_t&                                       agraph          ,
    const vector<int64_t>&                                int64_vec       ,
    const std::function< std::string(size_t, vitype_t) >& tensor_src
) const {
    assert(false && "adten_t version of src_gen called for index operator");
    return "";
}
} } // End ad_tensor::dev
