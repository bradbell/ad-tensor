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
// UNPACK
#define UNPACK \
    size_t arg_start = agraph.m_arg_start[op_index]; \
    size_t dim       = agraph.m_arg_value[arg_start]; \
    size_t n_ten     = agraph.m_arg_value[arg_start + 1]; \

//
// EXPECT_ARGUMENTS
#ifdef NDEBUG
#define EXPECT_ARGUMENTS
#else
#define EXPECT_ARGUMENTS \
    assert( agraph.m_arg_type[arg_start] == adtype_t::none ); \
    assert( agraph.m_arg_type[arg_start + 1] == adtype_t::none ); \
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start; \
    assert( n_arg == 2 + n_ten && "cat: n_arg != 2 + n_ten" );
#endif
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void cat_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<TensorType>&          par_all
) const {
    //
    // tensor_list
    vector<TensorType> tensor_list;
    //
    // arg_start, dim, n_ten
    UNPACK
    EXPECT_ARGUMENTS
    //
    // tensor_list
    tensor_list.resize(0);
    for(size_t i = 0; i < n_ten; ++i) {
        TensorType tensor = tensor_at_arg_index(
            arg_start + 2 + i, agraph, con_vec, par_all
        );
        tensor_list.push_back( std::move(tensor) );
    }
    //
    // par_all
    par_all[op_index] = cat(tensor_list, dim);
}
template void cat_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const;
template void cat_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void cat_op_t<TensorType>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    vector<TensorType>&          var_all
) const {
    //
    // tensor_list
    vector<TensorType> tensor_list;
    //
    // arg_start, dim, n_ten
    UNPACK
    EXPECT_ARGUMENTS
    //
    // tensor_list
    tensor_list.resize(0);
    for(size_t i = 0; i < n_ten; ++i) {
        TensorType tensor = tensor_at_arg_index(
            arg_start + 2 + i, agraph, con_vec, par_all, var_all
        );
        tensor_list.push_back( std::move(tensor) );
    }
    //
    // par_all
    var_all[op_index] = cat(tensor_list, dim);
}
template void cat_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
template void cat_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
// ------------------------------------------------------------------------
// forward_der
template<class TensorType>
void cat_op_t<TensorType>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          for_der
) const {
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 3 && "where: n_arg != 3" );
# endif
    // cond
    TensorType cond  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all, var_all
    );
    //
    // variable
    adtype_t  variable = adtype_t::variable;
    //
    // true_index, true_type, true_zero_der
    size_t   true_index    = agraph.m_arg_value[arg_start + 1];
    adtype_t true_type     = agraph.m_arg_type[arg_start + 1];
    bool     true_zero_der =
        true_type != variable || not_defined( for_der[true_index] );
    //
    // false_index, false_type, false_zero_der
    size_t   false_index    = agraph.m_arg_value[arg_start + 2];
    adtype_t false_type     = agraph.m_arg_type[arg_start + 2];
    bool     false_zero_der =
        false_type != variable || not_defined( for_der[false_index] );
    //
    if( true_zero_der && false_zero_der ) {
        return;
    }
    //
    // for_der[op_index]
    if( true_zero_der ) {
        assert( false_type == variable );
        TensorType zero   = TensorType( torch::zeros( {1} ) );
        for_der[op_index] = where(cond, zero, for_der[false_index] );
    } else if( false_zero_der ) {
        assert( true_type == variable );
        TensorType zero   = TensorType( torch::zeros( {1} ) );
        for_der[op_index] = where(cond, for_der[true_index], zero );
    } else {
        for_der[op_index] =
            where(cond, for_der[true_index], for_der[false_index] );
    }
}
template void cat_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
template void cat_op_t<adten_t>::forward_der(
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
void cat_op_t<TensorType>::reverse_der(
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
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 3 && "where: n_arg != 3" );
# endif
    // cond
    TensorType cond  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all, var_all
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
            var_all[op_index].sizes(), var_all[true_index].sizes(), array
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
            var_all[op_index].sizes(), var_all[false_index].sizes(), array
        );
        c10::IntArrayRef dim(array);
        //
        // rev_der[false_index] += rev_der[op_index]
        plus_equal(rev_der[false_index], rev_der[op_index], dim);
        //
    }
}
template void cat_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const;
template void cat_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const;
// ---------------------------------------------------------------------------
// src_gen
template <> std::string cat_op_t<at::Tensor>::src_gen(
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
    size_t arg_start = agraph.m_arg_start[op_index];
    //
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 3 && "where: n_arg != 3" );
# endif
    //
    // cond_src
    size_t   cond_index  = agraph.m_arg_value[arg_start];
    adtype_t cond_adtype = agraph.m_arg_type[arg_start];
    string   cond_src    = tensor_src(cond_index, cond_adtype);
    //
    // true_case_src
    size_t   true_case_index  = agraph.m_arg_value[arg_start + 1];
    adtype_t true_case_adtype = agraph.m_arg_type[arg_start + 1];
    string   true_case_src    = tensor_src(true_case_index, true_case_adtype);
    //
    // false_case_src
    size_t   false_case_index  = agraph.m_arg_value[arg_start + 2];
    adtype_t false_case_adtype = agraph.m_arg_type[arg_start + 2];
    string   false_case_src = tensor_src(false_case_index, false_case_adtype);
    //
    // target_src
    string target_src = tensor_src(op_index, op_adtype);
    //
    // src
    constexpr const char* fmt = "{} = where({}, {}, {});";
    string src = std::format(fmt,
        target_src, cond_src, true_case_src, false_case_src
    );
    //
    return src;
}
template <> std::string cat_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    adtype_t                                              op_adtype       ,
    const agraph_t&                                       agraph          ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    assert(false && "adten_t version of src_gen called for where operator");
    return "";
}
} } // End ad_tensor::dev
