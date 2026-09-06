// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/dev/src_gen_binary.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/broadcast.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/dev/tensor_at_index.hpp>
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void add_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<TensorType>&          par_all
) const {
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 && "add: n_arg != 2" );
# endif
    //
    // lhs_tensor, rhs_tensor
    TensorType lhs_tensor  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all
    );
    TensorType rhs_tensor  = tensor_at_arg_index(
        arg_start + 1, agraph, con_vec, par_all
    );
    //
    // par_all
    par_all[op_index] = lhs_tensor + rhs_tensor;
}
template void add_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const;
template void add_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void add_op_t<TensorType>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    vector<TensorType>&          var_all
) const {
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 && "add: n_arg != 2" );
# endif
    //
    // lhs_tensor, rhs_tensor
    TensorType lhs_tensor  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all, var_all
    );
    TensorType rhs_tensor  = tensor_at_arg_index(
        arg_start + 1, agraph, con_vec, par_all, var_all
    );
    //
    // var_all
    var_all[op_index] = lhs_tensor + rhs_tensor;
}
template void add_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
template void add_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
// ------------------------------------------------------------------------
// forward_der
template<class TensorType>
void add_op_t<TensorType>::forward_der(
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
    assert( n_arg == 2 && "add: n_arg != 2" );
# endif
    //
    // lhs_type, rhs_type
    adtype_t lhs_type = agraph.m_arg_type[arg_start];
    adtype_t rhs_type = agraph.m_arg_type[arg_start + 1];
    //
    // lhs_index, rhs_index
    size_t lhs_index = agraph.m_arg_value[arg_start];
    size_t rhs_index = agraph.m_arg_value[arg_start + 1];
    //
    if( lhs_type != adtype_t::variable ) {
        assert( rhs_type == adtype_t::variable );
        //
        // for_der
        for_der[op_index] = for_der[rhs_index];
        //
    } else if( rhs_type != adtype_t::variable ) {
        assert( lhs_type == adtype_t::variable );
        //
        // for_der
        for_der[op_index] = for_der[lhs_index];
        //
    } else {
        //
        // for_der
        for_der[op_index] = for_der[lhs_index] + for_der[rhs_index];
    };
}
template void add_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             for_der
) const;
template void add_op_t<at::Tensor>::forward_der(
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
void add_op_t<TensorType>::reverse_der(
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
    assert( n_arg == 2 && "add: n_arg != 2" );
# endif
    //
    // lhs_type, rhs_type
    adtype_t lhs_type = agraph.m_arg_type[arg_start];
    adtype_t rhs_type = agraph.m_arg_type[arg_start + 1];
    //
    // rev_der[lhs_index]
    if( lhs_type == adtype_t::variable ) {
        //
        // lhs_index
        size_t lhs_index = agraph.m_arg_value[arg_start];
        //
        // dim
        broadcast(
            var_all[op_index].sizes(), var_all[lhs_index].sizes(), array
        );
        c10::IntArrayRef dim(array);
        //
        // rev_der[lhs_index] += rev_der[op_index]
        plus_equal(rev_der[lhs_index], rev_der[op_index], dim);
        //
    }
    //
    // rev_der[rhs_index]
    if( rhs_type == adtype_t::variable ) {
        //
        // rhs_index
        size_t rhs_index = agraph.m_arg_value[arg_start + 1];
        //
        // dim
        broadcast(
            var_all[op_index].sizes(), var_all[rhs_index].sizes(), array
        );
        c10::IntArrayRef dim(array);
        //
        // rev_der[rhs_index] += rev_der[op_index]
        plus_equal(rev_der[rhs_index], rev_der[op_index], dim);
        //
    }
}
template void add_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const;
template void add_op_t<at::Tensor>::reverse_der(
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
std::string add_op_t<TensorType>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    return src_gen_binary(op_index, agraph, variable_agraph, tensor_src);
}
template std::string add_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const;
template std::string add_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const;
} } // End ad_tensor::dev
