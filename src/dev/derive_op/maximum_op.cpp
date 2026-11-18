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
#include <ad_tensor/defined.hpp>
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void maximum_op_t<TensorType>::forward_par(
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
    assert( n_arg == 2 && "maximum: n_arg != 2" );
# endif
    //
    // first_tensor, second_tensor
    TensorType first_tensor  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all
    );
    TensorType second_tensor  = tensor_at_arg_index(
        arg_start + 1, agraph, con_vec, par_all
    );
    //
    // par_all
    par_all[op_index] = first_tensor.maximum(second_tensor);
}
template void maximum_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    vector<at::Tensor>&          par_all
) const;
template void maximum_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    vector<adten_t>&             par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void maximum_op_t<TensorType>::forward_var(
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
    //
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 && "maximum: n_arg != 2" );
# endif
    //
    // first_tensor, second_tensor
    TensorType first_tensor  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all, var_all
    );
    TensorType second_tensor  = tensor_at_arg_index(
        arg_start + 1, agraph, con_vec, par_all, var_all
    );
    //
    // var_all
    var_all[op_index] = first_tensor.maximum(second_tensor);
}
template void maximum_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
template void maximum_op_t<adten_t>::forward_var(
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
void maximum_op_t<TensorType>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          for_der
) const {
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 && "maximum: n_arg != 2" );
# endif
    // variable
    vitype_t variable = vitype_t::variable;
    //
    // first_tensor, first_index, first_type, first_zero_der
    size_t   first_index    = agraph.m_arg_value[arg_start];
    vitype_t first_type     = agraph.m_arg_type[arg_start];
    bool     first_zero_der =
        first_type != variable || not_defined( for_der[first_index] );
    //
    // second_index, second_type, second_zero_der
    size_t   second_index    = agraph.m_arg_value[arg_start + 1];
    vitype_t second_type     = agraph.m_arg_type[arg_start + 1];
    bool     second_zero_der =
        second_type != variable || not_defined( for_der[second_index] );
    //
    if( first_zero_der && second_zero_der ) {
        return;
    }
    //
    // greater_than
    TensorType first_tensor  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all, var_all
    );
    TensorType second_tensor  = tensor_at_arg_index(
        arg_start + 1, agraph, con_vec, par_all, var_all
    );
    TensorType greater_than = first_tensor > second_tensor;
    //
    // for_der[op_index]
    if( first_zero_der ) {
        assert( second_type == vitype_t::variable );
        //
        // zero
        c10::IntArrayRef shape = for_der[second_index].sizes();
        TensorType       zero  = TensorType( torch::zeros(shape) );
        //
        // for_der
        for_der[op_index] = where(greater_than, zero, for_der[second_index]);
        //
    } else if( second_zero_der ) {
        assert( first_type == vitype_t::variable );
        //
        // zero
        c10::IntArrayRef shape = for_der[first_index].sizes();
        TensorType       zero  = TensorType( torch::zeros(shape) );
        //
        // for_der
        for_der[op_index] = where(greater_than, for_der[first_index], zero);
    } else {
        for_der[op_index] = where(
            greater_than, for_der[first_index], for_der[second_index]
        );
    };
}
template void maximum_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
template void maximum_op_t<adten_t>::forward_der(
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
void maximum_op_t<TensorType>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
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
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 && "maximum: n_arg != 2" );
# endif
    //
    // greater_than
    TensorType first_tensor  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all, var_all
    );
    TensorType second_tensor  = tensor_at_arg_index(
        arg_start + 1, agraph, con_vec, par_all, var_all
    );
    TensorType greater_than = first_tensor > second_tensor;
    //
    // first_type, second_type
    vitype_t first_type  = agraph.m_arg_type[arg_start];
    vitype_t second_type = agraph.m_arg_type[arg_start + 1];
    //
    // first_index, second_index
    size_t first_index = agraph.m_arg_value[arg_start];
    size_t second_index = agraph.m_arg_value[arg_start + 1];
    //
    // res_shape, zero
    c10::IntArrayRef res_shape = var_all[op_index].sizes();
    TensorType       zero      = TensorType( torch::zeros(res_shape) );
    //
    // rev_der[first_index]
    if( first_type == vitype_t::variable ) {
        //
        // first_rev
        TensorType first_rev = where(greater_than, rev_der[op_index], zero);
        //
        // dim
        broadcast(
            res_shape,
            var_all[first_index].sizes(),
            array
        );
        c10::IntArrayRef dim(array);
        //
        // rev_der[first_index] += first_rev
        plus_equal(rev_der[first_index], first_rev, dim);
    }
    //
    // rev_der[second_index]
    if( second_type == vitype_t::variable ) {
        //
        // second_rev
        TensorType second_rev = where(greater_than, zero, rev_der[op_index]);
        //
        // dim
        broadcast(
            res_shape,
            var_all[second_index].sizes(),
            array
        );
        c10::IntArrayRef dim(array);
        //
        // rev_der[second_index] += second_rev
        plus_equal(rev_der[second_index], second_rev, dim);
    }
}
template void maximum_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const;
template void maximum_op_t<adten_t>::reverse_der(
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
template <> std::string maximum_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    vitype_t                                              op_vitype       ,
    const agraph_t&                                       agraph          ,
    const vector<int64_t>&                                int64_vec       ,
    const std::function< std::string(size_t, vitype_t) >& tensor_src
) const {
    return src_gen_binary(op_index, op_vitype, agraph, tensor_src);
}
template <> std::string maximum_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    vitype_t                                              op_vitype       ,
    const agraph_t&                                       agraph          ,
    const vector<int64_t>&                                int64_vec       ,
    const std::function< std::string(size_t, vitype_t) >& tensor_src
) const {
    assert(false && "adten_t version of src_gen called for maximum operator");
    return "";
}
} } // End ad_tensor::dev
