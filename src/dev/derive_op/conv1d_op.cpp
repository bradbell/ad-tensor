// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tensor_at_index.hpp>
#include <ad_tensor/no_elements.hpp>
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void conv1d_op_t<TensorType>::forward_par(
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
    assert( n_arg == 6  );
    for(size_t i = 3; i < 6; ++i) {
        assert( agraph.m_arg_type[arg_start+i] == adtype_t::none );
    }
# endif
    //
    // input, weight, bias
    TensorType input  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all
    );
    TensorType weight  = tensor_at_arg_index(
        arg_start + 1, agraph, con_vec, par_all
    );
    TensorType bias  = tensor_at_arg_index(
        arg_start + 2, agraph, con_vec, par_all
    );
    //
    // options
    int64_t stride   = int64_t( agraph.m_arg_value[arg_start + 3] );
    int64_t dilation = int64_t( agraph.m_arg_value[arg_start + 4] );
    int64_t groups   = int64_t( agraph.m_arg_value[arg_start + 5] );
    auto options = torch::nn::functional::Conv1dFuncOptions()
        .stride(stride)
        .dilation(dilation)
        .groups(groups);
    //
    // par_all
    par_all[op_index] = conv1d(input, weight, bias, options);
}
template void conv1d_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const;
template void conv1d_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void conv1d_op_t<TensorType>::forward_var(
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
    assert( n_arg == 6  );
    for(size_t i = 3; i < 6; ++i) {
        assert( agraph.m_arg_type[arg_start+i] == adtype_t::none );
    }
# endif
    //
    // input, weight, bias
    TensorType input  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all, var_all
    );
    TensorType weight  = tensor_at_arg_index(
        arg_start + 1, agraph, con_vec, par_all, var_all
    );
    TensorType bias  = tensor_at_arg_index(
        arg_start + 2, agraph, con_vec, par_all, var_all
    );
    //
    // options
    int64_t stride   = int64_t( agraph.m_arg_value[arg_start + 3] );
    int64_t dilation = int64_t( agraph.m_arg_value[arg_start + 4] );
    int64_t groups   = int64_t( agraph.m_arg_value[arg_start + 5] );
    auto options = torch::nn::functional::Conv1dFuncOptions()
        .stride(stride)
        .dilation(dilation)
        .groups(groups);
    //
    // par_all
    var_all[op_index] = conv1d(input, weight, bias, options);
}
template void conv1d_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
template void conv1d_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
// ------------------------------------------------------------------------
// forward_der
template<class TensorType>
void conv1d_op_t<TensorType>::forward_der(
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
    assert( n_arg == 6  );
    for(size_t i = 3; i < 6; ++i) {
        assert( agraph.m_arg_type[arg_start+i] == adtype_t::none );
    }
# endif
    // variable
    adtype_t variable = adtype_t::variable;
    //
    // input_index, input_type, input_zero_der
    size_t   input_index    = agraph.m_arg_value[arg_start];
    adtype_t input_type     = agraph.m_arg_type[arg_start];
    bool     input_zero_der =
        input_type != variable || no_elements( for_der[input_index] );
    //
    // weight_index, weight_type, weight_zero_der
    size_t   weight_index    = agraph.m_arg_value[arg_start + 1];
    adtype_t weight_type     = agraph.m_arg_type[arg_start + 1];
    bool     weight_zero_der =
        weight_type != variable || no_elements( for_der[weight_index] );
    //
    // bias_index, bias_type, bias_zero_der
    size_t   bias_index    = agraph.m_arg_value[arg_start + 1];
    adtype_t bias_type     = agraph.m_arg_type[arg_start + 1];
    bool     bias_zero_der =
        bias_type != variable || no_elements( for_der[bias_index] );
    //
    if( input_zero_der && weight_zero_der && bias_zero_der) {
        return;
    }
    //
    // bias_shape
    TensorType bias  = tensor_at_arg_index(
        arg_start + 2, agraph, con_vec, par_all, var_all
    );
    c10::IntArrayRef bias_shape = bias.sizes();
    //
    // for_der[op_index]
    TensorType  doutput;
    if( bias_zero_der ) {
        doutput = TensorType( torch::zeros( bias_shape ) );
    } else {
        doutput = for_der[bias_index];
    }
    //
    // for_der[op_index]
    if( input_zero_der && weight_zero_der ) {
        for_der[op_index] = doutput;
        return;
    }
    //
    // options
    int64_t stride   = int64_t( agraph.m_arg_value[arg_start + 3] );
    int64_t dilation = int64_t( agraph.m_arg_value[arg_start + 4] );
    int64_t groups   = int64_t( agraph.m_arg_value[arg_start + 5] );
    auto options = torch::nn::functional::Conv1dFuncOptions()
        .stride(stride)
        .dilation(dilation)
        .groups(groups);
    //
    // no_bias
    TensorType no_bias = TensorType( torch::zeros( bias_shape ) );
    //
    // doutput
    if( ! input_zero_der ) {
        TensorType weight  = tensor_at_arg_index(
            arg_start + 1, agraph, con_vec, par_all, var_all
        );
        doutput += conv1d( for_der[input_index], weight, no_bias , options );
    }
    if( ! weight_zero_der ) {
        TensorType input  = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_all, var_all
        );
        doutput += conv1d( input, for_der[weight_index], no_bias , options );
    }
    //
    // for_der[op_index]
    for_der[op_index] = doutput;
    return;
}
template void conv1d_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             for_der
) const;
template void conv1d_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
// ------------------------------------------------------------------------
template<class TensorType>
void conv1d_op_t<TensorType>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          rev_der
) const {
    user_assert(false, "reverse_der not yet implemented for conv1d operator" );
}
template void conv1d_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const;
template void conv1d_op_t<at::Tensor>::reverse_der(
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
std::string conv1d_op_t<TensorType>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    user_assert(false, "src_gen not yet implemented for conv1d operator" );
    return "";
}
template std::string conv1d_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const;
template std::string conv1d_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const;
} } // End ad_tensor::dev
