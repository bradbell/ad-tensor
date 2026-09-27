// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/pad_enum.hpp>
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void pad_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<TensorType>&          par_all
) const {
    //
    // pad
    using torch::nn::functional::pad;
    //
    // arg_start
    size_t    arg_start = agraph.m_arg_start[op_index];
    //
    // input
    size_t     input_index = agraph.m_arg_value[arg_start];
    TensorType input       = par_all[input_index];
    //
    // constant_value
    size_t     constant_index = agraph.m_arg_value[arg_start + 1];
    at::Tensor constant_value = con_vec[constant_index];
    //
    // pad_mode
    size_t mode_size_t  = agraph.m_arg_value[arg_start + 2];
    pad_enum_t pad_mode = static_cast<pad_enum_t>(mode_size_t);
    //
    // n_sizes
    size_t n_sizes = agraph.m_arg_value[arg_start + 3];
    //
#ifndef NDEBUG
    //
    // adtype
    adtype_t input_type   = agraph.m_arg_type[arg_start];
    assert( input_type  == adtype_t::parameter );
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 4 + n_sizes && "pad_op: n_arg != 4 + n_sizes" );
    //
    for(size_t i = 2; i < n_arg; ++i) {
        assert( agraph.m_arg_type[arg_start +i] == adtype_t::none );
    }
#endif
    //
    // pad_sizes
    const size_t* begin = agraph.m_arg_value.data() + arg_start + 4;
    const size_t* end   = begin + n_sizes;
    vector<int64_t> pad_sizes(begin, end);
    //
    // options
    torch::nn::functional::PadFuncOptions options(pad_sizes);
    switch( pad_mode ) {
        //
        case pad_enum_t::circular:
        options.mode( torch::kCircular );
        break;
        //
        case pad_enum_t::constant:
        options.mode( torch::kConstant );
        options.value( constant_value.item<double>() );
        break;
        //
        case pad_enum_t::reflect:
        options.mode( torch::kReflect );
        break;
        //
        case pad_enum_t::replicate:
        options.mode( torch::kReplicate );
        break;
    }
    //
    // par_all
    par_all[op_index] = pad(input, options);
}
template void pad_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const;
template void pad_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void pad_op_t<TensorType>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    vector<TensorType>&          var_all
) const {
    //
    // pad
    using torch::nn::functional::pad;
    //
    // arg_start
    size_t    arg_start = agraph.m_arg_start[op_index];
    //
    // input
    size_t     input_index = agraph.m_arg_value[arg_start];
    TensorType input       = var_all[input_index];
    //
    // constant_value
    size_t     constant_index = agraph.m_arg_value[arg_start + 1];
    at::Tensor constant_value = con_vec[constant_index];
    //
    // pad_mode
    size_t mode_size_t  = agraph.m_arg_value[arg_start + 2];
    pad_enum_t pad_mode = static_cast<pad_enum_t>(mode_size_t);
    //
    // n_sizes
    size_t n_sizes = agraph.m_arg_value[arg_start + 3];
    //
#ifndef NDEBUG
    //
    // adtype
    adtype_t input_type   = agraph.m_arg_type[arg_start];
    assert( input_type  == adtype_t::variable );
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 4 + n_sizes && "pad_op: n_arg != 4 + n_sizes" );
    //
    for(size_t i = 2; i < n_arg; ++i) {
        assert( agraph.m_arg_type[arg_start +i] == adtype_t::none );
    }
#endif
    //
    // pad_sizes
    const size_t* begin = agraph.m_arg_value.data() + arg_start + 4;
    const size_t* end   = begin + n_sizes;
    vector<int64_t> pad_sizes(begin, end);
    //
    // options
    torch::nn::functional::PadFuncOptions options(pad_sizes);
    switch( pad_mode ) {
        //
        case pad_enum_t::circular:
        options.mode( torch::kCircular );
        break;
        //
        case pad_enum_t::constant:
        options.mode( torch::kConstant );
        options.value( constant_value.item<double>() );
        break;
        //
        case pad_enum_t::reflect:
        options.mode( torch::kReflect );
        break;
        //
        case pad_enum_t::replicate:
        options.mode( torch::kReplicate );
        break;
    }
    //
    // var_all
    var_all[op_index] = pad(input, options);
}
template void pad_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
template void pad_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
// ------------------------------------------------------------------------
// forward_der
template<class TensorType>
void pad_op_t<TensorType>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          for_der
) const {
    //
    // pad
    using torch::nn::functional::pad;
    //
    // arg_start
    size_t    arg_start = agraph.m_arg_start[op_index];
    //
    // input
    size_t     input_index = agraph.m_arg_value[arg_start];
    TensorType input_der   = for_der[input_index];
    //
    // pad_mode
    size_t mode_size_t  = agraph.m_arg_value[arg_start + 2];
    pad_enum_t pad_mode = static_cast<pad_enum_t>(mode_size_t);
    //
    // n_sizes
    size_t n_sizes = agraph.m_arg_value[arg_start + 3];
    //
#ifndef NDEBUG
    //
    // adtype
    adtype_t input_type   = agraph.m_arg_type[arg_start];
    assert( input_type  == adtype_t::variable );
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 4 + n_sizes && "pad_op: n_arg != 4 + n_sizes" );
    //
    for(size_t i = 2; i < n_arg; ++i) {
        assert( agraph.m_arg_type[arg_start +i] == adtype_t::none );
    }
#endif
    //
    // pad_sizes
    const size_t* begin = agraph.m_arg_value.data() + arg_start + 4;
    const size_t* end   = begin + n_sizes;
    vector<int64_t> pad_sizes(begin, end);
    //
    // options
    torch::nn::functional::PadFuncOptions options(pad_sizes);
    switch( pad_mode ) {
        //
        case pad_enum_t::circular:
        options.mode( torch::kCircular );
        break;
        //
        case pad_enum_t::constant:
        options.mode( torch::kConstant );
        options.value( 0.0 );
        break;
        //
        case pad_enum_t::reflect:
        options.mode( torch::kReflect );
        break;
        //
        case pad_enum_t::replicate:
        options.mode( torch::kReplicate );
        break;
    }
    //
    // for_der
    for_der[op_index] = pad(input_der, options);
}
template void pad_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
template void pad_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             for_der
) const;
// ------------------------------------------------------------------------
template<class TensorType>
void pad_op_t<TensorType>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          rev_der
) const {
    user_assert(false,
        "reverse_der not yet implemented for pad operator"
    );
}
template void pad_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const;
template void pad_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const;
// ---------------------------------------------------------------------------
// src_gen
template <> std::string pad_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    user_assert(false,
        "src_gen not yet implemented for pad operator"
    );
    return "";
}
template <> std::string pad_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    assert(false && "adten_t version of src_gen called for pad operator");
    return "";
}
} } // End ad_tensor::dev
