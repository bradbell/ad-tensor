// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/pad_enum.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
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
    // input_type, constant_type
    adtype_t input_type   = agraph.m_arg_type[arg_start];
    assert( input_type  == adtype_t::parameter );
    adtype_t constant_type   = agraph.m_arg_type[arg_start + 1];
    assert( constant_type  == adtype_t::constant );
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
    // input_type, constant_type
    adtype_t input_type   = agraph.m_arg_type[arg_start];
    assert( input_type  == adtype_t::parameter );
    adtype_t constant_type   = agraph.m_arg_type[arg_start + 1];
    assert( constant_type  == adtype_t::constant );
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
    // input_type, constant_type
    adtype_t input_type   = agraph.m_arg_type[arg_start];
    assert( input_type  == adtype_t::parameter );
    adtype_t constant_type   = agraph.m_arg_type[arg_start + 1];
    assert( constant_type  == adtype_t::constant );
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
    //
    // pad
    using torch::nn::functional::pad;
    //
    // arg_start
    size_t    arg_start = agraph.m_arg_start[op_index];
    //
    // input_index
    size_t     input_index = agraph.m_arg_value[arg_start];
    //
    // pad_mode
    size_t mode_size_t  = agraph.m_arg_value[arg_start + 2];
    pad_enum_t pad_mode = static_cast<pad_enum_t>(mode_size_t);
    if( pad_mode != pad_enum_t::constant ) {
        user_assert(false,
            "pad: so far only constant mode reverse_der has been implemented."
        );
    }
    //
    // n_sizes
    size_t n_sizes = agraph.m_arg_value[arg_start + 3];
    //
#ifndef NDEBUG
    //
    // input_type, constant_type
    adtype_t input_type   = agraph.m_arg_type[arg_start];
    assert( input_type  == adtype_t::parameter );
    adtype_t constant_type   = agraph.m_arg_type[arg_start + 1];
    assert( constant_type  == adtype_t::constant );
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
    // output_bar
    c10::IntArrayRef output_shape = var_all[op_index].sizes();
    TensorType input_bar          = rev_der[op_index];
    size_t n_dim = n_sizes / 2;
    for(size_t i = 0; i < n_dim; ++i) {
        int64_t dim   = int64_t( output_shape.size() - i - 1 );
        if( pad_sizes[2 * i] != 0 || pad_sizes[2 * i +1] != 0 ) {
            int64_t start = pad_sizes[2 * i];
            int64_t stop  = output_shape[dim] - pad_sizes[2 * i + 1];
            int64_t step  = 1;
            input_bar     = input_bar.slice(dim, start, stop, step);
        }
    }
    //
    // rev_der[input_index] += input_bar
    plus_equal(rev_der[input_index], input_bar);
};
template void pad_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const;
template <> void pad_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const {
    // TODO: Change this function to use the TensorType implementation above
    // once the following operators have forward_var adten_t implementations:
    // slice
    user_assert(false,
    "reverse_der not yet implemented for pad with adten_t arguments" );
}

;
// ---------------------------------------------------------------------------
// src_gen
template <> std::string pad_op_t<at::Tensor>::src_gen(
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
    size_t    arg_start = agraph.m_arg_start[op_index];
    //
    // input_src
    size_t   input_index   = agraph.m_arg_value[arg_start];
    adtype_t input_adtype  = agraph.m_arg_type[arg_start];
    string   input_src     = tensor_src(input_index, input_adtype);
    //
    // constant_src
    size_t constant_index = agraph.m_arg_value[arg_start + 1];
    string constant_src   = tensor_src(constant_index, adtype_t::constant);
    //
    // target_src
    size_t   target_index = op_index;
    adtype_t target_adtype  =
        variable_agraph ? adtype_t::variable : adtype_t::parameter;
    string   target_src   = tensor_src(target_index, target_adtype);
    //
    // pad_mode_src
    size_t mode_size_t  = agraph.m_arg_value[arg_start + 2];
    pad_enum_t pad_mode = static_cast<pad_enum_t>(mode_size_t);
    string pad_mode_src;
    switch( pad_mode ) {
        //
        case pad_enum_t::circular:
        pad_mode_src = "torch::kCircular";
        break;
        //
        case pad_enum_t::constant:
        pad_mode_src = "torch::kConstant";
        break;
        //
        case pad_enum_t::reflect:
        pad_mode_src = "torch::kReflect";
        break;
        //
        case pad_enum_t::replicate:
        pad_mode_src = "torch::kReplicate";
        break;
    }
    //
    // n_sizes
    size_t n_sizes = agraph.m_arg_value[arg_start + 3];
    //
#ifndef NDEBUG
    //
    // constant_type
    adtype_t constant_type   = agraph.m_arg_type[arg_start + 1];
    assert( constant_type  == adtype_t::constant );
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
    // pad_sizes_src
    const size_t* begin = agraph.m_arg_value.data() + arg_start + 4;
    string pad_sizes_src = "{" + std::to_string( begin[0] );
    for(size_t i = 1; i < n_sizes; ++i) {
        pad_sizes_src += "," + std::to_string( begin[i] );
    }
    pad_sizes_src += "}";
    //
    // options_src
    string options_src;
    {   constexpr const char* fmt = 
R"|({{   auto options = torch::nn::functional::PadFuncOptions( {} );
    options.mode( {} );
)|";
        options_src = std::format(fmt, pad_sizes_src, pad_mode_src);
    }
    if( pad_mode == pad_enum_t::constant )
    {   constexpr const char* fmt = 
            "    options.value( {}.item<double>() );\n";
        options_src += std::format(fmt, constant_src);
    }
    //
    // src
    string src = options_src;
    constexpr const char* fmt =
        "    {} = torch::nn::functional::pad( {}, options );";
    src += std::format(fmt, target_src, input_src );
    src += "\n}";
    //
    return src;
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
