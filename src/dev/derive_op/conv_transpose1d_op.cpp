// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tensor_at_index.hpp>
//
// set options
#define SET_OPTIONS \
    auto options = torch::nn::functional::ConvTranspose1dFuncOptions(); \
    { \
        int64_t stride   = int64_t( agraph.m_arg_value[arg_start + 3] ); \
        int64_t dilation = int64_t( agraph.m_arg_value[arg_start + 4] ); \
        int64_t groups   = int64_t( agraph.m_arg_value[arg_start + 5] ); \
        options.stride(stride); \
        options.dilation(dilation); \
        options.groups(groups); \
    }
//
// check arguments
#ifdef NDEBUG
    #define CHECK_ARGUMENTS
#else
    #define CHECK_ARGUMENTS \
    {   size_t arg_start = agraph.m_arg_start[op_index]; \
        size_t arg_end   = agraph.m_arg_start[op_index+1]; \
        size_t n_arg = arg_end - arg_start; \
        assert( n_arg == 6  ); \
        for(size_t i = 3; i < 6; ++i) { \
            assert( agraph.m_arg_type[arg_start+i] == adtype_t::none ); \
        } \
    }
#endif
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void conv_transpose1d_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<TensorType>&          par_all
) const {
    CHECK_ARGUMENTS
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
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
    SET_OPTIONS
    //
    // par_all
    par_all[op_index] = conv_transpose1d(input, weight, bias, options);
}
template void conv_transpose1d_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const;
template void conv_transpose1d_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void conv_transpose1d_op_t<TensorType>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    vector<TensorType>&          var_all
) const {
    CHECK_ARGUMENTS
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
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
    SET_OPTIONS
    //
    // par_all
    var_all[op_index] = conv_transpose1d(input, weight, bias, options);
}
template void conv_transpose1d_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
template void conv_transpose1d_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
// ------------------------------------------------------------------------
// forward_der
template<class TensorType>
void conv_transpose1d_op_t<TensorType>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          for_der
) const {
    user_assert(false,
        "forward_der not yet implemented for conv_transpose1d operator"
    );
    return;
}
template void conv_transpose1d_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
template void conv_transpose1d_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             for_der
) const;
// ------------------------------------------------------------------------
template<class TensorType>
void conv_transpose1d_op_t<TensorType>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          rev_der
) const {
    user_assert(false,
        "reverse_der not yet implemented for conv_transpose1d operator"
    );
}
template void conv_transpose1d_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const;
template void conv_transpose1d_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const;
// ---------------------------------------------------------------------------
// src_gen
template <> std::string conv_transpose1d_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    CHECK_ARGUMENTS
    //
    // string
    using std::string;
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
    // input_src
    size_t   input_index   = agraph.m_arg_value[arg_start];
    adtype_t input_adtype  = agraph.m_arg_type[arg_start];
    string   input_src     = tensor_src(input_index, input_adtype);
    //
    // weight_src
    size_t   weight_index   = agraph.m_arg_value[arg_start + 1];
    adtype_t weight_adtype  = agraph.m_arg_type[arg_start + 1];
    string   weight_src     = tensor_src(weight_index, weight_adtype);
    //
    // bias_src
    size_t   bias_index   = agraph.m_arg_value[arg_start + 2];
    adtype_t bias_adtype  = agraph.m_arg_type[arg_start + 2];
    string   bias_src     = tensor_src(bias_index, bias_adtype);
    //
    // target_src
    size_t   target_index = op_index;
    adtype_t target_adtype  =
        variable_agraph ? adtype_t::variable : adtype_t::parameter;
    string   target_src   = tensor_src(target_index, target_adtype);
    //
    // src
    size_t stride   = agraph.m_arg_value[arg_start + 3];
    size_t dilation = agraph.m_arg_value[arg_start + 4];
    size_t groups   = agraph.m_arg_value[arg_start + 5];
    constexpr const char* fmt1 =
R"|({{   auto options = torch::nn::functional::ConvTranspose1dFuncOptions()
        .stride({})
        .dilation({})
        .groups({});
)|";
    string src = std::format(fmt1, stride, dilation, groups);
    //
    // src
    constexpr const char* fmt2 =
        "    {} = ad_tensor::conv_transpose1d({}, {}, {}, options);";
    src += std::format(fmt2, target_src, input_src, weight_src, bias_src);
    src += "\n}";
    //
    return src;
}
template <> std::string conv_transpose1d_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    assert(false &&
        "adten_t version of src_gen called for conv_transpose1d operator"
    );
    return "";
}
// ---------------------------------------------------------------------------
} } // End ad_tensor::dev
