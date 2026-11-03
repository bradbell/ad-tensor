// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/to_string.hpp>
#include <ad_tensor/dev/src_gen_index.hpp>
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/dev/rev_sum_view.hpp>
#include <ad_tensor/defined.hpp>
//
// UNPACK
// arg_start, operand_index, keepdim, n_dim
// use 'git grep n_split *= *agraph' to see where this order is assumed.
#define UNPACK \
    size_t    arg_start     = agraph.m_arg_start[op_index]; \
    size_t    operand_index = agraph.m_arg_value[arg_start]; \
    size_t    dim           = agraph.m_arg_value[arg_start + 1]; \
    size_t    n_split       = agraph.m_arg_value[arg_start + 2];
//
// EXPECT_ARGUMENTS(operand_adtype)
#if NDEBUG
#define EXPECT_ARGUMENTS(operand_adtype)
#else
#define EXPECT_ARGUMENTS(operand_adtype) \
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start; \
    assert( n_arg == 3 + n_split && "split_op: n_arg != 3 + n_split" ); \
    if( operand_adtype == adtype_t::none ) { \
        assert(  \
            agraph.m_arg_type[arg_start] == adtype_t::parameter || \
            agraph.m_arg_type[arg_start] == adtype_t::variable \
        ); \
    } \
    for(size_t i = 1; i < n_arg; ++i) { \
        assert( agraph.m_arg_type[arg_start + i] == adtype_t::none ); \
    }
#endif
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void split_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<TensorType>&          par_all
) const {
    //
    // arg_start, operand_index, dim, n_split
    UNPACK
    EXPECT_ARGUMENTS( adtype_t::parameter );
    //
    // split_size
    const size_t* begin = agraph.m_arg_value.data() + arg_start + 3;
    const size_t* end   = begin + n_split;
    const vector<int64_t> split_size(begin, end);
    //
    // var_all
    vector<TensorType> vec = par_all[operand_index].split(split_size, dim);
    for(size_t i = 0; i < n_split; ++i) {
        par_all[op_index + i] = std::move( vec[i] );
    }
}
template void split_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const;
template void split_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void split_op_t<TensorType>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    vector<TensorType>&          var_all
) const {
    //
    // arg_start, operand_index, dim, n_split
    UNPACK
    EXPECT_ARGUMENTS( adtype_t::variable );
    //
    // split_size
    const size_t* begin = agraph.m_arg_value.data() + arg_start + 3;
    const size_t* end   = begin + n_split;
    const vector<int64_t> split_size(begin, end);
    //
    // var_all
    vector<TensorType> vec = var_all[operand_index].split(split_size, dim);
    for(size_t i = 0; i < n_split; ++i) {
        var_all[op_index + i] = std::move( vec[i] );
    }
}
template void split_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
template void split_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
// ------------------------------------------------------------------------
// forward_der
template<class TensorType>
void split_op_t<TensorType>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          for_der
) const {
    //
    // arg_start, operand_index, dim, n_split
    UNPACK
    EXPECT_ARGUMENTS( adtype_t::variable );
    //
    // split_size
    const size_t* begin = agraph.m_arg_value.data() + arg_start + 3;
    const size_t* end   = begin + n_split;
    const vector<int64_t> split_size(begin, end);
    //
    // var_all
    vector<TensorType> vec = for_der[operand_index].split(split_size, dim);
    for(size_t i = 0; i < n_split; ++i) {
        for_der[op_index + i] = std::move( vec[i] );
    }
}
template void split_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
template void split_op_t<adten_t>::forward_der(
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
void split_op_t<TensorType>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          rev_der
) const {
    //
    // arg_start, operand_index, dim, n_split
    UNPACK
    EXPECT_ARGUMENTS( adtype_t::variable );
    //
    // tensor_list
    vector<TensorType> tensor_list;
    for(size_t i = 0; i < n_split; ++i) {
        tensor_list.push_back( rev_der[op_index + i] );
    }
    //
    // rev_der[operand_index]
    rev_der[operand_index] = ad_tensor::cat(tensor_list, dim);
}
template void split_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const;
template void split_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const;
// ---------------------------------------------------------------------------
// src_gen
template <> std::string split_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    adtype_t                                              op_adtype       ,
    const agraph_t&                                       agraph          ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    //
    // arg_start, operand_index, dim, n_split
    UNPACK
    EXPECT_ARGUMENTS(adtype_t::none)
    //
    // string
    using std::string;
    //
    // openum_src
    string op_enum_src = to_string( agraph.m_op_seq[op_index] );
    //
    // operand_src
    adtype_t operand_adtype = agraph.m_arg_type[arg_start];
    string   operand_src    = tensor_src(operand_index, operand_adtype);
    //
    // src
    constexpr const char* fmt = "{{   vector<Tensor> at_split = {}.split({{";
    string src = std::format(fmt, operand_src);
    for(size_t i = 0; i < n_split; ++i) {
        size_t size = agraph.m_arg_value[arg_start + 3 + i];
        if( i == 0 ) {
            src += std::format("{}", size);
        } else {
            src += std::format(", {}", size);
        }
    }
    src += std::format("}}, {});\n", dim);
    for(size_t i = 0; i < n_split; ++i) {
        string ten_src     = tensor_src(op_index + i, operand_adtype);
        src += std::format("    {} = at_split[{}];\n", ten_src, i);
    }
    src += "}";
    //
    return src;
}
template <> std::string split_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    adtype_t                                              op_adtype       ,
    const agraph_t&                                       agraph          ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    assert(false && "adten_t version of src_gen called for sum operator");
    return "";
}
} } // End ad_tensor::dev
