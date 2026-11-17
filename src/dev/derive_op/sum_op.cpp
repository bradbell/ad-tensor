// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/int64_sub_vec.hpp>
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
#define UNPACK \
    size_t    arg_start     = agraph.m_arg_start[op_index]; \
    size_t    operand_index = agraph.m_arg_value[arg_start]; \
    bool      keepdim       = agraph.m_arg_value[arg_start + 1]; \
    size_t    offset        = agraph.m_arg_value[arg_start + 2];
//
// EXPECT_ARGUMENTS(operand_vitype)
#if NDEBUG
#define EXPECT_ARGUMENTS(operand_vitype)
#else
#define EXPECT_ARGUMENTS(operand_vitype) \
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start; \
    assert( n_arg == 3 && "sum_op: n_arg != 3" ); \
    if( operand_vitype == vitype_t::none ) { \
        assert(  \
            agraph.m_arg_type[arg_start] == vitype_t::parameter || \
            agraph.m_arg_type[arg_start] == vitype_t::variable \
        ); \
    } else { \
        assert( agraph.m_arg_type[arg_start] == operand_vitype ); \
    }
#endif
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void sum_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    vector<TensorType>&          par_all
) const {
    //
    // arg_start, operand_index, keepdim, n_dim
    UNPACK
    EXPECT_ARGUMENTS( vitype_t::parameter );
    //
    // dim
    c10::IntArrayRef dim = int64_sub_vec(offset, int64_vec);
    //
    // par_all
    par_all[op_index] = par_all[operand_index].sum(dim, keepdim);
}
template void sum_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    vector<at::Tensor>&          par_all
) const;
template void sum_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    vector<adten_t>&             par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void sum_op_t<TensorType>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<TensorType>&    par_all     ,
    vector<TensorType>&          var_all
) const {
    //
    // arg_start, operand_index, keepdim, n_dim
    UNPACK
    EXPECT_ARGUMENTS( vitype_t::variable );
    //
    // dim
    c10::IntArrayRef dim = int64_sub_vec(offset, int64_vec);
    //
    // var_all
    var_all[op_index] = var_all[operand_index].sum(dim, keepdim);
}
template void sum_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
template void sum_op_t<adten_t>::forward_var(
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
void sum_op_t<TensorType>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          for_der
) const {
    //
    // arg_start, operand_index, keepdim, n_dim
    UNPACK
    EXPECT_ARGUMENTS( vitype_t::variable );
    //
    if( not_defined(for_der[operand_index]) ) {
        return;
    }
    //
    // dim
    c10::IntArrayRef dim = int64_sub_vec(offset, int64_vec);
    //
    // for_der
    for_der[op_index] = for_der[operand_index].sum(dim, keepdim);
}
template void sum_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
template void sum_op_t<adten_t>::forward_der(
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
void sum_op_t<TensorType>::reverse_der(
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
    // arg_start, operand_index, keepdim, n_dim
    UNPACK
    EXPECT_ARGUMENTS( vitype_t::variable );
    assert( is_defined(rev_der[op_index]) );
    //
    // operand_shape
    c10::IntArrayRef operand_shape  = var_all[operand_index].sizes();
    //
    // rev_der[operand_index]
    if( rev_der[op_index].numel() == 1 || keepdim ) {
        if( not_defined(rev_der[operand_index]) ) {
            TensorType zeros = TensorType( torch::zeros(operand_shape) );
            rev_der[operand_index] = zeros + rev_der[op_index];
        } else {
            rev_der[operand_index] += rev_der[op_index];
        }
    } else {
        //
        // dim
        c10::IntArrayRef dim = int64_sub_vec(offset, int64_vec);
        assert( dim.size() != 0 );
        //
        // res_reshape
        rev_sum_view(
            dim,
            var_all[op_index].sizes(),
            var_all[operand_index].sizes(),
            array
        );
        c10::IntArrayRef res_reshape(array);
        if( not_defined(rev_der[operand_index]) ) {
            TensorType zeros = TensorType( torch::zeros(operand_shape) );
            rev_der[operand_index] = zeros + \
                rev_der[op_index].view(res_reshape);
        } else {
            rev_der[operand_index] += rev_der[op_index].view(res_reshape);
        }
    }
}
template void sum_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<int64_t>&       int64_vec   ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const;
template void sum_op_t<adten_t>::reverse_der(
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
template <> std::string sum_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    vitype_t                                              op_vitype       ,
    const agraph_t&                                       agraph          ,
    const vector<int64_t>&                                int64_vec       ,
    const std::function< std::string(size_t, vitype_t) >& tensor_src
) const {
    //
    // arg_start, operand_index, keepdim, n_dim
    UNPACK
    EXPECT_ARGUMENTS(vitype_t::none)
    //
    // dim
    c10::IntArrayRef dim = int64_sub_vec(offset, int64_vec);
    //
    // string
    using std::string;
    //
    // openum_src
    string op_enum_src = to_string( agraph.m_op_seq[op_index] );
    //
    // operand_src
    vitype_t operand_vitype = agraph.m_arg_type[arg_start];
    string   operand_src    = tensor_src(operand_index, operand_vitype);
    //
    // target_src
    string target_src = tensor_src(op_index, op_vitype);
    //
    // src
    string src = std::format(
        "{} = {}.{}({{", target_src, operand_src, op_enum_src
    );
    for(size_t i = 0; i < dim.size(); ++i) {
        if( i == 0 ) {
            src += std::format("{}", dim[i]);
        } else {
            src += std::format(", {}", dim[i]);
        }
    }
    if( keepdim ) {
        src += "}, true);";
    } else {
        src += "}, false);";
    }
    //
    return src;
}
template <> std::string sum_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    vitype_t                                              op_vitype       ,
    const agraph_t&                                       agraph          ,
    const vector<int64_t>&                                int64_vec       ,
    const std::function< std::string(size_t, vitype_t) >& tensor_src
) const {
    assert(false && "adten_t version of src_gen called for sum operator");
    return "";
}
} } // End ad_tensor::dev
