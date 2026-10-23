// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
//
// UNPACK
#define UNPACK \
    size_t  arg_start = agraph.m_arg_start[op_index];                 \
    int64_t dim       = int64_t( agraph.m_arg_value[arg_start + 1] ); \
    int64_t start     = int64_t( agraph.m_arg_value[arg_start + 2] ); \
    int64_t end       = int64_t( agraph.m_arg_value[arg_start + 3] ); \
    int64_t step      = int64_t( agraph.m_arg_value[arg_start + 4] );
//
// CHECK_ARGUMENTS
#ifdef NDEBUG
    #define CHECK_ARGUMENTS
#else
    #define CHECK_ARGUMENTS \
    {   size_t n_arg = agraph.m_arg_start[op_index + 1] - arg_start; \
        assert( n_arg == 5  ); \
        for(size_t i = 1; i < 5; ++i) { \
            assert( agraph.m_arg_type[arg_start+i] == adtype_t::none ); \
        } \
    }
#endif
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void slice_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<TensorType>&          par_all
) const {
    //
    // arg_start, dim, start, end, step
    UNPACK
    CHECK_ARGUMENTS
    assert( agraph.m_arg_type[arg_start] == (adtype_t::parameter) );
    //
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    // par_all
    par_all[op_index] = par_all[operand_index].slice(dim, start, end, step);
}
template void slice_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const;
template void slice_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void slice_op_t<TensorType>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    vector<TensorType>&          var_all
) const {
    //
    // arg_start, dim, start, end, step
    UNPACK
    CHECK_ARGUMENTS
    assert( agraph.m_arg_type[arg_start] == (adtype_t::variable) );
    //
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    // var_all
    var_all[op_index] = var_all[operand_index].slice(dim, start, end, step);
}
template void slice_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
template void slice_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
// ------------------------------------------------------------------------
// forward_der
template<class TensorType>
void slice_op_t<TensorType>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          for_der
) const {
    //
    // arg_start, dim, start, end, step
    UNPACK
    CHECK_ARGUMENTS
    assert( agraph.m_arg_type[arg_start] == (adtype_t::variable) );
    //
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    // var_all
    for_der[op_index] = for_der[operand_index].slice(dim, start, end, step);
}
template void slice_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
template void slice_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             for_der
) const;
// ------------------------------------------------------------------------
template<class TensorType>
void slice_op_t<TensorType>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          rev_der
) const {
    //
    // arg_start, dim, start, end, step
    UNPACK
    CHECK_ARGUMENTS
    assert( agraph.m_arg_type[arg_start] == (adtype_t::variable) );
    //
    if( step != 1 ) { user_assert(false ,
        "reverse_der: slice operation with step !=1 is not yet implemented"
    ); }
    //
    // operand_index, operand_shape
    size_t           operand_index = agraph.m_arg_value[arg_start];
    c10::IntArrayRef operand_shape = var_all[operand_index].sizes();
    //
    // operand_bar
    TensorType operand_bar;
    if( start == 0 && end == operand_shape[dim] ) {
        operand_bar = rev_der[op_index];
    } else {
        //
        // zero_start, zero_end
        vector<int64_t> shape( operand_shape.begin(), operand_shape.end() );
        shape[dim]            = start;
        TensorType zero_start = TensorType( torch::zeros( shape ) );
        shape[dim]            = operand_shape[dim] - end;
        TensorType zero_end   = TensorType( torch::zeros( shape ) );
        //
        // operand_bar
        if( start == 0 ) {
            operand_bar = torch::cat( {rev_der[op_index], zero_end}, dim );
        } else if( end == operand_shape[dim] ) {
            operand_bar = torch::cat( {zero_start, rev_der[op_index]}, dim );
        } else {
            operand_bar = torch::cat(
                {zero_start, rev_der[op_index], zero_end}, dim
            );
        }
    }
    //
    // rev_der[operand_index]
    plus_equal(rev_der[operand_index], operand_bar);
}
template void slice_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const;
// TODO: use template version of reverse_der to implement this once
// forward_var and forward_par are implemented for cat operator.
template <> void slice_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const {
    assert(false && "slice not yet implemented for adten_t version of slice");
}
// ---------------------------------------------------------------------------
// src_gen
template <> std::string slice_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    adtype_t                                              op_adtype       ,
    const agraph_t&                                       agraph          ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    //
    // string
    using std::string;
    //
    // arg_start, dim, start, end, step
    UNPACK
    CHECK_ARGUMENTS
    assert(
        agraph.m_arg_type[arg_start] == adtype_t::parameter ||
        agraph.m_arg_type[arg_start] == adtype_t::variable
    );
    //
    // operand_src
    size_t   operand_index  = agraph.m_arg_value[arg_start];
    adtype_t operand_adtype = agraph.m_arg_type[arg_start];
    string   operand_src    = tensor_src(operand_index, operand_adtype);
    //
    // target_src
    string target_src = tensor_src(op_index, op_adtype);
    //
    // src
    constexpr const char* fmt = "{} = {}.slice({}, {}, {}, {});";
    string src = std::format(
        fmt, target_src, operand_src, dim, start, end, step
    );
    //
    return src;
}
template <> std::string slice_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    adtype_t                                              op_adtype       ,
    const agraph_t&                                       agraph          ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    assert(false && "adten_t version of src_gen called for slice operator");
    return "";
}
} } // End ad_tensor::dev
