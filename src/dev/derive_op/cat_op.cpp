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
    par_all[op_index] = ad_tensor::cat(tensor_list, dim);
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
    // var_all
    var_all[op_index] = ad_tensor::cat(tensor_list, dim);
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
        size_t   tensor_index  = agraph.m_arg_value[arg_start + 2 + i];
        adtype_t tensor_adtype = agraph.m_arg_type[arg_start + 2 + i];
        switch( tensor_adtype ) {
            //
            case adtype_t::variable:
            tensor_list.push_back( for_der[tensor_index] );
            break;
            //
            case adtype_t::parameter: {
                c10::IntArrayRef shape = par_all[tensor_index].sizes();
                TensorType       zero  = TensorType( torch::zeros( shape ) );
                tensor_list.push_back( zero );
            }
            break;
            //
            case adtype_t::constant: {
                c10::IntArrayRef shape = con_vec[tensor_index].sizes();
                TensorType       zero  = TensorType( torch::zeros( shape ) );
                tensor_list.push_back( zero );
            }
            break;
            //
            default:
            assert(false && "cat: unexpected tensor type in tensor list");
        }
    }
    //
    // for_der
    for_der[op_index] = ad_tensor::cat(tensor_list, dim);
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
    // arg_start, dim, n_ten
    UNPACK
    EXPECT_ARGUMENTS
    //
    // start
    int64_t start = 0;
    for(size_t i = 0; i < n_ten; ++i) {
        //
        // shape
        c10::IntArrayRef shape;
        size_t   tensor_index  = agraph.m_arg_value[arg_start + 2 + i];
        adtype_t tensor_adtype = agraph.m_arg_type[arg_start + 2 + i];
        switch( tensor_adtype ) {
            //
            case adtype_t::variable:
            shape = var_all[tensor_index].sizes();
            break;
            //
            case adtype_t::parameter:
            shape = par_all[tensor_index].sizes();
            break;
            //
            case adtype_t::constant:
            shape = con_vec[tensor_index].sizes();
            break;
            //
            default:
            assert(false && "cat: unexpected tensor type in tensor list");
        }
        //
        // end
        int64_t end = start + shape[dim];
        //
        // rev_der[tensor_index]
        int64_t step = 1;
        TensorType tensor_bar = rev_der[op_index].slice(dim, start, end, step);
        plus_equal( rev_der[tensor_index], tensor_bar );
        //
        // start
        start = end;
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
    // arg_start, dim, n_ten
    UNPACK
    EXPECT_ARGUMENTS
    //
    // target_src
    string target_src = tensor_src(op_index, op_adtype);
    //
    // src
    string src = std::format("{} = ad_tensor::cat({{", target_src);
    for(size_t i = 0; i < n_ten; ++i) {
        size_t   operand_index  = agraph.m_arg_value[arg_start + 2 + i];
        adtype_t operand_adtype = agraph.m_arg_type[arg_start + 2 + i];
        string   operand_src    = tensor_src(operand_index, operand_adtype);
        if( i == 0 ) {
            src += std::format("{}", operand_src);
        } else {
            src += std::format(", {}", operand_src);
        }
    }
    src += std::format("}}, {});", dim);
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
