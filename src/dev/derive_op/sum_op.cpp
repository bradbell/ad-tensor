// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/dev/rev_sum_view.hpp>
#include <ad_tensor/no_elements.hpp>
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void sum_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<TensorType>&          par_all
) const {
    //
    // arg_start
    size_t    arg_start = agraph.m_arg_start[op_index];
    //
    // n_dim
    size_t n_dim = agraph.m_arg_value[arg_start + 1];
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 1] );
    //
#ifndef NDEBUG
    //
    // adtype
    adtype_t adtype   = agraph.m_arg_type[arg_start];
    assert( adtype  == adtype_t::parameter );
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 + n_dim && "sum_op: n_arg != 2 + n_dim" );
#endif
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    if( n_dim == 0 ) {
        // par_all
        par_all[op_index] = par_all[operand_index].sum();
    } else {
        //
        // dim
        const size_t* begin = agraph.m_arg_value.data() + arg_start + 2;
        const size_t* end   = begin + n_dim;
        const vector<int64_t>   dim(begin, end);
        //
        // par_all
        par_all[op_index] = par_all[operand_index].sum(dim);
    }
}
template void sum_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const;
template void sum_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void sum_op_t<TensorType>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    vector<TensorType>&          var_all
) const {
    //
    // arg_start
    size_t    arg_start = agraph.m_arg_start[op_index];
    //
    // n_dim
    size_t n_dim = agraph.m_arg_value[arg_start + 1];
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 1] );
    //
#ifndef NDEBUG
    //
    // adtype
    adtype_t adtype   = agraph.m_arg_type[arg_start];
    assert( adtype  == adtype_t::variable );
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 + n_dim && "sum_op: n_arg != 2 + n_dim" );
#endif
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    //
    if( n_dim == 0 ) {
        // var_all
        var_all[op_index] = var_all[operand_index].sum();
    } else {
        //
        // dim
        const size_t* begin = agraph.m_arg_value.data() + arg_start + 2;
        const size_t* end   = begin + n_dim;
        const vector<int64_t>   dim(begin, end);
        //
        // var_all
        var_all[op_index] = var_all[operand_index].sum(dim);
    }
}
template void sum_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
template void sum_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
// ------------------------------------------------------------------------
// forward_der
template<class TensorType>
void sum_op_t<TensorType>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          for_der
) const {
    //
    // arg_start
    size_t    arg_start = agraph.m_arg_start[op_index];
    //
    // n_dim
    size_t n_dim = agraph.m_arg_value[arg_start + 1];
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 1] );
    //
#ifndef NDEBUG
    //
    // adtype
    adtype_t adtype   = agraph.m_arg_type[arg_start];
    assert( adtype  == adtype_t::variable );
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 + n_dim && "sum_op: n_arg != 2 + n_dim" );
#endif
    // operand_index
    size_t operand_index  = agraph.m_arg_value[arg_start];
    if( no_elements(for_der[operand_index]) ) {
        return;
    }
    //
    if( n_dim == 0 ) {
        // for_der
        for_der[op_index] = for_der[operand_index].sum();
    } else {
        //
        // dim
        const size_t* begin = agraph.m_arg_value.data() + arg_start + 2;
        const size_t* end   = begin + n_dim;
        const vector<int64_t>   dim(begin, end);
        //
        // for_der
        for_der[op_index] = for_der[operand_index].sum(dim);
    }
}
template void sum_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             for_der
) const;
template void sum_op_t<at::Tensor>::forward_der(
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
void sum_op_t<TensorType>::reverse_der(
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
    // check for case where this operation is not connected to the range
    if( no_elements(rev_der[op_index]) ) {
        return;
    }
    //
    // arg_start
    size_t    arg_start = agraph.m_arg_start[op_index];
    //
    // n_dim
    size_t n_dim = agraph.m_arg_value[arg_start + 1];
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 1] );
    //
#ifndef NDEBUG
    //
    // adtype
    adtype_t adtype   = agraph.m_arg_type[arg_start];
    assert( adtype  == adtype_t::variable );
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 + n_dim && "sum_op: n_arg != 2 + n_dim" );
#endif
    // operand_index, operand_shape
    size_t           operand_index  = agraph.m_arg_value[arg_start];
    c10::IntArrayRef operand_shape  = var_all[operand_index].sizes();
    //
    // rev_der[operand_index]
    if( rev_der[op_index].numel() == 1 ) {
        if( no_elements(rev_der[operand_index]) ) {
            TensorType zeros   = TensorType( torch::zeros(operand_shape) );
            rev_der[operand_index] = zeros + rev_der[op_index];
        } else {
            rev_der[operand_index] += rev_der[op_index];
        }
    } else {
        assert( n_dim != 0 );
        //
        // dim
        const size_t* begin = agraph.m_arg_value.data() + arg_start + 2;
        const size_t* end   = begin + n_dim;
        const vector<int64_t>   dim(begin, end);
        //
        // res_shape
        rev_sum_view(
            dim,
            var_all[op_index].sizes(),
            var_all[operand_index].sizes(),
            array
        );
        c10::IntArrayRef res_shape(array);
        if( no_elements(rev_der[operand_index]) ) {
            TensorType zeros   = TensorType( torch::zeros(operand_shape) );
            rev_der[operand_index] = zeros + \
                rev_der[op_index].view(res_shape);
        } else {
            rev_der[operand_index] += rev_der[op_index].view(res_shape);
        }
    }
}
template void sum_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const;
template void sum_op_t<at::Tensor>::reverse_der(
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
std::string sum_op_t<TensorType>::src_gen(
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
    // n_dim
    size_t n_dim = agraph.m_arg_value[arg_start + 1];
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 1] );
    //
    //
    // adtype
    adtype_t adtype   = agraph.m_arg_type[arg_start];
    //
#ifndef NDEBUG
    if( variable_agraph ) {
        assert( adtype  == adtype_t::variable );
    } else {
        assert( adtype  == adtype_t::parameter );
    }
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 + n_dim && "sum_op: n_arg != 2 + n_dim" );
#endif
    //
    // operand_src
    size_t   operand_index  = agraph.m_arg_value[arg_start];
    adtype_t operand_adtype = agraph.m_arg_type[arg_start];
    string   operand_src    = tensor_src(operand_index, operand_adtype);
    //
    // target_src
    size_t   target_index  = op_index;
    adtype_t target_adtype = adtype;
    string   target_src    = tensor_src(target_index, target_adtype);
    //
    // src
    string src;
    if( n_dim == 0 ) {
        constexpr const char* fmt = "{} = {}.sum();";
        src = std::format(fmt, target_src, operand_src);
    } else {
        constexpr const char* fmt1 = "{{   std::array<int64_t,{}> dim = {{";
        src += std::format(fmt1, n_dim);
        for(size_t i = 0; i < n_dim; ++i) {
            assert( agraph.m_arg_type[arg_start + 2 + i] == adtype_t::none );
            size_t index = agraph.m_arg_value[arg_start + 2 + i];
            if( 0 < i ) {
                src += ", ";
            }
            src += std::to_string(index);
        }
        src += "};\n";
        constexpr const char* fmt2 = "    {} = {}.sum(dim);\n";
        src += std::format(fmt2, target_src, operand_src);
        src += "}";
    }
    return src;
}
template std::string sum_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const;
template std::string sum_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const;
} } // End ad_tensor::dev
