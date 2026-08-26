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
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<> void call_result_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const {
}
template<> void call_result_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const {
}
// ------------------------------------------------------------------------
// forward_var
template<> void call_result_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const {
}
template<> void call_result_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const {
}
// ------------------------------------------------------------------------
// forward_der
template<> void call_result_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             for_der
) const {
}
template<> void call_result_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const {
}
// ------------------------------------------------------------------------
// reverse_der
template<> void call_result_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const {
}
template<> void call_result_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const {
}
} } // End ad_tensor::dev
