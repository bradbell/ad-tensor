// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/ad.hpp>
//
namespace ad_tensor { namespace dev {
    // ----------------------------------------------------------------------
    // forward_par
    template<class TensorType>
    void dom_op_t<TensorType>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<TensorType>&          par_vec
    ) const {
    }
    template void dom_op_t<ad_t>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<ad_t>&                par_vec
    ) const;
    template void dom_op_t<at::Tensor>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<at::Tensor>&          par_vec
    ) const;
    // ----------------------------------------------------------------------
    // forward_var
    template<class TensorType>
    void dom_op_t<TensorType>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        ad_tensor::vector<TensorType>&          var_vec
    ) const {
    }
    template void dom_op_t<ad_t>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<ad_t>&          par_vec     ,
        ad_tensor::vector<ad_t>&                var_vec
    ) const;
    template void dom_op_t<at::Tensor>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        ad_tensor::vector<at::Tensor>&          var_vec
    ) const;
    // ----------------------------------------------------------------------
    // forward_der
    template<class TensorType>
    void dom_op_t<TensorType>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        const ad_tensor::vector<TensorType>&    var_vec     ,
        ad_tensor::vector<TensorType>&          for_der
    ) const {
    }
    template void dom_op_t<ad_t>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<ad_t>&          par_vec     ,
        const ad_tensor::vector<ad_t>&          var_vec     ,
        ad_tensor::vector<ad_t>&                for_der
    ) const;
    template void dom_op_t<at::Tensor>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        const ad_tensor::vector<at::Tensor>&    var_vec     ,
        ad_tensor::vector<at::Tensor>&          for_der
    ) const;
    // ----------------------------------------------------------------------
    // reverse_der
    template<class TensorType>
    void dom_op_t<TensorType>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        const ad_tensor::vector<TensorType>&    var_vec     ,
        ad_tensor::vector<TensorType>&          rev_der
    ) const {
    }
    template void dom_op_t<ad_t>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<ad_t>&          par_vec     ,
        const ad_tensor::vector<ad_t>&          var_vec     ,
        ad_tensor::vector<ad_t>&                rev_der
    ) const;
    template void dom_op_t<at::Tensor>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        const ad_tensor::vector<at::Tensor>&    var_vec     ,
        ad_tensor::vector<at::Tensor>&          rev_der
    ) const;
} }
