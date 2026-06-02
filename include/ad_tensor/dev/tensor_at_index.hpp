#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
namespace ad_tensor { namespace dev {
    template <class TensorType>
    const TensorType& tensor_at_arg_index(
        size_t                               arg_index ,
        const agraph_t&                      agraph    ,
        const ad_tensor::vector<TensorType>& con_vec   ,
        const ad_tensor::vector<TensorType>& par_vec   );
    //
    template <class TensorType>
    const TensorType& tensor_at_arg_index(
        size_t                               arg_index ,
        const agraph_t&                      agraph    ,
        const ad_tensor::vector<TensorType>& con_vec   ,
        const ad_tensor::vector<TensorType>& par_vec   ,
        const ad_tensor::vector<TensorType>& var_vec   );
} }
