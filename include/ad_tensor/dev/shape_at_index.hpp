#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/vector.hpp>
//
namespace ad_tensor { namespace dev {
    template <class TensorType>
    c10::IntArrayRef shape_at_arg_index(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<TensorType>& par_vec   );
    //
    template <class TensorType>
    c10::IntArrayRef shape_at_arg_index(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<TensorType>& par_vec   ,
        const vector<TensorType>& var_vec   );
} }
