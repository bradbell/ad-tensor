#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
//
// BEGIN_BROADCAST
namespace ad_tensor { namespace dev {
    c10::ArrayRef<long> broadcast(
        bool              lock                       ,
        const at::Tensor& res  = torch::empty( {0} ) ,
        const at::Tensor& arg  = torch::empty( {0} )
    );
} }
// END_BROADCAST
