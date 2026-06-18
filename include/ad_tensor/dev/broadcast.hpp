#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
//
namespace ad_tensor { namespace dev {
    void broadcast(
        const c10::IntArrayRef&    res_shape  ,
        const c10::IntArrayRef&    arg_shape  ,
        vector<int64_t>&           dim
    );
} }
