#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
//
namespace ad_tensor { namespace dev {
    vector<int64_t> rev_sum_view(
        const c10::IntArrayRef&     dim       = c10::IntArrayRef() ,
        const c10::IntArrayRef&     res_shape = c10::IntArrayRef() ,
        const c10::IntArrayRef&     arg_shape = c10::IntArrayRef()
    );
} }
