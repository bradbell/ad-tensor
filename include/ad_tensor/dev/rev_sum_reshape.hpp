#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
//
// BEGIN_REV_SUM_RESHAPE
namespace ad_tensor { namespace dev {
    // shape = rev_sum_reshape(lock, dim, res, arg)
    c10::ArrayRef<long> rev_sum_reshape(
        bool                        lock                        ,
        const c10::ArrayRef<long>&  dim       = c10::ArrayRef<long>() ,
        const c10::ArrayRef<long>&  res_shape = c10::ArrayRef<long>() ,
        const c10::ArrayRef<long>&  arg_shape = c10::ArrayRef<long>()
    );
} }
// END_REV_SUM_RESHAPE
