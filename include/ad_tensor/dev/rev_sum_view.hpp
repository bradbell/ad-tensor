#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
//
// BEGIN_REV_SUM_VIEW
namespace ad_tensor { namespace dev {
    // shape = rev_sum_view(lock, dim, res, arg)
    c10::IntArrayRef rev_sum_view(
        bool                        lock                        ,
        const c10::IntArrayRef&     dim       = c10::IntArrayRef() ,
        const c10::IntArrayRef&     res_shape = c10::IntArrayRef() ,
        const c10::IntArrayRef&     arg_shape = c10::IntArrayRef()
    );
} }
// END_REV_SUM_VIEW
