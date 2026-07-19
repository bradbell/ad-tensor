#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin at_ten_nan usr}

The Scalar NaN Tensor
#####################
{xrst_literal ,
    BEGIN_AT_TEN_NAN, END_AT_TEN_NAN
}
{xrst_end at_ten_nan}
*/
#include <limits>
#include <torch/torch.h>
// BEGIN_AT_TEN_NAN
namespace ad_tensor { inline at::Tensor at_ten_nan(void) {
    float nan = std::numeric_limits<float>::quiet_NaN();
    return torch::tensor(nan);
} }
// END_AT_TEN_NAN
