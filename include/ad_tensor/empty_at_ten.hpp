#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin empty_at_ten usr}

The Empty at::Tensor used by ad_tensor
######################################
{xrst_literal ,
    BEGIN_EMPTY_AT_TEN, END_EMPTY_AT_TEN
}
{xrst_end empty_at_ten}
*/
#include <limits>
#include <torch/torch.h>
// BEGIN_EMPTY_AT_TEN
namespace ad_tensor { inline at::Tensor empty_at_ten(void) {
    return at::Tensor();
} }
// END_EMPTY_AT_TEN
