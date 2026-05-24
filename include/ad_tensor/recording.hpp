#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
//
#include <ad_tensor/ad.hpp>
#include <ad_tensor/adfn.hpp>
//
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
//
// recording
class recording {
public:
    //
    // BEGIN_STOP
    // adfn = recording::stop(arnage)
    static adfn_t stop(const std::vector<ad_t>& arange);
    // END_STOP
};
//
} // END_NAMESPACE_AD_TENSOR
