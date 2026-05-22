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
// record_t
class record_t {
public:
    // BEGIN_START
    // auto [adom_par, adom_var] = record_t::start(dom_par, dom_var)
    static std::tuple< std::vector<ad_t>, std::vector<ad_t> > start(
        std::vector<at::Tensor>&& dom_par,
        std::vector<at::Tensor>&& dom_var
    );
    // END_START
    //
    // BEGIN_STOP
    // adfn = record_t::stop(arnage)
    static adfn_t stop(const std::vector<ad_t>& arange);
    // END_STOP
};
//
} // END_NAMESPACE_AD_TENSOR
