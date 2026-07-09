#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
//
namespace ad_tensor { namespace dev {
    //
    // rng_depend
    std::tuple< vector<bool>, vector<bool>, vector<bool> > rng_depend(
        const adfn_t* adfn
    );
} }
