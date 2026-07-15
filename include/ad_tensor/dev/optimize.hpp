#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin optimize_dev dev}

The Optimizer Implementation
############################

{xrst_toc_table
    src/dev/optimize/rng_depend.cpp
    src/dev/optimize/old2new.cpp
}

{xrst_end optimize_dev}
*/
#include <array>
#include <ad_tensor/adfn.hpp>
//
namespace ad_tensor { namespace dev {
    //
    // rng_depend
    std::array< vector<bool>, 3 > rng_depend(const adfn_t* adfn);
    //
    // old2new
    std::array< vector<size_t>, 3 > old2new(
        size_t              n_dom_par,
        size_t              n_dom_var,
        const vector<bool>& depend_con,
        const vector<bool>& depend_par,
        const vector<bool>& depend_var
    );
} }
