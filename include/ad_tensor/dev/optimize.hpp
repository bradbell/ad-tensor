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
    src/dev/optimize/optimize_con.cpp
    src/dev/optimize/old2new.cpp
    src/dev/optimize/new_call.cpp
    src/dev/optimize/new_agraph.cpp
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
    //
    // new_call
    size_t new_call(
        ad_type_t                 agraph_type,
        agraph_t&                 agraph_new,
        const agraph_t&           agraph_old,
        size_t                    op_index_old,
        const vector<bool>&       depend_old,
        vector<size_t>&           old2new,
        std::map<size_t, size_t>& hash2old_agraph
    );
    //
    // new_agraph
    std::tuple< agraph_t, vector<size_t> > new_agraph(
        ad_type_t                agraph_type     ,
        const agraph_t&          agraph_old      ,
        const vector<size_t>&    rng_index_old   ,
        const vector<ad_type_t>& rng_ad_type_old ,
        const vector<bool>&      depend_old
    );
} }
