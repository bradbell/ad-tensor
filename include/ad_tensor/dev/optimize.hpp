#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// TODO; split this up into separate include files for each function
// in the include/ad_tensor/dev/optimize directory.
#include <array>
#include <ad_tensor/adfn.hpp>
//
namespace ad_tensor { namespace dev {
    //
    // rng_depend
    std::array< vector<bool>, 4 > rng_depend(const adfn_t* adfn);
    //
    // new_call
    size_t new_call(
        vitype_t                  agraph_type,
        agraph_t&                 agraph_new,
        const agraph_t&           agraph_old,
        size_t                    op_index_old,
        const vector<bool>&       depend_old,
        vector<size_t>&           old2new,
        std::map<size_t, size_t>& hash2old_agraph
    );
    //
    // new_split
    size_t new_split(
        vitype_t                  agraph_type,
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
        vitype_t                 agraph_type     ,
        const agraph_t&          agraph_old      ,
        const vector<size_t>&    rng_index_old   ,
        const vector<vitype_t>&  rng_vitype_old ,
        const vector<bool>&      depend_old
    );
} }
