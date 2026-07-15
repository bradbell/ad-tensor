// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/dev/optimize.hpp>

namespace ad_tensor { // BEGIN_AD_TENSOR_NAMESPACE

adfn_t adfn_t::optimize(const adfn_t& adfn_old) {
    //
    // not_used
    size_t not_used = std::numeric_limits<size_t>::max();
    //
    // depend_con, depend_par, depend_var
    auto [depend_con, depend_par, depend_var] = dev::rng_depend( &adfn_old );
    //
    // n_dom_par, n_dom_var
    size_t n_dom_par = adfn_old.m_par.m_dom_shapes.size();
    size_t n_dom_var = adfn_old.m_var.m_dom_shapes.size();
    //
    // old2new_con, old2new_par, old2new_var
    auto [old2new_con, old2new_par, old2new_var] = dev::old2new(
        n_dom_par, n_dom_var, depend_con, depend_par, depend_var
    );
    //
    // new_adfn
    adfn_t new_adfn;
    //
    // new_adfn.m_con
    for(size_t i_old = 0; i_old < adfn_old.m_con.size(); ++i_old) {
        if( old2new_con[i_old] != not_used ) {
            assert( old2new_con[i_old] == new_adfn.m_con.size() );
            new_adfn.m_con.push_back( adfn_old.m_con[i_old] );
        }
    }
    return new_adfn;
}

} // END_AD_TENSOR_NAMESPACE
