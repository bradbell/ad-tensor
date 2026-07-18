// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/dev/optimize.hpp>
#include <ad_tensor/dev/to_string.hpp>

namespace {
    using std::cout;
    //
    void print_old2new(
        const std::string&               name,
        const ad_tensor::vector<size_t>& old2new) {
        size_t not_used = std::numeric_limits<size_t>::max();
        cout <<  name << " = [";
        for(size_t i = 0; i < old2new.size(); ++i) {
            if( old2new[i] == not_used ) {
                cout << "not_used";
            } else {
                cout << old2new[i];
            }
            if( i + 1 < old2new.size() ) {
                cout << ", ";
            }
        }
        cout << "]\n";
        return;
    }
}

namespace ad_tensor { // BEGIN_AD_TENSOR_NAMESPACE

adfn_t adfn_t::optimize(void) const {
    //
    // adfn_old
    const adfn_t& adfn_old = *this;
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
    if( adfn_old.get_trace() ) {
        cout << "Begin tracing " + adfn_old.get_name() + ".optimize\n";
        print_old2new( "old2new_con", old2new_con );
        print_old2new( "old2new_par", old2new_par );
        print_old2new( "old2new_var", old2new_var );
        cout << "End tracing " + adfn_old.get_name() + ".optimize\n";
    }


    //
    // adfn_new
    adfn_t adfn_new;
    //
    // adfn_new.m_con
    for(size_t i_old = 0; i_old < adfn_old.m_con.size(); ++i_old) {
        if( old2new_con[i_old] != not_used ) {
            assert( old2new_con[i_old] == adfn_new.m_con.size() );
            adfn_new.m_con.push_back( adfn_old.m_con[i_old] );
        }
    }
    //
    // adfn_new.m_par
    bool var_op = false;
    adfn_new.m_par = new_agraph(
        adfn_old.m_par, var_op, old2new_con, old2new_par, old2new_var
    );
    //
    // adfn_new.m_var
    var_op = true;
    adfn_new.m_var = new_agraph(
        adfn_old.m_var, var_op, old2new_con, old2new_par, old2new_var
    );
    //
    // adfn_new: m_rng_index, m_rng_ad_type, m_rng_shapes
    size_t n_range = adfn_old.m_rng_index.size();
    for(size_t i = 0; i < n_range; ++i) {
        const vector<int64_t>& shape     = adfn_old.m_rng_shapes[i];
        ad_type_t              ad_type   = adfn_old.m_rng_ad_type[i];
        size_t                 index_old = adfn_old.m_rng_index[i];
        size_t                 index_new = not_used;
        switch( ad_type ) {
            //
            case ad_type_t::constant:
            index_new = old2new_con[index_old];
            break;
            //
            case ad_type_t::parameter:
            index_new = old2new_par[index_old];
            break;
            //
            case ad_type_t::variable:
            index_new = old2new_var[index_old];
            break;
            //
            default:
            break;
        }
        assert( index_new != not_used );
        //
        adfn_new.m_rng_shapes.push_back( shape );
        adfn_new.m_rng_index.push_back( index_new );
        adfn_new.m_rng_ad_type.push_back( ad_type );
    }
    // m_name, trash
    adfn_new.m_name = adfn_old.m_name + "_optimize";
    adfn_new.set_trace( adfn_old.get_trace() );
    //
    return adfn_new;
}

} // END_AD_TENSOR_NAMESPACE
