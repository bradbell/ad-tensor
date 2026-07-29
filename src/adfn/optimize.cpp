// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn_optimize usr}

Reduce The Operations and Memory Used by an AD Function
#######################################################

Syntax
******
{xrst_code cpp}
    adfn_new  = adfn_old.optimize()
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_OPTIMIZE, END_OPTIMIZE
}

adfn_old
********
This is the original adfn_t representation of a function:

    range = adfn(dom_par, dom_var)

adfn_new
********
This is the new adfn_t representation of a function:

    range = adfn(dom_par, dom_var)

The optimization will remove unnecessary constants, parameters, and variables;
see :ref:`adfn@n_con` ,
see :ref:`adfn@n_par` ,
see :ref:`adfn@n_var` .

name
****
The name for adfn_new will be the name for adfn_old plus "_optimize" .

trace
*****
The trace setting for adfn_new (directly after the optimization)
will be the same as adfn_old.

{xrst_end adfn_optimize}
*/
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/dev/optimize.hpp>
#include <ad_tensor/dev/to_string.hpp>
//
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

// BEGIN_OPTIMIZE
adfn_t adfn_t::optimize(void)
{   // END_OPTIMIZE
    //
    // adfn_old
    const adfn_t& adfn_old = *this;
    //
    // depend_con, depend_par, depend_var
    auto [depend_con, depend_par, depend_var] = dev::rng_depend( &adfn_old );
    //
    // m_con, m_par, m_var
    optimize_con( depend_con );
    //
    // depend_con
    depend_con.resize( m_con.size() );
    for(size_t i = 0; i < m_con.size(); ++i) {
        depend_con[i] = true;
    }
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
        adfn_new.m_con.push_back( adfn_old.m_con[i_old] );
    }
    //
    // adfn_new.m_rng_ad_type, m_rng_shapes
    adfn_new.m_rng_ad_type  = adfn_old.m_rng_ad_type;
    adfn_new.m_rng_shapes   = adfn_old.m_rng_shapes;
    //
    // adfn_new: m_par, m_rng_index
    ad_type_t agraph_type = ad_type_t::parameter;
    std::tie(adfn_new.m_par, adfn_new.m_rng_index) = dev::new_agraph(
        agraph_type,
        adfn_old.m_par,
        adfn_old.m_rng_index,
        adfn_old.m_rng_ad_type,
        depend_par
    );
    //
    // adfn_new.m_var
    bool var_op = true;
    adfn_new.m_var = new_agraph(
        adfn_old.m_var, var_op, depend_var, old2new_par, old2new_var
    );
    //
    // adfn_new: m_rng_index
    size_t n_range = adfn_old.m_rng_index.size();
    for(size_t i = 0; i < n_range; ++i) {
        ad_type_t              ad_type   = adfn_old.m_rng_ad_type[i];
        size_t                 index_old = adfn_old.m_rng_index[i];
        switch( ad_type ) {
            //
            case ad_type_t::constant:
            case ad_type_t::parameter:
            break;
            //
            case ad_type_t::variable:
            adfn_new.m_rng_index[i] = old2new_var[index_old];
            break;
            //
            default:
            break;
        }
    }
    // m_name, trace
    adfn_new.m_name = adfn_old.m_name + "_optimize";
    adfn_new.set_trace( adfn_old.get_trace() );
    //
    return adfn_new;
}

} // END_AD_TENSOR_NAMESPACE
