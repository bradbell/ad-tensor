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
namespace ad_tensor { // BEGIN_AD_TENSOR_NAMESPACE
//
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
    if( adfn_old.get_trace() ) {
        std::cout << "Begin tracing " + adfn_old.get_name() + ".optimize\n";
        std::cout << "depend_con = " << dev::to_string( depend_con ) << "\n";
        std::cout << "depend_par = " << dev::to_string( depend_par ) << "\n";
        std::cout << "depend_var = " << dev::to_string( depend_var ) << "\n";
    }
    //
    // adfn_old: m_con, m_par, m_var, m_rng_index
    optimize_con( depend_con );
    //
    // adfn_new
    adfn_t adfn_new;
    //
    // adfn_new.m_rng_ad_type, m_rng_shapes
    adfn_new.m_con          = adfn_old.m_con;
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
    agraph_type = ad_type_t::variable;
    std::tie(adfn_new.m_var, adfn_new.m_rng_index) = dev::new_agraph(
        agraph_type,
        adfn_old.m_var,
        adfn_new.m_rng_index,
        adfn_old.m_rng_ad_type,
        depend_var
    );
    //
    // m_name, trace
    adfn_new.m_name = adfn_old.m_name + "_optimize";
    adfn_new.set_trace( adfn_old.get_trace() );
    //
    if( adfn_old.get_trace() ) {
        std::cout << "End tracing " + adfn_old.get_name() + ".optimize\n";
    }
    return adfn_new;
}

} // END_AD_TENSOR_NAMESPACE
