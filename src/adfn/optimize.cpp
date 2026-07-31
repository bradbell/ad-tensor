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
    adfn.optimize()
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_OPTIMIZE, END_OPTIMIZE
}

adfn
****
is the adfn_t that we are optimizing; i.e., the mapping below does not change:

    range = adfn(dom_par, dom_var)

The optimization will remove unnecessary constants, parameters, and variables;
see :ref:`adfn@n_con` ,
see :ref:`adfn@n_par` ,
see :ref:`adfn@n_var` .

name
****
The name for the function will be changed to its old name plus "_optimize" .

{xrst_end adfn_optimize}
*/
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/dev/optimize.hpp>
#include <ad_tensor/dev/to_string.hpp>
#include <ad_tensor/dev/move_swap.hpp>
//
namespace ad_tensor { // BEGIN_AD_TENSOR_NAMESPACE
//
// BEGIN_OPTIMIZE
void adfn_t::optimize(void)
{   // END_OPTIMIZE
    //
    // adfn_old
    const adfn_t& adfn_old = *this;
    //
    // depend_con, depend_par, depend_var
    auto [depend_con, depend_par, depend_var] = dev::rng_depend( this );
    //
    if( adfn_old.get_trace() ) {
        std::cout << "Begin tracing " + adfn_old.get_name() + ".optimize\n";
        std::cout << "depend_con = " << dev::to_string( depend_con ) << "\n";
        std::cout << "depend_par = " << dev::to_string( depend_par ) << "\n";
        std::cout << "depend_var = " << dev::to_string( depend_var ) << "\n";
    }
    //
    // m_con, m_par, m_var, m_rng_index
    // constants do not depend on parameters or variables so do this first
    optimize_con( depend_con );
    //
    //
    // m_par, m_rng_index
    // The constants have been mapped to their new values
    // Parameters do not have variable arguments so we optimize them next.
    {   // Start a block so that the old m_par and m_rng_index get freeded
        // before we optimize the variables...
        ad_type_t agraph_type = ad_type_t::parameter;
        auto [ agraph_new, rng_index_new ] = dev::new_agraph(
            agraph_type,
            m_par,
            m_rng_index,
            m_rng_ad_type,
            depend_par
        );
        dev::move_swap(m_par, agraph_new);
        dev::move_swap(m_rng_index, rng_index_new);
    }
    //
    // m_var, m_rng_index
    // Variables can have constant and parameter arguments. We optimize them
    // last to make detection of equivalent operators easier.
    ad_type_t agraph_type = ad_type_t::variable;
    auto [ agraph_new, rng_index_new ] = dev::new_agraph(
        agraph_type,
        m_var,
        m_rng_index,
        m_rng_ad_type,
        depend_var
    );
    dev::move_swap(m_var, agraph_new);
    dev::move_swap(m_rng_index, rng_index_new);
    //
    // m_name
    m_name = m_name + "_optimize";
    //
    if( get_trace() ) {
        std::cout << "End tracing " + adfn_old.get_name() + ".optimize\n";
    }
}

} // END_AD_TENSOR_NAMESPACE
