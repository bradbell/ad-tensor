// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin new_agraph dev}
{xrst_spell
    op
    rng
}

Create the Optimized Version of an Acyclic Graph
################################################

Syntax
******
{xrst_code cpp}
    tie(agraph_new, rng_index_new) = new_agraph(
        agraph_type, agraph_old, rng_index_old, rng_ad_type, depend_old
    )
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_NEW_AGRAPH, END_NEW_AGRAPH
}

agraph_type
***********
is the type for the graphs and must be ad_type_t::parameter
or ad_type_t::variable.

agraph_old
**********
The old graph representing the agraph_type values before optimization.

rng_index_old
*************
The :ref:`adfn_dev@m_rng_index` values for the function before optimization
of the agraph_type operations.

rng_ad_type
***********
The :ref:`adfn_dev@m_rng_ad_type` values for the function optimization.
This the same before and after optimization.

depend_old
**********
If depend_old[op_index] is true (false) then the range values for this
AD function depend (do not depend) on the agraph_type value corresponding
to op_index.

agraph_new
**********
The new graph representing the agraph_type values that are deemed necessary.

rng_index_new
*************
The range index values for the function after optimization of the
agraph_type operations. If rng_ad_type[i] is agraph_type,
rng_index_new[i] may be different from rng_index_old[i].

TEST
****
{xrst_literal ,
    tests/adfn/optimize.cpp
    BEGIN_OPTIMIZE_AGRAPH, END_OPTIMIZE_AGRAPH
}

{xrst_end new_agraph}
*/
#include <ad_tensor/dev/optimize.hpp>
#include <ad_tensor/dev/hash_operator.hpp>
//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
//
// )
// BEGIN_NEW_AGRAPH
std::tuple< agraph_t, vector<size_t> > new_agraph(
    ad_type_t                agraph_type     ,
    const agraph_t&          agraph_old      ,
    const vector<size_t>&    rng_index_old   ,
    const vector<ad_type_t>& rng_ad_type     ,
    const vector<bool>&      depend_old   )
{   // END_NEW_AGRAPH
    //
    // hash2old_op, const_iterator
    // multimap would be more complicated, but it might be better when
    // collisions occur.
    std::map<size_t, size_t>                         hash2old_agraph;
    typedef std::map<size_t, size_t>::const_iterator const_iterator;
    //
    // not_used
    size_t not_used = std::numeric_limits<size_t>::max();
    //
    // agraph_new
    agraph_t agraph_new;
    //
    // old2new
    size_t n_op_old = agraph_old.m_op_seq.size();
    vector<size_t> old2new(n_op_old, not_used);
    //
    // agraph_new.m_dom_shapes
    for(size_t j = 0; j < agraph_old.m_dom_shapes.size(); ++j) {
        agraph_new.m_dom_shapes.push_back( agraph_old.m_dom_shapes[j] );
    }
    //
    // itr, replace, hash, op_before
    const_iterator itr;
    bool           replace;
    size_t         hash;
    size_t         op_before;
    //
    // n_old, op_index_old
    size_t n_old = agraph_old.m_op_seq.size();
    size_t n_dom = agraph_old.m_dom_shapes.size();
    size_t op_index_old = 0;
    //
    // agraph_new: m_op_seq, m_arg_start
    assert( agraph_new.m_arg_value.size() == 0 );
    while(op_index_old < n_dom) {
        agraph_new.m_op_seq.push_back( dev::op_enum_t::dom );
        agraph_new.m_arg_start.push_back( 0 );
        old2new[op_index_old] = op_index_old;
        ++op_index_old;
    }
    while(op_index_old < n_old) { if( ! depend_old[op_index_old] ) {
        ++op_index_old;
    } else {
        // op_enum
        op_enum_t op_enum = agraph_old.m_op_seq[op_index_old];
        switch( op_enum ) {
            // --------------------------------------------------------------
            // agraph_new, op_index_old
            case op_enum_t::call:
            case op_enum_t::call_result:
            op_index_old = new_call(
                agraph_type,
                agraph_new,
                agraph_old,
                op_index_old,
                depend_old,
                old2new
            );
            break;
            //
            // ---------------------------------------------------------------
            default:
            //
            // hash, itr
            hash = hash_operator(
                agraph_type, agraph_old, old2new, op_index_old
            );
            itr       = hash2old_agraph.find(hash);
            //
            // op_before, replace
            replace   = false;
            op_before = std::numeric_limits<size_t>::max();
            if( itr != hash2old_agraph.end() ) {
                op_before = itr->second;
                assert( op_before <= op_index_old );
                assert( old2new[op_before] != not_used );
                replace = new_op_equal(
                    agraph_type, agraph_old, old2new, op_index_old, op_before
                );
            }
            // old2new[op_index_old]
            if( replace ) {
                old2new[op_index_old] = old2new[op_before];
            } else {
                old2new[op_index_old] = agraph_new.m_op_seq.size();
                hash2old_agraph[hash] = op_index_old;
                //
                // agraph_new: m_op_seq, m_arg_start
                agraph_new.m_op_seq.push_back(op_enum);
                agraph_new.m_arg_start.push_back(
                        agraph_new.m_arg_value.size()
                );
                //
                // agraph_new:: m_arg_type, m_arg_value
                size_t start = agraph_old.m_arg_start[op_index_old];
                size_t end   = agraph_old.m_arg_start[op_index_old + 1];
                for(size_t arg_index = start; arg_index < end; ++arg_index) {
                    //
                    ad_type_t ad_type = agraph_old.m_arg_type[arg_index];
                    agraph_new.m_arg_type.push_back( ad_type );
                    size_t value_old = agraph_old.m_arg_value[arg_index];
                    size_t value_new = value_old;
                    if( ad_type == agraph_type ) {
                        value_new = old2new[value_old];
                    }
                    agraph_new.m_arg_value.push_back( value_new );
                }
            }
            // op_index_old
            ++op_index_old;
            // ---------------------------------------------------------------
        }
    } }
#ifndef NDEBUG
    for(size_t i_old = 0; i_old < n_old; ++i_old) {
        size_t i_new = old2new[i_old];
        if( i_new != not_used ) {
            assert( i_new < agraph_new.m_op_seq.size() );
        }
    }
#endif
    // agraph_new.m_arg_start
    agraph_new.m_arg_start.push_back( agraph_new.m_arg_value.size() );
    //
    // rng_index_new
    vector<size_t> rng_index_new = rng_index_old;
    for(size_t i = 0; i < rng_index_old.size(); ++i) {
        if( rng_ad_type[i] == agraph_type ) {
            assert( depend_old[ rng_index_old[i] ] );
            rng_index_new[i] = old2new[ rng_index_old[i] ];
        }
    }
    return std::tuple< agraph_t, vector<size_t> >(agraph_new, rng_index_new);
}
} } // END_AD_TENSOR_DEV_NAMESPACE
