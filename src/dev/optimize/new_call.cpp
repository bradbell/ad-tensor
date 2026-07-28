// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin new_call dev}
{xrst_spell
    op
    agraph
    std
}

Add an Atomic Function Call to a New AD Function
################################################

Syntax
******
{xrst_code cpp}
    op_index_old = new_call(
        agraph_new,
        agraph_old,
        op_index_old,
        var_op,
        old2new_par,
        old2new_var
    )
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_NEW_CALL, END_NEW_CALL
}

not_used
********
We use the notation
{xrst_code cpp}
    not_used = std::numeric_limits<size_t>::max()
{xrst_code}

agraph_new
**********
The new graph up to but not including the call the an atomic function.

agraph_old
**********
The old graph that contains the call to an atomic function.

op_index_old
************
On input op_index_old is the old graph index for a call, or a call result,
that is used in the new graph.
Upon return, is op_index_old is the next operator index,
after this call,  in the old graph.

var_op
******
If this is true (false) the is a variable (parameter) graph.

old2new_par, old2par_var
************************
is the mapping from indices in the old graph to indices in the new graph
for parameters and variables (in that order).
If not_used is equal (not equal) to an old to new index,
the corresponding result is (is not) in the new graph.

Constants
*********
It is assumed that the constant with index zero is :ref:`empty_at_ten-name` .

{xrst_end new_call}
*/
#include <ad_tensor/dev/optimize.hpp>
#include <ad_tensor/dev/unpack_call.hpp>
//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
//
// BEGIN_NEW_CALL
size_t new_call(
    agraph_t&             agraph_new,
    const agraph_t&       agraph_old,
    size_t                op_index_old,
    bool                  var_op,
    const vector<size_t>& old2new_par,
    const vector<size_t>& old2new_var)
{   // END_NEW_CALL
    //
    //
    // not_used
    size_t not_used = std::numeric_limits<size_t>::max();
    //
    // old2new_res
    const vector<size_t>* old2new_res = nullptr;
    if( var_op ) {
        old2new_res = &old2new_var;
    } else {
        old2new_res = &old2new_par;
    }
    assert( (*old2new_res)[op_index_old] != not_used );
    //
    // op_index_old
    op_enum_t op_enum = agraph_old.m_op_seq[op_index_old];
    while( op_enum != op_enum_t::call ) {
        assert( op_enum == op_enum_t::call_result );
        op_enum = agraph_old.m_op_seq[--op_index_old];
    }
    // arg_start, atom_id, n_domain, n_range, n_result_old
    auto [arg_start, atom_id, n_domain, n_range, n_result_old] = unpack_call(
        op_index_old, agraph_old
    );
    //
    // n_result_new
    size_t n_result_new  = 0;
#ifndef NDEBUG
    size_t first_old2new = 0;
#endif
    for(size_t k = 0; k < n_result_old; ++k) {
        if( (*old2new_res)[op_index_old + k] != not_used ) {
#ifndef NDEBUG
            first_old2new = (*old2new_res)[op_index_old + k];
#endif
            ++n_result_new;
        }
    }
    assert( 0 < n_result_new );
    //
#ifndef NDEBUG
    size_t op_index_new = agraph_new.m_op_seq.size();
    assert( first_old2new == op_index_new );
#endif
    //
    // agraph_new: m_op_seq, m_arg_start
    agraph_new.m_op_seq.push_back( dev::op_enum_t::call );
    agraph_new.m_arg_start.push_back( agraph_new.m_arg_value.size() );
    //
    // agraph_new: m_arg_value, m_arg_type
    agraph_new.m_arg_value.push_back( atom_id );
    agraph_new.m_arg_value.push_back( n_domain );
    agraph_new.m_arg_value.push_back( n_range );
    agraph_new.m_arg_value.push_back( n_result_new );
    for(size_t k = 0; k < 4; ++k) {
        agraph_new.m_arg_type.push_back( ad_type_t::none );
    }
    // agraph_new: m_arg_value, m_arg_type
    for(size_t j = 0; j < n_domain; ++j) {
        size_t    arg_index     = arg_start + 4 + j;
        size_t    arg_value_old = agraph_old.m_arg_value[arg_index];
        ad_type_t arg_type      = agraph_old.m_arg_type[arg_index];
        switch( arg_type ) {
            //
            case ad_type_t::constant:
            agraph_new.m_arg_value.push_back( arg_value_old );
            break;
            //
            case ad_type_t::parameter:
            if( old2new_par[arg_value_old] == not_used ) {
                // empty_at_ten()
                agraph_new.m_arg_value.push_back( 0 );
                arg_type = ad_type_t::constant;
            } else {
                agraph_new.m_arg_value.push_back( old2new_par[arg_value_old] );
            }
            break;
            //
            case ad_type_t::variable:
            if( old2new_var[arg_value_old] == not_used ) {
                // empty_at_ten()
                agraph_new.m_arg_value.push_back( 0 );
                arg_type = ad_type_t::constant;
            } else {
                agraph_new.m_arg_value.push_back( old2new_var[arg_value_old] );
            }
            break;
            //
            default:
            assert( false );
        }
        agraph_new.m_arg_type.push_back( arg_type );
    }
    // agraph_new: m_arg_value, m_arg_type
    for(size_t k = 0; k < n_result_old; ++k) {
        if( (*old2new_res)[op_index_old + k] != not_used ) {
            size_t arg_index = arg_start + 4 + n_domain + k;
            size_t arg_value = agraph_old.m_arg_value[ arg_index ];
            agraph_new.m_arg_value.push_back( arg_value );
            agraph_new.m_arg_type.push_back( ad_type_t::none );
        }
    }
    //
    // agraph: m_op_seq, m_arg_start
    for(size_t k = 1; k < n_result_new; ++k) {
        agraph_new.m_op_seq.push_back( dev::op_enum_t::call_result );
        agraph_new.m_arg_start.push_back( agraph_new.m_arg_value.size() );
    }
    //
    return op_index_old + n_result_old;
}
//
} } // END_AD_TENSOR_DEV_NAMESPACE
