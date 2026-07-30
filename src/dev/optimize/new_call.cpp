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
        agraph_type,
        agraph_new,
        agraph_old,
        op_index_old,
        old2new,
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

agraph_type
***********
is the type for the graphs and must be ad_type_t::parameter
or ad_type_t::variable.

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

old2new
*******
is the mapping from indices in the old graph to indices in the new graph
for agraph_type.
If not_used is equal (not equal) to old2new for an old index,
the corresponding result is not (is) in the new graph.
On input (output), this is only defined for old indices less than or equal
the input (output) value of op_index_old.

{xrst_end new_call}
*/
#include <ad_tensor/dev/optimize.hpp>
#include <ad_tensor/dev/unpack_call.hpp>
//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
//
// BEGIN_NEW_CALL
size_t new_call(
    ad_type_t             agraph_type,
    agraph_t&             agraph_new,
    const agraph_t&       agraph_old,
    size_t                op_index_old,
    const vector<bool>&   depend_old,
    vector<size_t>&       old2new)
{   // END_NEW_CALL
    //
    assert( depend_old[op_index_old] );
    //
    // not_used
    size_t not_used = std::numeric_limits<size_t>::max();
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
    for(size_t k = 0; k < n_result_old; ++k) {
        old2new[op_index_old + k] = not_used;
        if( depend_old[op_index_old + k] ) {
            size_t op_index_new = agraph_new.m_op_seq.size() + n_result_new;
            old2new[op_index_old + k]  = op_index_new;
            ++n_result_new;
        }
    }
    assert( 0 < n_result_new );
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
        if( arg_type != agraph_type ) {
            agraph_new.m_arg_value.push_back( arg_value_old );
        } else {
            if( ! depend_old[arg_value_old] ) {
                // empty_at_ten()
                agraph_new.m_arg_value.push_back( 0 );
                arg_type = ad_type_t::constant;
            } else {
                agraph_new.m_arg_value.push_back( old2new[arg_value_old] );
            }
        }
        agraph_new.m_arg_type.push_back( arg_type );
    }
    // agraph_new: m_arg_value, m_arg_type
    for(size_t k = 0; k < n_result_old; ++k) {
        if( depend_old[op_index_old + k] ) {
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
