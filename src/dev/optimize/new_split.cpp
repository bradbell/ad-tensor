// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin new_split dev}
{xrst_spell
    op
    agraph
    std
}

Add a Split Operator to a New Acyclic Graph
###########################################

Syntax
******
{xrst_code cpp}
    op_index_old = new_split(
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
    BEGIN_NEW_SPLIT, END_NEW_SPLIT
}

not_used
********
We use the notation
{xrst_code cpp}
    not_used = std::numeric_limits<size_t>::max()
{xrst_code}

agraph_type
***********
is the type for the graphs and must be adtype_t::parameter
or adtype_t::variable.

agraph_new
**********
The new graph up to but not including the new split operator.

agraph_old
**********
The old graph that contains this split operation.

op_index_old
************
On input op_index_old is the old graph index this split, or split result,
that is used in the new graph.
Upon return, is op_index_old is the next operator index,
after the last split result for this split, in the old graph.

depend_old
**********
if depend[op_index] is true (false) the results in the old operation sequence
depend (do not depend) on the result for operation op_index.

old2new
*******
is the mapping from indices in the old graph to indices in the new graph
for agraph_type.
If not_used is equal (not equal) to old2new for an old index,
the corresponding result is not (is) in the new graph.
On input (output), this is only defined for old indices less than or equal
the input (output) value of op_index_old.

hash2old_agraph
***************
This maps operator hash codes to operator index in the old graph of a
previous operator that has this hash code.
This is both an input and an output.
To be more specific, if the hash code for this operation is not
in hash2old_agraph, it is placed there with op_index_old as the result.
If the hash code is in hash2old_agraph, but the previous old operator
is not equivalent to this one, the result for this has is changed to
op_index_old.

Test
****
{xrst_literal ,
    tests/adfn/optimize.cpp
    BEGIN_OPTIMIZE_CALL, END_OPTIMIZE_CALL
}

{xrst_end new_split}
*/
#include <map>
#include <limits>
#include <ad_tensor/dev/optimize.hpp>
#include <ad_tensor/dev/new_operator.hpp>
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
//
// BEGIN_NEW_SPLIT
size_t new_split(
    adtype_t                  agraph_type,
    agraph_t&                 agraph_new,
    const agraph_t&           agraph_old,
    size_t                    op_index_old,
    const vector<bool>&       depend_old,
    vector<size_t>&           old2new,
    std::map<size_t, size_t>& hash2old_agraph)
{   // END_NEW_SPLIT
    //
    assert( depend_old[op_index_old] );
    //
    // not_used
    size_t not_used = std::numeric_limits<size_t>::max();
    //
    // op_index_old
    op_enum_t op_enum = agraph_old.m_op_seq[op_index_old];
    while( op_enum != op_enum_t::split ) {
        assert( op_enum == op_enum_t::split_result );
        op_enum = agraph_old.m_op_seq[--op_index_old];
    }
    // n_split
    size_t arg_start = agraph_old.m_arg_start[op_index_old];
    size_t n_split   = agraph_old.m_arg_value[arg_start + 2];
    //
    // hash, itr
    size_t hash = new_op_hash(
        agraph_type, agraph_old, old2new, op_index_old
    );
    std::map<size_t, size_t>::const_iterator itr = hash2old_agraph.find(hash);
    //
    // op_before, replace
    bool   replace   = false;
    size_t op_before = not_used;
    if( itr != hash2old_agraph.end() ) {
        op_before = itr->second;
        assert( op_before <= op_index_old );
        replace = new_op_equal(
            agraph_type, agraph_old, old2new, op_index_old, op_before
        );
    }
    // old2new[op_index_old]
    if( replace ) {
        for(size_t k = 0; k < n_split; ++k) {
            assert( old2new[op_before + k] != not_used );
            old2new[op_index_old + k] = old2new[op_before + k];
        }
    } else {
        old2new[op_index_old] = agraph_new.m_op_seq.size();
        hash2old_agraph[hash] = op_index_old;
        //
        for(size_t k = 0; k < n_split; ++k) {
            size_t op_index_new = agraph_new.m_op_seq.size() + k;
            old2new[op_index_old + k]  = op_index_new;
        }
        //
        // agraph_new: m_op_seq, m_arg_start
        agraph_new.m_op_seq.push_back( dev::op_enum_t::split );
        agraph_new.m_arg_start.push_back( agraph_new.m_arg_value.size() );
        //
        // agraph_new:: m_arg_type, m_arg_value
        // TODO: new_agraph.cpp has very similar code, perhaps we should
        // refactor this as a separate function.
        size_t start = agraph_old.m_arg_start[op_index_old];
        size_t end   = agraph_old.m_arg_start[op_index_old + 1];
        for(size_t arg_index = start; arg_index < end; ++arg_index) {
            //
            adtype_t adtype = agraph_old.m_arg_type[arg_index];
            agraph_new.m_arg_type.push_back( adtype );
            size_t value_old = agraph_old.m_arg_value[arg_index];
            size_t value_new = value_old;
            if( adtype == agraph_type ) {
                value_new = old2new[value_old];
            }
            agraph_new.m_arg_value.push_back( value_new );
        }
        // agraph: m_op_seq, m_arg_start
        for(size_t k = 1; k < n_split; ++k) {
            agraph_new.m_op_seq.push_back( dev::op_enum_t::split_result );
            agraph_new.m_arg_start.push_back( agraph_new.m_arg_value.size() );
        }
    }
    //
    return op_index_old + n_split;
}
//
} } // End ad_tensor::dev
