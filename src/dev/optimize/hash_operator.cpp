// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin hash_operator dev}
{xrst_spell
    agraph
    op
}

The Operator Hash Code Function
###############################

Syntax
******
{xrst_code cpp}
    hash = hash_operator(ad_type, agraph_old, old2new, old_index)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_HASH_OPERATOR, END_HASH_OPERATOR
}

ad_type
*******
This is the type of agraph and must be either parameter or variable.

agraph_old
**********
This is the old graph that is being optimized

old2new
*******
The idea here is that equivalent operators in the old graph
will have the same operator index in the new (optimized) graph.
Hence we want them to be treated as equal when computing hash codes.

Suppose op_index is an operator index before old_index and
arg_index is a corresponding argument index with type ad_type; i.e.,

#.  If op_index < old_index
#.  agraph_old.m_arg_start[op_index] < arg_index
#.  arg_index <  agraph_old.m_arg_start[op_index + 1]
#.  agraph_old.m_arg_type[arg_index] == ad_type

Then old2new[op_index] is the index in the new graph of an operator
that is equivalent to the operator with index op_index in the old graph.

old_index
*********
This is the index of the operator that we are computing the hash code for.
Note that all of its arguments to this operator with type ad_type have indices
less than old_index (and hence old2new is defined for all such arguments).

{xrst_end hash_operator}
------------------------------------------------------------------------------
*/
#include <limits>
#include <map>
#include <cstdint>
#include <cassert>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/optimize.hpp>
//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
//
// BEGIN_HASH_OPERATOR
// stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
size_t hash_operator(
    ad_type_t            ad_type,
    const agraph_t&      agraph_old,
    const vector<size_t> old2new,
    size_t               old_index )
{   // END_HASH_OPERATOR
    //
    // op_enum, start, end
    op_enum_t op_enum  = agraph_old.m_op_seq[old_index];
    size_t start       = agraph_old.m_arg_start[old_index];
    size_t end         = agraph_old.m_arg_start[old_index + 1];
    //
    // hash
    size_t hash  = (end - start) + (size_t(op_enum) << 8);
    for(size_t arg_index = start; arg_index < end; ++arg_index) {
        ad_type_t  arg_type   = agraph_old.m_arg_type[arg_index];
        size_t     arg_value  = agraph_old.m_arg_value[arg_index];
        if( arg_type == ad_type ) {
            arg_value = old2new[arg_value];
        }
        //
        hash ^= size_t(arg_type)  + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= arg_value         + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}
//
// new_op_equal
bool new_op_equal(
    ad_type_t            ad_type,
    const agraph_t&      agraph_old,
    const vector<size_t> old2new ,
    size_t               old_index_1 ,
    size_t               old_index_2 ) {
    //
    // op_enum, narg, start_1
    op_enum_t op_enum = agraph_old.m_op_seq[old_index_1];
    size_t start_1 = agraph_old.m_arg_start[old_index_1];
    size_t end_1   = agraph_old.m_arg_start[old_index_1 + 1];
    size_t n_arg   = end_1 - start_1;
    //
    // start_2
    size_t start_2 = agraph_old.m_arg_start[old_index_2];
    size_t end_2   = agraph_old.m_arg_start[old_index_2 + 1];
    if( op_enum != agraph_old.m_op_seq[old_index_2] ) {
        return false;
    }
    if( n_arg != end_2 - start_2 ) {
        return false;
    }
    // i_arg
    for(size_t i_arg = 0; i_arg < n_arg; ++i_arg) {
        //
        ad_type_t  arg_type = agraph_old.m_arg_type[start_1 + i_arg];
        if( arg_type != agraph_old.m_arg_type[start_2 + i_arg] ) {
            return false;
        }
        size_t arg_value_1  = agraph_old.m_arg_value[start_1 + i_arg];
        size_t arg_value_2  = agraph_old.m_arg_value[start_2 + i_arg];
        if( arg_type == ad_type ) {
            arg_value_1 = old2new[arg_value_1];
            arg_value_2 = old2new[arg_value_2];
        }
        if( arg_value_1 != arg_value_2 ) {
            return false;
        }
    }
    return true;
}
} } // END_AD_TENSOR_DEV_NAMESPACE
