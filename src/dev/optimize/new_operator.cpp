// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin new_op_hash dev}
{xrst_spell
    agraph
}

The Operator Hash Code Function
###############################

Syntax
******
{xrst_code cpp}
    hash = new_op_hash(ad_type, agraph_old, old2new, old_index)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_NEW_OP_HASH, END_NEW_OP_HASH
}

ad_type
*******
This is the type of agraph_old and must be either parameter or variable.

agraph_old
**********
This is the old graph that is being optimized.

old2new
*******
The idea here is that if two operators have the same index in the new graph,
they can be treated as equivalent operators in the old graph.
Suppose that

#.  agraph_old.m_arg_start[old_index] < arg_index
#.  arg_index <  agraph_old.m_arg_start[old_index + 1]
#.  agraph_old.m_arg_type[arg_index] == ad_type
#.  op_index = agraph_old.m_arg_value[arg_index]

Then op_index < old_index and
old2new[op_index] is the index in the new graph of an operator
that is equivalent to the operator with index op_index in the old graph.

old_index
*********
This is the index of the operator that we are computing the hash code for.
Note that all of its arguments to this operator with type ad_type have indices
less than old_index (and hence old2new is defined for all such arguments).

hash
****
If two operators are :ref:`new_op_equal-name` ,
then will have the same hash code.
Otherwise, hash tries to separate operators.

{xrst_end new_op_hash}
------------------------------------------------------------------------------
{xrst_begin new_op_equal dev}
{xrst_spell
    agraph
}

Check if Two Operators are Equal in the Optimized Graph
#######################################################

Syntax
******
{xrst_code cpp}
    equal = new_op_equal(ad_type, agraph_old, old2new, old_index_1, old_index_2
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_NEW_OP_EQUAL, END_NEW_OP_EQUAL
}

ad_type
*******
This is the type of agraph_old and must be either parameter or variable.

agraph_old
**********
This is the old graph that is being optimized.

old2new
*******
The idea here is that if two operators have the same index in the new graph,
they can be treated as equivalent operators in the old graph.

Suppose old_index <= old_index_1 and old_index <= old_index_2 and

#.  agraph_old.m_arg_start[old_index] < arg_index
#.  arg_index <  agraph_old.m_arg_start[old_index + 1]
#.  agraph_old.m_arg_type[arg_index] == ad_type
#.  op_index = agraph_old.m_arg_value[arg_index]

Then op_index < old_index_1, op_index < old_index_2 and
old2new[op_index] is the index in the new graph of an operator
that is equivalent to the operator with index op_index in the old graph.


old_index_1, old_index_2
************************
are the indices of the operators on the old graph
that we are checking for equivalence; i.e,.
are all their arguments the same the new graph.

equal
*****
The return value is true if the :ref:`op_enum-name` is the
same and all the operator arguments are the same (in the new graph).
An operator's arguments are m_arg_type[arg_index] and m_arg_value[arg_index]
for all the arg_index values corresponding to an operator.

{xrst_end new_op_equal}
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
// BEGIN_NEW_OP_HASH
// stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
size_t new_op_hash(
    ad_type_t            ad_type,
    const agraph_t&      agraph_old,
    const vector<size_t> old2new,
    size_t               old_index )
{   // END_NEW_OP_HASH
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
// BEGIN_NEW_OP_EQUAL
bool new_op_equal(
    ad_type_t            ad_type,
    const agraph_t&      agraph_old,
    const vector<size_t> old2new ,
    size_t               old_index_1 ,
    size_t               old_index_2 )
{   // END_NEW_OP_EQUAL
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
