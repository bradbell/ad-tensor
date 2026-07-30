// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
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
// hash_operator
// stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
size_t hash_operator(
    ad_type_t            ad_type,
    const agraph_t&      agraph,
    const vector<size_t> old2new,
    size_t               op_index ) {
    //
    // op_enum, start, end
    op_enum_t op_enum  = agraph.m_op_seq[op_index];
    size_t start       = agraph.m_arg_start[op_index];
    size_t end         = agraph.m_arg_start[op_index + 1];
    //
    // hash
    size_t hash  = (end - start) + (size_t(op_enum) << 8);
    for(size_t arg_index = start; arg_index < end; ++arg_index) {
        ad_type_t  arg_type   = agraph.m_arg_type[arg_index];
        size_t     arg_value  = agraph.m_arg_value[arg_index];
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
    const agraph_t&      agraph,
    const vector<size_t> old2new ,
    size_t               op_index_1 ,
    size_t               op_index_2 ) {
    //
    // op_enum, narg, start_1
    op_enum_t op_enum = agraph.m_op_seq[op_index_1];
    size_t start_1 = agraph.m_arg_start[op_index_1];
    size_t end_1   = agraph.m_arg_start[op_index_1 + 1];
    size_t n_arg   = end_1 - start_1;
    //
    // start_2
    size_t start_2 = agraph.m_arg_start[op_index_2];
    size_t end_2   = agraph.m_arg_start[op_index_2 + 1];
    if( op_enum != agraph.m_op_seq[op_index_2] ) {
        return false;
    }
    if( n_arg != end_2 - start_2 ) {
        return false;
    }
    // i_arg
    for(size_t i_arg = 0; i_arg < n_arg; ++i_arg) {
        //
        ad_type_t  arg_type = agraph.m_arg_type[start_1 + i_arg];
        if( arg_type != agraph.m_arg_type[start_2 + i_arg] ) {
            return false;
        }
        size_t arg_value_1  = agraph.m_arg_value[start_1 + i_arg];
        size_t arg_value_2  = agraph.m_arg_value[start_2 + i_arg];
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
