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
//
// new_agraph
// auto [agraph_new, rng_index_new] = new_agraph(
//      agraph_type, agraph_old, rng_index_old, rng_ad_type_old, depend_old
// )
std::tuple< agraph_t, vector<size_t> > new_agraph(
    ad_type_t                agraph_type     ,
    const agraph_t&          agraph_old      ,
    const vector<size_t>     rng_index_old   ,
    const vector<ad_type_t>  rng_ad_type_old ,
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
    size_t op_index_old = 0;
    while(op_index_old < n_old ) { if( depend_old[op_index_old] ) {
        // op_enum
        op_enum_t op_enum = agraph_old.m_op_seq[op_index_old];
        switch( op_enum ) {
            // --------------------------------------------------------------
            // agraph_new
            case op_enum_t::call:
            case op_enum_t::call_result:
            //
            // op_index_old
            ++op_index_old;
            break;
            //
            // ---------------------------------------------------------------
            default:
            //
            // hash, itr, replace
            hash = hash_operator(
                agraph_type, agraph_old, old2new, op_index_old
            );
            itr       = hash2old_agraph.find(hash);
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
                //
                // agraph_new: m_op_seq, m_arg_start
                agraph_new.m_op_seq.push_back(op_enum);
                agraph_new.m_arg_start.push_back(
                        agraph_new.m_arg_value.size()
                );
                //
                // arg_index
                size_t start = agraph_old.m_arg_start[op_index_old];
                size_t end   = agraph_old.m_arg_start[op_index_old + 1];
                for(size_t arg_index = start; arg_index < end; ++arg_index) {
                    //
                    // agraph_new:: m_arg_type, m_arg_value
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
    //
    // rng_index_new
    vector<size_t> rng_index_new = rng_index_old;
    for(size_t i = 0; i < rng_index_old.size(); ++i) {
        if( rng_ad_type_old[i] == agraph_type ) {
            rng_index_new[i] = old2new[ rng_index_old[i] ];
        }
    }
    return std::tuple< agraph_t, vector<size_t> >(agraph_new, rng_index_new);
}
} } // END_AD_TENSOR_DEV_NAMESPACE
