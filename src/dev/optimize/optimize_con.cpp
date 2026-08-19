// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin optimize_con dev}{
{xrst_spell
    adfn
}

Remove Unnecessary Constants From an AD function
################################################

Syntax
******
{xrst_code cpp}
    adfn.optimized_con( depend_old )
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_OPTIMIZE_CON, END_OPTIMIZE_CON
}

depend_old
**********
If depend_old[i_old] is true (false) then the range values for adfn depend
on the constant with index i_old in the input value of adfn.m_con.
The size of this vector is equal the size of the input value of adfn.m_con.

adfn
****
This is the function we are removing unnecessary constants from.

m_con
=====
The unnecessary constants are removed from this vector.
It new size is less than or equal its old size.

m_par.m_arg_value, m_var.arg_value
==================================
The entries in these vectors, that have m_arg_type constant,
get mapped from their index in the old m_con to their index in the new m_con.

Test
****
{xrst_literal ,
    tests/adfn/optimize.cpp
    BEGIN_ADFN_OPTIMIZE_CON, END_ADFN_OPTIMIZE_CON
}

{xrst_end optimize_con}
*/
#include <map>
#include <ad_tensor/adten.hpp>
//
namespace ad_tensor { // BEGIN_AD_TENSOR_NAMESPACE
//
// BEGIN_OPTIMIZE_CON
void adfn_t::optimize_con(const vector<bool>& depend_old)
{   // END_OPTIMIZE_CON
    //
    // hash2new_con, const_iterator
    // multimap would be more complicated, but it might be better when
    // collisions occur.
    std::map<int64_t, size_t>                         hash2new_con;
    typedef std::map<int64_t, size_t>::const_iterator const_iterator;
    //
    // n_old
    size_t n_old = depend_old.size();
    assert( n_old == m_con.size() );
    //
    // not_used
    size_t not_used = std::numeric_limits<size_t>::max();
    //
    // old2new_con
    vector<size_t> old2new_con(n_old, not_used);
    //
    // old2new_con[0], hash2new_con[0]
    // always keep the undefined tensor at constant index 0
    assert( ! m_con[0].defined() );
    old2new_con[0]  = 0;
    hash2new_con[0] = 0;
    //
    // n_new, i_old
    size_t n_new = 1;
    for(size_t i_old = 1; i_old < n_old; ++i_old) { if( depend_old[i_old] ) {
        assert( n_new <= i_old );
        //
        // hash
        int64_t hash = 0;
        if( m_con[i_old].defined() ) {
            hash = torch::hash_tensor( m_con[i_old] ).item<int64_t>();
        }
        //
        //
        // itr, replace
        const_iterator itr     = hash2new_con.find(hash);
        bool           replace = false;
        if( hash != 0 && itr != hash2new_con.end() ) {
            size_t i_new = itr->second;
            assert( i_new < n_new );
            replace = m_con[i_old].equal( m_con[i_new] );
        }
        //
        if( replace ) {
            // old2new_con[i_old]
            old2new_con[i_old] = itr->second;
        } else {
            // old2new[i_old], m_con[n_new], hash2new_con[hash]
            assert( n_new <= i_old );
            if( n_new < i_old ) {
                m_con[n_new] = m_con[i_old];
            }
            old2new_con[i_old] = n_new;
            if( hash != 0 ) {
                hash2new_con[hash] = n_new;
            }
            ++n_new;
        }
    } }
    // m_con
    // shrink to fit is non-binding, so ensure this memory gets freed
    for(size_t i_old = n_new; i_old < n_old; ++i_old) {
        m_con[i_old] = at::Tensor();
    }
    m_con.resize(n_new);
    //
    // agraph
    dev::agraph_t* agraph = nullptr;
    for(size_t ig = 0; ig < 2; ++ig) {
        if( ig == 0 ) {
            agraph = &m_par;
        } else {
            agraph = &m_var;
        }
        //
        // agraph->m_arg_value
        size_t n_op = agraph->m_op_seq.size();
        for(size_t op_index = 0; op_index < n_op; ++op_index) {
            //
            // arg_index
            size_t start = agraph->m_arg_start[op_index];
            size_t end   = agraph->m_arg_start[op_index + 1];
            for(size_t arg_index = start; arg_index < end; ++arg_index) {
                //
                // arg_type
                adtype_t arg_type = agraph->m_arg_type[arg_index];
                if( arg_type == adtype_t::constant ) {
                    //
                    // argraph->m_arg_value[arg_index]
                    size_t arg_value = agraph->m_arg_value[arg_index];
                    arg_value        = old2new_con[arg_value];
                    agraph->m_arg_value[arg_index] = arg_value;
                    assert( arg_value < n_new );
                }
            }
        }
    }
    //
    // m_rng_index
    for(size_t i = 0; i < m_rng_index.size(); ++i) {
        if( m_rng_adtype[i] == adtype_t::constant ) {
            m_rng_index[i] = old2new_con[ m_rng_index[i] ];
        }
    }
}
} // END_AD_TENSOR_NAMESPACE
