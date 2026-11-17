// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin optimize_int64 dev}{
{xrst_spell
    adfn
    op
    seq
}

Remove Unnecessary Integer Vectors From an AD function
######################################################

Syntax
******
{xrst_code cpp}
    adfn.optimize_int64( depend_int64, depend_par, depend_var )
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_OPTIMIZE_INT64, END_OPTIMIZE_INT64
}

depend_int64
************
If depend_int64[i_old] is true (false) then the range values for adfn depend
on the sub-vector that starts at index i_old in the input value of
:ref:`adfn_dev@m_int64_vec` .

depend_par
**********
If depend_par[op_index] is true (false) then the range values for adfn depend
on the parameter with index op_seq in the value of adfn.m_par.

depend_var
**********
If depend_var[op_index] is true (false) then the range values for adfn depend
on the variable with index op_seq in the value of adfn.m_var.

adfn
****
This is the function we are removing unnecessary constants from.

m_int64_vec
===========
The unnecessary integer sub-vectors are removed from this vector.
It new size is less than or equal its old size.

m_par.m_arg_value, m_var.arg_value
==================================
The entries in these vectors, that have m_arg_type int64vector,
get mapped from their index in the old m_con_vec to their index in the
new m_con_vec.

Test
****
{xrst_literal ,
    tests/adfn/optimize.cpp
    BEGIN_ADFN_OPTIMIZE_INT64, END_ADFN_OPTIMIZE_INT64
}

{xrst_end optimize_int64}
*/
#include <map>
#include <boost/functional/hash.hpp>
//
#include <ad_tensor/adten.hpp>
#include <ad_tensor/defined.hpp>
//
namespace ad_tensor { // Begin ad_tensor
//
// BEGIN_OPTIMIZE_INT64
void adfn_t::optimize_int64(
    const vector<bool>&    depend_int64 ,
    const vector<bool>&    depend_par   ,
    const vector<bool>&    depend_var   )
{   assert( depend_int64.size() == m_int64_vec.size() );
    assert( depend_par.size() == m_par.m_op_seq.size() );
    assert( depend_var.size() == m_var.m_op_seq.size() );
    // END_OPTIMIZE_INT64
    //
    // hash2new_int64, const_iterator
    // multimap would be more complicated, but it might be better when
    // collisions occur.
    std::map<size_t, size_t>                         hash2new_int64;
    typedef std::map<size_t, size_t>::const_iterator const_iterator;
    //
    // n_old
    size_t n_old = depend_int64.size();
    //
    // not_used
    size_t not_used = std::numeric_limits<size_t>::max();
    //
    // old2new_int64
    vector<size_t> old2new_int64(n_old, not_used);
    //
    // i_new, i_old
    size_t i_new = 0;
    size_t i_old = 0;
    while( i_old < n_old ) {
        assert( i_new <= i_old );
        //
        // length
        size_t   length  = size_t( m_int64_vec[i_old] );
        assert( 0 <= length );
        //
        if( depend_int64[i_old] ) {
            //
            // hash
            int64_t* begin   = m_int64_vec.data() + i_old + 1;
            int64_t* end     = m_int64_vec.data() + i_old + 1 + length;
            size_t   hash    = boost::hash_range(begin, end);
            //
            // match
            const_iterator itr   = hash2new_int64.find(hash);
            bool           match = false;
            if( itr != hash2new_int64.end() ) {
                size_t i_previous = itr->second;
                assert( i_previous < i_new );
                match = true;
                for(size_t i = 0; i <= length; ++i) {
                    match &= m_int64_vec[i_old + i] == m_int64_vec[i_new + i];
                }
            }
            //
            // old2new_int64, m_int64_vec, i_new, i_old
            if( match ) {
                // old2new_int64[i_old]
                old2new_int64[i_old] = itr->second;
            } else {
                // old2new[i_old], m_int64_vec[i_new], hash2new_int64[hash]
                old2new_int64[i_old] = i_new;
                hash2new_int64[hash] = i_new;
                if( i_new < i_old ) {
                    for(size_t i = 0; i <= length; ++i) {
                        m_int64_vec[i_new + i] = m_int64_vec[i_old + i];
                    }
                }
                i_new += length + 1;
            }
        }
        i_old += length + 1;
    }
    //
    // m_int64_vec
    // TODO: Perhaps we should shrink_to_fit instead of resize ?
    size_t n_new = i_new;
    m_int64_vec.resize(n_new);
    //
    // agraph
    dev::agraph_t*      agraph        = nullptr;
    const vector<bool>* depend_agraph = nullptr;
    for(size_t ig = 0; ig < 2; ++ig) {
        if( ig == 0 ) {
            agraph        = &m_par;
            depend_agraph = &depend_par;
        } else {
            agraph = &m_var;
            depend_agraph = &depend_var;
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
                // argraph->m_arg_value[arg_index]
                vitype_t arg_type = agraph->m_arg_type[arg_index];
                if( arg_type == vitype_t::int64vector ) {
                    if( ! (*depend_agraph)[op_index] ) {
                        agraph->m_arg_value[arg_index] = not_used;
                    } else {
                        size_t old_value = agraph->m_arg_value[arg_index];
                        size_t new_value = old2new_int64[old_value];
                        agraph->m_arg_value[arg_index] = new_value;
                        assert( new_value <= old_value );
                        assert( new_value < n_new );
                    }
                }
            }
        }
    }
}
} // End ad_tensor::dev
